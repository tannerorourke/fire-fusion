"""
Spatiotemporal ConvFormer:
- dynamic weather and land-state channels run through the encoder and its attention stack.
- static terrain and settlement maps condition that path through FiLM.
"""
from typing import Dict
import torch
import torch.nn as nn
from .modules import (
    SpatialEncoder, StaticFiLMBranch,
    WindowedSpatialAttention, ChannelMixingAttention, TemporalMixingAttention,
    BiHeadDecoder
)


class FireFusionModel(nn.Module):
    """
    Given dynamic feature channels (C_dyn) over T days plus static maps, compute the
    risk of wildfire ignition across a H x W grid.

    Steps:
        - Turn the static maps and the date into per-level FiLM (gamma, beta)
        - Encode (downsample) Spatial patterns with basic ResNet MLP-style CNN encoder,
          one stem per dynamic modality group, FiLM applied at each resolution
        - Run self-attention over larger HxW windows than Encoder (these generalize well)
        - Run self-attention over channels (features)
        - Run self-attention over time
        - Decode (upsample) into a (B, 1, H, W) grid

    Modality dropout trains the model on random subsets of its input groups; a
    product outage at inference degrades the prediction instead of voiding it.
    """
    def __init__(self, dyn_channels: int, static_channels: int, mp: Dict):
        super().__init__()
        ws_params   =mp["win_spatial_mixing"]
        cm_params   =mp["channel_mixing"]
        tm_params   =mp["temporal_mixing"]
        
        embed_dim   =mp["embed_dim"]

        ws_heads    =ws_params['num_heads']
        ws_win_size =ws_params['window_size']
        ws_dropout  =ws_params['dropout'];

        cm_heads    =cm_params['num_heads']
        cm_d_model  =cm_params['d_model']
        cm_mlp_ratio=cm_params['mlp_ratio']
        cm_dropout  =cm_params['dropout']

        tm_heads    =tm_params['num_heads']
        tm_mlp_ratio=tm_params['mlp_ratio']
        tm_dropout  =tm_params['dropout']

        cm_chunk    =cm_params.get('chunk_size', 4096)
        n_causes    =mp['n_cause_classes']
        depth       =mp.get('encoder_depth', 1)

        dyn_groups  =mp['dyn_groups']
        film_width  =mp['static_film']['width']
        self.modality_dropout = mp['modality_dropout']

        self.encoder = SpatialEncoder(dyn_channels, embed_dim, dyn_groups, depth=depth)
        # -- the trailing static channel is the date plane, consumed as a scalar
        self.static_branch = StaticFiLMBranch(static_channels - 1, self.encoder.base_ch, embed_dim,
                                              depth, width=film_width)
        self.ws_attn = WindowedSpatialAttention(embed_dim, num_heads=ws_heads, window_size=ws_win_size, dropout=ws_dropout)
        self.cm_attn = ChannelMixingAttention(num_heads=cm_heads, num_channels=embed_dim, d_model=cm_d_model, mlp_ratio=cm_mlp_ratio, dropout=cm_dropout, chunk_size=cm_chunk)
        self.tm_attn = TemporalMixingAttention(embed_dim, num_heads=tm_heads, mlp_ratio=tm_mlp_ratio, dropout=tm_dropout)
        self.decoder = BiHeadDecoder(embed_dim, n_cause_classes=n_causes, depth=depth, base_ch=self.encoder.base_ch)

        # The backbone ("main") produces the shared representation; the decoder
        # ("heads") turns it into the ignition and cause maps. Grouping them here
        # lets a training stage freeze one group and specialize the other.
        self._main_modules = [self.encoder, self.static_branch, self.ws_attn, self.cm_attn, self.tm_attn]
        self._head_modules = [self.decoder]
        self._frozen_main = False
        self._frozen_heads = False

    def set_frozen(self, freeze_main: bool = False, freeze_heads: bool = False):
        """ Toggle gradient flow for the backbone and decoder groups.

            - a frozen group also switches to eval; dropout stays fixed while
              the other group trains
        """
        self._frozen_main = freeze_main
        self._frozen_heads = freeze_heads

        for module in self._main_modules:
            for p in module.parameters():
                p.requires_grad = not freeze_main
        for module in self._head_modules:
            for p in module.parameters():
                p.requires_grad = not freeze_heads

        # re-assert train/eval; a freeze applied mid-run takes effect at once
        self.train(self.training)
        return self

    def train(self, mode: bool = True):
        """ Set train/eval mode, keeping any frozen backbone or decoder group in eval. """
        super().train(mode)
        if self._frozen_main:
            for module in self._main_modules:
                module.eval()
        if self._frozen_heads:
            for module in self._head_modules:
                module.eval()
        return self

    def _drop_masks(self, B: int, device: torch.device):
        """ Sample a per-group drop mask and a per-sample keep mask for modality dropout,
            or (None, None) outside training or when `modality_dropout` is 0.
        """
        p = self.modality_dropout
        if not (self.training and p > 0):
            return None, None
        drop = {g: torch.rand(B, device=device) < p for g in self.encoder.group_names}
        keep = (torch.rand(B, device=device) >= p).float()
        return drop, keep

    def forward(self, x_dyn: torch.Tensor, x_static: torch.Tensor):
        """ (B, T, C_dyn, H, W) dynamic input plus (B, C_static, H, W) static maps -> (B, 1, H, W)
            ignition logits and (B, n_cause_classes, H, W) cause logits.
        """
        drop, keep = self._drop_masks(x_dyn.shape[0], x_dyn.device)

        # -- the date plane is spatially constant; the mean recovers it exactly
        doy = x_static[:, -1].mean(dim=(-2, -1))
        film = self.static_branch(x_static[:, :-1], doy, keep=keep)

        y, skips = self.encoder(x_dyn, film=film, drop=drop)
        y = self.ws_attn(y)
        y = self.cm_attn(y)
        y = self.tm_attn(y)

        # Only decode the prediction from the last day
        y = y[:, -1]

        outputs = self.decoder(y, skips, film=film)
        return outputs


