"""
Spatiotemporal ConvFormer building blocks: 
Per-group spatial encoder, Static FiLM branch that conditions it, 
spatial/channel/temporal attention blocks, and the two-headed decoder.

The dynamic path has one stem per modality group; a missing group is a learned
token. The static path enters as FiLM at every encoder resolution.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """ Return a GroupNorm sized to the largest group count dividing `channels`;
        it carries no running statistics.
    """
    return nn.GroupNorm(math.gcd(channels, max_groups), channels)


class ConvResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size = 3, stride = 1, padding = 1, dropout = 0.0):
        """ Build a two-conv residual block with GroupNorm, projecting the identity
            when the residual and trunk shapes disagree.
        """
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.norm1 = group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=1, bias=False)
        self.norm2 = group_norm(out_ch)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace=True)

        # only projected when the residual and trunk shapes disagree
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                group_norm(out_ch),
            )
            if stride != 1 or (in_ch != out_ch)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ (N, in_ch, H, W) -> (N, out_ch, H', W')
            two convolutions w/GroupNorm + residual, downsampling by `stride`.
        """
        identity = x
        out = self.norm1(self.conv1(x))
        out = self.relu(out)
        out = self.norm2(self.conv2(out))
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)
        return out


FilmLevels = Sequence[Tuple[torch.Tensor, torch.Tensor]]


def apply_film(x: torch.Tensor, level: Tuple[torch.Tensor, torch.Tensor], T: int = 1) -> torch.Tensor:
    """ gamma * x + beta, with a per-sample pair shared across a folded time axis. """
    gamma, beta = level
    B = gamma.shape[0]
    out = gamma.unsqueeze(1) * x.view(B, T, *x.shape[1:]) + beta.unsqueeze(1)
    return out.reshape(B * T, *out.shape[2:])


class SpatialEncoder(nn.Module):
    """
    CNN with Residual Blocks over (H x W), extracting spatial features
    per time step T (we call H' and W')

    Shape: (B, T, C_dyn, H, W) --> (B, T, embed_dim, H', W')

    `dyn_groups`: modality group -> channel indices. One stem per group; a
                  group's absence is a learned token (zeros are a legal
                  normalized value).

    `depth`: num of stride-2 stages, each doubling RF (in grid cells)
             A dataset at 1/2 ground resolution needs one more stage to reach
             the same distance in kilometres. Results are compared at a fixed
             distance across resolutions.
    """
    def __init__(self, dyn_channels, embed_dim, dyn_groups: Dict[str, List[int]], depth: int = 1):
        """ Build per-group stems, missing-group tokens, and the stride-2 encoder stages. """
        super().__init__()

        self.base_ch            = 64
        self.depth              = depth
        self.down1_dropout      = 0.01
        self.down2_dropout      = 0.01

        self.group_names = list(dyn_groups)
        self.group_idx = {g: list(dyn_groups[g]) for g in self.group_names}
        assert sum(len(i) for i in self.group_idx.values()) == dyn_channels, \
            "SpatialEncoder: dyn_groups must cover every dynamic channel exactly once"

        # -- remainders spread over the leading groups; the concatenation is
        #    exactly base_ch wide for any group count
        n = len(self.group_names)
        widths = {g: self.base_ch // n + (i < self.base_ch % n) for i, g in enumerate(self.group_names)}

        self.stems = nn.ModuleDict({
            g: nn.Sequential(
                nn.Conv2d(len(self.group_idx[g]), widths[g], kernel_size=3, padding=1, bias=False),
                group_norm(widths[g]),
                nn.ReLU(inplace=True)
            )
            for g in self.group_names
        })
        # -- group identity, and the vector a dropped group is represented by
        self.group_embed = nn.ParameterDict({g: nn.Parameter(torch.zeros(widths[g])) for g in self.group_names})
        self.missing_token = nn.ParameterDict({
            g: nn.Parameter(torch.randn(widths[g]) * 0.02) for g in self.group_names
        })

        self.down1 = nn.Sequential(
            ConvResidualBlock(self.base_ch, self.base_ch, stride=1, dropout=self.down1_dropout),
            ConvResidualBlock(self.base_ch, self.base_ch, stride=1, dropout=self.down1_dropout)
        )

        self.stages = nn.ModuleList([
            nn.Sequential(
                ConvResidualBlock(self.base_ch if i == 0 else embed_dim, embed_dim,
                                  stride=2, dropout=self.down2_dropout),
                ConvResidualBlock(embed_dim, embed_dim, stride=1, dropout=self.down2_dropout)
            )
            for i in range(depth)
        ])

    def _stem(self, x: torch.Tensor, B: int, T: int, drop: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        feats = []
        for g in self.group_names:
            h = self.stems[g](x[:, self.group_idx[g]]) + self.group_embed[g].view(1, -1, 1, 1)
            if drop is not None:
                # -- a dropped group is substituted wholesale, every day alike
                gone = drop[g].view(B, 1, 1, 1, 1).expand(B, T, 1, 1, 1).reshape(B*T, 1, 1, 1)
                h = torch.where(gone, self.missing_token[g].view(1, -1, 1, 1), h)
            feats.append(h)
        return torch.cat(feats, dim=1)

    def forward(self, x: torch.Tensor, film: Optional[FilmLevels] = None,
                drop: Optional[Dict[str, torch.Tensor]] = None):
        """ (B, T, C_dyn, H, W) -> (B, T, embed_dim, H', W'), plus the per-resolution
            feature maps for the final day as skip connections.
        """
        B, T, C, H, W = x.shape

        # -- every day is encoded independently; T rides along in the batch axis
        out = self._stem(x.reshape(B*T, C, H, W), B, T, drop)
        out = self.down1(out)
        if film is not None:
            out = apply_film(out, film[0], T)

        # -- the resolution entering each stage, kept for the decoder to fuse; only
        # -- the predicted day is retained (all the decoder consumes)
        skips = [out.view(B, T, *out.shape[1:])[:, -1]]
        for i, stage in enumerate(self.stages):
            out = stage(out)
            if film is not None:
                out = apply_film(out, film[i + 1], T)
            if i < self.depth - 1:
                skips.append(out.view(B, T, *out.shape[1:])[:, -1])

        e_dim, Hp, Wp = out.shape[1], out.shape[2], out.shape[3]
        return out.reshape(B, T, e_dim, Hp, Wp), skips


class StaticFiLMBranch(nn.Module):
    """
    Terrain, settlement and date, turned into per-level FiLM for the trunk.

    Input:  static maps (B, C_s, H, W) and a day-of-year scalar (B,)
    Output: [(gamma, beta)] per encoder level, full resolution first

    Terrain does not change across a look-back window; the temporal path would
    carry T copies of a constant. FiLM modulation of the dynamic features keeps
    the interaction (a wind on a steep south slope) without the duplication.
    Heads are zero-initialized: conditioning starts at identity and the branch
    gains influence as gradient arrives.
    """
    def __init__(self, static_channels: int, base_ch: int, embed_dim: int, depth: int, width: int = 32):
        """ Build the trunk, stride-2 downsampling levels, and zero-initialized
            per-level FiLM heads. """
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(static_channels, width, kernel_size=3, padding=1, bias=False),
            group_norm(width),
            nn.GELU(),
        )
        # -- mirrors the encoder's stride-2 stages; every level lands on the
        #    same extent as the map it modulates, odd sizes included
        self.downs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1, bias=False),
                group_norm(width),
                nn.GELU(),
            )
            for _ in range(depth)
        ])

        level_ch = [base_ch] + [embed_dim] * depth
        # -- the date enters as a channel; seasonality reweights the same map
        self.heads = nn.ModuleList([nn.Conv2d(width + 1, 2 * c, kernel_size=1) for c in level_ch])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, static: torch.Tensor, doy: torch.Tensor,
                keep: Optional[torch.Tensor] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """ (B, C_s, H, W) static maps and (B,) day-of-year -> per-level (gamma, beta) FiLM
            pairs, zeroed by `keep` for a sample whose static input dropped.
        """
        f = self.trunk(static)
        levels = [f]
        for down in self.downs:
            f = down(f)
            levels.append(f)

        out = []
        for head, f in zip(self.heads, levels):
            date_plane = doy.view(-1, 1, 1, 1).expand(-1, 1, *f.shape[-2:])
            params = head(torch.cat([f, date_plane], dim=1))
            if keep is not None:
                params = params * keep.view(-1, 1, 1, 1)
            gamma_offset, beta = params.chunk(2, dim=1)
            out.append((1.0 + gamma_offset, beta))
        return out


class WindowedSpatialAttention(nn.Module):
    """
    Windowed spatial self-attention (Owerko et al. 2024, https://arxiv.org/html/2306.08191v2)
    Mixes Spatial attributes (H' x W') at a larger resolution than H' and W'

    Shape:  (B, T, C, H', W') --> (B, T, C, H', W') (no change)
    
    Tokens within a window carry a learned position. Attention is permutation
    equivariant; without one the block cannot tell upslope from downslope
    within its window.

    The feature map is padded up to a whole number of windows and cropped after;
    window_size need not divide the grid. Padded tokens masked out.
    """
    def __init__(self, embed_dim, num_heads, window_size, dropout):
        """ Build the window attention module with a learned per-window position embedding. """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn_chunk = 32768

        self.pos_embed = nn.Parameter(torch.randn(window_size * window_size, embed_dim) * 0.02)

        self.norm = nn.LayerNorm(embed_dim)
        self.window_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ (B, T, C, H', W') -> (B, T, C, H', W'); pads to a whole number of windows,
            runs windowed self-attention in chunks, and crops back to the input extent.
        """
        B, T, C, Hp, Wp = x.shape
        ws = self.window_size

        x = x.permute(0, 1, 3, 4, 2).contiguous().view(B*T, Hp, Wp, C)

        pad_h = (ws - Hp % ws) % ws
        pad_w = (ws - Wp % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hpad, Wpad = Hp + pad_h, Wp + pad_w
        nH, nW = Hpad // ws, Wpad // ws

        x_windows = x.view(B*T, nH, ws, nW, ws, C).permute(0, 1, 3, 2, 4, 5)
        x_windows = x_windows.reshape(B*T*nH*nW, ws*ws, C)

        pad_mask = None
        if pad_h or pad_w:
            valid = torch.zeros(Hpad, Wpad, dtype=torch.bool, device=x.device)
            valid[:Hp, :Wp] = True
            valid = valid.view(nH, ws, nW, ws).permute(0, 2, 1, 3).reshape(nH*nW, ws*ws)
            # -- a window entirely outside the grid masks every key and yields NaN
            valid[~valid.any(dim=1)] = True
            pad_mask = (~valid).repeat(B*T, 1)

        x_w = x_windows
        x_norm = self.norm(x_windows) + self.pos_embed

        # -- SDPA maps the window-batch axis onto a CUDA grid dimension capped at
        #    65535; a full-grid batch exceeds it. Attention runs in chunks
        outs = []
        for i in range(0, x_norm.shape[0], self.attn_chunk):
            chunk = x_norm[i:i + self.attn_chunk]
            m = pad_mask[i:i + self.attn_chunk] if pad_mask is not None else None
            # need_weights=False keeps the attention matrix unmaterialized
            # ++ lets torch dispatch its fused kernels; the weights are discarded regardless
            o, _ = self.window_attn(chunk, chunk, chunk, need_weights=False, key_padding_mask=m)
            outs.append(o)
        out = self.proj(torch.cat(outs, dim=0)) + x_w

        out = out.view(B*T, nH, nW, ws, ws, C).permute(0, 1, 3, 2, 4, 5)
        out = out.reshape(B*T, Hpad, Wpad, C)[:, :Hp, :Wp]
        return out.view(B, T, Hp, Wp, C).permute(0, 1, 4, 2, 3).contiguous()


class ChannelMixingAttention(nn.Module):
    """
    Multi-Head Attention over CHANNELS, for fixed (B, T, H', W')

    Shape: (B, T, embed_dim, H', W') --> (B, T, embed_dim, H', W') (no change)
    Steps:
        - Tokenizes channels d_model vectors (value scale + identity)
        - Apply MH self-attention over channels
        - Apply MLP
        - Project back to a scalar per channel

    Each (b, t, h', w') location is an independent attention problem; the
    N = B*T*H'*W' locations are processed in chunks. This block lifts every
    channel to a d_model vector, d_model times wider than the residual stream
    around it. Chunking bounds that peak without altering the result.
    """
    # The fused kernels map the leading chunk_size*num_heads onto a CUDA grid
    # dimension and the launch fails past its cap. Measured on sm_86: 196608
    # passes, 262144 raises 'invalid configuration argument'.
    MAX_ATTN_BATCH = 196608

    def __init__(self, num_channels, d_model, num_heads, mlp_ratio, dropout, chunk_size=4096):
        """ Build the per-channel tokenizer, attention block, and channel-mixing MLP,
            validating `chunk_size` against the CUDA batch cap.
        """
        super().__init__()
        if chunk_size * num_heads > self.MAX_ATTN_BATCH:
            raise ValueError(
                f"channel_mixing chunk_size {chunk_size} x num_heads {num_heads} = "
                f"{chunk_size * num_heads} exceeds the {self.MAX_ATTN_BATCH} attention "
                f"batch limit; cap chunk_size at {self.MAX_ATTN_BATCH // num_heads}"
            )
        self.num_channels = num_channels
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.dropout_p = dropout

        # Per-channel tokenizer: each channel gets its own Linear(1, d_model),
        # carrying channel identity into the permutation-equivariant attention.
        self.value_scale = nn.Parameter(torch.randn(num_channels, d_model) * 0.02)
        self.channel_embed = nn.Parameter(torch.randn(num_channels, d_model) * 0.02)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, 1)

        self.norm1 = nn.LayerNorm(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def _tokenize(self, x_chunk: torch.Tensor) -> torch.Tensor:
        """ (n, embed_dim) -> (n, embed_dim, d_model). """
        return x_chunk.unsqueeze(-1) * self.value_scale + self.channel_embed

    def _mix(self, h: torch.Tensor) -> torch.Tensor:
        """ (n, embed_dim, d_model) -> (n, embed_dim, d_model). """
        n, C, D = h.shape
        h_norm = self.norm1(h)

        qkv = self.qkv(h_norm).view(n, C, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        # scaled_dot_product_attention keeps the (C, C) attention matrix out of
        # memory; nn.MultiheadAttention materializes it by default
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )
        h = h + self.attn_proj(attn.transpose(1, 2).reshape(n, C, D))
        return h + self.mlp(self.norm2(h))

    def _block(self, x_chunk: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self._mix(self._tokenize(x_chunk))).squeeze(-1)

    def forward(self, x: torch.Tensor):
        """ (B, T, embed_dim, H', W') -> (B, T, embed_dim, H', W'); recomputes each chunk
            in backward under gradient checkpointing during training.
        """
        B, T, embed_dim, Hp, Wp = x.shape
        assert embed_dim == self.num_channels, "ChannelMixBlock: num_channels doesn't match incoming embed_dim"

        # -- one independent attention problem per spatial-temporal location
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(B*T*Hp*Wp, embed_dim)

        recompute = self.training and torch.is_grad_enabled()
        outs = []
        for i in range(0, x_flat.shape[0], self.chunk_size):
            x_chunk = x_flat[i:i + self.chunk_size]
            # checkpointing: each chunk is recomputed in backward; the widened
            # tokens are not retained for every chunk at once
            outs.append(
                checkpoint(self._block, x_chunk, use_reentrant=False)
                if recompute else self._block(x_chunk)
            )
        out_flat = torch.cat(outs, dim=0)

        # Reshape back to (B, T, embed_dim, H', W')
        return out_flat.view(B, T, Hp, Wp, embed_dim).permute(0, 1, 4, 2, 3).contiguous()


class TemporalMixingAttention(nn.Module):
    """
    Multi-Head Attention over time T, for fixed dims B, embed_dim, H', and W'

    Shape: (B, T, embed_dim, H', W') --> (B, T, embed_dim, H', W')

    Attention is permutation equivariant; without a positional term the block
    reads the look-back window as an unordered bag of days. Fire weather is a
    sequence (a drying trend, a wind buildup, rain three days ago versus rain
    yesterday). A learned per-day embedding added to the residual stream
    carries day identity.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout, max_window: int = 64):
        """ Build the per-day time embedding and the temporal self-attention block. """
        super().__init__()
        # indexed from the end of the window; position 0 is the day being
        # predicted from, for any window length
        self.time_embed = nn.Parameter(torch.randn(max_window, embed_dim) * 0.02)
        self.max_window = max_window
        self.attn_chunk = 32768

        self.norm = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(p=dropout/2)
        )

    def forward(self, f):
        """ (B, T, C, H', W') -> (B, T, C, H', W'); runs attention over time in chunks
            along the flattened pixel axis.
        """
        B, T, C, Hp, Wp = f.shape
        if T > self.max_window:
            raise ValueError(f"window of {T} days exceeds max_window={self.max_window}")

        # collapse B/H'/W': each pixel for each channel across time
        f_permute = f.permute(0, 3, 4, 1, 2).contiguous()
        x = f_permute.view(B*Hp*Wp, T, C)

        # -- reversed so the predicted-from day holds index 0 for any window length
        x = x + self.time_embed[:T].flip(0)

        # pre-norm: each sub-block normalizes its own input and leaves the
        # residual stream itself untouched, with a LayerNorm per sub-block
        x_norm = self.norm(x)
        # -- SDPA maps the pixel-batch axis onto a CUDA grid dimension capped at
        #    65535; a full-grid batch exceeds it. Attention runs in chunks
        outs = []
        for i in range(0, x_norm.shape[0], self.attn_chunk):
            chunk = x_norm[i:i + self.attn_chunk]
            o, _ = self.attn(chunk, chunk, chunk, need_weights=False)
            outs.append(o)
        x = x + torch.cat(outs, dim=0)

        out_ffn = self.mlp(self.norm2(x))
        x = x + out_ffn

        # --> back to (B, T, embed_dim, H', W')
        out = x.view(B, Hp, Wp, T, C).permute(0, 3, 4, 1, 2).contiguous() 
        
        
        return out
    

class BiHeadDecoder(nn.Module):
    """
    Convert spatiotemporal features into H x W risk map.
    Input:  (B, embed_dim, H', W'), time dimension collapsed to last day,
            plus the encoder's per-resolution feature maps for that day
    Output: (B, 1, H, W) and (B, num_classes, H, W) per two heads

    One upsample step inverts each encoder stride-2 stage, fusing the encoder
    map at that resolution. The fused maps carry structure below the stride
    the encoder ended on.

    Each step resizes to the shape of the map it fuses, not by a fixed factor;
    odd extents survive the round trip and one model trains on crops and
    predicts over the full domain.
    """
    def __init__(self, embed_dim, n_cause_classes: int, depth: int = 1,
                 base_ch: int = 64, head_ch: int = 64):
        """ Build the upsample-and-fuse stages, deepest first, plus the ignition
            and cause output heads.
        """
        super().__init__()
        self.n_cause_classes = n_cause_classes
        self.depth = depth

        skip_ch = [base_ch] + [embed_dim] * (depth - 1)
        in_ch = embed_dim
        fuse = []
        # -- built deepest first, matching the order the skips are consumed
        for level in reversed(range(depth)):
            out_ch = head_ch if level == 0 else embed_dim
            fuse.append(nn.Sequential(
                nn.Conv2d(in_ch + skip_ch[level], out_ch, 3, padding=1),
                nn.GELU(),
            ))
            in_ch = out_ch
        self.fuse = nn.ModuleList(fuse)

        self.ignition_head = nn.Conv2d(head_ch, 1, kernel_size=1)

        self.cause_head = nn.Conv2d(head_ch, self.n_cause_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, skips, film: Optional[FilmLevels] = None):
        """ (B, embed_dim, H', W') plus skip maps -> (B, 1, H, W) ignition logits and
            (B, n_cause_classes, H, W) cause logits.
        """
        f = x
        # -- deepest first; the level the encoder ended on was already conditioned there
        for i, (block, skip) in enumerate(zip(self.fuse, reversed(skips))):
            f = F.interpolate(f, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            f = block(torch.cat([f, skip], dim=1))
            if film is not None:
                f = apply_film(f, film[self.depth - 1 - i])

        ignition_logits = self.ignition_head(f) # (B, 1, H, W)
        cause_logits = self.cause_head(f)  # (B, num_classes, H, W)

        return ignition_logits, cause_logits
