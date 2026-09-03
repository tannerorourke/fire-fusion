"""
Splits store one pre-stacked (time, channel, y, x) float32 array "X" plus
per-day label and mask arrays. A sample splits X's channel axis into a
dynamic block (met + state channels, read across the full window) and a
static block (terrain, infrastructure, and vegetation-context channels plus
a day-of-year scalar plane, read once at the window's final day). 
Channel grouping comes from feature_config.channel_group_indices; window, crop.
Halo bookkeeping comes from the dataset's manifest.json.
"""
import json
import platform
import random
from typing import Dict, Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
import xarray as xr

from ..config.dataset_config import DatasetConfig, get_dataset_config
from ..config.feature_config import channel_group_indices


# -- Utility functions ------------------------------------------------------
def crop_align(encoder_depth: int, attn_window: int) -> int:
    """ A crop whose origin is not a multiple of their product shifts the window partition relative
        to a full-grid pass and changes the prediction for the same cell
    """
    
    return (2 ** encoder_depth) * attn_window


def crop_halo(encoder_depth: int, attn_window: int) -> int:
    """ Avoid edges training on padding by cropping a halo of context
        equal to the cell's receptive field.
        Radius: stem and residual 5, stride-2 stages 7*(2^d - 1), window 2^d*(ws - 1), decoder 2^(d+1) - 1
    """
    d, align = encoder_depth, crop_align(encoder_depth, attn_window)
    radius = 5 + 7 * (2 ** d - 1) + (2 ** d) * (attn_window - 1) + (2 ** (d + 1) - 1)
    return -(-radius // align) * align

from dask import config as daskconfig
# zarr reads happen inside DataLoader workers; nested dask threads only add overhead
daskconfig.set(scheduler='synchronous')


def _seed_worker(worker_id: int) -> None:
    """ A worker gets a pickled copy of the dataset.
        torch derives each worker's initial seed from the loader's generator.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

    info = get_worker_info()
    if info is not None:
        info.dataset._rng = np.random.default_rng(seed)

# --------------------------------------------------------------------------
class FireDataset(Dataset):
    """ Yields spatiotemporal windows as ((x_dyn, x_static), labels, masks):
        - x_dyn: (T, 26, H, W) float32, met + state channels across the window
        - x_static: (13, H, W) float32, terrain/infrastructure/vegetation-context
          channels at the window's final day, with a day-of-year scalar plane
          appended last
        - labels/masks: (H, W) at the window's final day (the prediction target
          is a fresh ignition within the forward horizon of that day)
    """
    def __init__(
        self,
        ds_config: DatasetConfig,
        split: Literal['train', 'eval', 'test'],
        window_size: int = 10,
        window_stride: int = 2,
        crop_size: int | None = None,
        crop_seed: int | None = None,
        encoder_depth: int = 1,
        attn_window: int = 2,
    ):
        """ Open a split's zarr store and derive channel groupings, crop geometry,
            and window starts from its manifest. """
        super().__init__()
        self.manifest = json.loads(ds_config.manifest_path.read_text())

        path = ds_config.split_path(split)
        print(f"opening >> {path}")
        self.ds = xr.open_zarr(path)
        self.X = self.ds["X"]

        self.feature_names = list(self.manifest["channels"])

        groups = channel_group_indices(self.feature_names)
        self._dyn_idx = sorted(groups["MET"] + groups["STATE"])
        self._static_idx = sorted(groups["STATIC"] + groups["QUASI_STATIC"])
        scalar_idx = groups["SCALAR"]
        assert len(scalar_idx) == 1, f"SCALAR group must be a single channel, got {scalar_idx}"
        self._scalar_idx = scalar_idx[0]
        self.dyn_channels = len(self._dyn_idx)
        self.static_channels = len(self._static_idx) + 1  # + the appended scalar plane

        # -- positions within x_dyn's channel axis of MET/STATE branches
        dyn_pos = {c: i for i, c in enumerate(self._dyn_idx)}
        self.dyn_groups = {
            name: sorted(dyn_pos[c] for c in groups[name]) for name in ("MET", "STATE")
        }

        self.label_names = list(self.manifest["labels"])
        self.mask_names = list(self.manifest["masks"])
        self.in_channels = int(self.manifest["in_channels"])
        self.out_size = (
            int(self.manifest["grid"]["height"]),
            int(self.manifest["grid"]["width"]),
        )
        # -- An extent that is not a multiple of the stride-window product partitions
        # -- raggedly in the last encoder stage, and its far edge sits past the last
        # -- legal crop origin. Every split trims to same aligned extent
        self.crop_align = crop_align(encoder_depth, attn_window)
        H, W = (n - n % self.crop_align for n in self.out_size)
        if (H, W) != self.out_size:
            print(f"trimming {self.out_size} -> {(H, W)} for alignment {self.crop_align}")
        self.out_size = (H, W)

        self.n_cause_classes = int(self.manifest["n_cause_classes"])
        self.ign_pos_weight = float(self.manifest["ign_pos_weight"])
        self.cause_counts = [int(c) for c in self.manifest["cause_counts"]]

        if self.X.sizes["channel"] != self.in_channels:
            raise ValueError(
                f"{path} has {self.X.sizes['channel']} channels; "
                f"manifest says {self.in_channels}"
            )

        self.window_size = window_size
        self.window_stride = window_stride
        self.n_timesteps = self.ds.sizes["time"]
        starts = np.arange(
            0, max(self.n_timesteps - window_size + 1, 0),
            window_stride,
            dtype=int,
        )
        # -- Consecuitive positions are not always consecutive days (seasonally windowed dataset).
        #    Straddling thje gap would present fire seasons as one sequence.
        if len(starts) and self.n_timesteps > 1:
            days = np.asarray(self.ds.indexes["time"], dtype="datetime64[D]")
            step = np.diff(days).astype(int)
            block = np.concatenate([[0], np.cumsum(step != 1)])
            keep = block[starts] == block[starts + window_size - 1]
            n_dropped = int((~keep).sum())
            if n_dropped:
                print(f"dropped {n_dropped} window(s) spanning a season gap")
            starts = starts[keep]
        self.window_starts = starts

        # ; The sample read is crop_size is the supervised extent plus a halo on every side
        self.crop_size = crop_size
        self.crop_halo = crop_halo(encoder_depth, attn_window)
        self._rng = np.random.default_rng(crop_seed)
        if crop_size is not None:
            if crop_size % self.crop_align:
                raise ValueError(f"crop_size {crop_size} must be a multiple of {self.crop_align}")
            self.read_size = crop_size + 2 * self.crop_halo
            if self.read_size > H or self.read_size > W:
                raise ValueError(
                    f"crop_size {crop_size} needs a {self.read_size}px read "
                    f"({self.crop_halo}px halo at encoder depth {encoder_depth}), "
                    f"larger than the {H}x{W} grid"
                )

    def __len__(self) -> int:
        return len(self.window_starts)

    def _crop_origin(self) -> Tuple[int, int]:
        """ Random (y, x) crop origin, aligned to crop_align, from the alignable
            range of the grid. """
        H, W = self.out_size
        align = self.crop_align
        y = self._rng.integers(0, (H - self.read_size) // align + 1) * align
        x = self._rng.integers(0, (W - self.read_size) // align + 1) * align
        return int(y), int(x)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict, Dict]:
        """ Read one window and return ((x_dyn, x_static), labels, masks).

            Uncropped reads take the full grid; a cropped read picks a random
            aligned origin, then zeroes halo cells out of every mask.
        """
        t0 = int(self.window_starts[idx])
        t1 = t0 + self.window_size
        last = t1 - 1

        H, W = self.out_size
        if self.crop_size is None:
            ysel, xsel = slice(0, H), slice(0, W)
            keep = None
        else:
            y0, x0 = self._crop_origin()
            ysel = slice(y0, y0 + self.read_size)
            xsel = slice(x0, x0 + self.read_size)
            keep = (self._keep_span(y0, H), self._keep_span(x0, W))

        # (T, 26, H, W) float32
        x_dyn = torch.from_numpy(np.ascontiguousarray(
            self.X.isel(time=slice(t0, t1), channel=self._dyn_idx, y=ysel, x=xsel).values
        ))

        # (13, H, W) float32
        static_idx = self._static_idx + [self._scalar_idx]  # scalar plane last
        x_static = torch.from_numpy(np.ascontiguousarray(
            self.X.isel(time=last, channel=static_idx, y=ysel, x=xsel).values
        ))

        labels = {
            name: torch.as_tensor(self.ds[name].isel(time=last, y=ysel, x=xsel).values)
            for name in self.label_names
        }
        masks = {
            name: torch.as_tensor(self.ds[name].isel(time=last, y=ysel, x=xsel).values)
            for name in self.mask_names
        }

        if keep is not None:
            # -- Drop halo days from every mask to ensure no loss is taken on those cells.
            masks = {
                name: self._halo_masked(m, keep) for name, m in masks.items()
            }
        return (x_dyn, x_static), labels, masks

    def _keep_span(self, origin: int, extent: int) -> slice:
        """ a crop side on the domain edge loses no context: its padding is what
            full-grid inference sees and its padding is supervised.
        """
        lo = 0 if origin == 0 else self.crop_halo
        hi = self.read_size if origin + self.read_size == extent else self.read_size - self.crop_halo
        return slice(lo, hi)

    @staticmethod
    def _halo_masked(mask: torch.Tensor, keep: Tuple[slice, slice]) -> torch.Tensor:
        out = torch.zeros_like(mask)
        out[keep[0], keep[1]] = mask[keep[0], keep[1]]
        return out


def init_data_loader(
    split: Literal['train', 'eval', 'test'],
    dataset_name: str = "wa2000",
    num_workers: int = 0,
    batch_size: int = 1,
    window_size: int = 10,
    window_stride: int = 2,
    crop_size: int | None = None,
    seed: int | None = None,
    encoder_depth: int = 1,
    attn_window: int = 2,
    fold: str = "full",
):
    """ Build a FireDataset for split and wrap it in a torch DataLoader.

        Cropping applies only to the train split; shuffling and a per-worker
        seed follow the same rule.
    """
    ds = FireDataset(
        get_dataset_config(dataset_name, fold),
        split,
        window_size=window_size,
        window_stride=window_stride,
        crop_size=crop_size if split == "train" else None,
        crop_seed=seed,
        encoder_depth=encoder_depth,
        attn_window=attn_window,
    )

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        # -- WSL2 caps the pinned-host pool and prefetched full-grid windows
        #    exhaust it; native Linux hosts pin normally
        pin_memory="microsoft" not in platform.uname().release.lower(),
        persistent_workers=(num_workers > 0),
        generator=generator,
        worker_init_fn=_seed_worker if seed is not None else None,
    )
