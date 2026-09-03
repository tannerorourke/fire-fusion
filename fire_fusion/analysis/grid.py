"""
The shared evaluation grid: which ground a claim is scored over, and how each
tier's cells are pooled onto it.

Cross-resolution comparison is scored over identical ground. The tiers do not
share any by default: `cascades500` covers a corridor whose ignition rate is
roughly three times the state average. Any claim comparing resolutions passes
the reference tier's footprint in; claims about a single tier on its own extent
do not.

Membership is computed from projected cell-center coordinates against a fixed
origin; tiers of different resolution and extent land on identical coarse
cells. The noisy-OR aggregate matches an 'at least one event in the coarse cell'
target, with mean pooling as a robustness check.
"""
from typing import Dict, Tuple

import numpy as np
from pyproj import Transformer
from scipy import sparse

from ..config.dataset_config import get_dataset_config
from ..dataset.grid import GRID_CRS, LATTICE_M


def reference_envelope(reference: str) -> Tuple[float, float, float, float]:
    """ Projected (x0, x1, y0, y1) envelope of a tier's configured bounds. """
    cfg = get_dataset_config(reference)
    tf = Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True)
    lat0, lat1 = min(cfg.lat_bounds), max(cfg.lat_bounds)
    lon0, lon1 = min(cfg.lon_bounds), max(cfg.lon_bounds)
    corners = [tf.transform(lo, la) for lo in (lon0, lon1) for la in (lat0, lat1)]
    xs, ys = zip(*corners)
    return min(xs), max(xs), min(ys), max(ys)


def evaluation_grid(footprint_env: Tuple[float, float, float, float],
                    coarse_res: float) -> Tuple[Tuple[float, float], Tuple[int, int]]:
    """ (origin, shape) of a coarse grid covering the envelope, on the tier lattice.

        Every tier's edges sit on the lattice. The origin is a lattice multiple
        and the resolution divides the lattice, holding every tier's cells whole.
        An origin at the envelope corner cuts cells.
    """
    x0, x1, y0, y1 = footprint_env
    assert LATTICE_M % coarse_res == 0, f"{coarse_res} m does not divide the {LATTICE_M} m lattice"
    sx = np.floor(x0 / LATTICE_M) * LATTICE_M
    sy = np.ceil(y1 / LATTICE_M) * LATTICE_M
    shape = (int(np.ceil((sy - y0) / coarse_res)), int(np.ceil((x1 - sx) / coarse_res)))
    return (float(sx), float(sy)), shape


def build_membership(x: np.ndarray, y: np.ndarray, origin: Tuple[float, float],
                     coarse_res: float, coarse_shape: Tuple[int, int]) -> sparse.csr_matrix:
    """ (Hc*Wc, H*W) 0/1 matrix mapping fine cells into coarse cells. """
    x0, y1 = origin
    Hc, Wc = coarse_shape
    ix = np.floor((x - x0) / coarse_res).astype(int)
    # -- y decreases row-wise in the projected grids; rows count down from
    #    the origin's top edge
    iy = np.floor((y1 - y) / coarse_res).astype(int)
    IX, IY = np.meshgrid(ix, iy)
    valid = (IX >= 0) & (IX < Wc) & (IY >= 0) & (IY < Hc)
    fine_idx = np.flatnonzero(valid.ravel())
    coarse_idx = (IY.ravel()[fine_idx] * Wc + IX.ravel()[fine_idx])
    data = np.ones(len(fine_idx), dtype=np.float64)
    return sparse.csr_matrix((data, (coarse_idx, fine_idx)),
                             shape=(Hc * Wc, len(x) * len(y)))


def pool_fields(p: np.ndarray, labels: np.ndarray, mask: np.ndarray,
                member: sparse.csr_matrix, mode: str = "noisy_or") -> Dict[str, np.ndarray]:
    """ Pool (D, H, W) fields; unsupervised fine cells contribute nothing. """
    D = p.shape[0]
    flat_m = mask.reshape(D, -1).astype(np.float64)
    n_sup = flat_m @ member.T

    if mode == "noisy_or":
        log_keep = np.log1p(-np.clip(p, 0.0, 1.0 - 1e-12)) * mask
        agg = 1.0 - np.exp(log_keep.reshape(D, -1) @ member.T)
    elif mode == "mean":
        agg = (p * mask).reshape(D, -1) @ member.T / np.maximum(n_sup, 1.0)
    else:
        raise ValueError(f"unknown mode '{mode}'")

    y_pool = ((labels * mask).reshape(D, -1) @ member.T) > 0
    return {"p": agg, "y": y_pool.astype(np.uint8), "n_supervised": n_sup}


def common_supervision(n_sup_by_tier: Dict[str, np.ndarray]) -> np.ndarray:
    """ Bool intersection of supervised coarse cells across every tier in
        'n_sup_by_tier', the coarse cell-days every tier scores.
    """
    out = None
    for n_sup in n_sup_by_tier.values():
        m = n_sup > 0
        out = m if out is None else (out & m)
    return out
