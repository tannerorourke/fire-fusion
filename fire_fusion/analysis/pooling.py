"""
Pool per-cell probability fields from any tier's grid onto a shared coarse
evaluation grid. Membership is computed from projected cell-center coordinates
against a fixed origin,. Tiers with different resolutions and extents land
on the identical coarse cells; the noisy-OR aggregate matches an 'at least one
event in the coarse cell' target, with mean pooling as a robustness check.
"""
from typing import Dict, Tuple

import numpy as np
from scipy import sparse


def build_membership(x: np.ndarray, y: np.ndarray, origin: Tuple[float, float],
                     coarse_res: float, coarse_shape: Tuple[int, int]) -> sparse.csr_matrix:
    """ (Hc*Wc, H*W) 0/1 matrix mapping fine cells into coarse cells. """
    x0, y1 = origin
    Hc, Wc = coarse_shape
    ix = np.floor((x - x0) / coarse_res).astype(int)
    # -- y decreases row-wise in the projected grids, so rows count down from
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
    # -- intersect so every tier scores the identical coarse cell-days
    out = None
    for n_sup in n_sup_by_tier.values():
        m = n_sup > 0
        out = m if out is None else (out & m)
    return out
