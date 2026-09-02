"""
Spatial verification: fractions skill score over neighborhood scales, and
sub-cell localization of pooled events.

FSS binarizes the probability field at the frequency-matched threshold (the
exceedance count equals the observed event count on the scored days), so tiers
with different calibration binarize to comparable event frequencies.
"""
from typing import Dict, Sequence

import numpy as np
from scipy import sparse
from scipy.ndimage import uniform_filter


def frequency_matched_threshold(p: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    vals = p[mask.astype(bool)]
    n_events = int(y[mask.astype(bool)].sum())
    if n_events == 0 or n_events >= vals.size:
        return float("inf")
    return float(np.partition(vals, vals.size - n_events)[vals.size - n_events])


def fss_curve(p: np.ndarray, y: np.ndarray, mask: np.ndarray,
              scales_px: Sequence[int]) -> Dict[str, list]:
    """ FSS per odd window size, fractions normalized by supervised coverage. """
    thr = frequency_matched_threshold(p, y, mask)
    m = mask.astype(np.float64)
    fb = (p >= thr).astype(np.float64) * m
    ob = (y > 0).astype(np.float64) * m

    out = {"scale_px": [], "fss": [], "threshold": thr}
    for k in scales_px:
        num = den = 0.0
        for d in range(p.shape[0]):
            cov = uniform_filter(m[d], size=k, mode="constant")
            valid = (m[d] > 0) & (cov > 0)
            pf = uniform_filter(fb[d], size=k, mode="constant")[valid] / cov[valid]
            po = uniform_filter(ob[d], size=k, mode="constant")[valid] / cov[valid]
            num += ((pf - po) ** 2).sum()
            den += (pf ** 2).sum() + (po ** 2).sum()
        out["scale_px"].append(int(k))
        out["fss"].append(1.0 - num / den if den > 0 else float("nan"))
    return out


def skillful_scale(scales_px: Sequence[int], fss: Sequence[float],
                   event_fraction: float) -> float:
    # -- smallest scale reaching the useful threshold 0.5 + f/2
    target = 0.5 + event_fraction / 2.0
    for k, v in zip(scales_px, fss):
        if not np.isnan(v) and v >= target:
            return float(k)
    return float("nan")


def localization_scores(p: np.ndarray, y: np.ndarray, mask: np.ndarray,
                        member: sparse.csr_matrix) -> Dict[str, float]:
    """ Ignorance of the induced sub-cell distribution, over coarse cell-days
        holding at least one event, against the uniform in-cell baseline. """
    D = p.shape[0]
    pm = (p * mask).reshape(D, -1)
    ym = (y * mask).reshape(D, -1).astype(np.float64)
    msup = mask.reshape(D, -1).astype(np.float64)

    p_tot = pm @ member.T
    y_tot = ym @ member.T
    n_sup = msup @ member.T

    lign_sum = 0.0
    uniform_sum = 0.0
    n_cells = 0
    rows, cols = member.nonzero()
    order = np.argsort(rows)
    rows, cols = rows[order], cols[order]
    starts = np.searchsorted(rows, np.arange(member.shape[0] + 1))

    for d in range(D):
        hit = np.flatnonzero((y_tot[d] > 0) & (n_sup[d] > 1) & (p_tot[d] > 0))
        for R in hit:
            j = cols[starts[R]:starts[R + 1]]
            q = pm[d, j] / p_tot[d, R]
            e = ym[d, j] / y_tot[d, R]
            nz = e > 0
            lign_sum += float(-(e[nz] * np.log2(np.maximum(q[nz], 1e-15))).sum())
            uniform_sum += float(np.log2(n_sup[d, R]))
            n_cells += 1

    if n_cells == 0:
        return {"lign": float("nan"), "uniform": float("nan"),
                "skill": float("nan"), "n_cell_days": 0}
    lign = lign_sum / n_cells
    uniform = uniform_sum / n_cells
    return {"lign": lign, "uniform": uniform,
            "skill": 1.0 - lign / uniform, "n_cell_days": n_cells}
