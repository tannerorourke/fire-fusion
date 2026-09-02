"""
Probabilistic verification scores as pure array functions, plus a year-block
bootstrap for confidence intervals. Nothing here loads a model or a dataset;
callers pass flat probability/label arrays (or per-day partial sums) in.
"""
from typing import Callable, Dict, Sequence

import numpy as np

LN2 = float(np.log(2.0))


def ignorance_bits(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return -(y * np.log2(p) + (1.0 - y) * np.log2(1.0 - p))


def ignorance_bits_from_logits(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    # -- softplus form stays exact where sigmoid saturates in float64
    return (np.logaddexp(0.0, z) - y * z) / LN2


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p - y.astype(np.float64)) ** 2


def skill(model_score: float, ref_score: float) -> Dict[str, float]:
    return {
        "model": model_score, "reference": ref_score,
        "resolved": ref_score - model_score,
        "skill_score": 1.0 - model_score / ref_score if ref_score > 0 else float("nan"),
    }


def murphy_decomposition(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> Dict[str, float]:
    """ Brier = REL - RES + UNC over quantile bins of p. """
    y = y.astype(np.float64)
    order = np.argsort(p)
    edges = np.linspace(0, len(p), n_bins + 1).astype(int)
    o_bar = y.mean()
    rel = res = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        idx = order[lo:hi]
        pk, ok, nk = p[idx].mean(), y[idx].mean(), hi - lo
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - o_bar) ** 2
    n = len(p)
    return {"rel": rel / n, "res": res / n, "unc": o_bar * (1.0 - o_bar),
            "brier": float(brier(p, y).mean()), "n_bins": n_bins}


def cox_calibration(z: np.ndarray, y: np.ndarray, max_iter: int = 50) -> Dict[str, float]:
    # -- logit-space calibration line y ~ sigmoid(a*z + b), Newton-IRLS
    y = y.astype(np.float64)
    a, b = 1.0, 0.0
    for _ in range(max_iter):
        eta = np.clip(a * z + b, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1.0 - mu)
        g = np.array([((mu - y) * z).sum(), (mu - y).sum()])
        h00, h01, h11 = (w * z * z).sum(), (w * z).sum(), w.sum()
        det = h00 * h11 - h01 * h01
        if det <= 0:
            break
        da = (h11 * g[0] - h01 * g[1]) / det
        db = (h00 * g[1] - h01 * g[0]) / det
        a, b = a - da, b - db
        if abs(da) < 1e-8 and abs(db) < 1e-8:
            break
    return {"slope": float(a), "intercept": float(b)}


def per_year_sums(values: np.ndarray, day_years: np.ndarray,
                  day_counts: np.ndarray) -> Dict[int, np.ndarray]:
    out = {}
    for yr in np.unique(day_years):
        m = day_years == yr
        out[int(yr)] = np.array([values[m].sum(), day_counts[m].sum()])
    return out


def year_block_bootstrap(per_year: Dict[int, Dict[str, np.ndarray]],
                         stat_fn: Callable[[Dict[str, np.ndarray]], float],
                         n_boot: int = 2000, seed: int = 0,
                         ci: Sequence[float] = (2.5, 97.5)) -> Dict[str, float]:
    """ Resample whole years with replacement; stat_fn sees summed partials. """
    years = sorted(per_year)
    keys = list(per_year[years[0]])
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(len(years), size=len(years), replace=True)
        sums = {k: sum(per_year[years[j]][k] for j in pick) for k in keys}
        stats[i] = stat_fn(sums)
    point = stat_fn({k: sum(per_year[y][k] for y in years) for k in keys})
    lo, hi = np.percentile(stats, ci)
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "n_boot": n_boot, "n_years": len(years)}
