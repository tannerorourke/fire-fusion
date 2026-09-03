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


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p - y.astype(np.float64)) ** 2


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
    """ Fitted logit-space calibration line 'y ~ sigmoid(a*z + b)' by Newton-IRLS,
        as {'slope': a, 'intercept': b}.
    """
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


def year_block_bootstrap(per_year: Dict[int, Dict[str, np.ndarray]],
                         stat_fn: Callable[[Dict[str, np.ndarray]], float],
                         n_boot: int = 2000, seed: int = 0,
                         ci: Sequence[float] = (2.5, 97.5),
                         block_days: int = 30) -> Dict[str, float]:
    """ Two-level resample: whole years with replacement, then within each drawn
        year contiguous blocks of 'block_days' with replacement. Years carry the
        interannual variance; blocks add within-season resolution beyond the 35
        outcomes of four whole test years. Each entry of per_year holds per-day
        partial sums as (D,) arrays, re-summed per block. block_days=0 falls
        back to plain year blocks.
    """
    years = sorted(per_year)
    keys = list(per_year[years[0]])
    rng = np.random.default_rng(seed)
    blocks = {}
    for y in years:
        d = np.asarray(per_year[y][keys[0]]).size
        edges = list(range(0, d, block_days)) + [d] if block_days and d > 1 else [0, d]
        blocks[y] = [(a, b) for a, b in zip(edges[:-1], edges[1:])]

    def draw_year(y):
        if block_days == 0 or len(blocks[y]) == 1:
            return {k: np.asarray(per_year[y][k]).sum() for k in keys}
        pick = rng.integers(0, len(blocks[y]), size=len(blocks[y]))
        return {k: sum(np.asarray(per_year[y][k])[blocks[y][j][0]:blocks[y][j][1]].sum()
                       for j in pick) for k in keys}

    stats = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(len(years), size=len(years), replace=True)
        sums = {k: 0.0 for k in keys}
        for j in pick:
            part = draw_year(years[j])
            for k in keys:
                sums[k] += part[k]
        stats[i] = stat_fn(sums)
    point = stat_fn({k: sum(np.asarray(per_year[y][k]).sum() for y in years) for k in keys})
    lo, hi = np.nanpercentile(stats, ci)
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "n_boot": n_boot, "n_years": len(years), "block_days": block_days,
            "n_blocks": int(sum(len(b) for b in blocks.values()))}


def cause_ignorance_bits(logits: np.ndarray, labels: np.ndarray,
                         train_counts: Sequence[int],
                         a: Sequence[float] | None = None,
                         b: Sequence[float] | None = None) -> Dict[str, float]:
    """ Multiclass ignorance of the cause head against the train-year class prior.

    The head trains under inverse-frequency class weights, whose population
    optimum tilts the posterior. The calibrator's vector scaling is the fitted
    inverse, applied before scoring. The reference is the train-year marginal
    frequency, what a forecaster knows without a model.
    """
    z = logits.astype(np.float64)
    if a is not None and b is not None:
        z = z * np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    log_q = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    y = labels.astype(int)
    n = len(y)

    prior = np.asarray(train_counts, dtype=np.float64)
    prior = prior / prior.sum()

    model_bits = float(-log_q[np.arange(n), y].sum() / n / LN2)
    prior_bits = float(-np.log(prior[y]).sum() / n / LN2)
    pred = log_q.argmax(axis=1)
    n_classes = log_q.shape[1]
    f1 = []
    for c in range(n_classes):
        tp = float(((pred == c) & (y == c)).sum())
        fp = float(((pred == c) & (y != c)).sum())
        fn = float(((pred != c) & (y == c)).sum())
        denom = 2 * tp + fp + fn
        f1.append(2 * tp / denom if denom > 0 else float("nan"))
    return {"n_cells": n, "bits": model_bits, "bits_prior": prior_bits,
            "skill_score": 1 - model_bits / prior_bits if prior_bits > 0 else float("nan"),
            "macro_f1": float(np.nanmean(f1)), "per_class_f1": f1,
            "class_counts": [int((y == c).sum()) for c in range(n_classes)]}
