"""
How well a field places a day's ignitions, given how many there are.

Each day's supervised cells are renormalized into one distribution, and the
score is the multinomial ignorance of the cells that ignited, in bits per
ignition. Uniform over m cells reads log2(m): the score becomes the number of equally
likely cells the field has narrowed an ignition to.

A constant added to a day's field cancels in the normalization. Nothing here
moves with the overall ignition rate. Annual ignition counts span two orders of
magnitude, driven by seasonal climate outside the look-back window, and a
per-cell score spends most of its range on that quantity.

Capture at budget is the same field read as a decision: the share of ignitions
falling inside the top-ranked fraction of the map on the day they occurred.

Localization asks the same question one level down: given a coarse cell that
burned, how well does the fine field place the event inside it.

Two references and two strata. Climatology cannot place spread. The persistence
blend, climatology mixed with a dilation of the cells burning today, is the
reference where fire is already on the ground. Its ring is held in kilometres
so every tier is scored against the same physical reach.
Every event is scored in one of two strata: 'spread', an active cell within
SPREAD_REACH_M on the day, or 'ignition', none. The ignition stratum is the
placement question with no fire nearby to point at.

A static map, the field time-averaged over its supervised days, is the object
the ignition stratum's skill turns out to consist of; scoring it beside the
field asks whether the field moves for any reason the events reward.
"""
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.ndimage import maximum_filter

DEFAULT_BUDGETS = (0.01, 0.05, 0.10)
SPREAD_REACH_M = 2000.0
PERSIST_KM = 8.0
TEMPERATURES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
PERSIST_W = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
PERSIST_R = (1, 2, 3, 5)


def cells(km: float, res_m: float) -> int:
    return max(1, int(round(km * 1000.0 / res_m)))


def dilate(active: np.ndarray, r: int) -> np.ndarray:
    # -- (D, H, W) bool, within r cells of an active cell; a separable max
    #    filter, since a square structuring element is quadratic in r
    return maximum_filter(active.astype(np.uint8), size=(1, 2 * r + 1, 2 * r + 1),
                          mode="constant", cval=0) > 0


def strata(active: np.ndarray, res_m: float) -> np.ndarray:
    # -- (D, H, W) bool, True where an event would count as spread
    return dilate(active, cells(SPREAD_REACH_M / 1000.0, res_m))


def static_map(p: np.ndarray, sup: np.ndarray) -> np.ndarray:
    # -- (H, W) time average over each cell's supervised days; zero where never
    wsum, wcnt = (p * sup).sum(0), sup.sum(0)
    return np.where(wcnt > 0, wsum / np.maximum(wcnt, 1), 0.0)


def persistence(clim: np.ndarray, active: np.ndarray, sup: np.ndarray,
                w: float, r: int) -> np.ndarray:
    """ Per-day blend of the climatology allocation and a uniform allocation
        over the dilated active cells; each part normalized over supervised
        cells first. w is a share of probability, not of amplitude. """
    D = clim.shape[0]
    near = dilate(active, r) & sup.astype(bool)
    c = np.where(sup, clim, 0.0).reshape(D, -1)
    c = c / np.maximum(c.sum(1, keepdims=True), 1e-300)
    n = near.reshape(D, -1).astype(np.float64)
    has = n.sum(1, keepdims=True) > 0
    n = np.where(has, n / np.maximum(n.sum(1, keepdims=True), 1.0), 0.0)
    out = np.where(has, (1 - w) * c + w * n, c)
    return out.reshape(clim.shape)


def fit_persistence(clim: np.ndarray, active: np.ndarray, y: np.ndarray, sup: np.ndarray,
                    r: Optional[int] = None) -> Dict:
    # -- (w, r) minimizing allocation bits on the given days; a given r fixes
    #    the ring and fits only the share
    best = None
    for r in (PERSIST_R if r is None else (r,)):
        for w in PERSIST_W:
            bits = day_sums(persistence(clim, active, sup, w, r), y, sup, ())["bits"].sum()
            if best is None or bits < best["bits"]:
                best = {"w": w, "r": r, "bits": float(bits)}
    return best


def fit_temperature(p: np.ndarray, y: np.ndarray, sup: np.ndarray) -> float:
    """ s minimizing allocation bits of q proportional to p**s. """
    bits = {s: day_sums(np.power(p, s), y, sup, ())["bits"].sum() for s in TEMPERATURES}
    return float(min(bits, key=bits.get))


def day_sums(p: np.ndarray, y: np.ndarray, sup: np.ndarray,
             budgets: Sequence[float] = DEFAULT_BUDGETS,
             events: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """ Per-day partial sums as (D,) arrays, resampled by the year-block bootstrap.

    Args:
        p:      (D, ...) non-negative field over the evaluation grid
        y:      (D, ...) labels in {0, 1}
        sup:    (D, ...) supervised selection; the normalizer of q
        events: (D, ...) optional selection of which positives to score; a
                stratum shares the whole's normalizer
    """
    D = p.shape[0]
    p, y, sup = (a.reshape(D, -1) for a in (p, y, sup))
    events = y if events is None else (events.reshape(D, -1) & (y > 0))
    out = {k: np.zeros(D) for k in ("bits", "uniform", "events")}
    out.update({f"cap_{b:g}": np.zeros(D) for b in budgets})

    for d in range(D):
        cells = sup[d].astype(bool)
        hit = events[d][cells].astype(bool)
        if not hit.any():
            continue
        pd_ = np.clip(p[d][cells].astype(np.float64), 1e-300, None)
        q = pd_ / pd_.sum()
        out["bits"][d] = float(-np.log2(np.maximum(q[hit], 1e-300)).sum())
        out["uniform"][d] = float(hit.sum() * np.log2(cells.sum()))
        out["events"][d] = float(hit.sum())
        order = np.argsort(pd_)[::-1]
        for b in budgets:
            k = max(1, int(round(b * pd_.size)))
            out[f"cap_{b:g}"][d] = float(hit[order[:k]].sum())
    return out


def report(model: Dict[str, np.ndarray], refs: Dict[str, Dict[str, np.ndarray]],
           budgets: Sequence[float] = DEFAULT_BUDGETS) -> Dict:
    """ Model against each named reference on the same events. """
    n = model["events"].sum()
    if n == 0:
        return {"n_events": 0}
    out = {
        "n_events": float(n),
        "bits_per_event": float(model["bits"].sum() / n),
        "bits_per_event_uniform": float(model["uniform"].sum() / n),
        "skill_vs_uniform": float(1 - model["bits"].sum() / model["uniform"].sum()),
        "capture": {f"top_{b:g}": {"model": float(model[f"cap_{b:g}"].sum() / n)} for b in budgets},
    }
    for name, ref in refs.items():
        out[f"bits_per_event_{name}"] = float(ref["bits"].sum() / n)
        out[f"skill_vs_{name}"] = float(1 - model["bits"].sum() / ref["bits"].sum())
        out[f"{name}_skill_vs_uniform"] = float(1 - ref["bits"].sum() / ref["uniform"].sum())
        for b in budgets:
            out["capture"][f"top_{b:g}"][name] = float(ref[f"cap_{b:g}"].sum() / n)
    return out


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
