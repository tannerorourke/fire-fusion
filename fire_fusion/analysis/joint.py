"""
Joint intensity-times-allocation score: what the rate and the placement are
worth together, per cell, on the native grid.

p(i, t) = lambda(t) * q(i | t) under every pairing of a count factor (the
model's own field sum, the climatology sum, the fitted negative-binomial rate,
the seasonal profile, the observed count as an oracle) with a placement factor
(the model's normalized field, the climatology's). Ignorance is scored as the
native stage scores it, against the climatology product, with the two-level
year-block bootstrap; the 'model' row equals the native report's skill. The
model's own sum is a worse count than the seasonal profile, and the per-cell
score charges placement for it; the decomposition separates the two.
"""
import json

import numpy as np

from .archive import climatology_for, load_archive
from .rate import daily_rate, rate_path
from .scores import by_year, ignorance_bits, per_day_sums, year_block_bootstrap

PAIRINGS = (
    ("rate_x_model", "rate", "model"), ("rate_x_clim", "rate", "clim"),
    ("seasonal_x_model", "seasonal", "model"), ("clim_sum_x_model", "clim_sum", "model"),
    ("model_sum_x_clim", "model_sum", "clim"),
    ("oracle_x_model", "observed", "model"), ("oracle_x_clim", "observed", "clim"),
)


def _aligned_rate(arc: dict) -> dict:
    # -- the rate's per-day arrays on the archive's window-final days
    rate = daily_rate(arc["dataset"], arc["sidecar"].get("fold", "full"), arc["sidecar"]["split"])
    pos = {d: i for i, d in enumerate(rate["dates"])}
    sel = np.array([pos[d] for d in arc["dates"]])
    return {k: rate[k][sel] for k in ("lambda", "seasonal")}


def joint_fields(p: np.ndarray, clim: np.ndarray, y: np.ndarray, mask: np.ndarray,
                 lam_rate: np.ndarray, lam_seasonal: np.ndarray) -> tuple:
    """ (fields, counts): the nine (D, H, W) probability fields of the pairings
        and the per-day count factors, over the supervised cells.
    """
    m = mask.astype(bool)

    def total(f):
        return (f * m).reshape(f.shape[0], -1).sum(1)

    counts = {"model_sum": total(p), "clim_sum": total(clim), "observed": total(y),
              "rate": lam_rate, "seasonal": lam_seasonal}
    q = {"model": p / np.maximum(counts["model_sum"], 1e-12)[:, None, None],
         "clim": clim / np.maximum(counts["clim_sum"], 1e-12)[:, None, None]}
    fields = {"model": p, "clim": clim}
    for key, lam, placement in PAIRINGS:
        fields[key] = np.clip(counts[lam][:, None, None] * q[placement], 1e-9, 1.0 - 1e-9)
    return fields, counts


def stage_joint(experiments, split, bandwidth_km, n_boot, seed, calibrated) -> dict:
    # -- per-cell skill against the climatology product for every pairing
    report = {}
    for exp in experiments:
        arc = load_archive(exp, split, calibrated)
        fold = arc["sidecar"].get("fold", "full")
        rate = _aligned_rate(arc)
        y = arc["y"].astype(np.float64)
        fields, counts = joint_fields(arc["p"].astype(np.float64),
                                      climatology_for(arc, bandwidth_km), y, arc["mask"],
                                      rate["lambda"], rate["seasonal"])
        ign = {k: ignorance_bits(f, y) for k, f in fields.items()}
        day = per_day_sums({**ign, "pos": y}, arc["mask"])
        per_year = by_year(day, arc["years"])
        n, events = day["n"].sum(), day["pos"].sum()
        obs = counts["observed"]

        entry = {
            "dataset": arc["dataset"], "split": split, "fold": fold, "calibrated": calibrated,
            "n_days": int(len(arc["dates"])), "n_cell_days": float(n), "n_events": float(events),
            "rate": {k: v for k, v in json.loads(rate_path(arc["dataset"], fold).with_suffix(".json")
                                                 .read_text()).items() if k != "splits"},
            "mean_daily_count": {k: float(v.mean()) for k, v in counts.items()},
            "count_corr": {k: float(np.corrcoef(v, obs)[0, 1]) for k, v in counts.items()
                           if k != "observed"},
            "fields": {},
        }
        for k in fields:
            entry["fields"][k] = {
                "ign_bits": float(day[k].sum() / n),
                "skill_vs_clim": year_block_bootstrap(
                    per_year, lambda s, k=k: 1.0 - s[k] / s["clim"], n_boot=n_boot, seed=seed),
                "bits_per_event_resolved": float((day["clim"].sum() - day[k].sum()) / events),
            }
        report[exp] = entry
    return report
