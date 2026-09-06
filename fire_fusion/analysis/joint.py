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

With `--regions`, the count is a rate per lattice region and the placement is
the field renormalized within each region: p(i, t) = lambda_r(t) q(i | r, t).
The 'regional_rate_x_model' and 'regional_seasonal_x_model' pairings, run on
a static arm, are the product's ignition factor and its no-weather reference;
their paired bits per event on the ignition stratum is the reading. The
region allocation score asks the count question alone: which region gets
today's events, against the seasonal share of each region.
"""
import json

import numpy as np

from .allocation import strata
from .archive import climatology_for, load_archive
from .rate import daily_rate, membership, rate_path
from .scores import by_year, ignorance_bits, per_day_sums, year_block_bootstrap
from ..config.dataset_config import get_dataset_config

PAIRINGS = (
    ("rate_x_model", "rate", "model"), ("rate_x_clim", "rate", "clim"),
    ("seasonal_x_model", "seasonal", "model"), ("clim_sum_x_model", "clim_sum", "model"),
    ("model_sum_x_clim", "model_sum", "clim"),
    ("oracle_x_model", "observed", "model"), ("oracle_x_clim", "observed", "clim"),
)
REGIONAL_PAIRINGS = (
    ("regional_rate_x_model", "lambda", "model"), ("regional_seasonal_x_model", "seasonal", "model"),
    ("regional_rate_x_clim", "lambda", "clim"), ("regional_seasonal_x_clim", "seasonal", "clim"),
)
PAIRED = {"rate_over_seasonal": ("rate_x_model", "seasonal_x_model"),
          "rate_over_clim_placement": ("rate_x_model", "rate_x_clim")}
REGIONAL_PAIRED = {"regional_rate_over_seasonal": ("regional_rate_x_model", "regional_seasonal_x_model"),
                   "regional_over_statewide_rate": ("regional_rate_x_model", "rate_x_model"),
                   "regional_seasonal_over_statewide_seasonal": ("regional_seasonal_x_model", "seasonal_x_model")}


def _aligned_rate(arc: dict, regions_km=None) -> dict:
    # -- the rate's per-day (D, R) arrays on the archive's window-final days,
    #    and the region map trimmed to the archive's aligned grid
    rate = daily_rate(arc["dataset"], arc["sidecar"].get("fold", "full"), arc["sidecar"]["split"], regions_km)
    pos = {d: i for i, d in enumerate(rate["dates"])}
    sel = np.array([pos[d] for d in arc["dates"]])
    H, W = arc["p"].shape[1:]
    out = {k: rate[k][sel] for k in ("lambda", "seasonal", "seasonal_share", "events", "n_sup")}
    out["regions"] = rate["regions"][:H, :W]
    return out


def joint_fields(p: np.ndarray, clim: np.ndarray, y: np.ndarray, mask: np.ndarray,
                 lam_rate: np.ndarray, lam_seasonal: np.ndarray) -> tuple:
    """ (fields, counts): the (D, H, W) probability fields of the statewide
        pairings and the per-day count factors, over the supervised cells.
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


def regional_product(p: np.ndarray, mask: np.ndarray, regions: np.ndarray,
                     lam: np.ndarray) -> np.ndarray:
    """ (D, H, W) field lambda_r(t) q(i | r, t): p renormalized over the
        supervised cells of each region, scaled by that region's count. """
    D = p.shape[0]
    M = membership(regions, lam.shape[1])
    flat = (p * mask).reshape(D, -1)
    mass = flat @ M.T                                   # (D, R)
    scale = (lam / np.maximum(mass, 1e-12)) @ M         # (D, H*W)
    return np.clip((flat * scale).reshape(p.shape), 1e-9, 1.0 - 1e-9)


def regional_fields(p, clim, mask, rate: dict) -> dict:
    return {key: regional_product(p if placement == "model" else clim, mask, rate["regions"], rate[lam])
            for key, lam, placement in REGIONAL_PAIRINGS}


def _score_fields(fields: dict, y: np.ndarray, sel: np.ndarray, years: np.ndarray,
                  paired: dict, n_boot: int, seed: int) -> dict:
    """ Per-field skill against the climatology product over the cells in
        `sel`, plus paired bits per event for the named pairs. """
    ign = {k: ignorance_bits(f, y) for k, f in fields.items()}
    day = per_day_sums({**ign, "pos": y}, sel)
    per_year = by_year(day, years)
    n, events = day["n"].sum(), day["pos"].sum()
    out = {"n_cell_days": float(n), "n_events": float(events), "fields": {}, "paired": {},
           "per_year": per_year}
    for k in fields:
        out["fields"][k] = {
            "ign_bits": float(day[k].sum() / n),
            "skill_vs_clim": year_block_bootstrap(
                per_year, lambda s, k=k: 1.0 - s[k] / s["clim"], n_boot=n_boot, seed=seed),
            "bits_per_event_resolved": float((day["clim"].sum() - day[k].sum()) / events),
        }
    for name, (a, b) in paired.items():
        # -- bits per event that a resolves beyond b, positive favouring a
        out["paired"][name] = year_block_bootstrap(
            per_year, lambda s, a=a, b=b: (s[b] - s[a]) / max(s["pos"], 1.0), n_boot=n_boot, seed=seed)
    return out


def _merge_years(blocks: list) -> dict:
    # -- one per-year table from experiments scored on disjoint years
    out = {}
    for b in blocks:
        assert not set(b) & set(out), "pooled experiments must score disjoint years"
        out.update(b)
    return out


def _pool(entries: list, paired: dict, n_boot: int, seed: int) -> dict:
    """ The fields and paired blocks of several _score_fields outputs, re-read
        under one bootstrap over the union of their years. """
    merged = _merge_years([e["per_year"] for e in entries])
    keys = [k for k in next(iter(merged.values())) if k not in ("n", "pos")]
    years = sorted(merged)
    n = sum(merged[y]["n"].sum() for y in years)
    events = sum(merged[y]["pos"].sum() for y in years)
    total = {k: sum(merged[y][k].sum() for y in years) for k in keys}
    out = {"n_years": len(years), "n_cell_days": float(n), "n_events": float(events), "fields": {}, "paired": {}}
    for k in keys:
        out["fields"][k] = {
            "ign_bits": float(total[k] / n),
            "skill_vs_clim": year_block_bootstrap(merged, lambda s, k=k: 1.0 - s[k] / s["clim"], n_boot=n_boot, seed=seed),
            "bits_per_event_resolved": float((total["clim"] - total[k]) / events),
        }
    for name, (a, b) in paired.items():
        out["paired"][name] = year_block_bootstrap(
            merged, lambda s, a=a, b=b: (s[b] - s[a]) / max(s["pos"], 1.0), n_boot=n_boot, seed=seed)
    return out


def _strip(entry: dict) -> dict:
    entry.pop("per_year", None)
    return entry


def region_allocation(lam: np.ndarray, refs: dict, events: dict, years: np.ndarray,
                      n_boot: int, seed: int) -> dict:
    """ Which region gets today's events: multinomial ignorance over regions of
        the rate's shares against the seasonal share, the region's own seasonal
        profile and exposure, one block per event set. """
    exposed = refs["n_sup"] > 0
    shares = {"model": lam, "seasonal_share": refs["seasonal_share"],
              "seasonal_own": refs["seasonal"], "exposure": refs["n_sup"]}
    log_q = {}
    for k, v in shares.items():
        v = np.where(exposed, v, 0.0)
        log_q[k] = np.log2(np.maximum(v / np.maximum(v.sum(1, keepdims=True), 1e-300), 1e-300))
    out = {}
    for name, e in events.items():
        day = {k: -(e * lq).sum(1) for k, lq in log_q.items()}
        day["events"] = e.sum(1)
        per_year = by_year(day, years)
        n = day["events"].sum()
        block = {"n_events": float(n), "n_regions": int(lam.shape[1]),
                 "bits_per_event": {k: float(day[k].sum() / max(n, 1)) for k in log_q}}
        for ref in ("seasonal_share", "seasonal_own", "exposure"):
            block[f"skill_vs_{ref}"] = year_block_bootstrap(
                per_year, lambda s, ref=ref: 1.0 - s["model"] / s[ref], n_boot=n_boot, seed=seed)
        out[name] = block
    return out


def stage_joint(experiments, split, bandwidth_km, n_boot, seed, calibrated, regions=(),
                pool=False) -> dict:
    report, raw = {}, {}
    for exp in experiments:
        arc = load_archive(exp, split, calibrated)
        fold = arc["sidecar"].get("fold", "full")
        res = get_dataset_config(arc["dataset"]).resolution
        rate = _aligned_rate(arc)
        y = arc["y"].astype(np.float64)
        p, clim, mask = arc["p"].astype(np.float64), climatology_for(arc, bandwidth_km), arc["mask"].astype(bool)
        fields, counts = joint_fields(p, clim, y, mask, rate["lambda"][:, 0], rate["seasonal"][:, 0])
        ignition = mask & ~strata(arc["active"], res)
        obs = counts["observed"]

        entry = {
            "dataset": arc["dataset"], "split": split, "fold": fold, "calibrated": calibrated,
            "n_days": int(len(arc["dates"])),
            "rate": {k: v for k, v in json.loads(rate_path(arc["dataset"], fold).with_suffix(".json")
                                                 .read_text()).items() if k != "splits"},
            "mean_daily_count": {k: float(v.mean()) for k, v in counts.items()},
            "count_corr": {k: float(np.corrcoef(v, obs)[0, 1]) for k, v in counts.items()
                           if k != "observed"},
        }
        scored = _score_fields(fields, y, mask, arc["years"], PAIRED, n_boot, seed)
        ign_scored = _score_fields(fields, y, ignition, arc["years"], PAIRED, n_boot, seed)
        entry.update({k: scored[k] for k in ("n_cell_days", "n_events", "fields", "paired")})
        entry["ignition_stratum"] = _strip(dict(ign_scored))
        raw[exp] = {"all": scored, "ignition_stratum": ign_scored, "regions": {}}

        entry["regions"] = {}
        for km in regions:
            reg = _aligned_rate(arc, km)
            rf = regional_fields(p, clim, mask, reg)
            both = {**{k: fields[k] for k in ("clim", "rate_x_model", "seasonal_x_model")}, **rf}
            block = {"n_regions": int(reg["lambda"].shape[1]),
                     "rate": {k: v for k, v in json.loads(rate_path(arc["dataset"], fold, km).with_suffix(".json")
                                                          .read_text()).items() if k != "splits"}}
            raw[exp]["regions"][f"{km:g}"] = {}
            for name, sel in (("all", mask), ("ignition_stratum", ignition)):
                s = _score_fields(both, y, sel, arc["years"], REGIONAL_PAIRED, n_boot, seed)
                raw[exp]["regions"][f"{km:g}"][name] = s
                block[name] = {"n_events": s["n_events"],
                               "fields": {k: s["fields"][k] for k in rf},
                               "paired": s["paired"]}
            M = membership(reg["regions"], reg["lambda"].shape[1])
            events = {"all": reg["events"],
                      "ignition_stratum": (y * ignition).reshape(y.shape[0], -1) @ M.T}
            block["region_allocation"] = region_allocation(reg["lambda"], reg, events, arc["years"], n_boot, seed)
            entry["regions"][f"{km:g}"] = block
        report[exp] = entry

    if pool and len(experiments) > 1:
        # -- one bootstrap over every experiment's years; each archive is one
        #    fold scored on its own test years
        pooled = {"experiments": list(experiments),
                  "all": _pool([raw[e]["all"] for e in experiments], PAIRED, n_boot, seed),
                  "ignition_stratum": _pool([raw[e]["ignition_stratum"] for e in experiments], PAIRED, n_boot, seed),
                  "regions": {}}
        for km in regions:
            key = f"{km:g}"
            pooled["regions"][key] = {
                name: _pool([raw[e]["regions"][key][name] for e in experiments], REGIONAL_PAIRED, n_boot, seed)
                for name in ("all", "ignition_stratum")}
        report["pooled"] = pooled
    return report
