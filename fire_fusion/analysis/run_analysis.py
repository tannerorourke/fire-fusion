"""
Score extracted prediction archives: native per-tier skill against climatology,
cross-tier comparison on a pooled common grid, spatial verification, and
sub-cell localization. Stages are independent; each reads the extraction
archives under PRED_DIR and writes one JSON report under REPORTS_DIR.

  python -m fire_fusion.analysis.run_analysis native --experiments wa2000-s1 --split test
  python -m fire_fusion.analysis.run_analysis pooled --experiments wa2000-s1 wa1000-s1 cascades500-s1 --split test
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ..config.dataset_config import get_dataset_config
from ..config.path_config import PRED_DIR, REPORTS_DIR
from .extract import extract, load_experiment
from .footprint import reference_envelope
from .pooling import build_membership, common_supervision, pool_fields
from .reference import build_reference, reference_probs
from .scores import (
    cox_calibration, ignorance_bits, murphy_decomposition, year_block_bootstrap,
)
from .spatial import fss_curve, localization_scores, skillful_scale


def load_archive(experiment: str, split: str, calibrated: bool = True) -> dict:
    path = PRED_DIR / f"{experiment}_{split}.npz"
    if not path.exists():
        extract(experiment, split)
    arc = np.load(path)
    side = json.loads((PRED_DIR / f"{experiment}_{split}.json").read_text())

    a, b = 1.0, float(np.log(side["neg_keep_rate"]))
    if calibrated and side.get("calibration"):
        a, b = side["calibration"]["a"], side["calibration"]["b"]
    z = a * arc["logits"].astype(np.float64) + b
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    dates = arc["dates"].astype("datetime64[D]")
    return {
        "experiment": experiment, "dataset": side["dataset"], "sidecar": side,
        "p": p.astype(np.float32), "z": z.astype(np.float32),
        "y": arc["labels"], "mask": arc["masks"], "dates": dates,
        "years": dates.astype("datetime64[Y]").astype(int) + 1970,
        "x": arc["x_coords"], "y_coords": arc["y_coords"],
    }


def _climatology_for(arc: dict, bandwidth_km: float) -> np.ndarray:
    tag = f"{arc['dataset']}_clim_bw{bandwidth_km:g}km"
    ref_path = PRED_DIR / f"{tag}.npz"
    if not ref_path.exists():
        build_reference(arc["dataset"], bandwidth_km)
    return reference_probs(str(ref_path), arc["dates"], arc["p"].shape[1:])


def _per_day_sums(fields: dict, mask: np.ndarray) -> dict:
    m = mask.astype(bool)
    out = {"n": m.reshape(m.shape[0], -1).sum(axis=1).astype(np.float64)}
    for k, v in fields.items():
        out[k] = (v * m).reshape(v.shape[0], -1).sum(axis=1)
    return out


def _by_year(day_sums: dict, years: np.ndarray) -> dict:
    return {int(yr): {k: v[years == yr].sum() for k, v in day_sums.items()}
            for yr in np.unique(years)}


def _skill_stats(per_year: dict, num: str, ref: str, n_boot: int, seed: int) -> dict:
    resolved = year_block_bootstrap(
        per_year, lambda s: (s[ref] - s[num]) / s["n"], n_boot=n_boot, seed=seed)
    ss = year_block_bootstrap(
        per_year, lambda s: 1.0 - s[num] / s[ref], n_boot=n_boot, seed=seed)
    return {"resolved_bits": resolved, "skill_score": ss}


def stage_native(experiments, split, bandwidth_km, n_boot, seed, calibrated) -> dict:
    report = {}
    for exp in experiments:
        arc = load_archive(exp, split, calibrated)
        clim = _climatology_for(arc, bandwidth_km)
        m = arc["mask"].astype(bool)

        ign_model = ignorance_bits(arc["p"].astype(np.float64), arc["y"])
        ign_clim = ignorance_bits(clim, arc["y"])
        bs_model = (arc["p"].astype(np.float64) - arc["y"]) ** 2
        bs_clim = (clim - arc["y"]) ** 2

        day = _per_day_sums(
            {"ign_m": ign_model, "ign_c": ign_clim, "bs_m": bs_model,
             "bs_c": bs_clim, "pos": arc["y"].astype(np.float64)}, m)
        per_year = _by_year(day, arc["years"])

        flat_p, flat_y = arc["p"][m].astype(np.float64), arc["y"][m].astype(np.float64)
        entry = {
            "dataset": arc["dataset"], "split": split, "calibrated": calibrated,
            "n_cell_days": float(day["n"].sum()), "n_events": float(day["pos"].sum()),
            "prevalence": float(day["pos"].sum() / day["n"].sum()),
            "ign_bits": float(day["ign_m"].sum() / day["n"].sum()),
            "ign_bits_clim": float(day["ign_c"].sum() / day["n"].sum()),
            "ignorance": _skill_stats(per_year, "ign_m", "ign_c", n_boot, seed),
            "brier": _skill_stats(per_year, "bs_m", "bs_c", n_boot, seed),
            "murphy": murphy_decomposition(flat_p, flat_y),
            "cox": cox_calibration(arc["z"][m].astype(np.float64), flat_y),
        }
        entry["murphy"]["rel_over_res"] = (
            entry["murphy"]["rel"] / entry["murphy"]["res"]
            if entry["murphy"]["res"] > 0 else float("nan"))
        report[exp] = entry
    return report


def _restrict_to_footprint(arc: dict, footprint: str) -> None:
    x0, x1, y0, y1 = reference_envelope(footprint)
    inside = ((arc["y_coords"] >= y0) & (arc["y_coords"] <= y1))[:, None] \
        & ((arc["x"] >= x0) & (arc["x"] <= x1))[None, :]
    arc["mask"] = arc["mask"] & inside[None, :, :]


def stage_compare(experiments, split, bandwidth_km, footprint, n_boot, seed,
                  calibrated) -> dict:
    # -- paired year-block bootstrap: one resample drives every tier, so the
    #    difference CI reflects shared fire years rather than independent noise
    per_year, native = {}, {}
    for exp in experiments:
        arc = load_archive(exp, split, calibrated)
        _restrict_to_footprint(arc, footprint)
        clim = _climatology_for(arc, bandwidth_km)
        m = arc["mask"].astype(bool)
        day = _per_day_sums(
            {"ign_m": ignorance_bits(arc["p"].astype(np.float64), arc["y"]),
             "ign_c": ignorance_bits(clim, arc["y"]),
             "pos": arc["y"].astype(np.float64)}, m)
        per_year[exp] = _by_year(day, arc["years"])
        native[exp] = {
            "resolved_bits": float((day["ign_c"].sum() - day["ign_m"].sum()) / day["n"].sum()),
            "n_events": float(day["pos"].sum()), "n_cell_days": float(day["n"].sum()),
        }

    years = sorted(per_year[experiments[0]])
    rng = np.random.default_rng(seed)
    base, others = experiments[0], experiments[1:]

    def resolved(exp, pick):
        s = {k: sum(per_year[exp][years[j]][k] for j in pick) for k in ("ign_m", "ign_c", "n")}
        return (s["ign_c"] - s["ign_m"]) / s["n"]

    report = {"footprint": footprint, "split": split, "base": base,
              "native": native, "pairs": {}}
    for exp in others:
        deficits = np.empty(n_boot)
        retained = np.empty(n_boot)
        for i in range(n_boot):
            pick = rng.choice(len(years), size=len(years), replace=True)
            rb, ro = resolved(base, pick), resolved(exp, pick)
            deficits[i] = rb - ro
            retained[i] = ro / rb if rb != 0 else float("nan")
        d_pt = native[base]["resolved_bits"] - native[exp]["resolved_bits"]
        r_pt = (native[exp]["resolved_bits"] / native[base]["resolved_bits"]
                if native[base]["resolved_bits"] != 0 else float("nan"))
        report["pairs"][exp] = {
            "deficit_bits": {"point": d_pt, "lo": float(np.percentile(deficits, 2.5)),
                             "hi": float(np.percentile(deficits, 97.5))},
            "retained_fraction": {"point": r_pt, "lo": float(np.nanpercentile(retained, 2.5)),
                                  "hi": float(np.nanpercentile(retained, 97.5))},
        }
    return report


def stage_pooled(experiments, split, bandwidth_km, footprint, coarse_res,
                 n_boot, seed, calibrated) -> dict:
    x0, x1, y0, y1 = reference_envelope(footprint)
    Hc = int(np.ceil((y1 - y0) / coarse_res))
    Wc = int(np.ceil((x1 - x0) / coarse_res))

    pooled, clim_pooled, membership = {}, {}, {}
    dates_ref = None
    for exp in experiments:
        arc = load_archive(exp, split, calibrated)
        member = build_membership(arc["x"], arc["y_coords"], (x0, y1), coarse_res, (Hc, Wc))
        membership[exp] = member
        pooled[exp] = pool_fields(arc["p"].astype(np.float64), arc["y"], arc["mask"], member)
        clim = _climatology_for(arc, bandwidth_km)
        clim_pooled[exp] = pool_fields(clim, arc["y"], arc["mask"], member)["p"]
        if dates_ref is None:
            dates_ref, years_ref = arc["dates"], arc["years"]
        elif not np.array_equal(arc["dates"], dates_ref):
            raise SystemExit(f"{exp} target dates differ; tiers must share scored days")

    sup = common_supervision({e: v["n_supervised"] for e, v in pooled.items()})
    y_union = np.zeros(sup.shape, dtype=np.uint8)
    for v in pooled.values():
        y_union |= v["y"]

    report = {"coarse_grid": [Hc, Wc], "coarse_res_m": coarse_res,
              "footprint": footprint, "n_cell_days": float(sup.sum()),
              "n_events": float((y_union * sup).sum()), "tiers": {},
              "label_agreement": {}}
    for exp in experiments:
        agree = float((pooled[exp]["y"][sup] == y_union[sup]).mean())
        report["label_agreement"][exp] = agree
        day = {
            "ign_m": (ignorance_bits(pooled[exp]["p"], y_union) * sup).sum(axis=1),
            "ign_c": (ignorance_bits(clim_pooled[exp], y_union) * sup).sum(axis=1),
            "pos": (y_union.astype(np.float64) * sup).sum(axis=1),
            "n": sup.sum(axis=1).astype(np.float64),
        }
        per_year = _by_year(day, years_ref)
        report["tiers"][exp] = {
            "ign_bits": float(day["ign_m"].sum() / day["n"].sum()),
            "ign_bits_clim": float(day["ign_c"].sum() / day["n"].sum()),
            "ignorance": _skill_stats(per_year, "ign_m", "ign_c", n_boot, seed),
        }
    return report


def stage_fss(experiments, split, scales_km, calibrated) -> dict:
    report = {}
    for exp in experiments:
        arc = load_archive(exp, split, calibrated)
        res = get_dataset_config(arc["dataset"]).resolution
        scales_px = sorted({max(1, 2 * round(k * 1000 / res / 2) + 1) for k in scales_km})
        curve = fss_curve(arc["p"].astype(np.float64), arc["y"], arc["mask"], scales_px)
        m = arc["mask"].astype(bool)
        f = float(arc["y"][m].sum() / m.sum())
        curve["scale_km"] = [round(k * res / 1000, 3) for k in curve["scale_px"]]
        curve["event_fraction"] = f
        curve["skillful_scale_km"] = skillful_scale(
            curve["scale_km"], curve["fss"], f)
        report[exp] = curve
    return report


def stage_localization(experiments, split, footprint, coarse_res, calibrated) -> dict:
    x0, x1, y0, y1 = reference_envelope(footprint)
    Hc = int(np.ceil((y1 - y0) / coarse_res))
    Wc = int(np.ceil((x1 - x0) / coarse_res))
    report = {}
    for exp in experiments:
        arc = load_archive(exp, split, calibrated)
        member = build_membership(arc["x"], arc["y_coords"], (x0, y1), coarse_res, (Hc, Wc))
        report[exp] = localization_scores(
            arc["p"].astype(np.float64), arc["y"], arc["mask"], member)
        report[exp]["coarse_res_m"] = coarse_res
    return report


def main():
    ap = argparse.ArgumentParser(description="Score extracted prediction archives")
    ap.add_argument("stage", choices=["extract", "native", "pooled", "fss",
                                      "localization", "compare"])
    ap.add_argument("--experiments", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=["train", "eval", "test"])
    ap.add_argument("--bandwidth-km", type=float, default=20.0)
    ap.add_argument("--footprint", default="cascades500")
    ap.add_argument("--coarse-res", type=float, default=2000.0)
    ap.add_argument("--scales-km", type=float, nargs="+",
                    default=[2, 4, 8, 12, 16, 24, 32, 48])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw", action="store_true",
                    help="analytic subsampling shift instead of the fitted calibrator")
    args = ap.parse_args()
    calibrated = not args.raw

    if args.stage == "extract":
        for exp in args.experiments:
            extract(exp, args.split)
        return

    if args.stage == "native":
        report = stage_native(args.experiments, args.split, args.bandwidth_km,
                              args.n_boot, args.seed, calibrated)
    elif args.stage == "pooled":
        report = stage_pooled(args.experiments, args.split, args.bandwidth_km,
                              args.footprint, args.coarse_res, args.n_boot,
                              args.seed, calibrated)
    elif args.stage == "fss":
        report = stage_fss(args.experiments, args.split, args.scales_km, calibrated)
    elif args.stage == "compare":
        report = stage_compare(args.experiments, args.split, args.bandwidth_km,
                               args.footprint, args.n_boot, args.seed, calibrated)
    else:
        report = stage_localization(args.experiments, args.split, args.footprint,
                                    args.coarse_res, calibrated)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{args.stage}_{args.split}_{'-'.join(args.experiments)}.json"
    out = REPORTS_DIR / name
    out.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))
    print(f"[run_analysis] wrote {out}")


if __name__ == "__main__":
    main()
