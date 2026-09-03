"""
Score extracted prediction archives, one stage per question: native per-tier
skill against climatology, cross-tier comparison on a pooled common grid,
allocation on a fixed evaluation grid, cause skill over the class prior, and
sub-cell localization.

Stages are independent, every one reads through archive.load_archive, and each
writes a single JSON report under REPORTS_DIR.

  python -m fire_fusion.analysis.score native --experiments wa2000-s1 --split test
  python -m fire_fusion.analysis.score pooled --experiments wa2000-s1 wa1000-s1 cascades500-s1
  python -m fire_fusion.analysis.score allocation --experiments cascades500-optimal --footprint self
"""
import argparse
import json

import numpy as np

from ..config.path_config import REPORTS_DIR
from .allocation import (DEFAULT_BUDGETS, day_sums, fit_persistence, fit_temperature,
                         localization_scores, persistence, report as allocation_report,
                         strata)
from .archive import climatology_for, coarse_grid, load_archive, restrict_to_footprint
from .extract import extract
from .grid import (build_membership, common_supervision, evaluation_grid, pool_fields,
                   reference_envelope)
from .scores import (
    cause_ignorance_bits, cox_calibration, ignorance_bits, murphy_decomposition,
    year_block_bootstrap,
)


def _per_day_sums(fields: dict, mask: np.ndarray) -> dict:
    """ Per-day masked sums of supervised cell count 'n' and each field in 'fields'. """
    m = mask.astype(bool)
    out = {"n": m.reshape(m.shape[0], -1).sum(axis=1).astype(np.float64)}
    for k, v in fields.items():
        out[k] = (v * m).reshape(v.shape[0], -1).sum(axis=1)
    return out


def _by_year(day_sums: dict, years: np.ndarray) -> dict:
    """ Per-day sums split into per-year arrays for the bootstrap to resample in blocks. """
    return {int(yr): {k: np.asarray(v)[years == yr] for k, v in day_sums.items()}
            for yr in np.unique(years)}


def _paired(per_year_by_exp: dict) -> dict:
    """ Per-year table merging every experiment's sums under experiment-prefixed
        keys, for a single resample across tiers.
    """
    years = sorted(next(iter(per_year_by_exp.values())))
    return {y: {f"{e}/{k}": v for e, t in per_year_by_exp.items() for k, v in t[y].items()}
            for y in years}


def _skill_stats(per_year: dict, num: str, ref: str, n_boot: int, seed: int) -> dict:
    """ Bootstrap resolved-bits and skill-score statistics comparing 'num' against 'ref'. """
    resolved = year_block_bootstrap(
        per_year, lambda s: (s[ref] - s[num]) / s["n"], n_boot=n_boot, seed=seed)
    ss = year_block_bootstrap(
        per_year, lambda s: 1.0 - s[num] / s[ref], n_boot=n_boot, seed=seed)
    return {"resolved_bits": resolved, "skill_score": ss}


def stage_native(experiments, split, bandwidth_km, n_boot, seed, calibrated) -> dict:
    """ Native per-tier skill report: ignorance and Brier skill against climatology,
        Murphy decomposition, and calibration line, one entry per experiment.
    """
    report = {}
    for exp in experiments:
        arc = load_archive(exp, split, calibrated)
        clim = climatology_for(arc, bandwidth_km)
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


def stage_compare(experiments, split, bandwidth_km, footprint, n_boot, seed,
                  calibrated) -> dict:
    """ Cross-tier comparison report on a shared footprint: each other experiment's
        resolved-bits deficit and retained fraction against the first experiment,
        from a paired year-block bootstrap whose difference confidence interval
        reflects shared fire years.
    """
    per_year, native = {}, {}
    for exp in experiments:
        arc = load_archive(exp, split, calibrated)
        restrict_to_footprint(arc, footprint)
        clim = climatology_for(arc, bandwidth_km)
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

    base, others = experiments[0], experiments[1:]
    paired = _paired(per_year)

    def resolved(exp, s):
        return (s[f"{exp}/ign_c"] - s[f"{exp}/ign_m"]) / s[f"{exp}/n"]

    report = {"footprint": footprint, "split": split, "base": base,
              "native": native, "pairs": {}}
    for exp in others:
        report["pairs"][exp] = {
            "deficit_bits": year_block_bootstrap(
                paired, lambda s: resolved(base, s) - resolved(exp, s), n_boot, seed),
            "retained_fraction": year_block_bootstrap(
                paired, lambda s: resolved(exp, s) / resolved(base, s)
                if resolved(base, s) != 0 else float("nan"), n_boot, seed),
        }
    return report


def stage_pooled(experiments, split, bandwidth_km, footprint, coarse_res,
                 n_boot, seed, calibrated) -> dict:
    """ Cross-tier comparison report on a pooled common grid: ignorance skill
        against climatology per tier, plus label agreement between each tier's
        pooled labels and the union across tiers.
    """
    origin, shape = evaluation_grid(reference_envelope(footprint), coarse_res)

    pooled, clim_pooled, membership = {}, {}, {}
    dates_ref = None
    for exp in experiments:
        arc = load_archive(exp, split, calibrated)
        member = build_membership(arc["x"], arc["y_coords"], origin, coarse_res, shape)
        membership[exp] = member
        pooled[exp] = pool_fields(arc["p"].astype(np.float64), arc["y"], arc["mask"], member)
        clim = climatology_for(arc, bandwidth_km)
        clim_pooled[exp] = pool_fields(clim, arc["y"], arc["mask"], member)["p"]
        if dates_ref is None:
            dates_ref, years_ref = arc["dates"], arc["years"]
        elif not np.array_equal(arc["dates"], dates_ref):
            raise SystemExit(f"{exp} target dates differ; tiers must share scored days")

    sup = common_supervision({e: v["n_supervised"] for e, v in pooled.items()})
    y_union = np.zeros(sup.shape, dtype=np.uint8)
    for v in pooled.values():
        y_union |= v["y"]

    report = {"coarse_grid": list(shape), "coarse_res_m": coarse_res,
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



def _pooled_arm(exp, split, bandwidth_km, footprint, coarse_res, calibrated):
    """ One experiment's fields pooled onto the footprint's coarse grid: probabilities,
        labels, supervision count, climatology, and active-cell indicator, each
        reshaped to the grid, plus years and dates.
    """
    arc = load_archive(exp, split, calibrated)
    member, grid = coarse_grid(arc, footprint, coarse_res)
    clim = climatology_for(arc, bandwidth_km)
    out = pool_fields(arc["p"].astype(np.float64), arc["y"], arc["mask"], member)
    out["clim"] = pool_fields(clim, arc["y"], arc["mask"], member)["p"]
    # -- active pools by any: a coarse cell is active when any fine cell is
    out["active"] = pool_fields(arc["active"].astype(np.float64), arc["y"],
                                np.ones_like(arc["mask"]), member, mode="mean")["p"] > 0
    # -- pooled fields come back flat; the strata dilation needs the grid shape
    for k in ("p", "y", "n_supervised", "clim", "active"):
        out[k] = out[k].reshape(out[k].shape[0], *grid)
    out["years"], out["grid"], out["dates"] = arc["years"], grid, arc["dates"]
    return out


def stage_allocation(experiments, split, bandwidth_km, footprint, coarse_res,
                     n_boot, seed, calibrated, label_source=None,
                     calibrate_split="eval") -> dict:
    """ Placement skill on a fixed evaluation grid, invariant to the burn rate.

        A named footprint scores every model on the same ground; 'self' keeps
        each model's own extent. `label_source`, required on a named footprint,
        is the coarsest tier scored; its pooled labels are the target. (w, r)
        and s are fitted on `calibrate_split`. Numbers reported overall
        and per stratum.
    """
    if footprint != "self" and label_source is None:
        raise SystemExit("a named footprint needs --label-source, the coarsest tier scored")
    report = {"footprint": footprint, "coarse_res_m": coarse_res, "split": split,
              "calibrate_split": calibrate_split, "budgets": list(DEFAULT_BUDGETS), "tiers": {}}
    pooled = {e: _pooled_arm(e, split, bandwidth_km, footprint, coarse_res, calibrated)
              for e in experiments}
    cal = {e: _pooled_arm(e, calibrate_split, bandwidth_km, footprint, coarse_res, calibrated)
           for e in experiments}

    if footprint == "self":
        supervision = {e: pooled[e]["n_supervised"] > 0 for e in experiments}
        targets = {e: pooled[e]["y"] for e in experiments}
        cal_sup = {e: cal[e]["n_supervised"] > 0 for e in experiments}
        cal_y = {e: cal[e]["y"] for e in experiments}
    else:
        if label_source not in pooled:
            raise SystemExit(f"label source '{label_source}' is not among the scored experiments")
        shared = common_supervision({e: v["n_supervised"] for e, v in pooled.items()})
        y_target = pooled[label_source]["y"]
        supervision = {e: shared for e in experiments}
        targets = {e: y_target for e in experiments}
        c_shared = common_supervision({e: v["n_supervised"] for e, v in cal.items()})
        cal_sup = {e: c_shared for e in experiments}
        cal_y = {e: cal[label_source]["y"] for e in experiments}
        report["label_source"] = label_source
        # -- agreement over positives (Jaccard). Over all cell-days it reads
        #    0.999 for any tier; nearly every cell-day is negative
        report["label_agreement"] = {}
        for e in experiments:
            own, tgt = pooled[e]["y"][shared].astype(bool), y_target[shared].astype(bool)
            report["label_agreement"][e] = float((own & tgt).sum() / max((own | tgt).sum(), 1))
        report["n_cell_days"] = float(shared.sum())
        report["n_events"] = float((y_target * shared).sum())

    for exp in experiments:
        sup, y = supervision[exp], targets[exp]
        active = pooled[exp]["active"]
        spread = strata(active, coarse_res)
        # -- fitted on the calibration split, applied unchanged here
        pers = fit_persistence(cal[exp]["clim"], cal[exp]["active"], cal_y[exp], cal_sup[exp])
        temp = fit_temperature(cal[exp]["p"], cal_y[exp], cal_sup[exp])
        fields = {
            "model": pooled[exp]["p"],
            "model_tempered": np.power(pooled[exp]["p"], temp),
            "clim": pooled[exp]["clim"],
            "persistence": persistence(pooled[exp]["clim"], active, sup, pers["w"], pers["r"]),
        }
        entry = {"grid": list(pooled[exp]["grid"]), "temperature": temp,
                 "persistence": {"w": pers["w"], "r_cells": pers["r"]},
                 "mean_supervised_cells_per_day": float(sup.sum((1, 2)).mean())}
        for stratum, sel in (("all", None), ("spread", spread), ("ignition", ~spread)):
            sums = {k: day_sums(f, y, sup, events=sel) for k, f in fields.items()}
            block = allocation_report(sums["model"], {k: sums[k] for k in ("clim", "persistence")})
            tempered = allocation_report(sums["model_tempered"], {k: sums[k] for k in ("clim", "persistence")})
            block["tempered"] = {k: tempered[k] for k in ("skill_vs_clim", "skill_vs_persistence",
                                                           "skill_vs_uniform")} if tempered.get("n_events") else {}
            if block.get("n_events"):
                per_year = _by_year({"bits_m": sums["model"]["bits"], "bits_c": sums["clim"]["bits"],
                                     "bits_p": sums["persistence"]["bits"],
                                     "uniform": sums["model"]["uniform"], "n": sums["model"]["events"]},
                                    pooled[exp]["years"])
                block["skill_vs_clim_ci"] = year_block_bootstrap(
                    per_year, lambda s: 1 - s["bits_m"] / s["bits_c"], n_boot, seed)
                block["skill_vs_persistence_ci"] = year_block_bootstrap(
                    per_year, lambda s: 1 - s["bits_m"] / s["bits_p"], n_boot, seed)
                block["skill_vs_uniform_ci"] = year_block_bootstrap(
                    per_year, lambda s: 1 - s["bits_m"] / s["uniform"], n_boot, seed)
            entry[stratum] = block
        report["tiers"][exp] = entry
    return report


def stage_cause(experiments, split, calibrated) -> dict:
    """ Cause skill over the train-year class prior, on known-cause ignitions. """
    report = {"split": split, "tiers": {}}
    for exp in experiments:
        arc = load_archive(exp, split, calibrated)
        calib = arc["sidecar"].get("calibration") or {}
        a = calib.get("cause_a") if calibrated else None
        b = calib.get("cause_b") if calibrated else None
        entry = cause_ignorance_bits(
            arc["cause_logits"], arc["cause_labels"],
            arc["sidecar"]["cause_counts_train"], a, b)
        entry["calibrated"] = bool(a is not None and b is not None)
        report["tiers"][exp] = entry
    return report


def stage_localization(experiments, split, footprint, coarse_res, calibrated) -> dict:
    """ Sub-cell localization score report per experiment on the footprint's coarse grid. """
    origin, shape = evaluation_grid(reference_envelope(footprint), coarse_res)
    report = {}
    for exp in experiments:
        arc = load_archive(exp, split, calibrated)
        member = build_membership(arc["x"], arc["y_coords"], origin, coarse_res, shape)
        report[exp] = localization_scores(
            arc["p"].astype(np.float64), arc["y"], arc["mask"], member)
        report[exp]["coarse_res_m"] = coarse_res
    return report


def main():
    """ Parse CLI arguments, run the requested scoring stage, and write the report JSON. """
    ap = argparse.ArgumentParser(description="Score extracted prediction archives")
    ap.add_argument("stage", choices=["extract", "native", "pooled", "allocation",
                                      "cause", "localization", "compare"])
    ap.add_argument("--experiments", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=["train", "eval", "test"])
    ap.add_argument("--bandwidth-km", type=float, default=20.0)
    ap.add_argument("--footprint", default="cascades500")
    ap.add_argument("--label-source", default=None,
                    help="experiment whose pooled labels define the scored target on a "
                         "named footprint; required there, and the coarsest tier scored")
    ap.add_argument("--calibrate-split", default="eval", choices=["train", "eval"],
                    help="split whose archive fits the persistence blend and temperature")
    ap.add_argument("--coarse-res", type=float, default=2000.0)
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
    elif args.stage == "allocation":
        report = stage_allocation(args.experiments, args.split, args.bandwidth_km,
                                  args.footprint, args.coarse_res, args.n_boot,
                                  args.seed, calibrated, args.label_source,
                                  args.calibrate_split)
    elif args.stage == "cause":
        report = stage_cause(args.experiments, args.split, calibrated)
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
    print(f"[score] wrote {out}")


if __name__ == "__main__":
    main()
