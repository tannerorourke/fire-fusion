"""
The product forecast as one archive, so 'allocation', 'joint' and 'cause'
score it like any arm.

p(i, t) = lambda_r(t) q(i | r, t): the regional rate times the static arm's
field renormalized within each region. Within the persistence ring's reach of
active fire the spread arm's field replaces it. Cause is the cause arm's head
at the known-cause ignition cells, else the train-year class prior. The
archive is written under PRED_DIR as forecast_r<km>_<static>[_<spread>] with
a sidecar that records its parts; its logits are the product's log-odds and
carry no subsampling shift.
"""
import json

import numpy as np

from ..config.dataset_config import get_dataset_config
from ..config.path_config import PRED_DIR
from .allocation import PERSIST_KM, cells, dilate
from .archive import load_archive
from .joint import _aligned_rate, regional_product


def forecast_name(static_exp: str, regions_km: float, spread_exp: str | None = None) -> str:
    return f"forecast_r{regions_km:g}_{static_exp}" + (f"_{spread_exp}" if spread_exp else "")


def _cause_cells(exp: str, split: str, day_index: dict) -> tuple:
    # -- sparse cause rows of an archive, re-indexed onto the forecast's days
    raw = np.load(PRED_DIR / f"{exp}_{split}.npz")
    dates = raw["dates"].astype("datetime64[D]")
    idx, logits, labels = raw["cause_index"], raw["cause_logits"], raw["cause_labels"]
    days = dates.astype(np.int64)
    keep = np.array([days[i] in day_index for i in idx[:, 0]], dtype=bool)
    idx = idx[keep].copy()
    idx[:, 0] = [day_index[days[i]] for i in idx[:, 0]]
    return idx, logits[keep], labels[keep]


def assemble(static_exp: str, split: str, regions_km: float, spread_exp: str | None = None,
             cause_exp: str | None = None, persist_km: float = PERSIST_KM,
             calibrated: bool = True) -> str:
    """ Write the product archive for one split and return its experiment name. """
    arc = load_archive(static_exp, split, calibrated)
    res = get_dataset_config(arc["dataset"]).resolution
    mask = arc["mask"].astype(bool)
    rate = _aligned_rate(arc, regions_km)
    p = regional_product(arc["p"].astype(np.float64), mask, rate["regions"], rate["lambda"])

    keep = np.ones(len(arc["dates"]), dtype=bool)
    if spread_exp:
        # -- the dynamic arm's window drops season-opening days; the product
        #    covers the days both arms scored
        dyn = load_archive(spread_exp, split, calibrated)
        pos = {int(d): i for i, d in enumerate(dyn["dates"].astype(np.int64))}
        keep = np.array([int(d) in pos for d in arc["dates"].astype(np.int64)], dtype=bool)
        if not keep.any():
            raise SystemExit(f"{static_exp} and {spread_exp} share no {split} days; their target days "
                             "sit on different stride lattices, re-extract the spread arm")
        sel = np.array([pos[int(d)] for d in arc["dates"][keep].astype(np.int64)], dtype=np.int64)
        near = dilate(arc["active"][keep], cells(persist_km, res)) & mask[keep]
        p = p[keep]
        p[near] = dyn["p"][sel].astype(np.float64)[near]
    dates = arc["dates"][keep]
    day_index = {int(d): i for i, d in enumerate(dates.astype(np.int64))}

    side_s = arc["sidecar"]
    if cause_exp:
        c_idx, c_logits, c_labels = _cause_cells(cause_exp, split, day_index)
    else:
        c_idx, _, c_labels = _cause_cells(static_exp, split, day_index)
        prior = np.asarray(side_s["cause_counts_train"], dtype=np.float64)
        c_logits = np.broadcast_to(np.log(prior / prior.sum()), (len(c_labels), prior.size)).astype(np.float32)

    name = forecast_name(static_exp, regions_km, spread_exp)
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    np.savez_compressed(
        PRED_DIR / f"{name}_{split}.npz",
        logits=np.log(p / (1.0 - p)).astype(np.float32), labels=arc["y"][keep],
        masks=mask[keep], active=arc["active"][keep],
        cause_index=c_idx, cause_logits=c_logits, cause_labels=c_labels,
        dates=dates.astype("datetime64[D]").astype(np.int64),
        y_coords=arc["y_coords"], x_coords=arc["x"],
    )
    sidecar = {
        "experiment": name, "intervention": None, "dataset": arc["dataset"], "split": split,
        "checkpoint": None, "seed": None, "neg_keep_rate": 1.0, "calibration": None,
        "n_windows": int(keep.sum()), "grid": list(p.shape[1:]), "fold": side_s.get("fold", "full"),
        "n_cause_classes": side_s["n_cause_classes"], "n_cause_cells": int(c_labels.size),
        "cause_counts_train": side_s["cause_counts_train"],
        "components": {"static": static_exp, "spread": spread_exp, "cause": cause_exp or "prior",
                       "regions_km": regions_km, "persist_km": persist_km,
                       "n_regions": int(rate["lambda"].shape[1]), "calibrated": calibrated},
    }
    (PRED_DIR / f"{name}_{split}.json").write_text(json.dumps(sidecar, indent=1))
    print(f"[forecast] wrote {name}_{split}.npz ({keep.sum()} days, {sidecar['components']['n_regions']} regions)")
    return name
