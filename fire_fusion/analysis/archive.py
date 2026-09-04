"""
The one reader of an extraction archive, and the definitions every consumer of a
store shares with it.

An archive is <experiment>_<split>.npz plus its JSON sidecar under PRED_DIR.
The calibrator, the subsampling shift and the sidecar schema are applied once
here for every question scored off the same file. A missing archive is
extracted on demand.

`last_day` and `supervised` live here, shared by everything that touches a
store, extract.py and reference.py included. Those two modules are imported
inside the functions that reach down to them.
"""
import json

import numpy as np

from ..config.path_config import PRED_DIR, REF_DIR
from .grid import build_membership, evaluation_grid, reference_envelope


def last_day(t):
    # -- final-day slice of a (B, T, H, W) window field; non-4D passes through
    return t[:, -1] if t.ndim == 4 else t


def supervised(ds):
    # -- the ignition head's population: land with no active fire, read by name
    #    from a store, a loader's mask dict, or an archive
    return (ds["land_mask"] == 1) & (ds["no_act_fire_mask"] == 1)


def load_archive(experiment: str, split: str, calibrated: bool = True) -> dict:
    """ Probabilities, labels, supervision and cause cells for one archive.

        The head trains against subsampled negatives; log(neg_keep_rate) is
        the analytic shift on the raw logits. A fitted calibrator replaces that
        shift outright.
    """
    path = PRED_DIR / f"{experiment}_{split}.npz"
    if not path.exists():
        # -- lazy import: a cache miss is the only path that loads a checkpoint.
        #    '<exp>@<group>_<mode>' names an intervention archive of <exp>
        from .extract import extract
        base, _, spec = experiment.partition("@")
        extract(base, split, intervene=spec.replace("_", ":") or None)
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
        "active": arc["active"],
        "years": dates.astype("datetime64[Y]").astype(int) + 1970,
        "x": arc["x_coords"], "y_coords": arc["y_coords"],
        # -- cause is stored sparsely, at the known-cause ignition cells alone
        "cause_logits": arc["cause_logits"], "cause_labels": arc["cause_labels"],
    }


def climatology_for(arc: dict, bandwidth_km: float) -> np.ndarray:
    """ Climatology probabilities for the archive's dates, building the archive's
        fold's cached reference if it is missing.
    """
    from .reference import build_reference, reference_probs, reference_tag

    fold = arc["sidecar"].get("fold", "full")
    ref_path = REF_DIR / f"{reference_tag(arc['dataset'], bandwidth_km, fold)}.npz"
    if not ref_path.exists():
        build_reference(arc["dataset"], bandwidth_km, fold=fold)
    return reference_probs(str(ref_path), arc["dates"], arc["p"].shape[1:])


def restrict_to_footprint(arc: dict, footprint: str) -> None:
    # -- narrows the supervision mask in place to the footprint's envelope
    x0, x1, y0, y1 = reference_envelope(footprint)
    inside = ((arc["y_coords"] >= y0) & (arc["y_coords"] <= y1))[:, None] \
        & ((arc["x"] >= x0) & (arc["x"] <= x1))[None, :]
    arc["mask"] = arc["mask"] & inside[None, :, :]


def coarse_grid(arc: dict, footprint: str, coarse_res: float):
    # -- (membership, shape) of a coarse grid: 'self' keeps the archive's extent,
    #    a named footprint restricts supervision to its envelope
    if footprint == "self":
        env = (float(arc["x"].min()), float(arc["x"].max()),
               float(arc["y_coords"].min()), float(arc["y_coords"].max()))
    else:
        env = reference_envelope(footprint)
        restrict_to_footprint(arc, footprint)
    origin, shape = evaluation_grid(env, coarse_res)
    member = build_membership(arc["x"], arc["y_coords"], origin, coarse_res, shape)
    return member, shape
