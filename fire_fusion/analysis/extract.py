"""
Run a saved checkpoint over one split and persist per-day prediction fields.

Writes <experiment>_<split>.npz under PRED_DIR: raw ignition logits, labels,
supervision masks, and target dates for every stride-aligned window, plus a
JSON sidecar recording checkpoint identity, calibration, and seed. Any
downstream score is reproducible from the archive alone.

Cause is stored sparsely, at the known-cause ignition cells the cause claim is
conditional on. Those cells number in the thousands against tens of millions
of grid cells.

An intervention re-runs the same checkpoint with one input group interfered
with and writes <experiment>@<group>_<mode>_<split>.npz, which every scoring
stage accepts as an experiment name:

  met:year     MET from the window one season earlier; season and geography
               survive, the year's actual weather is wrong
  met:mean     MET replaced by its per-channel mean over the split
  state:year   the same shift on STATE, which carries the active-fire signal
  wind:swap    the two wind components exchanged

  python -m fire_fusion.analysis.extract --experiment wa2000-s1 --split test
  python -m fire_fusion.analysis.extract --experiment wa4000-s1 --intervene met:year
"""
import argparse
import hashlib
import json

import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..config.path_config import MODEL_DIR, MODEL_SAVE_DIR, PRED_DIR
from ..dataset.data_loader import init_data_loader
from ..model.model import FireFusionModel
from ..training.utils import checkpoint_name, get_device_config, load_model, set_global_seed
from .archive import last_day, supervised


def load_experiment(experiment: str) -> dict:
    with open(MODEL_DIR / "params.json") as f:
        data = json.load(f)
    if experiment not in data:
        raise SystemExit(f"Unknown experiment '{experiment}'. Options: {sorted(data)}")
    return data[experiment]


class Intervened(Dataset):
    """ The split's windows with one dynamic group interfered with. Year modes
        read the window `shift` positions earlier, one season at the stride;
        mean mode holds the group at its split mean; swap exchanges two channels.
    """
    def __init__(self, ds, group: str, mode: str):
        self.ds, self.mode = ds, mode
        n_years = len(np.unique(np.asarray(ds.ds.indexes["time"], dtype="datetime64[Y]")))
        self.shift = max(1, len(ds) // n_years)
        if group == "wind":
            pos = {c: i for i, c in enumerate(ds._dyn_idx)}
            names = ds.feature_names
            self.idx = [pos[names.index("wind_dir_ew")], pos[names.index("wind_dir_ns")]]
        else:
            self.idx = ds.dyn_groups[group.upper()]
        if mode == "mean":
            # -- mean over the windows' days, each window weighted alike
            day = ds.X.isel(channel=[ds._dyn_idx[i] for i in self.idx]).mean(("y", "x")).values
            T = ds.window_size
            self.mean = torch.as_tensor(np.stack([day[s:s + T].mean(0) for s in ds.window_starts]).mean(0),
                                        dtype=torch.float32).view(1, -1, 1, 1)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        (x_dyn, x_static), golds, mk = self.ds[i]
        if self.mode == "year":
            x_dyn[:, self.idx] = self.ds[(i - self.shift) % len(self.ds)][0][0][:, self.idx]
        elif self.mode == "mean":
            x_dyn[:, self.idx] = self.mean
        else:
            x_dyn[:, self.idx] = x_dyn[:, self.idx[::-1]]
        return (x_dyn, x_static), golds, mk


def extract(experiment: str, split: str, dataset: str | None = None,
            checkpoint: str | None = None, batch_size: int = 1,
            intervene: str | None = None) -> str:
    """ Run the experiment's checkpoint over the split and write the prediction archive.

        Persists per-day ignition logits, labels, supervision and active masks, and
        the sparse cause logits and labels at known-cause ignition cells, plus a JSON
        sidecar of checkpoint identity, calibration, and seed. Returns the archive path.
    """
    params = load_experiment(experiment)
    name = experiment if intervene is None else f"{experiment}@{intervene.replace(':', '_')}"
    tp, mp = params["training"], dict(params["model"])
    dataset = dataset or params["dataset"]
    seed = tp["seed"]
    set_global_seed(seed)
    device, num_workers = get_device_config(maximum=tp.get("max_workers", 8))

    loader = init_data_loader(
        split, dataset, num_workers, batch_size,
        window_size=tp.get("window_size", 10), window_stride=tp.get("window_stride", 2),
        seed=seed, encoder_depth=mp.get("encoder_depth", 1),
        attn_window=mp["win_spatial_mixing"]["window_size"],
        fold=tp.get("fold", "full"),
    )
    ds = loader.dataset
    mp["n_cause_classes"] = ds.n_cause_classes
    mp["dyn_groups"] = ds.dyn_groups

    model = FireFusionModel(ds.dyn_channels, ds.static_channels, mp=mp).to(device)
    ckpt = checkpoint or checkpoint_name(experiment)
    ckpt_path = MODEL_SAVE_DIR / f"{ckpt}.th"
    load_model(model, f"{ckpt}.th", map_location=device)
    model.eval()

    if intervene is not None:
        group, mode = intervene.split(":")
        loader = DataLoader(Intervened(ds, group, mode), batch_size=batch_size,
                            num_workers=loader.num_workers, pin_memory=loader.pin_memory)

    days = np.asarray(ds.ds.indexes["time"], dtype="datetime64[D]")
    target_days = days[ds.window_starts + ds.window_size - 1]

    H, W = ds.out_size
    n = len(ds)
    logits = np.empty((n, H, W), dtype=np.float32)
    labels = np.empty((n, H, W), dtype=np.uint8)
    masks = np.empty((n, H, W), dtype=bool)
    active = np.empty((n, H, W), dtype=bool)
    cause_index, cause_logits, cause_labels = [], [], []

    use_amp = device.type == "cuda"
    i = 0
    with torch.inference_mode():
        for (x_dyn, x_static), golds, mk in tqdm(loader, desc=f"extract {name}/{split}"):
            x_dyn, x_static = x_dyn.to(device), x_static.to(device)
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                z, zc = model(x_dyn, x_static)
            z = z.squeeze(1).float().cpu().numpy()
            b = z.shape[0]
            logits[i:i + b] = z
            ign = last_day(golds["burn_next"]).numpy().astype(np.uint8)
            cause = last_day(golds["burn_next_cause"]).numpy()
            mk_last = {k: last_day(v).numpy() for k, v in mk.items()}
            sup = supervised(mk_last)
            labels[i:i + b] = ign
            masks[i:i + b] = sup
            active[i:i + b] = mk_last["active"].astype(bool)

            keep = (ign == 1) & (cause != -1) & sup
            if keep.any():
                bi, yi, xi = np.nonzero(keep)
                cause_index.append(np.stack([bi + i, yi, xi], axis=1).astype(np.int32))
                cause_logits.append(
                    zc.permute(0, 2, 3, 1).float().cpu().numpy()[keep].astype(np.float32)
                )
                cause_labels.append(cause[keep].astype(np.int8))
            i += b

    n_cause = mp["n_cause_classes"]
    cause_index = (np.concatenate(cause_index) if cause_index
                   else np.empty((0, 3), dtype=np.int32))
    cause_logits = (np.concatenate(cause_logits) if cause_logits
                    else np.empty((0, n_cause), dtype=np.float32))
    cause_labels = (np.concatenate(cause_labels) if cause_labels
                    else np.empty((0,), dtype=np.int8))

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"{name}_{split}.npz"
    np.savez_compressed(
        out, logits=logits, labels=labels, masks=masks, active=active,
        cause_index=cause_index, cause_logits=cause_logits, cause_labels=cause_labels,
        dates=target_days.astype("datetime64[D]").astype(np.int64),
        y_coords=np.asarray(ds.ds["y"].values, dtype=np.float64)[:H],
        x_coords=np.asarray(ds.ds["x"].values, dtype=np.float64)[:W],
    )

    calib_path = MODEL_SAVE_DIR / f"{ckpt}.calib.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else None
    sidecar = {
        "experiment": experiment, "intervention": intervene, "dataset": dataset, "split": split,
        "checkpoint": ckpt, "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest()[:16],
        "seed": seed, "neg_keep_rate": tp.get("neg_keep_rate", 1.0),
        "calibration": calib, "n_windows": n, "grid": [H, W],
        "fold": tp.get("fold", "full"),
        "n_cause_classes": n_cause,
        "n_cause_cells": int(cause_labels.size),
        "cause_counts_train": list(ds.cause_counts),
        "torch": torch.__version__,
    }
    (PRED_DIR / f"{name}_{split}.json").write_text(json.dumps(sidecar, indent=1))
    print(f"[extract] wrote {out} ({n} windows, {H}x{W}, "
          f"{cause_labels.size} known-cause cells)")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Persist a checkpoint's per-day prediction fields")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--split", default="test", choices=["train", "eval", "test"])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--intervene", default=None,
                    choices=["met:year", "met:mean", "state:year", "wind:swap"],
                    help="re-run with one input group interfered with; archive named <exp>@<group>_<mode>")
    args = ap.parse_args()
    extract(args.experiment, args.split, args.dataset, args.checkpoint, args.batch_size, args.intervene)


if __name__ == "__main__":
    main()
