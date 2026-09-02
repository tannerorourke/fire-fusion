"""
Run a saved checkpoint over one split and persist per-day prediction fields.

Writes <experiment>_<split>.npz under PRED_DIR: raw ignition logits, labels,
supervision masks, and target dates for every stride-aligned window, plus a
JSON sidecar recording checkpoint identity, calibration, and seed so any
downstream score is reproducible from the archive alone.

  python -m fire_fusion.analysis.extract --experiment wa2000-s1 --split test
"""
import argparse
import hashlib
import json

import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from tqdm import tqdm

from ..config.path_config import MODEL_DIR, MODEL_SAVE_DIR, PRED_DIR
from ..dataset.data_loader import init_data_loader
from ..model.model import FireFusionModel
from ..train_utils import checkpoint_name, get_device_config, load_model, set_global_seed


def _last_day(t: torch.Tensor) -> torch.Tensor:
    return t[:, -1] if t.ndim == 4 else t


def load_experiment(experiment: str) -> dict:
    with open(MODEL_DIR / "params.json") as f:
        data = json.load(f)
    if experiment not in data:
        raise SystemExit(f"Unknown experiment '{experiment}'. Options: {sorted(data)}")
    return data[experiment]


def extract(experiment: str, split: str, dataset: str | None = None,
            checkpoint: str | None = None, batch_size: int = 1) -> str:
    params = load_experiment(experiment)
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
    )
    ds = loader.dataset
    mp["n_cause_classes"] = ds.n_cause_classes
    mp["dyn_groups"] = ds.dyn_groups

    model = FireFusionModel(ds.dyn_channels, ds.static_channels, mp=mp).to(device)
    ckpt = checkpoint or checkpoint_name(experiment)
    ckpt_path = MODEL_SAVE_DIR / f"{ckpt}.th"
    load_model(model, f"{ckpt}.th", map_location=device)
    model.eval()

    days = np.asarray(ds.ds.indexes["time"], dtype="datetime64[D]")
    target_days = days[ds.window_starts + ds.window_size - 1]

    H, W = ds.out_size
    n = len(ds)
    logits = np.empty((n, H, W), dtype=np.float32)
    labels = np.empty((n, H, W), dtype=np.uint8)
    masks = np.empty((n, H, W), dtype=bool)

    use_amp = device.type == "cuda"
    i = 0
    with torch.inference_mode():
        for (x_dyn, x_static), golds, mk in tqdm(loader, desc=f"extract {experiment}/{split}"):
            x_dyn, x_static = x_dyn.to(device), x_static.to(device)
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                z, _ = model(x_dyn, x_static)
            z = z.squeeze(1).float().cpu().numpy()
            b = z.shape[0]
            logits[i:i + b] = z
            labels[i:i + b] = _last_day(golds["ign_next"]).numpy().astype(np.uint8)
            masks[i:i + b] = (
                (_last_day(mk["land_mask"]).numpy() == 1)
                & (_last_day(mk["no_act_fire_mask"]).numpy() == 1)
            )
            i += b

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"{experiment}_{split}.npz"
    np.savez_compressed(
        out, logits=logits, labels=labels, masks=masks,
        dates=target_days.astype("datetime64[D]").astype(np.int64),
        y_coords=np.asarray(ds.ds["y"].values, dtype=np.float64)[:H],
        x_coords=np.asarray(ds.ds["x"].values, dtype=np.float64)[:W],
    )

    calib_path = MODEL_SAVE_DIR / f"{ckpt}.calib.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else None
    sidecar = {
        "experiment": experiment, "dataset": dataset, "split": split,
        "checkpoint": ckpt, "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest()[:16],
        "seed": seed, "neg_keep_rate": tp.get("neg_keep_rate", 1.0),
        "calibration": calib, "n_windows": n, "grid": [H, W],
        "torch": torch.__version__,
    }
    (PRED_DIR / f"{experiment}_{split}.json").write_text(json.dumps(sidecar, indent=1))
    print(f"[extract] wrote {out} ({n} windows, {H}x{W})")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description="Persist a checkpoint's per-day prediction fields")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--split", default="test", choices=["train", "eval", "test"])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()
    extract(args.experiment, args.split, args.dataset, args.checkpoint, args.batch_size)


if __name__ == "__main__":
    main()
