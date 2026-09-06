"""Turn a trained FireFusion checkpoint into per-cell ignition probabilities.

Input:
    - Dynamic cube (B, T, C_dyn, H, W) spanning days [t_0..t_n]
    - static maps (B, C_static, H, W) read at t_n;

Output:
    - A (B, 1, H, W) map of P(fresh ignition within 7 days of t_n) in [0, 1]. '

The model emits raw logits; a fitted Platt calibrator maps them to probabilities. 
Absent a fitted sidecar, the analytic correction for the training-time negative
subsampling stands in.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .analysis.archive import last_day
from .config.dataset_config import get_dataset_config
from .config.feature_config import DYNAMIC_GROUPS, STATIC_GROUPS, channel_group_indices
from .config.path_config import MODEL_DIR, PLOTS_DIR
from .dataset.data_loader import init_data_loader
from .model.model import FireFusionModel
from .training.calibration import PlattScaler
from .training.utils import checkpoint_name, get_device_config, load_calibration, load_model


parser = argparse.ArgumentParser(description="Predict per-cell ignition probability for t_{n+1}")
parser.add_argument("--experiment", default="smoke",
                    help="params.json experiment the checkpoint was trained with")
parser.add_argument("--dataset", default=None,
                    help="override the dataset the experiment names")
parser.add_argument("--checkpoint", default=None,
                    help="defaults to the experiment's own checkpoint")
parser.add_argument("--calib", default=None,
                    help="calibration sidecar name; defaults to the checkpoint's")
parser.add_argument("--split", default="eval", choices=["train", "eval", "test"])
parser.add_argument("--batches", type=int, default=1,
                    help="how many batches to summarize and plot")


class FirePredictor:
    """ A checkpoint plus its calibrator, applied to input cubes. """
    def __init__(self, model: FireFusionModel, calibrator: PlattScaler, device: torch.device):
        self.model = model.eval()
        self.calibrator = calibrator
        self.device = device

    @torch.no_grad()
    def predict_proba(
        self, x_dyn: torch.Tensor, x_static: torch.Tensor,
        land_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """ (B, T, C_dyn, H, W) + (B, C_static, H, W) -> (B, 1, H, W) probabilities.

            - land_mask: 1 where usable, marks non-land cells NaN.
        """
        x_dyn = x_dyn.to(self.device)
        x_static = x_static.to(self.device)
        ign_logits, _ = self.model(x_dyn, x_static)   # (B, 1, H, W)
        probs = self.calibrator.probs(ign_logits.float())

        if land_mask is not None:
            lm = land_mask.to(probs.device)
            if lm.dim() == probs.dim() - 1:           # (B, H, W) -> (B, 1, H, W)
                lm = lm.unsqueeze(1)
            probs = probs.masked_fill(lm != 1, float("nan"))
        return probs


def load_predictor(
    dataset_name: str | None = None,
    experiment: str = "smoke",
    checkpoint: str | None = None,
    calib: str | None = None,
    device: torch.device | None = None,
) -> FirePredictor:
    """ Rebuild the model, load weights, and attach a calibrator. """
    if device is None:
        device, _ = get_device_config(maximum=1)

    with open(f"{MODEL_DIR}/params.json") as f:
        params = json.load(f)[experiment]
    if dataset_name is None:
        dataset_name = params["dataset"]
    if checkpoint is None:
        checkpoint = f"{checkpoint_name(experiment)}.th"

    fold = params["training"].get("fold", "full")
    manifest = json.loads(get_dataset_config(dataset_name, fold).manifest_path.read_text())

    # -- Derive the model's static and dynamic channel counts
    groups = channel_group_indices(list(manifest["channels"]))
    dyn_idx = sorted(i for g in DYNAMIC_GROUPS for i in groups[g])
    dyn_pos = {c: i for i, c in enumerate(dyn_idx)}
    static_channels = sum(len(groups[g]) for g in STATIC_GROUPS) + len(groups["SCALAR"])

    model_params = dict(params["model"])
    model_params["n_cause_classes"] = int(manifest["n_cause_classes"])
    model_params["dyn_groups"] = {
        name: sorted(dyn_pos[c] for c in groups[name]) for name in DYNAMIC_GROUPS
    }

    model = FireFusionModel(len(dyn_idx), static_channels, model_params).to(device)
    load_model(model, checkpoint, map_location=device)
    model.eval()

    # -- the head trains against subsampled negatives. 1/r inverts it exactly
    prior_pos_weight = 1.0 / params["training"].get("neg_keep_rate", 1.0)
    scaler = PlattScaler(prior_pos_weight=prior_pos_weight).to(device)
    sidecar = calib if calib is not None else Path(checkpoint).stem
    calib_params = load_calibration(sidecar)
    if calib_params is not None:
        scaler.load_state(calib_params)
        print(f"[predict] calibration a={calib_params['a']:.4f} b={calib_params['b']:.4f} "
              f"(ECE {calib_params.get('ece_before', float('nan')):.4f} -> "
              f"{calib_params.get('ece_after', float('nan')):.4f})")
    else:
        print(f"[predict] no calibration sidecar for '{sidecar}'; analytic prior "
              f"b=log(neg_keep_rate)={-math.log(prior_pos_weight):.4f}")

    return FirePredictor(model, scaler, device)


def plot_XY_grid(
    grid_2d: np.ndarray,
    land_mask: torch.Tensor | np.ndarray | None = None,
    title: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    save_path: str | None = None,
):
    """ Heatmap of a continuous field [H, W]; everything the land mask excludes
        renders black. """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    data = np.array(grid_2d, dtype=float)
    if land_mask is not None:
        if isinstance(land_mask, torch.Tensor):
            land_mask = land_mask.detach().cpu().numpy()
        data = np.ma.masked_where(~np.array(land_mask).astype(bool), data)

    cmap = matplotlib.colormaps["coolwarm"].copy()
    cmap.set_bad(color="black")
    vmin = np.nanmin(data) if vmin is None else vmin
    vmax = np.nanmax(data) if vmax is None else vmax

    plt.figure()
    im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Value")
    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()


def main():
    """ Load a predictor, run prediction batches from the configured split, and
        save probability heatmaps under PLOTS_DIR.
    """
    args = parser.parse_args()

    with open(f"{MODEL_DIR}/params.json") as f:
        params = json.load(f)[args.experiment]
    dataset = args.dataset or params["dataset"]

    predictor = load_predictor(dataset, args.experiment, args.checkpoint, args.calib)
    loader = init_data_loader(
        args.split, dataset, num_workers=0, batch_size=1,
        encoder_depth=params["model"]["encoder_depth"],
        attn_window=params["model"]["win_spatial_mixing"]["window_size"],
        fold=params["training"].get("fold", "full"),
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for i, ((x_dyn, x_static), _golds, masks) in enumerate(loader):
        land = last_day(masks["land_mask"])
        probs = predictor.predict_proba(x_dyn, x_static, land_mask=land)   # (B, 1, H, W)

        finite = probs[torch.isfinite(probs)]
        print(f"[predict] batch {i}: P(fire) over land  min={finite.min():.3e}  "
              f"mean={finite.mean():.3e}  max={finite.max():.3e}")

        grid = probs[0, 0].cpu().numpy()
        vmax = float(np.nanmax(grid)) if np.isfinite(grid).any() else 1.0
        plot_XY_grid(
            grid, land_mask=land[0],
            title=f"P(fire at t_+1)  [{args.split} #{i}]",
            vmin=0.0, vmax=vmax,
            save_path=str(PLOTS_DIR / f"pred_proba_{args.split}_{i}.png"),
        )

        if i + 1 >= args.batches:
            break


if __name__ == "__main__":
    main()
