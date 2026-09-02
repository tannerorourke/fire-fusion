"""
Separable space-by-season climatology reference forecast for one dataset tier.

The spatial factor is a Gaussian-kernel occurrence rate over train-year events
at a fixed physical bandwidth in km, so every tier smooths over the same ground
regardless of cell size. The seasonal factor is a smoothed day-of-year rate
profile from the same years. Their product is renormalized to the train event
rate and floored so reference ignorance stays bounded on zero-history cells.

Built once per (dataset, bandwidth, floor) and cached under PRED_DIR.

  python -m fire_fusion.analysis.reference --dataset wa2000
"""
import argparse
import json

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from ..config.dataset_config import get_dataset_config
from ..config.path_config import PRED_DIR


def _supervised(ds: xr.Dataset):
    return (ds["land_mask"] == 1) & (ds["no_act_fire_mask"] == 1)


def build_reference(dataset: str, bandwidth_km: float = 20.0, floor: float = 1e-8,
                    seasonal_sigma_days: float = 7.0) -> str:
    cfg = get_dataset_config(dataset)
    ds = xr.open_zarr(cfg.split_path("train"))
    sup = _supervised(ds)
    ign = ds["ign_next"].astype("float64") * sup

    ev_yx = ign.sum("time").values
    exp_yx = sup.sum("time").astype("float64").values
    ev_t = ign.sum(("y", "x")).values
    sup_t = sup.sum(("y", "x")).astype("float64").values
    days = np.asarray(ds.indexes["time"], dtype="datetime64[D]")
    doy = (days - days.astype("datetime64[Y]")).astype(int) + 1

    # -- kernel-smoothed rate: smoothing counts and exposure separately keeps
    #    coastal and masked borders unbiased where the kernel is truncated
    sigma_px = bandwidth_km * 1000.0 / cfg.resolution
    s_ev = gaussian_filter(ev_yx, sigma=sigma_px, mode="constant")
    s_exp = gaussian_filter(exp_yx, sigma=sigma_px, mode="constant")
    spatial = np.where(s_exp > 0, s_ev / np.maximum(s_exp, 1e-12), 0.0)

    ev_doy = np.bincount(doy, weights=ev_t, minlength=367)[1:367]
    sup_doy = np.bincount(doy, weights=sup_t, minlength=367)[1:367]
    ev_doy = gaussian_filter1d(ev_doy, seasonal_sigma_days, mode="wrap")
    sup_doy = gaussian_filter1d(sup_doy, seasonal_sigma_days, mode="wrap")
    rate = ev_yx.sum() / exp_yx.sum()
    seasonal = np.where(sup_doy > 0, ev_doy / np.maximum(sup_doy, 1e-12), rate) / rate

    # -- exact renormalization over the train supervised population, so the
    #    separable product cannot drift from the observed event rate
    mean_spatial = (spatial * exp_yx).sum() / exp_yx.sum()
    mean_seasonal = (seasonal[doy - 1] * sup_t).sum() / sup_t.sum()
    scale = rate / max(mean_spatial * mean_seasonal, 1e-30)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{dataset}_clim_bw{bandwidth_km:g}km"
    out = PRED_DIR / f"{tag}.npz"
    np.savez_compressed(out, spatial=spatial.astype(np.float64), seasonal=seasonal,
                        scale=np.float64(scale), floor=np.float64(floor))
    meta = {
        "dataset": dataset, "bandwidth_km": bandwidth_km, "floor": floor,
        "seasonal_sigma_days": seasonal_sigma_days, "train_rate": rate,
        "train_events": float(ev_yx.sum()), "train_cell_days": float(exp_yx.sum()),
    }
    (PRED_DIR / f"{tag}.json").write_text(json.dumps(meta, indent=1))
    print(f"[reference] wrote {out} (rate={rate:.3e}, events={ev_yx.sum():.0f})")
    return str(out)


def reference_probs(ref_path: str, dates: np.ndarray, shape: tuple) -> np.ndarray:
    """ (D, H, W) climatology probabilities for datetime64[D] target dates. """
    ref = np.load(ref_path)
    days = dates.astype("datetime64[D]")
    doy = (days - days.astype("datetime64[Y]")).astype(int) + 1
    H, W = shape
    field = ref["spatial"][:H, :W] * float(ref["scale"])
    p = field[None, :, :] * ref["seasonal"][doy - 1][:, None, None]
    return np.clip(p, float(ref["floor"]), 0.5).astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description="Build the climatology reference for a tier")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--bandwidth-km", type=float, default=20.0)
    ap.add_argument("--floor", type=float, default=1e-8)
    ap.add_argument("--seasonal-sigma-days", type=float, default=7.0)
    args = ap.parse_args()
    build_reference(args.dataset, args.bandwidth_km, args.floor, args.seasonal_sigma_days)


if __name__ == "__main__":
    main()
