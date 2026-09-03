"""
Separable space-by-season climatology reference forecast for one dataset tier.

The spatial factor is a Gaussian-kernel occurrence rate over train-year events
at a fixed physical bandwidth in km; every tier smooths over the same ground
regardless of cell size. The seasonal factor is a smoothed day-of-year rate
profile from the same years. Their product is renormalized to the train event
rate and floored, bounding reference ignorance on zero-history cells.

Built once per (dataset, bandwidth, floor) and cached under REF_DIR.

  python -m fire_fusion.analysis.reference --dataset wa2000
"""
import argparse
import json

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from ..config.dataset_config import get_dataset_config
from ..config.path_config import REF_DIR
from .archive import supervised


def seasonal_profile(doy: np.ndarray, events: np.ndarray, exposure: np.ndarray,
                     sigma_days: float = 7.0) -> np.ndarray:
    """ Smoothed day-of-year event rate per cell-day, indexed by day 1..366.

        Counts and exposure are smoothed separately; partial exposure at a
        season window's edges does not read as a rate cliff.
    """
    ev = np.bincount(doy, weights=events, minlength=367)[1:367]
    ex = np.bincount(doy, weights=exposure, minlength=367)[1:367]
    ev = gaussian_filter1d(ev, sigma_days, mode="wrap")
    ex = gaussian_filter1d(ex, sigma_days, mode="wrap")
    rate = events.sum() / exposure.sum()
    return np.where(ex > 0, ev / np.maximum(ex, 1e-12), rate)


def reference_tag(dataset: str, bandwidth_km: float, fold: str = "full") -> str:
    """ Cache filename stem for the dataset's climatology reference, keyed by
        fold, with the default fold carrying no suffix.
    """
    fold_part = "" if fold == "full" else f"_{fold}"
    return f"{dataset}{fold_part}_clim_bw{bandwidth_km:g}km"


def build_reference(dataset: str, bandwidth_km: float = 20.0, floor: float = 1e-8,
                    seasonal_sigma_days: float = 7.0, fold: str = "full") -> str:
    """ Build the dataset's climatology reference for the fold's train years and
        write it under REF_DIR.

        Writes the smoothed spatial rate, the seasonal profile, and the
        renormalizing scale to an npz file, plus a JSON metadata sidecar, and
        returns the npz path.
    """
    cfg = get_dataset_config(dataset, fold)
    ds = xr.open_zarr(cfg.split_path("train"))
    sup = supervised(ds)
    ign = ds["burn_next"].astype("float64") * sup

    ev_yx = ign.sum("time").values
    exp_yx = sup.sum("time").astype("float64").values
    ev_t = ign.sum(("y", "x")).values
    sup_t = sup.sum(("y", "x")).astype("float64").values
    days = np.asarray(ds.indexes["time"], dtype="datetime64[D]")
    doy = (days - days.astype("datetime64[Y]")).astype(int) + 1

    # -- kernel-smoothed rate: counts and exposure smoothed separately, unbiased
    #    at coastal and masked borders where the kernel is truncated
    sigma_px = bandwidth_km * 1000.0 / cfg.resolution
    s_ev = gaussian_filter(ev_yx, sigma=sigma_px, mode="constant")
    s_exp = gaussian_filter(exp_yx, sigma=sigma_px, mode="constant")
    spatial = np.where(s_exp > 0, s_ev / np.maximum(s_exp, 1e-12), 0.0)

    rate = ev_yx.sum() / exp_yx.sum()
    seasonal = seasonal_profile(doy, ev_t, sup_t, seasonal_sigma_days) / rate

    # -- exact renormalization of the separable product to the observed event
    #    rate over the train supervised population
    mean_spatial = (spatial * exp_yx).sum() / exp_yx.sum()
    mean_seasonal = (seasonal[doy - 1] * sup_t).sum() / sup_t.sum()
    scale = rate / max(mean_spatial * mean_seasonal, 1e-30)

    REF_DIR.mkdir(parents=True, exist_ok=True)
    tag = reference_tag(dataset, bandwidth_km, fold)
    out = REF_DIR / f"{tag}.npz"
    np.savez_compressed(out, spatial=spatial.astype(np.float64), seasonal=seasonal,
                        scale=np.float64(scale), floor=np.float64(floor))
    meta = {
        "dataset": dataset, "fold": fold, "bandwidth_km": bandwidth_km, "floor": floor,
        "seasonal_sigma_days": seasonal_sigma_days, "train_rate": rate,
        "train_events": float(ev_yx.sum()), "train_cell_days": float(exp_yx.sum()),
    }
    (REF_DIR / f"{tag}.json").write_text(json.dumps(meta, indent=1))
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
    """ Parse CLI arguments and build the climatology reference for one dataset. """
    ap = argparse.ArgumentParser(description="Build the climatology reference for a tier")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--bandwidth-km", type=float, default=20.0)
    ap.add_argument("--floor", type=float, default=1e-8)
    ap.add_argument("--seasonal-sigma-days", type=float, default=7.0)
    ap.add_argument("--fold", default="full")
    args = ap.parse_args()
    build_reference(args.dataset, args.bandwidth_km, args.floor, args.seasonal_sigma_days, args.fold)


if __name__ == "__main__":
    main()
