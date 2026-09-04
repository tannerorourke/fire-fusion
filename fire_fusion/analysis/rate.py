"""
Domain ignition rate: how many cells ignite across the whole grid on a given day.

This is the factor the spatiotemporal model does not carry. Annual ignition
counts span two orders of magnitude, driven by cumulative dryness and seasonal
climate outside a ten-day look-back over a 35 km receptive field. A per-cell
score fitted to the training mean over-predicts an ordinary year several fold.
The rate is fit and scored here as its own number, separate from the placement
objective.

A negative-binomial regression on domain-mean dynamic channels plus day-of-year
harmonics, with log supervised-cell count as a fixed offset; the coefficients
describe a rate per cell-day. The count is overdispersed: a single ignition
lights many cells of a fine grid across a seven-day window, the daily total
clusters, and a Poisson PIT comes out sharply U-shaped. The dispersion is
fitted alongside the coefficients. The reference is the same
smoothed seasonal profile the climatology forecast uses.

  python -m fire_fusion.analysis.rate --dataset wa2000 --fold fold3
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import xarray as xr

from ..config.dataset_config import get_dataset_config
from ..config.feature_config import channel_group_indices
from ..config.path_config import RATE_DIR
from .archive import supervised
from .reference import seasonal_profile

N_HARMONICS = 2


def daily_frame(dataset: str, split: str, fold: str) -> Dict[str, np.ndarray]:
    """ One row per day: domain-mean dynamic channels, the supervised cell count,
        and the observed ignition count. """
    cfg = get_dataset_config(dataset, fold)
    ds = xr.open_zarr(cfg.split_path(split))
    manifest = json.loads(cfg.manifest_path.read_text())
    names = list(manifest["channels"])
    groups = channel_group_indices(names)
    dyn_idx = sorted(groups["MET"] + groups["STATE"])

    sup = supervised(ds)
    n_sup = sup.sum(("y", "x")).astype("float64")
    events = (ds["burn_next"].astype("float64") * sup).sum(("y", "x"))

    # -- masked domain mean: a channel's daily level over the cells that count
    x = ds["X"].isel(channel=dyn_idx).astype("float32")
    means = (x.where(sup).mean(("y", "x")))

    # -- fire already on the ground: cells active today and cells burned in the
    #    last 7 days
    land = ds["land_mask"] == 1
    active = (ds["active"] * land).sum(("y", "x")).astype("float64")
    burns = (ds["burns"] * land).sum(("y", "x")).astype("float64")
    recent = burns.rolling(time=7, min_periods=1).sum()
    fire_state = np.log1p(np.stack([active.values, recent.values], axis=1))

    days = np.asarray(ds.indexes["time"], dtype="datetime64[D]")
    doy = (days - days.astype("datetime64[Y]")).astype(int) + 1
    return {
        "features": np.concatenate([np.nan_to_num(means.values.astype(np.float64)), fire_state], axis=1),
        "feature_names": [names[i] for i in dyn_idx] + ["log1p_active_cells", "log1p_burns_7d"],
        "n_sup": n_sup.values.astype(np.float64),
        "events": events.values.astype(np.float64),
        "doy": doy,
        "dates": days,
        "years": days.astype("datetime64[Y]").astype(int) + 1970,
    }


def harmonics(doy: np.ndarray, n: int = N_HARMONICS) -> np.ndarray:
    """ (n_days, 2*n) stacked sin/cos day-of-year harmonics up to order 'n'. """
    ang = 2.0 * np.pi * doy / 365.25
    return np.concatenate([
        np.stack([np.sin(k * ang), np.cos(k * ang)], axis=1) for k in range(1, n + 1)
    ], axis=1)


def design(frame: Dict[str, np.ndarray], mu: np.ndarray | None = None,
           sd: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Standardized (x, mu, sd) design matrix from the frame's features and
        day-of-year harmonics, fitting mu and sd from 'x' when not given.
    """
    x = np.concatenate([frame["features"], harmonics(frame["doy"])], axis=1)
    if mu is None:
        mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (x - mu) / sd, mu, sd


class CountRate:
    """ log E[count] = offset + w . x + b, under a Poisson or negative-binomial
        likelihood. The negative binomial carries a fitted log-dispersion:
        Var = mu + alpha * mu^2, and alpha -> 0 recovers the Poisson. """

    def __init__(self, n_features: int, l2: float = 1e-2, family: str = "nb"):
        """ Zero-initialized weights, bias, and log-dispersion for a Poisson or
            negative-binomial rate model.
        """
        self.w = torch.zeros(n_features, dtype=torch.float64, requires_grad=True)
        self.b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        self.log_alpha = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        self.l2 = l2
        self.family = family

    def _eta(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return offset + x @ self.w + self.b

    def _nll(self, eta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Mean negative log-likelihood of 'y' under 'eta', Poisson or negative-binomial
        if self.family == "poisson":
            return (torch.exp(eta) - y * eta).mean()
        mu = torch.exp(eta)
        r = 1.0 / torch.exp(self.log_alpha)
        return -(torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.0)
                 + r * (torch.log(r) - torch.log(r + mu))
                 + y * (torch.log(mu) - torch.log(r + mu))).mean()

    @property
    def alpha(self) -> float:
        return float(torch.exp(self.log_alpha).item())

    def fit(self, x: np.ndarray, y: np.ndarray, offset: np.ndarray, steps: int = 400):
        """ Fit weights, bias, and dispersion to '(x, y, offset)' by LBFGS, returning self. """
        xt = torch.as_tensor(x, dtype=torch.float64)
        yt = torch.as_tensor(y, dtype=torch.float64)
        ot = torch.as_tensor(offset, dtype=torch.float64)
        params = [self.w, self.b] + ([] if self.family == "poisson" else [self.log_alpha])
        opt = torch.optim.LBFGS(params, max_iter=steps,
                                tolerance_grad=1e-10, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = self._nll(self._eta(xt, ot), yt) + self.l2 * (self.w ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return self

    @torch.no_grad()
    def predict(self, x: np.ndarray, offset: np.ndarray) -> np.ndarray:
        # Predicted mean count for '(x, offset)'.
        eta = self._eta(torch.as_tensor(x, dtype=torch.float64),
                        torch.as_tensor(offset, dtype=torch.float64))
        return torch.exp(eta).numpy()


def seasonal_reference(train: Dict[str, np.ndarray], doy: np.ndarray,
                       n_sup: np.ndarray, sigma_days: float = 7.0) -> np.ndarray:
    """ Expected event count per day under the same seasonal profile the
        climatology field uses, scaled by 'n_sup'.
    """
    profile = seasonal_profile(train["doy"], train["events"], train["n_sup"], sigma_days)
    return profile[doy - 1] * n_sup


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.maximum(mu, 1e-12)
    term = np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0.0)
    return float(2.0 * (term - (y - mu)).mean())


def pit_histogram(y: np.ndarray, mu: np.ndarray, alpha: float = 0.0,
                  n_bins: int = 10) -> Sequence[float]:
    """ Randomized PIT for a count forecast; a calibrated rate reads flat.
        U-shaped means too narrow, humped in the middle means too wide. """
    from scipy.stats import nbinom, poisson
    rng = np.random.default_rng(0)
    if alpha > 0:
        r = 1.0 / alpha
        prob = r / (r + mu)
        lo, hi = nbinom.cdf(y - 1, r, prob), nbinom.cdf(y, r, prob)
    else:
        lo, hi = poisson.cdf(y - 1, mu), poisson.cdf(y, mu)
    u = lo + rng.random(len(y)) * (hi - lo)
    counts, _ = np.histogram(u, bins=n_bins, range=(0.0, 1.0))
    return (counts / max(len(y), 1)).tolist()


def rate_path(dataset: str, fold: str) -> Path:
    return RATE_DIR / f"rate_{dataset}_{fold}"


def run(dataset: str, fold: str, l2: float, family: str = "nb") -> Tuple[Dict, Dict]:
    """ Fit the count-rate model on the train split and score it on every split.
        Returns the report and the per-day arrays (dates, lambda, seasonal,
        events, n_sup) per split, the rate the joint score reads.
    """
    frames = {s: daily_frame(dataset, s, fold) for s in ("train", "eval", "test")}
    xtr, mu, sd = design(frames["train"])
    model = CountRate(xtr.shape[1], l2=l2, family=family).fit(
        xtr, frames["train"]["events"], np.log(frames["train"]["n_sup"]))
    alpha = 0.0 if family == "poisson" else model.alpha

    report = {"dataset": dataset, "fold": fold, "l2": l2, "family": family,
              "dispersion_alpha": alpha,
              "n_features": int(xtr.shape[1]), "splits": {}}
    arrays = {}
    for split, frame in frames.items():
        x, _, _ = design(frame, mu, sd)
        pred = model.predict(x, np.log(frame["n_sup"]))
        ref = seasonal_reference(frames["train"], frame["doy"], frame["n_sup"])
        y = frame["events"]
        arrays[split] = {"dates": frame["dates"].astype(np.int64), "lambda": pred,
                         "seasonal": ref, "events": y, "n_sup": frame["n_sup"]}
        d_model, d_ref = poisson_deviance(y, pred), poisson_deviance(y, ref)
        report["splits"][split] = {
            "n_days": int(len(y)),
            "mean_observed": float(y.mean()),
            "mean_predicted": float(pred.mean()),
            "mean_reference": float(ref.mean()),
            "deviance": d_model,
            "deviance_reference": d_ref,
            "deviance_skill": 1 - d_model / d_ref if d_ref > 0 else float("nan"),
            "corr_predicted": float(np.corrcoef(pred, y)[0, 1]),
            "corr_reference": float(np.corrcoef(ref, y)[0, 1]),
            "pit": pit_histogram(y, pred, alpha),
            "pit_reference": pit_histogram(y, ref, alpha),
        }
    return report, arrays


def write_rate(dataset: str, fold: str, l2: float = 1e-2, family: str = "nb") -> Dict:
    """ Fit, score, and cache the rate: the report as JSON and the per-day arrays
        as an npz keyed '<split>_<name>', both under RATE_DIR.
    """
    report, arrays = run(dataset, fold, l2, family)
    RATE_DIR.mkdir(parents=True, exist_ok=True)
    base = rate_path(dataset, fold)
    base.with_suffix(".json").write_text(json.dumps(report, indent=1))
    np.savez_compressed(base.with_suffix(".npz"),
                        **{f"{s}_{k}": v for s, arr in arrays.items() for k, v in arr.items()})
    print(f"[rate] wrote {base}.json and .npz")
    return report


def daily_rate(dataset: str, fold: str, split: str) -> Dict[str, np.ndarray]:
    """ One split's per-day rate arrays from the cache, fitting the rate when absent. """
    path = rate_path(dataset, fold).with_suffix(".npz")
    if not path.exists():
        write_rate(dataset, fold)
    arc = np.load(path)
    out = {k: arc[f"{split}_{k}"] for k in ("dates", "lambda", "seasonal", "events", "n_sup")}
    out["dates"] = out["dates"].astype("datetime64[D]")
    return out


def main():
    ap = argparse.ArgumentParser(description="Fit and score the domain ignition rate")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", default="full")
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--family", default="nb", choices=["nb", "poisson"])
    args = ap.parse_args()
    print(json.dumps(write_rate(args.dataset, args.fold, args.l2, args.family), indent=1))


if __name__ == "__main__":
    main()
