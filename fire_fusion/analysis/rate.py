"""
Ignition rate: how many cells ignite in a region on a given day.

This is the factor the spatiotemporal model does not carry. Annual ignition
counts span two orders of magnitude, driven by cumulative dryness and seasonal
climate outside a ten-day look-back over a 35 km receptive field. The rate is
fit and scored here as its own number, separate from the placement objective.

The grid is partitioned into a lattice of square regions (`--regions` km; the
statewide fit is the one-region case). One negative-binomial regression is
pooled across regions: region-mean dynamic channels, region fire state,
season-to-date dryness, day-of-year harmonics, a region intercept each, and
log supervised-cell count as a fixed offset. The count is overdispersed: a
single fire lights many cells of a fine grid across a seven-day window, the
daily total clusters, and a Poisson PIT comes out sharply U-shaped. The
dispersion is fitted alongside the coefficients.

Three references per region: the statewide seasonal profile times the
region's train-year share of events, the statewide fitted rate times that
share, and the region's own seasonal profile. Deviance skill and PIT are
reported pooled and per region. Blocks under 5% of the median supervised
count merge into their largest neighbour.

  python -m fire_fusion.analysis.rate --dataset wa4000 --fold fold3
  python -m fire_fusion.analysis.rate --dataset wa4000 --fold fold3 --regions 100
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import xarray as xr
from scipy import sparse

from ..config.dataset_config import get_dataset_config
from ..config.feature_config import DYNAMIC_GROUPS, channel_group_indices
from ..config.path_config import RATE_DIR
from .archive import supervised
from .reference import seasonal_profile

N_HARMONICS = 2
SEVERITY_CHANNELS = ("dead_fmo_1000hr", "dead_fmo_100hr", "precip_5d")
MERGE_FRACTION = 0.05
OWN_PROFILE_FLOOR = 0.1
TIME_CHUNK = 64


def region_map(sup_mean: np.ndarray, res_m: float, regions_km: float | None) -> np.ndarray:
    """ (H, W) int region label per cell, -1 where never supervised. Square
        blocks of `regions_km` on the tier's grid; one region when None. Blocks
        whose mean supervised count is under MERGE_FRACTION of the median merge
        into their largest 4-neighbour, or the nearest block by centroid when
        they have none. Labels are dense, in scan order.
    """
    H, W = sup_mean.shape
    if not regions_km:
        lab = np.where(sup_mean > 0, 0, -1)
        return lab.astype(np.int32)
    block = max(1, int(round(regions_km * 1000.0 / res_m)))
    nbx = -(-W // block)
    rows, cols = np.indices((H, W))
    lab = (rows // block) * nbx + cols // block
    lab = np.where(sup_mean > 0, lab, -1)

    def size(l):
        return float(sup_mean[lab == l].sum())

    while True:
        ids = [l for l in np.unique(lab) if l >= 0]
        sizes = {l: size(l) for l in ids}
        median = float(np.median(list(sizes.values())))
        small = [l for l in ids if sizes[l] < MERGE_FRACTION * median]
        if len(small) == 0 or len(ids) == 1:
            break
        l = min(small, key=sizes.get)
        cells = lab == l
        # -- 4-neighbour adjacency: any other label touching this block's cells
        touch = np.zeros_like(cells)
        touch[1:] |= cells[:-1]; touch[:-1] |= cells[1:]
        touch[:, 1:] |= cells[:, :-1]; touch[:, :-1] |= cells[:, 1:]
        neigh = [n for n in np.unique(lab[touch & ~cells]) if n >= 0]
        if neigh:
            target = max(neigh, key=sizes.get)
        else:
            cy, cx = rows[cells].mean(), cols[cells].mean()
            others = [n for n in ids if n != l]
            target = min(others, key=lambda n: (rows[lab == n].mean() - cy) ** 2
                         + (cols[lab == n].mean() - cx) ** 2)
        lab[cells] = target

    dense = {l: i for i, l in enumerate(l for l in np.unique(lab) if l >= 0)}
    out = np.full(lab.shape, -1, dtype=np.int32)
    for l, i in dense.items():
        out[lab == l] = i
    return out


def membership(regions: np.ndarray, n_regions: int | None = None) -> sparse.csr_matrix:
    # -- (R, H*W) 0/1 matrix from a region map; R is given when the map was
    #    trimmed to an aligned grid and may have lost whole blocks
    flat = regions.ravel()
    idx = np.flatnonzero(flat >= 0)
    R = int(flat.max()) + 1 if n_regions is None else n_regions
    return sparse.csr_matrix((np.ones(idx.size), (flat[idx], idx)), shape=(R, flat.size))


def season_blocks(days: np.ndarray) -> np.ndarray:
    # -- (D,) season id, incremented at every gap in the daily index
    step = np.diff(days.astype("datetime64[D]")).astype(int)
    return np.concatenate([[0], np.cumsum(step != 1)])


def season_to_date_mean(x: np.ndarray, season: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    for s in np.unique(season):
        sel = season == s
        n = np.arange(1, sel.sum() + 1).reshape(-1, *([1] * (x.ndim - 1)))
        out[sel] = np.cumsum(x[sel], axis=0) / n
    return out


def daily_frame(dataset: str, split: str, fold: str,
                regions_km: float | None = None) -> Dict[str, np.ndarray]:
    """ One row per (day, region): region-mean dynamic channels, region fire
        state, season-to-date dryness, the supervised cell count and the
        observed ignition count. Regions come from the fold's train-split
        supervision when the split is not train.
    """
    cfg = get_dataset_config(dataset, fold)
    ds = xr.open_zarr(cfg.split_path(split))
    manifest = json.loads(cfg.manifest_path.read_text())
    names = list(manifest["channels"])
    groups = channel_group_indices(names)
    dyn_idx = sorted(i for g in DYNAMIC_GROUPS for i in groups[g])
    dyn_names = [names[i] for i in dyn_idx]

    sup = supervised(ds)
    train_sup = sup if split == "train" else supervised(xr.open_zarr(cfg.split_path("train")))
    regions = region_map(train_sup.mean("time").values, cfg.resolution, regions_km)
    M = membership(regions)
    R = M.shape[0]

    D = ds.sizes["time"]
    C = len(dyn_idx)
    sums = np.zeros((D, C, R)); n_sup = np.zeros((D, R)); events = np.zeros((D, R))
    active = np.zeros((D, R)); burns = np.zeros((D, R))
    land = (ds["land_mask"] == 1)
    for t0 in range(0, D, TIME_CHUNK):
        sl = slice(t0, min(t0 + TIME_CHUNK, D))
        s = sup.isel(time=sl).values.reshape(-1, M.shape[1]).astype(np.float64)
        x = ds["X"].isel(time=sl, channel=dyn_idx).values.astype(np.float64)
        x = np.nan_to_num(x).reshape(x.shape[0], C, -1)
        n_sup[sl] = s @ M.T
        sums[sl] = ((x * s[:, None, :]).reshape(-1, M.shape[1]) @ M.T).reshape(-1, C, R)
        events[sl] = (ds["burn_next"].isel(time=sl).values.reshape(-1, M.shape[1]) * s) @ M.T
        l = land.isel(time=sl).values.reshape(-1, M.shape[1]).astype(np.float64)
        active[sl] = (ds["active"].isel(time=sl).values.reshape(-1, M.shape[1]) * l) @ M.T
        burns[sl] = (ds["burns"].isel(time=sl).values.reshape(-1, M.shape[1]) * l) @ M.T

    # -- masked region mean; a region with no supervised cell that day takes
    #    the statewide mean
    means = sums / np.maximum(n_sup, 1)[:, None, :]
    state = sums.sum(2) / np.maximum(n_sup.sum(1), 1)[:, None]
    means = np.where(n_sup[:, None, :] > 0, means, state[:, :, None])
    means = means.transpose(0, 2, 1)                       # (D, R, C)

    # -- fire already on the ground: cells active today and cells burned in
    #    the last 7 days
    recent = np.stack([burns[max(0, t - 6):t + 1].sum(0) for t in range(D)])
    fire_state = np.log1p(np.stack([active, recent], axis=2))     # (D, R, 2)

    days = np.asarray(ds.indexes["time"], dtype="datetime64[D]")
    season = season_blocks(days)
    sev_idx = [dyn_names.index(c) for c in SEVERITY_CHANNELS]
    severity = season_to_date_mean(means[:, :, sev_idx], season)  # (D, R, 3)

    doy = (days - days.astype("datetime64[Y]")).astype(int) + 1
    return {
        "features": np.concatenate([means, fire_state, severity], axis=2),
        "feature_names": dyn_names + ["log1p_active_cells", "log1p_burns_7d"]
                         + [f"season_mean_{c}" for c in SEVERITY_CHANNELS],
        "n_sup": n_sup, "events": events, "doy": doy, "dates": days,
        "years": days.astype("datetime64[Y]").astype(int) + 1970,
        "regions": regions, "regions_km": regions_km,
    }


def harmonics(doy: np.ndarray, n: int = N_HARMONICS) -> np.ndarray:
    ang = 2.0 * np.pi * doy / 365.25
    return np.concatenate([
        np.stack([np.sin(k * ang), np.cos(k * ang)], axis=1) for k in range(1, n + 1)
    ], axis=1)


def design(frame: Dict[str, np.ndarray], mu: np.ndarray | None = None,
           sd: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Standardized (x, mu, sd) design of shape (D, R, F): the frame's
        features and day-of-year harmonics, fitting mu and sd over the
        exposed region-days when not given.
    """
    D, R, _ = frame["features"].shape
    h = np.broadcast_to(harmonics(frame["doy"])[:, None, :], (D, R, 2 * N_HARMONICS))
    x = np.concatenate([frame["features"], h], axis=2)
    if mu is None:
        rows = x[frame["n_sup"] > 0]
        mu, sd = rows.mean(0), rows.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (x - mu) / sd, mu, sd


class CountRate:
    """ log E[count] = offset + w . x + b + b_r, under a Poisson or
        negative-binomial likelihood; w is pooled across regions and b_r is
        the region intercept. The negative binomial carries a fitted
        log-dispersion: Var = mu + alpha * mu^2, and alpha -> 0 recovers
        the Poisson. """

    def __init__(self, n_features: int, n_regions: int = 1, l2: float = 1e-2, family: str = "nb"):
        self.w = torch.zeros(n_features, dtype=torch.float64, requires_grad=True)
        self.b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        self.b_r = torch.zeros(n_regions, dtype=torch.float64, requires_grad=True)
        self.log_alpha = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        self.n_regions = n_regions
        self.l2 = l2
        self.family = family

    def _eta(self, x: torch.Tensor, offset: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        return offset + x @ self.w + self.b + self.b_r[region]

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

    def fit(self, x: np.ndarray, y: np.ndarray, offset: np.ndarray, region: np.ndarray,
            steps: int = 400):
        """ Fit weights, intercepts and dispersion to the rows by LBFGS, returning self. """
        xt = torch.as_tensor(x, dtype=torch.float64)
        yt = torch.as_tensor(y, dtype=torch.float64)
        ot = torch.as_tensor(offset, dtype=torch.float64)
        rt = torch.as_tensor(region, dtype=torch.long)
        params = [self.w, self.b] + ([self.b_r] if self.n_regions > 1 else []) \
            + ([] if self.family == "poisson" else [self.log_alpha])
        opt = torch.optim.LBFGS(params, max_iter=steps,
                                tolerance_grad=1e-10, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            # -- intercepts centred on b: the region term carries departures only
            loss = self._nll(self._eta(xt, ot, rt), yt) + self.l2 * ((self.w ** 2).sum() + (self.b_r ** 2).mean())
            loss.backward()
            return loss

        opt.step(closure)
        return self

    @torch.no_grad()
    def predict(self, x: np.ndarray, offset: np.ndarray, region: np.ndarray) -> np.ndarray:
        # Predicted mean count per row
        eta = self._eta(torch.as_tensor(x, dtype=torch.float64),
                        torch.as_tensor(offset, dtype=torch.float64),
                        torch.as_tensor(region, dtype=torch.long))
        return torch.exp(eta).numpy()


def fit_frame(train: Dict[str, np.ndarray], l2: float, family: str):
    """ (model, mu, sd) fitted on the exposed region-days of a train frame. """
    x, mu, sd = design(train)
    exposed = train["n_sup"] > 0
    D, R, F = x.shape
    region = np.broadcast_to(np.arange(R)[None, :], (D, R))
    model = CountRate(F, R, l2=l2, family=family).fit(
        x[exposed], train["events"][exposed], np.log(train["n_sup"][exposed]), region[exposed])
    return model, mu, sd


def predict_frame(model: CountRate, frame: Dict[str, np.ndarray], mu, sd) -> np.ndarray:
    """ (D, R) expected count; zero where the region has no supervised cell. """
    x, _, _ = design(frame, mu, sd)
    D, R, _ = x.shape
    region = np.broadcast_to(np.arange(R)[None, :], (D, R))
    out = np.zeros((D, R))
    exposed = frame["n_sup"] > 0
    out[exposed] = model.predict(x[exposed], np.log(frame["n_sup"][exposed]), region[exposed])
    return out


def seasonal_references(train: Dict[str, np.ndarray], frame: Dict[str, np.ndarray],
                        statewide: np.ndarray, sigma_days: float = 7.0) -> Dict[str, np.ndarray]:
    """ The three (D, R) references: the statewide seasonal profile times the
        region's train-year share of events, the statewide fitted rate times
        that share, and the region's own seasonal profile floored at
        OWN_PROFILE_FLOOR of the first.
    """
    share = train["events"].sum(0) / max(train["events"].sum(), 1.0)
    profile = seasonal_profile(train["doy"], train["events"].sum(1), train["n_sup"].sum(1), sigma_days)
    state_expected = profile[frame["doy"] - 1] * frame["n_sup"].sum(1)
    seasonal_share = state_expected[:, None] * share[None, :]
    statewide_share = statewide[:, None] * share[None, :]
    own = np.stack([seasonal_profile(train["doy"], train["events"][:, r], train["n_sup"][:, r], sigma_days)
                    [frame["doy"] - 1] * frame["n_sup"][:, r] for r in range(share.size)], axis=1)
    own = np.maximum(own, OWN_PROFILE_FLOOR * seasonal_share)
    return {"seasonal_share": seasonal_share, "statewide_share": statewide_share, "seasonal": own}


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


def rate_path(dataset: str, fold: str, regions_km: float | None = None) -> Path:
    tag = "" if not regions_km else f"_r{regions_km:g}km"
    return RATE_DIR / f"rate_{dataset}_{fold}{tag}"


def run(dataset: str, fold: str, l2: float, family: str = "nb",
        regions_km: float | None = None) -> Tuple[Dict, Dict, np.ndarray]:
    """ Fit the count-rate model on the train split and score it on every split.
        Returns the report, the per-day arrays per split (dates, lambda, the
        three references, events, n_sup; (D, R) each) and the region map.
    """
    frames = {s: daily_frame(dataset, s, fold, regions_km) for s in ("train", "eval", "test")}
    model, mu, sd = fit_frame(frames["train"], l2, family)
    alpha = 0.0 if family == "poisson" else model.alpha
    R = frames["train"]["n_sup"].shape[1]
    if R > 1:
        state_frames = {s: daily_frame(dataset, s, fold, None) for s in frames}
        state_model, smu, ssd = fit_frame(state_frames["train"], l2, family)
        statewide = {s: predict_frame(state_model, f, smu, ssd)[:, 0] for s, f in state_frames.items()}
    else:
        statewide = None

    report = {"dataset": dataset, "fold": fold, "l2": l2, "family": family,
              "regions_km": regions_km, "n_regions": int(R),
              "dispersion_alpha": alpha, "n_features": int(model.w.numel()),
              "region_intercepts": model.b_r.detach().numpy().round(4).tolist() if R > 1 else None,
              "splits": {}}
    arrays = {}
    for split, frame in frames.items():
        pred = predict_frame(model, frame, mu, sd)
        refs = seasonal_references(frames["train"], frame,
                                   statewide[split] if statewide else pred[:, 0])
        if R == 1:
            refs["statewide_share"] = pred.copy()
        y, exposed = frame["events"], frame["n_sup"] > 0
        arrays[split] = {"dates": frame["dates"].astype(np.int64), "lambda": pred,
                         "events": y, "n_sup": frame["n_sup"], **refs}
        d_model = poisson_deviance(y[exposed], pred[exposed])
        entry = {
            "n_days": int(len(frame["dates"])),
            "n_region_days": int(exposed.sum()),
            "mean_observed": float(y[exposed].mean()),
            "mean_predicted": float(pred[exposed].mean()),
            "deviance": d_model,
            "corr_predicted": float(np.corrcoef(pred[exposed], y[exposed])[0, 1]),
            "pit": pit_histogram(y[exposed], pred[exposed], alpha),
            "references": {},
        }
        for name, ref in refs.items():
            d_ref = poisson_deviance(y[exposed], ref[exposed])
            entry["references"][name] = {
                "mean": float(ref[exposed].mean()), "deviance": d_ref,
                "deviance_skill": 1 - d_model / d_ref if d_ref > 0 else float("nan"),
                "corr": float(np.corrcoef(ref[exposed], y[exposed])[0, 1]),
                "pit": pit_histogram(y[exposed], ref[exposed], alpha),
            }
        if R > 1:
            # -- per region, against the region's own seasonal profile
            per = []
            for r in range(R):
                e = exposed[:, r]
                dm, dr = poisson_deviance(y[e, r], pred[e, r]), poisson_deviance(y[e, r], refs["seasonal"][e, r])
                per.append({"events": float(y[e, r].sum()),
                            "mean_n_sup": float(frame["n_sup"][e, r].mean()) if e.any() else 0.0,
                            "deviance_skill": 1 - dm / dr if dr > 0 else float("nan")})
            entry["per_region"] = per
        report["splits"][split] = entry
    return report, arrays, frames["train"]["regions"]


def write_rate(dataset: str, fold: str, l2: float = 1e-2, family: str = "nb",
               regions_km: float | None = None) -> Dict:
    """ Fit, score, and cache the rate: the report as JSON and the per-day arrays
        as an npz keyed '<split>_<name>' plus the 'regions' map, under RATE_DIR.
    """
    report, arrays, regions = run(dataset, fold, l2, family, regions_km)
    RATE_DIR.mkdir(parents=True, exist_ok=True)
    base = rate_path(dataset, fold, regions_km)
    base.with_suffix(".json").write_text(json.dumps(report, indent=1))
    np.savez_compressed(base.with_suffix(".npz"), regions=regions,
                        **{f"{s}_{k}": v for s, arr in arrays.items() for k, v in arr.items()})
    print(f"[rate] wrote {base}.json and .npz ({report['n_regions']} regions)")
    return report


def daily_rate(dataset: str, fold: str, split: str,
               regions_km: float | None = None) -> Dict[str, np.ndarray]:
    """ One split's per-day (D, R) rate arrays and the region map from the
        cache, fitting the rate when absent or in an older layout. """
    path = rate_path(dataset, fold, regions_km).with_suffix(".npz")
    if not path.exists() or "regions" not in np.load(path):
        write_rate(dataset, fold, regions_km=regions_km)
    arc = np.load(path)
    keys = ("dates", "lambda", "seasonal", "seasonal_share", "statewide_share", "events", "n_sup")
    out = {k: arc[f"{split}_{k}"] for k in keys}
    out["dates"] = out["dates"].astype("datetime64[D]")
    out["regions"] = arc["regions"]
    return out


def main():
    ap = argparse.ArgumentParser(description="Fit and score the ignition rate per region")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", default="full")
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--family", default="nb", choices=["nb", "poisson"])
    ap.add_argument("--regions", type=float, default=None,
                    help="lattice region size in km; omitted is the statewide fit")
    args = ap.parse_args()
    print(json.dumps(write_rate(args.dataset, args.fold, args.l2, args.family, args.regions), indent=1))


if __name__ == "__main__":
    main()
