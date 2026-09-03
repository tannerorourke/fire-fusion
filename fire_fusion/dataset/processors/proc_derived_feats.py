from typing import List, Optional, Sequence, Tuple
import dask
import pandas as pd
import xarray as xr
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import lfilter

from fire_fusion.config.feature_config import IGN_HORIZON_DAYS


# Full time series working sizes for lightning, fuel-moisture, and ignition blocks per task
LIGHTNING_BLOCK_BYTES = 256 * 1024 ** 2
FUEL_BLOCK_BYTES = 96 * 1024 ** 2
IGN_BLOCK_BYTES = 128 * 1024 ** 2

# NFDRS 1978 dead fuel moisture (Bradshaw/Deeming, GTR INT-169; response factors and boundary constants
#  match the WIMS/FireFamilyPlus operational code).
# - Temperature in Farenheit (matches EMC boundary conditions)
# - Per-day response fractions encode the 24h/timelag ratio and are applied once per daily step.
FM100_RESPONSE = 1.0 - 0.87 * np.exp(-0.24)    # 0.315634
FM1000_RESPONSE = 1.0 - 0.82 * np.exp(-0.168)  # 0.306811
FM1000_SEED = 30.0                             # operational spin-up default (%)

# -- Fire state fusion. Source codes in priority order: a discovery point is the
#    ignition itself, an MCD64 day pinned by a detection is exact, an unpinned
#    MCD64 day carries the pixel's uncertainty, a detection alone is coarse, and
#    a polygon start day is the last resort for fires no satellite saw.

# Fire state fusion source codes in priority order
# - point: USFS point ignition itself exact to day
# - pinned: MCD64A1 burn day exact day pinned by a detection
# - mcd64: unpinned MCD64A1 day with uncertainty clipped to 1-7 days
# - firms: FIRMS detection where MCD64A1 saw nothing
# - polygon: USFS perimeter as last resort; same perimeter on start/end date
BURN_SRC = {"point": 1, "pinned": 2, "mcd64": 3, "firms": 4, "polygon": 5}
ACTIVE_PERSIST_DAYS = 3        # -- days an observation keeps a cell active
UNC_MAX_DAYS = 7               # -- MCD64 date uncertainty clip days
CROPLAND_FRAC = 0.5            # -- Cell fraction at which MCD64A1 burns above this cropland fraction are dropped
BURN_REFRACTORY_DAYS = 30      # -- Max days a cell is assumed to be burning after a discovery
CAUSE_REACH_M = 20_000.0       # -- Max distance and days burn takes the cause of the nearest discovery within
CAUSE_REACH_DAYS = 60


# -- Utility functions ------------------------------------------------------
def _abs_days(times) -> np.ndarray:
    return np.asarray(times).astype("datetime64[D]").astype(np.int64)


def _emc(Tf: np.ndarray, H: np.ndarray) -> np.ndarray:
    """ Equilibrium moisture content (% gravimetric) from temperature (F) and
        relative humidity (%). Three humidity branches; the mid branch carries no
        H*T cross term. Constants are the NFDRS operational values.
    """
    emc_low = 0.03229 + 0.281073 * H - 0.000578 * H * Tf
    emc_mid = 2.22749 + 0.160107 * H - 0.014784 * Tf
    emc_high = 21.0606 + 0.005565 * H * H - 0.00035 * H * Tf - 0.483199 * H
    return np.where(H < 10.0, emc_low, np.where(H < 50.0, emc_mid, emc_high))


def _pdur_hours(precip_mm: np.ndarray) -> np.ndarray:
    """ Precipitation duration (hours) estimated from daily precip amount.
        NFDRS ingests observed duration, not amount; with amount-only daily
        records this steps duration with rainfall and caps at the 8h
        state-of-weather reporting limit. This is a modeling choice, not an
        NFDRS constant.
    """
    conds = [precip_mm <= 0.0, precip_mm < 2.5, precip_mm < 5.0, precip_mm < 10.0, precip_mm < 25.0]
    vals = [0.0, 1.0, 2.0, 4.0, 6.0]
    return np.select(conds, vals, default=8.0).astype("float32")


def _fuel_boundaries(tmin_f, tmax_f, hmin, hmax, precip_mm):
    """ Daily 100h and 1000h boundary conditions (%). EMCbar is the 1978
        simple average of the hot-dry and cool-moist EMC endpoints; the wet
        term differs between the two timelag classes.
    """
    emc_hot = _emc(tmax_f, hmin)    # hot, dry -> lowest moisture
    emc_cool = _emc(tmin_f, hmax)   # cool, moist -> highest moisture
    emcbar = 0.5 * (emc_hot + emc_cool)
    pdur = _pdur_hours(precip_mm)
    d100 = ((24.0 - pdur) * emcbar + pdur * (0.5 * pdur + 41.0)) / 24.0
    d1000 = ((24.0 - pdur) * emcbar + pdur * (2.7 * pdur + 76.0)) / 24.0
    return d100.astype("float32"), d1000.astype("float32")


def _fm100_scan(tmin_f, tmax_f, hmin, hmax, precip_mm, reset):
    """ 100h fuel moisture recursion along axis 0 (time). `reset` marks the first
        day of each contiguous time block (year gaps in a seasonal index); the
        recursion reseeds at the day-0 boundary there rather than carrying the
        prior block's state across the gap. The 100h class forgets its seed
        within ~2 weeks.
    """
    d100, _ = _fuel_boundaries(tmin_f, tmax_f, hmin, hmax, precip_mm)
    out = np.empty_like(d100)
    prev = None
    for t in range(d100.shape[0]):
        if prev is None or reset[t]:
            cur = d100[t]
        else:
            cur = prev + (d100[t] - prev) * FM100_RESPONSE
        out[t] = cur
        prev = cur
    return np.clip(out, 0.0, 60.0).astype("float32")


def _fm1000_scan(tmin_f, tmax_f, hmin, hmax, precip_mm, reset):
    """ 1000h fuel moisture recursion along time axis. """
    _, d1000 = _fuel_boundaries(tmin_f, tmax_f, hmin, hmax, precip_mm)
    out = np.empty_like(d1000)
    prev = None
    window: List[np.ndarray] = []
    for t in range(d1000.shape[0]):
        if prev is None or reset[t]:
            prev = np.full_like(d1000[t], FM1000_SEED)
            window = []
        window.append(d1000[t])
        if len(window) > 7:
            window.pop(0)
        dbar = sum(window) / len(window)
        cur = prev + (dbar - prev) * FM1000_RESPONSE
        out[t] = cur
        prev = cur
    return np.clip(out, 0.0, 60.0).astype("float32")


class DerivedProcessor:
    """ Build labels, masks, and derived features. Steps in order:
        1. Build labels and masks
        2. Build other features. Group so extracted data can be (conditionally) dropped cleanly
    
    Notes:
    - Derivations that reference a statistic of the record (rather than a single day) take that statistic from the train split only;
    - `train_yrs` carries the year set in. Leaving it None uses the whole record and is only appropriate outside of a modelling context.
    """
    def __init__(self, train_yrs: Optional[Sequence[int]] = None):
        self.train_yrs = train_yrs

    def _train_slice(self, da: xr.DataArray) -> xr.DataArray:
        if self.train_yrs is None or "time" not in da.dims:
            return da
        return da.sel(time=da["time"].dt.year.isin(list(self.train_yrs)))

    # -- Fire state ----------------------------------------------------------------------------
    # All quantities below are built from sparse (day, cell) event tables rather than dense daily cubesfor memory efficiency
    # Days are absolute (days since epoch) because the master index has winter gaps;
    def _sparse(self, subds: xr.Dataset, names: Sequence[str]) -> dict:
        """ Nonzero (day, cell, value) per variable, read one time chunk at a time. """
        days = _abs_days(subds["time"].values)
        W = subds.sizes["x"]
        tables = {n: [] for n in names}
        step = subds[names[0]].chunks[0][0] if subds[names[0]].chunks else 64
        for t0 in range(0, len(days), step):
            block = subds[list(names)].isel(time=slice(t0, t0 + step)).load()
            for n in names:
                v = block[n].values
                t, y, x = np.nonzero(v)
                tables[n].append(np.stack([days[t0 + t], y * W + x, v[t, y, x].astype(np.int64)], 1))
        return {n: (np.concatenate(v) if v else np.zeros((0, 3), np.int64)) for n, v in tables.items()}

    def _values_at(self, da: xr.DataArray, day: np.ndarray, cell: np.ndarray) -> np.ndarray:
        """ da[(day, cell)] read one time chunk at a time. """
        days = _abs_days(da["time"].values)
        pos = np.searchsorted(days, day)
        out = np.zeros(len(day), dtype=np.float64)
        step = da.chunks[0][0] if da.chunks else 64
        for t0 in range(0, len(days), step):
            sel = np.flatnonzero((pos >= t0) & (pos < t0 + step))
            if sel.size == 0:
                continue
            block = da.isel(time=slice(t0, t0 + step)).values.reshape(min(step, len(days) - t0), -1)
            out[sel] = block[pos[sel] - t0, cell[sel]]
        return out

    def _scatter(self, like: xr.DataArray, day: np.ndarray, cell: np.ndarray, value: np.ndarray,
                 name: str, dtype, fill=0) -> xr.DataArray:
        """ Lazy (time, y, x) array holding 'value' at (day, cell), fill elsewhere. """
        import dask.array as da
        days = _abs_days(like["time"].values)
        H, W = like.sizes["y"], like.sizes["x"]
        pos = np.searchsorted(days, day)
        keep = (pos < len(days)) & (days[np.minimum(pos, len(days) - 1)] == day)
        pos, cell, value = pos[keep], cell[keep], value[keep]
        order = np.argsort(pos, kind="stable")
        pos, cell, value = pos[order], cell[order], value[order]
        step = like.chunks[0][0] if like.chunks else 64

        def block(t0, t1):
            out = np.full((t1 - t0, H * W), fill, dtype=dtype)
            i0, i1 = np.searchsorted(pos, [t0, t1])
            out[pos[i0:i1] - t0, cell[i0:i1]] = value[i0:i1]
            return out.reshape(t1 - t0, H, W)

        parts = [da.from_delayed(dask.delayed(block)(t0, min(t0 + step, len(days))),
                                 shape=(min(t0 + step, len(days)) - t0, H, W), dtype=dtype)
                 for t0 in range(0, len(days), step)]
        arr = xr.DataArray(da.concatenate(parts, axis=0), coords=like.coords, dims=like.dims, name=name)
        return arr

    def build_fire_state(self, subds: xr.Dataset, names: Sequence[str],
                         window: int = ACTIVE_PERSIST_DAYS) -> xr.Dataset:
        """ burns, burns_early, burns_late, burned, active, burn_src.

            burns: the cell burns on this day, fused from the sources in BURN_SRC order:
                1. A FIRMS detection inside an MCD64 pixel's date window pins the day; 
                2. an MCD64A1 day without FIRMS detection stands alone and its early/late variants shift it by the pixel's own uncertainty.
                3. A detection with no MCD64 burn counts only when it repeats the next day or a discovery point sits within 2 km.
                4. A USFS polygon with no satellite evidence contributes its start day only when it is at most one 2 km cell in area.
                5. MCD64A1 burns on cropland are dropped.
                6. A cell burns at most once per BURN_REFRACTORY_DAYS days; later days fold into active.
            active: what a forecaster sees on the day, a detection or discovery within the last 'window' days,
                    or an undetected small polygon between its start and end.
            burned: burned earlier this calendar year by any non-point source.
        """
        res = float(abs(subds["x"].values[1] - subds["x"].values[0]))
        H, W = subds.sizes["y"], subds.sizes["x"]
        like = subds["ign_occ"]
        sp = self._sparse(subds, ["ign_occ", "modis_burn", "modis_burn_unc", "firms_detect", "perimeter_id"])

        # -- candidates as (day, cell, src, unc) rows
        pts = sp["ign_occ"]
        firms = sp["firms_detect"]
        unc_at = dict(zip(map(tuple, sp["modis_burn_unc"][:, :2]), sp["modis_burn_unc"][:, 2]))
        mcd = sp["modis_burn"][:, :2]
        crop = self._values_at(subds["cropland_frac"], mcd[:, 0], mcd[:, 1])
        n_crop = int((crop > CROPLAND_FRAC).sum())
        mcd = mcd[crop <= CROPLAND_FRAC]
        u = np.array([np.clip(unc_at.get(tuple(r), 0), 1, UNC_MAX_DAYS) for r in mcd], dtype=np.int64)

        # -- pin MCD64 days to the earliest detection in the pixel's window
        firms_by_cell = pd.DataFrame(firms[:, :2], columns=["day", "cell"]).groupby("cell")["day"].apply(np.sort)
        day_pin = mcd[:, 0].copy(); src = np.full(len(mcd), BURN_SRC["mcd64"], np.int64)
        for i, (d, c) in enumerate(mcd):
            fd = firms_by_cell.get(c)
            if fd is None:
                continue
            inside = fd[(fd >= d - u[i]) & (fd <= d + u[i])]
            if inside.size:
                day_pin[i] = inside[0]; src[i] = BURN_SRC["pinned"]
        rows = [np.column_stack([pts[:, 0], pts[:, 1], np.full(len(pts), BURN_SRC["point"]), np.zeros(len(pts), np.int64)]),
                np.column_stack([day_pin, mcd[:, 1], src, u])]

        # -- detections with no MCD64 burn: the first day of a run of at least
        #    two, or any day with a discovery point within 2 km in the window
        reach = int(round(2000.0 / res))
        key = lambda d, c: d * (H * W) + c
        fkeys = np.sort(key(firms[:, 0], firms[:, 1]))
        oy, ox = np.meshgrid(np.arange(-reach, reach + 1), np.arange(-reach, reach + 1), indexing="ij")
        py, px = pts[:, 1] // W, pts[:, 1] % W
        ny = (py[:, None] + oy.ravel()[None, :]); nx = (px[:, None] + ox.ravel()[None, :])
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        ncell = (ny * W + nx)[ok]; nday = np.repeat(pts[:, 0], oy.size)[ok.ravel()]
        near_keys = np.unique((nday[:, None] + np.arange(window + 1)[None, :]).ravel() * (H * W)
                              + np.repeat(ncell, window + 1))
        fd, fc = firms[:, 0], firms[:, 1]
        in_mcd = np.isin(fc, np.unique(mcd[:, 1]))
        run_start = np.isin(key(fd + 1, fc), fkeys) & ~np.isin(key(fd - 1, fc), fkeys)
        near_pt = np.isin(key(fd, fc), near_keys)
        take = ~in_mcd & (run_start | near_pt)
        rows.append(np.column_stack([fd[take], fc[take], np.full(take.sum(), BURN_SRC["firms"]),
                                     np.zeros(take.sum(), np.int64)]))

        # -- undetected small polygons: start day for every painted cell
        per = pd.DataFrame(sp["perimeter_id"], columns=["day", "cell", "fid"])
        evid = set(map(tuple, np.concatenate(rows)[:, :2]))
        max_cells = max(1, int(4e6 / res ** 2))
        poly_rows, paint_rows = [], []
        for fid, g in per.groupby("fid"):
            cells = g.cell.unique()
            seen = any((d, c) in evid for d, c in zip(g.day, g.cell))
            if seen or len(cells) > max_cells:
                continue
            d0 = int(g.day.min())
            poly_rows += [(d0, int(c), BURN_SRC["polygon"], 0) for c in cells]
            paint_rows.append(g[["day", "cell"]].to_numpy())
        rows.append(np.array(poly_rows, dtype=np.int64).reshape(-1, 4))
        ev = pd.DataFrame(np.concatenate(rows), columns=["day", "cell", "src", "unc"])

        # -- one event per (cell, day), best source wins; then one per 30 days
        ev = ev.sort_values(["cell", "day", "src"]).drop_duplicates(["cell", "day"])
        keep = np.ones(len(ev), bool); last = {}
        for i, (c, d) in enumerate(zip(ev.cell.values, ev.day.values)):
            if c in last and d - last[c] <= BURN_REFRACTORY_DAYS:
                keep[i] = False
            else:
                last[c] = d
        ev = ev[keep]
        shift = np.where(ev.src == BURN_SRC["mcd64"], ev.unc, 0)

        out = xr.Dataset()
        out["burns"] = self._scatter(like, ev.day.values, ev.cell.values, np.ones(len(ev)), "burns", np.uint8)
        out["burns_early"] = self._scatter(like, (ev.day - shift).values, ev.cell.values, np.ones(len(ev)), "burns_early", np.uint8)
        out["burns_late"] = self._scatter(like, (ev.day + shift).values, ev.cell.values, np.ones(len(ev)), "burns_late", np.uint8)
        out["burn_src"] = self._scatter(like, ev.day.values, ev.cell.values, ev.src.values, "burn_src", np.uint8)

        # -- burned: from the event day onward within the same calendar year
        days = _abs_days(like["time"].values)
        years = pd.DatetimeIndex(like["time"].values).year.values
        nonpt = ev[ev.src != BURN_SRC["point"]]
        b_day, b_cell = [], []
        for d, c in zip(nonpt.day.values, nonpt.cell.values):
            i = np.searchsorted(days, d)
            if i >= len(days) or days[i] != d:
                continue
            j = i + 1
            while j < len(days) and years[j] == years[i]:
                j += 1
            b_day.append(days[i + 1:j]); b_cell.append(np.full(j - i - 1, c))
        b_day = np.concatenate(b_day) if b_day else np.zeros(0, np.int64)
        b_cell = np.concatenate(b_cell) if b_cell else np.zeros(0, np.int64)
        out["burned"] = self._scatter(like, b_day, b_cell, np.ones(len(b_day)), "burned", np.uint8)

        # -- active: observations persisted forward, plus undetected small polygons
        obs = np.concatenate([pts[:, :2], firms[:, :2]])
        a_day = (obs[:, 0][:, None] + np.arange(window + 1)[None, :]).ravel()
        a_cell = np.repeat(obs[:, 1], window + 1)
        if paint_rows:
            paint = np.concatenate(paint_rows)
            a_day = np.concatenate([a_day, paint[:, 0]]); a_cell = np.concatenate([a_cell, paint[:, 1]])
        out["active"] = self._scatter(like, a_day, a_cell, np.ones(len(a_day)), "active", np.uint8)
        print(f"[fire_state] events {len(ev):,}: " + ", ".join(
            f"{k} {int((ev.src == v).sum()):,}" for k, v in BURN_SRC.items())
            + f"; MCD64 burns on cropland dropped {n_crop:,}")
        return out

    def build_no_act_fire_mask(self, subds: xr.Dataset, name: str) -> xr.DataArray:
        """ 1 where the cell can burn today: nothing seen burning in it and it
            has not burned earlier this calendar year.
        """
        clear = ((subds["active"] == 0) & (subds["burned"] == 0)).astype("uint8")
        clear.name = name
        return clear

    def build_burn_next(self, subds: xr.Dataset, name: str, source: str = "burns",
                        horizon: int = IGN_HORIZON_DAYS) -> xr.DataArray:
        """ 1 where the cell is clear today and burns on any of T+1 .. T+horizon. """
        burns_t = subds[source] > 0
        if burns_t.chunks is not None:
            nt = burns_t.sizes["time"]
            edge = max(1, int(np.sqrt(IGN_BLOCK_BYTES / nt)))
            burns_t = burns_t.chunk({"time": -1, "y": min(burns_t.sizes["y"], edge),
                                     "x": min(burns_t.sizes["x"], edge)})
        future = burns_t.shift(time=-1, fill_value=False)
        for k in range(2, horizon + 1):
            future = future | burns_t.shift(time=-k, fill_value=False)
        label = ((subds["no_act_fire_mask"] == 1) & future).astype("uint8")
        label.name = name
        return label

    def build_burn_next_early(self, subds, name):
        return self.build_burn_next(subds, name, source="burns_early")

    def build_burn_next_late(self, subds, name):
        return self.build_burn_next(subds, name, source="burns_late")

    def build_burn_cause_day(self, subds: xr.Dataset, name: str) -> xr.DataArray:
        """ Cause class of each burn event, from the nearest discovery within
            CAUSE_REACH_M and the preceding CAUSE_REACH_DAYS; -1 when none. A
            point event finds its own discovery at distance zero.
        """
        from scipy.spatial import cKDTree
        res = float(abs(subds["x"].values[1] - subds["x"].values[0]))
        W = subds.sizes["x"]
        like = subds["burns"]
        ev = self._sparse(subds, ["burns"])["burns"][:, :2]
        planes = subds["ign_cause"]
        n_cls = planes.sizes["burn_cause"]
        
        # -- discoveries with a cause index, or -1 for an unmapped cause
        disc = self._sparse(subds, ["ign_occ"])["ign_occ"][:, :2]
        cause = np.full(len(disc), -1, np.int64)
        for k in range(n_cls):
            sk = self._sparse(planes.isel(burn_cause=k).to_dataset(name="p"), ["p"])["p"][:, :2]
            hit = pd.MultiIndex.from_arrays(disc.T).get_indexer(pd.MultiIndex.from_arrays(sk.T))
            cause[hit[hit >= 0]] = k
        order = np.argsort(disc[:, 0], kind="stable")
        disc, cause = disc[order], cause[order]
        d_days = disc[:, 0]
        out = np.full(len(ev), -1, np.int64)
        for d in np.unique(ev[:, 0]):
            i0, i1 = np.searchsorted(d_days, [d - CAUSE_REACH_DAYS, d + 1])
            if i1 == i0:
                continue
            cells = disc[i0:i1, 1]
            tree = cKDTree(np.column_stack([cells // W, cells % W]) * res)
            sel = np.flatnonzero(ev[:, 0] == d)
            q = np.column_stack([ev[sel, 1] // W, ev[sel, 1] % W]) * res
            dist, idx = tree.query(q, distance_upper_bound=CAUSE_REACH_M)
            ok = np.isfinite(dist)
            out[sel[ok]] = cause[i0:i1][idx[ok]]
        return self._scatter(like, ev[:, 0], ev[:, 1], out, name, np.int8, fill=-1)

    def build_burn_next_cause(self, subds: xr.Dataset, name: str,
                              horizon: int = IGN_HORIZON_DAYS) -> xr.DataArray:
        """ Cause of the earliest burn event in T+1 .. T+horizon where the
            label is 1, else -1.
        """
        cause_t = subds["burn_cause_day"]
        if cause_t.chunks is not None:
            nt = cause_t.sizes["time"]
            edge = max(1, int(np.sqrt(IGN_BLOCK_BYTES / (nt * 8))))
            cause_t = cause_t.chunk({"time": -1, "y": min(cause_t.sizes["y"], edge),
                                     "x": min(cause_t.sizes["x"], edge)})
        burns_t = subds["burns"].chunk(cause_t.chunks) if cause_t.chunks is not None else subds["burns"]
        nxt = xr.full_like(cause_t, -1)
        for k in range(horizon, 0, -1):
            hit = burns_t.shift(time=-k, fill_value=0) > 0
            nxt = xr.where(hit, cause_t.shift(time=-k, fill_value=-1), nxt)
        out = xr.where(subds["burn_next"] == 1, nxt, -1).astype("int8")
        out.name = name
        return out

    def build_valid_cause_mask(self, subds: xr.Dataset, name: str) -> xr.DataArray:
        mask = (subds["burn_next_cause"] >= 0).astype("uint8")
        mask.name = name
        return mask

    def build_fire_spatial_rolling(self, subds: xr.Dataset, name: str, kernel = 3, t_window = 3) -> xr.DataArray:
        """ 3x3 kernel max of active fires at time = T
            (ie, is there an active fire next to me?)
        """
        burning_t = subds["active"] > 0
        # rolling pads with a dtype-dependent fill value; bool input breaks under dask
        burn_rolling = (
            burning_t.astype("float32")
            .rolling(time=t_window, min_periods=1).max()
            .fillna(0)
            .astype("float32")
        )

        assert burn_rolling.dims == ("time", "y", "x"), f"Unexpected dims: {burn_rolling.dims}"

        # kernel spans spatial dims only, per-chunk application is exact as long as chunks cover the full 
        # spatial extent (rolling may have split them; a 3x3 max over partial spatial chunks would miss neighbors)
        if burn_rolling.chunks is not None:
            burn_rolling = burn_rolling.chunk({"y": -1, "x": -1})

        burn_filter = xr.apply_ufunc(
            maximum_filter,
            burn_rolling,
            kwargs={"size": (1, kernel, kernel)},
            dask="parallelized",
            output_dtypes=[np.float32],
        )
        burn_filter.name = name
        return burn_filter

    def build_land_mask(self, subds: xr.Dataset, name: str) -> xr.DataArray:
        """ 1 where the cell is land. MODIS flags deep water; anything it does
            not flag (including cells it never observed) is treated as land.
        """
        land_mask = (
            subds["modis_water_mask"].fillna(0) == 0
        ).astype("uint8")
        land_mask.name = name
        return land_mask


    # -- Other Features ---------------------------------------------------------------------------
    def build_ndvi_anomaly(self, subds: xr.Dataset, name: str) -> xr.DataArray:
        """ NDVI minus its day-of-year climatology, where the climatology is an
            average over the train years only. A climatology spanning the whole
            record would carry eval/test vegetation into every training sample
            and let each held-out day contribute to the mean it is measured
            against.
        """
        ndvi = subds['modis_ndvi']
        doy = ndvi["time"].dt.dayofyear

        # Materialize the day-of-year climatology once and subtract it via a
        # per-chunk positional lookup. A groupby subtraction shatters the time
        # axis into per-day chunks, which explodes the task graph downstream.
        # Assumes spatial dims are unchunked (the build keeps full-extent
        # spatial chunks throughout).
        ref = self._train_slice(ndvi)
        clim = ref.groupby(ref["time"].dt.dayofyear).mean("time").compute()
        clim_np = clim.values.astype("float32")

        # Days the train split never observed (Feb 29 when no train year is a
        # leap year) fall back to the nearest day-of-year that it did.
        observed = clim["dayofyear"].values
        right = np.searchsorted(observed, doy.values).clip(0, len(observed) - 1)
        left = (right - 1).clip(0, len(observed) - 1)
        take_left = np.abs(observed[left] - doy.values) <= np.abs(observed[right] - doy.values)
        pos = np.where(take_left, left, right)
        pos_da = xr.DataArray(pos, dims=("time",), coords={"time": ndvi["time"]})

        def _subtract_clim(nd, p):
            return (nd - clim_np[p.reshape(p.shape[0])]).astype("float32")

        ndvi_anom = xr.apply_ufunc(
            _subtract_clim,
            ndvi, pos_da,
            dask="parallelized",
            output_dtypes=[np.float32],
        )
        ndvi_anom.name = name
        return ndvi_anom
    


    def build_precip_cum(self, subds: xr.Dataset, names: List[str]) -> xr.Dataset:
        p2d = subds['precip_mm'].rolling(time=2, min_periods=1, center=False).sum().fillna(0)
        p5d = subds['precip_mm'].rolling(time=5, min_periods=1, center=False).sum().fillna(0)

        return xr.Dataset({ names[0]: p2d, names[1]: p5d })

    def build_dead_fuel_derived(self, subds: xr.Dataset, names: List[str]) -> xr.Dataset:
        """ NFDRS 100h and 1000h dead fuel moisture (%). Inputs arrive in NFDRS
            units -- temperature in F, humidity in %, precip in mm. EMCbar uses
            the diurnal extremes: rel_humidity carries the daily-min (hot-dry)
            branch, rh_max the daily-max (cool-moist) branch.

            The recursions run along time, splitting y/x is exact; the
            staging cube's time-chunked full-extent layout is rechunked to the
            opposite (full time, blocked space) to bound each task.
        """
        tmin = subds["temp_min"].fillna(60.0).astype("float32")
        tmax = subds["temp_max"].fillna(60.0).astype("float32")
        hmin = subds["rel_humidity"].fillna(50.0).clip(0.0, 100.0).astype("float32")
        hmax = subds["rh_max"].fillna(50.0).clip(0.0, 100.0).astype("float32")
        precip = subds["precip_mm"].fillna(0.0).astype("float32")

        # First day of each contiguous block. A seasonally windowed index carries
        # year-to-year gaps; the recursions reseed there rather than carry the
        # prior block's fuel state across the break.
        t = pd.DatetimeIndex(tmin.indexes["time"])
        gap = np.diff(t.values).astype("timedelta64[D]").astype("int64") != 1
        reset = np.concatenate([[True], gap])

        inputs = [tmin, tmax, hmin, hmax, precip]
        if tmin.chunks is not None:
            nt = tmin.sizes["time"]
            edge = max(1, int(np.sqrt(FUEL_BLOCK_BYTES / (nt * 4))))
            chunking = {"time": -1, "y": min(tmin.sizes["y"], edge), "x": min(tmin.sizes["x"], edge)}
            inputs = [a.chunk(chunking) for a in inputs]

        fm100 = xr.apply_ufunc(
            _fm100_scan, *inputs,
            kwargs={"reset": reset},
            dask="parallelized",
            output_dtypes=[np.float32],
        )
        fm1000 = xr.apply_ufunc(
            _fm1000_scan, *inputs,
            kwargs={"reset": reset},
            dask="parallelized",
            output_dtypes=[np.float32],
        )
        return xr.Dataset({ names[0]: fm100, names[1]: fm1000 })
         
    def build_lightning_load(self, subds: xr.Dataset, name: str, half_life: float = 4.0) -> xr.DataArray:
        """ Exponentially-decayed running sum of daily CG strike counts.
        
        `load[t] = strikes[t] + alpha * load[t-1],  alpha = 0.5 ** (1/half_life)`
        
        - Contribution halves every `half_life` days.
        - The pre-season halo is sized so carry-over decays below 0.1% by the first supervised day.
        """
        strikes = subds["lightning_strikes"].fillna(0.0).astype("float32")
        alpha = float(0.5 ** (1.0 / half_life))

        # The recursion needs each cell's full time series in one chunk. The
        # staging cube chunks time and holds full spatial extent.
        if strikes.chunks is not None:
            nt = strikes.sizes["time"]
            edge = max(1, int(np.sqrt(LIGHTNING_BLOCK_BYTES / (nt * 4))))
            strikes = strikes.chunk({
                "time": -1,
                "y": min(strikes.sizes["y"], edge),
                "x": min(strikes.sizes["x"], edge),
            })

        def _decay(arr):
            return lfilter([1.0], [1.0, -alpha], arr, axis=0).astype("float32")

        load = xr.apply_ufunc(
            _decay,
            strikes,
            dask="parallelized",
            output_dtypes=[np.float32],
        )
        load.name = name
        return load

    def build_wind_ew_ns(self, subds: xr.Dataset, names: List[str]) -> xr.Dataset:
        rads = xr.apply_ufunc(np.deg2rad, subds["wind_dir"], dask="allowed")
        val_ew = - xr.apply_ufunc(np.sin, rads, dask="allowed").astype("float32")
        val_ns = - xr.apply_ufunc(np.cos, rads, dask="allowed").astype("float32")

        return xr.Dataset({ names[0]:val_ew, names[1]:val_ns })
    
    def build_aspect_ew_ns(self, subds: xr.Dataset, names: List[str]) -> xr.Dataset:
        rads = xr.apply_ufunc(np.deg2rad, subds["lf_aspect"], dask="allowed")
        val_ew = xr.apply_ufunc(np.sin, rads, dask="allowed").astype("float32")
        val_ns = xr.apply_ufunc(np.cos, rads, dask="allowed").astype("float32")

        return xr.Dataset({ names[0]:val_ew, names[1]:val_ns })
    


    def build_ffwi(self, subds: xr.Dataset, name: str) -> xr.DataArray:
        """
        FFWI = n sqrt(1 + U^2) / 0.3002, where
        - n = 1 - 2x + 1.5x^2 - 0.5x^3
        - x = EMC/30
        - EMC:
            if H < 10%:     EMC = 0.03229 + (0.281073 * H) - (0.000578 * H% & T(Far))
            if H in 10-50%: EMC = 2.22749 + (0.160107 * H) - (0.01478 * T(Far)) 
            if H >= 50%:    EMC = 21.0606 + (0.005565 * H^2) - (0.00035 * H * T(Far)) - (0.483199 * H%) 
        """
        Tf = subds["temp_avg"]
        H = subds["rel_humidity"]
        Ws = subds["wind_mph"]

        EMC_p1   = 0.03229 + (0.281073 * H) - (0.000578 * H * Tf)
        EMC_p1p5 = 2.22749 + (0.160107 * H) - (0.01478 * Tf) 
        EMC_p5   = 21.0606 + (0.005565 * (H ** 2)) - (0.00035 * H * Tf) - (0.483199 * H)

        EMC = xr.where(
            H < 10, EMC_p1, 
            xr.where(H < 50, EMC_p1p5, EMC_p5)
        )
        x = EMC.clip(0.0, 30.0) / 30.0
        eta = 1 - (2.0 * x) + 1.5 * (x ** 2) - 0.5 * (x ** 3)
        ffwi = eta * np.sqrt(1.0 + Ws ** 2) / 0.3002
        ffwi.name = name
        return ffwi
        


    def build_doy_sin(self, subds: xr.Dataset, name: str, gridref: xr.DataArray) -> xr.DataArray:
        # time index to numpy
        time_index = pd.DatetimeIndex(subds.indexes["time"])
        doy = time_index.dayofyear.to_numpy(dtype="float32")

        # sin encoding on 0, 2pi
        phase = 2.0 * np.pi * (doy - 1.0) / 365.0
        time_signal = xr.DataArray(
            np.sin(phase).astype("float32"),
            dims=("time",),
            coords={"time": time_index},
        )

        # Broadcast against a (time, y, x) variable from the dataset result inherits its layout
        template = next(
            da for da in subds.data_vars.values() if da.dims == ("time", "y", "x")
        )
        doy_map = (time_signal * xr.ones_like(template, dtype="float32"))
        doy_map = doy_map.transpose("time", "y", "x")
        doy_map.name = name
        return doy_map
