"""
Interactive day-stepping viewer for one tier's spatiotemporal store.

Draws a single day at a time over a static terrain base: occurrence points,
perimeters, burn-day arrivals, label fields, mask complements, a cycled scalar
channel, and optionally a model probability field from a prediction archive.
Layers are discovered from the store's variables through an alias table; an
absent variable is never offered a hotkey. Days are read one at a time (the
500 m tier stays responsive); the base layer is built once.

  python -m fire_fusion.analysis.viewer --dataset wa4000
  python -m fire_fusion.analysis.viewer --dataset wa2000 --store dataset --year 2019
  python -m fire_fusion.analysis.viewer --dataset wa2000 --store split --split test --archive wa2000-s1
"""
import argparse
import base64
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_dilation

from ..config.dataset_config import get_dataset_config
from ..config.path_config import RAW_DATA_DIR, REF_DIR, VIEWER_DIR
from ..dataset.grid import GRID_CRS
from .reference import reference_probs

# -- canonical layer name -> variable names accepted for it, in resolution order
ALIASES = {
    "ign_occ": ("usfs_burn_occ", "ign_occ"),
    "firms_detect": ("firms_detect",),
    "perimeter": ("usfs_perimeter", "perimeter_id"),
    "modis_burn": ("modis_burn",),
    "burns": ("burns",),
    "burn_src": ("burn_src",),
    "active": ("active",),
    "burned": ("burned",),
    "burn_next": ("ign_next", "burn_next"),
    "burn_next_early": ("burn_next_early",),
    "burn_next_late": ("burn_next_late",),
    "unsupervised": ("no_act_fire_mask",),
    "land": ("land_mask",),
    "water": ("modis_water_mask",),
    "elevation": ("lf_elevation",),
    "ndvi": ("modis_ndvi",),
}

# -- (canonical, label, colour, style). "invert" draws the mask's complement.
LAYERS = [
    ("ign_occ", "discovery points", "#e8112d", "points"),
    ("firms_detect", "firms detections", "#ff8c00", "points"),
    ("perimeter", "perimeter", "#7c0a10", "outline"),
    ("modis_burn", "modis burn day", "#ff00ff", "points"),
    ("burns", "burn events, coloured by days from today", "#b000b0", "arrival"),
    ("active", "active fire", "#ffd400", "fill"),
    ("burned", "burned", "#8b5a2b", "fill"),
    ("unsupervised", "unsupervised cells", "#9a9a9a", "invert_hatch"),
    ("land", "off-grid (not land)", "#3c3c50", "invert_hatch"),
    ("burn_next", "label positives", "#22cc44", "outline"),
    ("burn_next_early", "label early", "#00ffa0", "outline"),
    ("burn_next_late", "label late", "#40a0ff", "outline"),
]
HOTKEY_POOL = "1234567890qwert"
GEOMAC_DIR = RAW_DATA_DIR / "geomac"   # -- validation-only perimeters, never extracted
FPA_FOD = RAW_DATA_DIR / "fpa_fod" / "fires.parquet"
USFS_POINTS = RAW_DATA_DIR / "usfs" / "National_USFS_Fire_Occurrence_Point_(Feature_Layer).csv"

KEYMAP = """
 left/right      step one day            [ / ]  or shift+arrow   step seven days
 home/end        first/last day of the current year block
 y / Y           previous / next year block
 n / p           next / previous day with an event inside the view
 f / F           cycle fires by size, descending / ascending
 c               cycle the scalar channel heatmap (off, then each channel)
 a / k           toggle archive probability / climatology
 g               toggle GeoMAC perimeter brackets for the year
 h               reset the view      ?   print this map      scroll   zoom
"""


def _cmap(name):
    from matplotlib import colormaps
    return colormaps[name]


def _edges(m):
    """ Returns the boundary cells of mask m: cells inside m adjacent to a cell
        with a differing value.
    """
    e = np.zeros(m.shape, bool)
    e[:-1] |= m[:-1] != m[1:]
    e[1:] |= m[1:] != m[:-1]
    e[:, :-1] |= m[:, :-1] != m[:, 1:]
    e[:, 1:] |= m[:, 1:] != m[:, :-1]
    return e & m


def _stripe(shape):
    yy, xx = np.indices(shape)
    return ((yy + xx) % 5) < 2


class Store:
    """ One opened zarr store, read lazily one day at a time. """

    def __init__(self, cfg, kind, split):
        """ Opens the store of the given kind (cube, dataset, or split) for the
            dataset config, and derives its time index, coordinates, extent, and
            channel list.
        """
        path = {"cube": cfg.staging_path, "dataset": cfg.published_path,
                "split": cfg.split_path(split)}[kind]
        self.path, self.kind, self.res = path, kind, cfg.resolution
        self.ds = xr.open_zarr(path)
        self.times = np.asarray(self.ds.indexes["time"], dtype="datetime64[D]")
        self.years = self.times.astype("datetime64[Y]").astype(int) + 1970
        self.xc, self.yc = self.ds.x.values, self.ds.y.values
        h = cfg.resolution / 2.0
        self.extent = (self.xc[0] - h, self.xc[-1] + h, self.yc[-1] - h, self.yc[0] + h)
        self.shape = (self.ds.sizes["y"], self.ds.sizes["x"])
        self.channels = [str(c) for c in self.ds.channel.values] if "channel" in self.ds.coords else [
            n for n, v in self.ds.data_vars.items()
            if v.dims == ("time", "y", "x") and v.dtype.kind == "f"]

    def name(self, canon):
        for n in ALIASES[canon]:
            if n in self.ds.data_vars or n in self.channels:
                return n
        return None

    def has(self, canon):
        return self.name(canon) is not None

    def raw(self, name, t):
        if name in self.ds.data_vars:
            return np.asarray(self.ds[name].isel(time=t).values)
        return np.asarray(self.ds["X"].isel(time=t, channel=self.channels.index(name)).values)

    def day(self, canon, t):
        name = self.name(canon)
        return None if name is None else self.raw(name, t)

    def mask(self, canon, t):
        a = self.day(canon, t)
        return None if a is None else np.nan_to_num(a) > 0


def base_rgb(store):
    """ Static grayscale relief with water in light blue, built once. """
    from matplotlib.colors import LightSource
    field = store.day("elevation", 0)
    field = store.day("ndvi", 0) if field is None else field
    z = np.nan_to_num(np.asarray(field, np.float64))
    shade = LightSource(315, 45).hillshade(z, vert_exag=6.0, dx=store.res, dy=store.res)
    rgb = np.repeat((0.35 + 0.6 * shade)[..., None], 3, axis=2)
    water = store.mask("water", 0)
    if water is None:
        land = store.mask("land", 0)
        water = None if land is None else ~land
    if water is not None:
        rgb[water] = (0.62, 0.76, 0.90)
    return np.clip(rgb, 0, 1)


def load_fires(store, cfg):
    """ Discovery points inside the grid bounds and years, largest first. """
    lo_lat, hi_lat = cfg.lat_bounds
    lo_lon, hi_lon = cfg.lon_bounds
    if FPA_FOD.exists():
        df = pd.read_parquet(FPA_FOD)
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"discovery_date": "date", "fire_size": "size"})
        df["name"] = df.get("fire_name", pd.Series(["fire"] * len(df), index=df.index))
    else:
        cols = ["X", "Y", "discoverydatetime", "totalacres", "firename", "statcause"]
        df = pd.read_csv(USFS_POINTS, usecols=cols, low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"x": "lon", "y": "lat", "discoverydatetime": "date",
                                "totalacres": "size", "firename": "name",
                                "statcause": "cause"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df.dropna(subset=["date", "lat", "lon"])
    yrs = df["date"].dt.year
    df = df[(df.lat.between(lo_lat, hi_lat)) & (df.lon.between(lo_lon, hi_lon))
            & yrs.between(int(store.years.min()), int(store.years.max()))]

    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4326", GRID_CRS, always_xy=True)
    df = df.assign(size=pd.to_numeric(df["size"], errors="coerce").fillna(0.0))
    px, py = tf.transform(df.lon.values, df.lat.values)
    df = df.assign(px=px, py=py)
    return df.sort_values("size", ascending=False).reset_index(drop=True)


class Viewer:
    def __init__(self, store, cfg, args):
        """ Builds the figure, layer artists, and archive/climatology sources for
            the given store and CLI args, and draws the first frame.
        """
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors, rcParams

        for k in [k for k in rcParams if k.startswith("keymap.")]:
            rcParams[k] = []
        self.store, self.cfg, self.args = store, cfg, args
        self.blocks = sorted(set(int(y) for y in store.years))
        self.t = self._year_start(args.year) if args.year else 0
        self.fires, self.fire_i, self.fire_label = None, -1, ""
        self.chan_i, self.geomac_on, self.geomac_warned = -1, False, False
        self.event_days = None

        self.layers = [(k, spec) for k, spec in
                       zip(HOTKEY_POOL, [s for s in LAYERS if store.has(s[0])])]
        self.on = {spec[0]: spec[3] != "invert_hatch" for _, spec in self.layers}

        self.arc, self.clim_path = None, None
        if args.archive:
            from .archive import load_archive
            self.arc = load_archive(args.archive, args.split)
            self.arc_idx = {d: i for i, d in enumerate(self.arc["dates"])}
        clim = REF_DIR / f"{args.dataset}_clim_bw20km.npz"
        self.clim_path = clim if clim.exists() else None
        self.show_prob, self.show_clim = self.arc is not None, False

        self.fig = plt.figure(figsize=(15, 9))
        self.ax = self.fig.add_axes([0.03, 0.05, 0.66, 0.88])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.base = base_rgb(store)
        self.ax.imshow(self.base, extent=store.extent, origin="upper", interpolation="nearest")
        self.home_lim = (self.ax.get_xlim(), self.ax.get_ylim())

        self.rasters, self.points = {}, {}
        for _, (canon, _label, colour, style) in self.layers:
            if style == "points":
                self.points[canon] = self.ax.scatter(
                    [], [], s=34, c=colour, marker="o", edgecolors="black",
                    linewidths=0.4, zorder=6)
            else:
                self.rasters[canon] = self._sheet(4)
        self.scalar_art, self.prob_art, self.top_art = (self._sheet(z) for z in (2, 3, 5))
        self.geomac_art = []
        self.rgb = {canon: mcolors.to_rgb(colour) for canon, _l, colour, _s in LAYERS}

        self.panel = self.fig.text(0.71, 0.93, "", va="top", ha="left", family="monospace",
                                   fontsize=9)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.refresh()

    def _sheet(self, zorder):
        """ A full-grid RGBA overlay pinned to the store's projected extent. """
        return self.ax.imshow(self.blank, extent=self.store.extent, origin="upper",
                              interpolation="nearest", zorder=zorder)

    @property
    def blank(self):
        return np.zeros(self.store.shape + (4,), np.float32)

    # -- time helpers -------------------------------------------------------
    @property
    def date(self):
        return self.store.times[self.t]

    @property
    def year(self):
        return int(self.store.years[self.t])

    def _year_start(self, year):
        idx = np.flatnonzero(self.store.years == int(year))
        return int(idx[0]) if len(idx) else 0

    def _step(self, n):
        self.t = int(np.clip(self.t + n, 0, len(self.store.times) - 1))

    def _goto_date(self, day):
        d = np.datetime64(pd.Timestamp(day).date())
        self.t = int(np.argmin(np.abs(self.store.times - d)))

    def _year_hop(self, n):
        i = int(np.clip(self.blocks.index(self.year) + n, 0, len(self.blocks) - 1))
        self.t = self._year_start(self.blocks[i])

    def _event_candidates(self):
        """ Returns, caching on first call, the time indices where any discovery
            point or label positive occurs somewhere on the grid.
        """
        if self.event_days is None:
            names = [self.store.name(c) for c in ("ign_occ", "burn_next")]
            total = None
            for n in [n for n in names if n and n in self.store.ds.data_vars]:
                s = (self.store.ds[n] > 0).sum(("y", "x")).values
                total = s if total is None else total + s
            self.event_days = np.flatnonzero(total > 0) if total is not None else np.array([], int)
        return self.event_days

    def _seek_event(self, step):
        """ Moves the current day, in the given step direction, to the next or
            previous day with a discovery point or label positive inside the
            current view.
        """
        x0, x1 = sorted(self.ax.get_xlim())
        y0, y1 = sorted(self.ax.get_ylim())
        ys = (self.store.yc >= y0) & (self.store.yc <= y1)
        xs = (self.store.xc >= x0) & (self.store.xc <= x1)
        view = np.outer(ys, xs)
        cands = self._event_candidates()
        cands = cands[cands > self.t] if step > 0 else cands[cands < self.t][::-1]
        for t in cands:
            hit = [self.store.mask(c, int(t)) for c in ("ign_occ", "burn_next")]
            if any(m is not None and (m & view).any() for m in hit):
                self.t = int(t)
                return

    # -- fires --------------------------------------------------------------
    def _cycle_fire(self, step):
        """ Steps to the next or previous fire in the discovery list, jumps the
            current day to it, and zooms the view to its extent.
        """
        if self.fires is None:
            self.fires = load_fires(self.store, self.cfg)
            print(f"fire list: {len(self.fires)} discovery points in bounds")
        if not len(self.fires):
            return
        self.fire_i = (self.fire_i + step) % len(self.fires)
        row = self.fires.iloc[self.fire_i]
        self._goto_date(row["date"])
        half = float(np.clip(np.sqrt(max(row["size"], 1.0)) * 220.0, 8000.0, 90000.0))
        self.ax.set_xlim(row.px - half, row.px + half)
        self.ax.set_ylim(row.py - half, row.py + half)
        self.fire_label = (f" | {row['name']} {row['size']:.0f} ac "
                           f"{row.get('cause', 'unknown')} {pd.Timestamp(row['date']).date()}")

    # -- drawing ------------------------------------------------------------
    def _layer_mask(self, canon, style, t):
        a = self.store.day(canon, t)
        if a is None:
            return None
        m = np.nan_to_num(a) > 0
        return ~m if style.startswith("invert") else m

    def _rgba(self, m, colour, alpha):
        """ Returns an RGBA array painting colour at the given alpha where m is
            true, transparent elsewhere.
        """
        out = np.zeros(m.shape + (4,), np.float32)
        out[..., :3] = colour
        out[..., 3] = np.where(m, alpha, 0.0)
        return out

    def _arrival_rgba(self, canon, t):
        """ Cells burning within a week either side of today, coloured by the
            signed day offset, showing a front's direction of travel;
            today's cells in the layer's own colour. """
        n = len(self.store.dates)
        delta = np.full(self.store.shape, np.nan, np.float32)
        for k in range(7, -8, -1):
            if 0 <= t + k < n:
                hit = np.nan_to_num(self.store.day(canon, t + k)) > 0
                delta[hit] = k
        valid = np.isfinite(delta)
        rgba = _cmap("coolwarm")((np.nan_to_num(delta) + 7.0) / 14.0).astype(np.float32)
        rgba[..., 3] = np.where(valid, 0.85, 0.0)
        rgba[valid & (delta == 0)] = list(self.rgb[canon]) + [0.95]
        return rgba

    def _draw_layers(self):
        """ Updates every layer artist (points and rasters) for the current day,
            clearing any layer that is toggled off.
        """
        for _, (canon, _label, _colour, style) in self.layers:
            if canon in self.points:
                art = self.points[canon]
                m = self._layer_mask(canon, style, self.t) if self.on[canon] else None
                if m is None or not m.any():
                    art.set_offsets(np.empty((0, 2)))
                else:
                    ys, xs = np.nonzero(m)
                    art.set_offsets(np.c_[self.store.xc[xs], self.store.yc[ys]])
                continue
            art = self.rasters[canon]
            if not self.on[canon]:
                art.set_data(self.blank)
                continue
            if style == "arrival":
                art.set_data(self._arrival_rgba(canon, self.t))
                continue
            m = self._layer_mask(canon, style, self.t)
            if style.endswith("outline"):
                m = _edges(m)
            elif style.endswith("hatch"):
                m = m & _stripe(m.shape)
            art.set_data(self._rgba(m, self.rgb[canon], 0.8 if style == "fill" else 0.9))

    def _draw_scalar(self):
        """ Updates the scalar channel heatmap overlay for the current day and
            selected channel, or clears it when no channel is selected.
        """
        if self.chan_i < 0:
            self.scalar_art.set_data(self.blank)
            return
        a = np.nan_to_num(self.store.raw(self.store.channels[self.chan_i], self.t).astype(np.float64))
        lo, hi = np.percentile(a, [2, 98])
        rgba = _cmap("viridis")(np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)).astype(np.float32)
        rgba[..., 3] = 0.55
        self.scalar_art.set_data(rgba)

    def _prob_field(self):
        """ Returns the probability field and its validity mask for the current
            day, drawn from the climatology reference or the loaded archive, or
            (None, None) when neither is available.
        """
        if self.show_clim and self.clim_path is not None:
            return reference_probs(str(self.clim_path), np.array([self.date]), self.store.shape)[0], None
        if self.arc is None or not self.show_prob:
            return None, None
        i = self.arc_idx.get(self.date)
        return (None, None) if i is None else (self.arc["p"][i], self.arc["mask"][i].astype(bool))

    def _draw_prob(self):
        """ Updates the probability overlay and its top-percentile outline for the
            current day, or clears both when no probability field is available.
        """
        q, m = self._prob_field()
        if q is None:
            self.prob_art.set_data(self.blank)
            self.top_art.set_data(self.blank)
            return
        sup = self.supervised() if m is None else m
        q = np.asarray(q, np.float64)
        hi = np.percentile(q[sup], 99.9) if sup.any() else q.max()
        rgba = _cmap("magma")(np.clip(q / max(hi, 1e-12), 0, 1)).astype(np.float32)
        rgba[..., 3] = np.where(sup, 0.6, 0.0)
        self.prob_art.set_data(rgba)
        thr = np.percentile(q[sup], 99.0) if sup.any() else np.inf
        self.top_art.set_data(self._rgba(_edges(sup & (q >= thr)), (1, 1, 1), 0.95))

    def _draw_geomac(self):
        """ Redraws the GeoMAC perimeter outlines for the current year up to the
            current date, clearing prior artists first.
        """
        for art in self.geomac_art:
            art.remove()
        self.geomac_art = []
        if not self.geomac_on:
            return
        path = GEOMAC_DIR / f"{self.year}.parquet"
        if not path.exists():
            if not self.geomac_warned:
                print(f"no GeoMAC brackets under {GEOMAC_DIR}")
                self.geomac_warned = True
            return
        import geopandas as gpd
        gdf = gpd.read_parquet(path).to_crs(GRID_CRS)
        col = next((c for c in gdf.columns if "date" in c.lower()), None)
        if col is not None:
            keep = pd.to_datetime(gdf[col], errors="coerce").values.astype("datetime64[D]") <= self.date
            gdf = gdf[keep]
        for geom in gdf.geometry:
            for poly in getattr(geom, "geoms", [geom]):
                xs, ys = poly.exterior.xy
                self.geomac_art += self.ax.plot(xs, ys, color="#00e5ff", lw=1.0, zorder=7)

    def supervised(self):
        land = self.store.mask("land", self.t)
        land = np.ones(self.store.shape, bool) if land is None else land
        free = self.store.mask("unsupervised", self.t)
        return land if free is None else (land & free)

    def _panel_text(self):
        """ Returns the multi-line status panel text for the current day. """
        s, t = self.store, self.t
        land = s.mask("land", t)
        land = ~s.mask("water", t) if land is None and s.has("water") else land
        land = np.ones(s.shape, bool) if land is None else land
        sup = self.supervised()
        n_land = int(land.sum())
        lines = [f"{self.date}   t={t}/{len(s.times) - 1}   block {self.year}",
                 f"grid {s.shape[0]}x{s.shape[1]} @ {s.res:g} m   store {s.kind}",
                 "",
                 f"land cells      {n_land}",
                 f"supervised      {int(sup.sum())}  ({sup.sum() / max(n_land, 1):.3f} of land)",
                 f"masked land     {1.0 - sup.sum() / max(n_land, 1):.3f}",
                 "", "events today"]
        for key, (canon, label, _c, _s) in self.layers:
            m = s.mask(canon, t)
            flag = "x" if self.on[canon] else " "
            lines.append(f" [{flag}] {key} {label:<20s} {int(m.sum()) if m is not None else 0}")
        pos = s.mask("burn_next", t)
        act = s.mask("active", t)
        if pos is not None and act is not None:
            near = binary_dilation(act, iterations=max(int(round(2000.0 / s.res)), 1))
            lines += ["", f"label spread    {int((pos & near).sum())}",
                      f"label ignition  {int((pos & ~near).sum())}"]
        chan = "off" if self.chan_i < 0 else s.channels[self.chan_i]
        lines += ["", f"c channel       {chan}"]
        q, m = self._prob_field()
        if q is not None and pos is not None:
            sup_q = sup if m is None else m
            thr = np.percentile(np.asarray(q)[sup_q], 99.0) if sup_q.any() else np.inf
            ev = pos & sup_q
            frac = float((ev & (q >= thr)).sum()) / max(int(ev.sum()), 1)
            src = "climatology" if self.show_clim else self.args.archive
            lines += [f"{src}", f" events {int(ev.sum())}   in top 1% {frac:.3f}"]
        return "\n".join(lines)

    def refresh(self):
        """ Redraws layers, scalar, probability, and GeoMAC overlays, and updates
            the title and status panel.
        """
        self._draw_layers()
        self._draw_scalar()
        self._draw_prob()
        self._draw_geomac()
        self.ax.set_title(f"{self.store.path.name}  {self.date}{self.fire_label}", fontsize=11)
        self.panel.set_text(self._panel_text())
        self.fig.canvas.draw_idle()

    # -- input --------------------------------------------------------------
    def on_scroll(self, event):
        if event.inaxes is not self.ax:
            return
        f = 0.8 if event.button == "up" else 1.25
        for lim, setter, c in ((self.ax.get_xlim(), self.ax.set_xlim, event.xdata),
                               (self.ax.get_ylim(), self.ax.set_ylim, event.ydata)):
            setter(c + (lim[0] - c) * f, c + (lim[1] - c) * f)
        self.fig.canvas.draw_idle()

    def _home(self):
        self.ax.set_xlim(*self.home_lim[0])
        self.ax.set_ylim(*self.home_lim[1])

    def _toggle(self, attr):
        setattr(self, attr, not getattr(self, attr))

    def on_key(self, event):
        """ Handles a key press, toggling a layer's visibility or running the
            matching view action, then refreshes the display.
        """
        k = event.key
        actions = {
            "right": lambda: self._step(1), "left": lambda: self._step(-1),
            "]": lambda: self._step(7), "shift+right": lambda: self._step(7),
            "[": lambda: self._step(-7), "shift+left": lambda: self._step(-7),
            "home": lambda: setattr(self, "t", self._year_start(self.year)),
            "end": lambda: setattr(
                self, "t", int(np.flatnonzero(self.store.years == self.year)[-1])),
            "y": lambda: self._year_hop(-1), "Y": lambda: self._year_hop(1),
            "n": lambda: self._seek_event(1), "p": lambda: self._seek_event(-1),
            "f": lambda: self._cycle_fire(1), "F": lambda: self._cycle_fire(-1),
            "c": lambda: setattr(self, "chan_i", -1 if self.chan_i + 1 >= len(
                self.store.channels) else self.chan_i + 1),
            "a": lambda: self._toggle("show_prob"), "k": lambda: self._toggle("show_clim"),
            "g": lambda: self._toggle("geomac_on"), "h": self._home,
        }
        hot = {key: canon for key, (canon, _l, _c, _s) in self.layers}
        if k == "?":
            print(KEYMAP)
            return
        if k in hot:
            self.on[hot[k]] = not self.on[hot[k]]
        elif k in actions:
            actions[k]()
        else:
            return
        self.refresh()

    # -- export -------------------------------------------------------------
    def _export_window(self):
        """ Returns the pixel window to export: the full grid when it is 200x200
            cells or smaller, otherwise the current view cropped to at most 256
            cells per axis, plus whether cropping occurred.
        """
        H, W = self.store.shape
        if H <= 200 and W <= 200:
            y0, x0, y1, x1 = 0, 0, H, W
        else:
            xl, yl = sorted(self.ax.get_xlim()), sorted(self.ax.get_ylim())
            xs = np.flatnonzero((self.store.xc >= xl[0]) & (self.store.xc <= xl[1]))
            ys = np.flatnonzero((self.store.yc >= yl[0]) & (self.store.yc <= yl[1]))
            x0, x1 = (int(xs[0]), int(xs[-1]) + 1) if len(xs) else (0, W)
            y0, y1 = (int(ys[0]), int(ys[-1]) + 1) if len(ys) else (0, H)
        cropped = False
        for lo, hi, n, axis in ((y0, y1, H, "y"), (x0, x1, W, "x")):
            if hi - lo > 256:
                cropped = True
                c = (lo + hi) // 2
                lo = int(np.clip(c - 128, 0, n - 256))
                if axis == "y":
                    y0, y1 = lo, lo + 256
                else:
                    x0, x1 = lo, lo + 256
        return y0, y1, x0, x1, cropped

    def export(self, out_path):
        """ Writes a standalone HTML export of the current view, with the base
            image and a 21-day window of each layer's mask baked in as data.
        """
        import matplotlib.image as mimage
        y0, y1, x0, x1, cropped = self._export_window()
        ts = [t for t in range(self.t - 10, self.t + 11) if 0 <= t < len(self.store.times)]
        buf = io.BytesIO()
        mimage.imsave(buf, self.base[y0:y1, x0:x1], format="png")
        base64_png = base64.b64encode(buf.getvalue()).decode()

        specs = []
        for _key, (canon, label, colour, style) in self.layers:
            days = []
            for t in ts:
                m = self._layer_mask(canon, style, t)
                m = np.zeros((y1 - y0, x1 - x0), bool) if m is None else m[y0:y1, x0:x1]
                days.append(base64.b64encode(np.packbits(m).tobytes()).decode())
            specs.append({"n": label, "c": colour, "d": days})

        rows = ",".join(
            '{n:"%s",c:"%s",on:true,d:[%s]}' % (s["n"], s["c"], ",".join(f'"{d}"' for d in s["d"]))
            for s in specs)
        dates = ",".join(f'"{self.store.times[t]}"' for t in ts)
        note = (" Extent cropped to 256x256 cells around the view centre."
                if cropped else "")
        html = _HTML.replace("__W__", str(x1 - x0)).replace("__H__", str(y1 - y0))
        html = html.replace("__BASE__", base64_png).replace("__LAYERS__", rows)
        html = html.replace("__DATES__", dates).replace("__START__", str(ts.index(self.t)))
        html = html.replace("__TITLE__", f"{self.store.path.name} {self.date}")
        html = html.replace("__NOTE__", note)
        Path(out_path).write_text(html)
        print(f"wrote {out_path}  {Path(out_path).stat().st_size / 1e6:.2f} MB  "
              f"{x1 - x0}x{y1 - y0} cells  {len(ts)} days")


_HTML = """<!doctype html><meta charset="utf-8"><title>__TITLE__</title>
<style>body{font:13px system-ui;margin:12px;background:#111;color:#ddd}
canvas{image-rendering:pixelated;border:1px solid #444;width:760px}
label{display:block;margin:2px 0}#side{float:right;width:230px}</style>
<h3>__TITLE__</h3><div id="side"></div><div id="d"></div><canvas id="cv"></canvas>
<p>Arrow keys step days.__NOTE__</p>
<script>
const W=__W__,H=__H__,DATES=[__DATES__],LAYERS=[__LAYERS__];let day=__START__;
const cv=document.getElementById("cv");cv.width=W;cv.height=H;
const ctx=cv.getContext("2d");const tmp=document.createElement("canvas");
tmp.width=W;tmp.height=H;const tctx=tmp.getContext("2d");
const img=new Image();img.src="data:image/png;base64,__BASE__";
function bits(s){const b=atob(s),o=new Uint8Array(W*H);
 for(let i=0;i<W*H;i++)o[i]=(b.charCodeAt(i>>3)>>(7-(i&7)))&1;return o;}
function rgb(h){return [parseInt(h.substr(1,2),16),parseInt(h.substr(3,2),16),parseInt(h.substr(5,2),16)];}
function render(){ctx.clearRect(0,0,W,H);ctx.drawImage(img,0,0,W,H);
 for(const L of LAYERS){if(!L.on)continue;const m=bits(L.d[day]),c=rgb(L.c);
  const im=tctx.createImageData(W,H);
  for(let i=0;i<W*H;i++){if(m[i]){im.data[4*i]=c[0];im.data[4*i+1]=c[1];
   im.data[4*i+2]=c[2];im.data[4*i+3]=215;}}
  tctx.putImageData(im,0,0);ctx.drawImage(tmp,0,0);}
 document.getElementById("d").textContent=DATES[day]+"  ("+(day+1)+"/"+DATES.length+")";}
const side=document.getElementById("side");
LAYERS.forEach((L,i)=>{const l=document.createElement("label");
 l.innerHTML='<input type="checkbox" checked> <span style="color:'+L.c+'">&#9632;</span> '+L.n;
 l.querySelector("input").onchange=e=>{L.on=e.target.checked;render();};side.appendChild(l);});
addEventListener("keydown",e=>{if(e.key=="ArrowRight")day=Math.min(day+1,DATES.length-1);
 else if(e.key=="ArrowLeft")day=Math.max(day-1,0);else return;e.preventDefault();render();});
img.onload=render;
</script>"""


def main():
    """ Parses CLI args, opens the requested store, and launches or exports the
        day-stepping viewer.
    """
    ap = argparse.ArgumentParser(description="Step through a tier's store day by day")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--store", default="cube", choices=("cube", "dataset", "split"))
    ap.add_argument("--split", default="test", choices=("train", "eval", "test"))
    ap.add_argument("--fold", default="full")
    ap.add_argument("--year", type=int)
    ap.add_argument("--archive")
    ap.add_argument("--export", nargs="?", const="",
                    help="write the HTML export; a bare flag names it after the store")
    ap.add_argument("--no-show", action="store_true", help="render headless, skip the window")
    args = ap.parse_args()

    import matplotlib
    if not args.no_show and not os.environ.get("MPLBACKEND"):
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    cfg = get_dataset_config(args.dataset, args.fold)
    store = Store(cfg, args.store, args.split)
    print(f"opened {store.path}  {len(store.times)} days  {store.shape}")
    view = Viewer(store, cfg, args)
    print("layers:", ", ".join(f"{k}={c}" for k, (c, _l, _col, _s) in view.layers))
    if args.export is not None:
        VIEWER_DIR.mkdir(parents=True, exist_ok=True)
        out = args.export or str(VIEWER_DIR / f"{store.path.name}_{view.date}.html")
        view.export(out)
    if not args.no_show:
        print(KEYMAP)
        plt.show()


if __name__ == "__main__":
    main()
