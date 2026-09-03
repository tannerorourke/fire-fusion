""" NCEI SWDI (Severe Weather Data Inventory) NLDN cloud-to-ground lightning.

Vaisala's National Lightning Detection Network, published by NCEI as a daily count of 
CG flashes per 0.1 degree tile (~11 km). A tile-day appears when at least one strike was detected.

The tiles form a regular lon/lat lattice. Rather than reproject the full daily stack, the master grid's 
cell centers are mapped once to their containing tile and the daily counts are gathered through that map 
(an exact nearest-neighbour resample from the coarse tile grid onto the fine master grid).
"""
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

from .processor import Processor
from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import NCEI_SWDI_DIR


# SWDI tile size / centroid spacing
TILE_DEG = 0.1  


class Lightning(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)
        self.mt_ix = self.gridref.attrs["time_index"]

    def build_feature(self, f_cfg: Feature) -> xr.Dataset:
        print("[NCEI SWDI] Counting cloud-to-ground strikes...")

        native, lat0, lon0, nlat, nlon = self._load_native_tiles()
        strikes = self._gather_to_mgrid(native, lat0, lon0, nlat, nlon, f_cfg.name)
        return strikes.to_dataset(name=f_cfg.name)

    def _window_years(self):
        return range(self.mt_ix[0].year, self.mt_ix[-1].year + 1)

    def _load_native_tiles(self):
        """ Read the per-year tile CSVs, clip to the master extent, and scatter
            daily strike counts onto a regular 0.1 degree lon/lat lattice.
        """
        # a margin to ensure master edge cells always fall inside the native extent
        margin = 0.2
        lat_min = self.gridref.attrs["lat_min"] - margin
        lat_max = self.gridref.attrs["lat_max"] + margin
        lon_min = self.gridref.attrs["lon_min"] - margin
        lon_max = self.gridref.attrs["lon_max"] + margin

        years = set(self._window_years())
        frames = []
        for fp in sorted(NCEI_SWDI_DIR.glob("nldn-tiles*.csv")):
            try:
                year = int(fp.stem.replace(".", "-").split("-")[-1])
            except ValueError:
                continue
            if year not in years:
                continue

            df = pd.read_csv(
                fp, comment="#", header=None,
                names=["ZDAY", "CENTERLON", "CENTERLAT", "TOTAL_COUNT"],
                dtype={"ZDAY": str},
            )
            df = df[
                (df["CENTERLON"] >= lon_min) & (df["CENTERLON"] <= lon_max) &
                (df["CENTERLAT"] >= lat_min) & (df["CENTERLAT"] <= lat_max)
            ]
            frames.append(df)

        tiles = pd.concat(frames, ignore_index=True)

        # UTC day string -> master time index position. Membership, not a range
        # test: a season-windowed master index has gaps inside [first, last], and
        # a strike on a dropped day has no position to scatter into.
        dates = pd.to_datetime(tiles["ZDAY"], format="%Y%m%d").dt.floor("D")
        in_window = dates.isin(self.mt_ix)
        tiles = tiles.loc[in_window]
        dates = dates.loc[in_window]

        time2index = pd.Series(np.arange(len(self.mt_ix)), index=self.mt_ix)
        t_idx = time2index.reindex(dates).to_numpy().astype("int64")

        # centroids sit on 0.1 degree multiples; snap to integer lattice indices
        lon0 = float(np.round(tiles["CENTERLON"].min() / TILE_DEG) * TILE_DEG)
        lat0 = float(np.round(tiles["CENTERLAT"].min() / TILE_DEG) * TILE_DEG)
        lon_idx = np.round((tiles["CENTERLON"].to_numpy() - lon0) / TILE_DEG).astype("int64")
        lat_idx = np.round((tiles["CENTERLAT"].to_numpy() - lat0) / TILE_DEG).astype("int64")
        nlon = int(lon_idx.max()) + 1
        nlat = int(lat_idx.max()) + 1

        native = np.zeros((len(self.mt_ix), nlat, nlon), dtype="float32")
        np.add.at(
            native,
            (t_idx, lat_idx, lon_idx),
            tiles["TOTAL_COUNT"].to_numpy(dtype="float32"),
        )
        return native, lat0, lon0, nlat, nlon

    def _gather_to_mgrid(self, native, lat0, lon0, nlat, nlon, name) -> xr.DataArray:
        """ Map every master cell centre to its containing tile, then gather the
            daily counts through that map -- a nearest-neighbour resample that
            preserves each tile's true count without inventing sub-tile gradients.
        """
        mx = self.gridref.coords["x"].values
        my = self.gridref.coords["y"].values
        xx, yy = np.meshgrid(mx, my)  # (ny, nx) in master CRS metres

        to_geo = Transformer.from_crs(self.mCRS, "EPSG:4326", always_xy=True)
        lon_m, lat_m = to_geo.transform(xx.ravel(), yy.ravel())

        lat_i = np.clip(np.round((lat_m - lat0) / TILE_DEG).astype("int64"), 0, nlat - 1)
        lon_i = np.clip(np.round((lon_m - lon0) / TILE_DEG).astype("int64"), 0, nlon - 1)

        ny, nx = len(my), len(mx)
        gathered = native[:, lat_i, lon_i].reshape(len(self.mt_ix), ny, nx)

        da = xr.DataArray(
            gathered,
            name=name,
            coords={"time": self.mt_ix, "y": my, "x": mx},
            dims=("time", "y", "x"),
        )
        da = da.rio.write_crs(self.gridref.rio.crs)
        da = da.rio.write_transform(self.gridref.rio.transform())
        return da
