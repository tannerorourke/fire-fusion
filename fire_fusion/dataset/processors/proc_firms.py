"""
FIRMS MODIS active-fire detections as a daily presence layer on the master grid.

One CSV per year of point detections with an along-scan pixel size. 
Each kept detection paints a circular footprint of that pixel's radius, and a cell is set
on the acquisition date when its centre falls inside one.
"""
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from pyproj import Transformer
from rasterio.features import rasterize

from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import FIRMS_DIR
from ..build_utils import print_layer_stats, release_memory
from .processor import Processor

# -- type 0 is a presumed vegetation fire; 1-3 are volcanoes, static sources and offshore
FIRE_TYPE = 0
# -- below this the detection is as likely to be a false alarm as a fire
MIN_CONFIDENCE = 25
# -- 'scan' is the along-scan pixel width in km, 1.0 at nadir; the footprint is
#    the circle of that diameter around the reported centre
SCAN_KM_TO_RADIUS_M = 500.0
# -- degrees of slack on the lat/lon prefilter, wider than any footprint radius
BOX_MARGIN_DEG = 0.05

CSV_COLUMNS = ["latitude", "longitude", "acq_date", "confidence", "scan", "type"]


class Firms(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)
        self.mt_ix = self.gridref.attrs["time_index"]
        self.to_mcrs = Transformer.from_crs("EPSG:4326", self.mCRS, always_xy=True)

    def build_feature(self, f_cfg: Feature) -> xr.Dataset:
        if f_cfg.key != "MODIS_AF":
            print(f"[FIRMS] Unknown key {f_cfg.key}???")
            return xr.Dataset()

        print("[FIRMS] Asking the satellite what is on fire right now..")
        ny, nx = self.gridref.sizes["y"], self.gridref.sizes["x"]
        detect = np.zeros((len(self.mt_ix), ny, nx), dtype="uint8")
        transform = self.gridref.rio.transform()

        for year in self.gridref.attrs["years"]:
            fp = FIRMS_DIR / f"modis_{year}_United_States.csv"
            if not fp.exists():
                print(f"[FIRMS] no detections file for {year}")
                continue

            df = self._read_year(fp)
            if df.empty:
                continue

            for t_idx, group in df.groupby("t_idx"):
                footprints = shapely.buffer(
                    shapely.points(group["mx"].to_numpy(), group["my"].to_numpy()),
                    group["radius_m"].to_numpy(),
                )
                detect[int(t_idx)] = np.maximum(
                    detect[int(t_idx)],
                    rasterize(
                        [(geom, 1) for geom in footprints],
                        out_shape=(ny, nx),
                        transform=transform,
                        all_touched=False,
                        fill=0,
                        dtype="uint8",
                    ),
                )
            print(f"[FIRMS] {year}: {len(df):,} detections over {df['t_idx'].nunique()} days")
            release_memory()

        layer = xr.DataArray(
            detect,
            name=f_cfg.name,
            coords={
                "time": self.mt_ix,
                "y": self.gridref.coords["y"].values,
                "x": self.gridref.coords["x"].values,
            },
            dims=("time", "y", "x"),
        )
        print_layer_stats(f_cfg.name, layer)

        out = layer.to_dataset(name=f_cfg.name)
        out = out.rio.write_crs(self.gridref.rio.crs)
        return out.rio.write_transform(transform)

    def _read_year(self, fp) -> pd.DataFrame:
        df = pd.read_csv(fp, usecols=CSV_COLUMNS)
        df = df[(df["type"] == FIRE_TYPE) & (df["confidence"] >= MIN_CONFIDENCE)]

        lat0, lat1 = self.gridref.attrs["lat_min"], self.gridref.attrs["lat_max"]
        lon0, lon1 = self.gridref.attrs["lon_min"], self.gridref.attrs["lon_max"]
        df = df[
            df["latitude"].between(lat0 - BOX_MARGIN_DEG, lat1 + BOX_MARGIN_DEG)
            & df["longitude"].between(lon0 - BOX_MARGIN_DEG, lon1 + BOX_MARGIN_DEG)
        ].copy()
        if df.empty:
            return df

        time2index = pd.Series(np.arange(len(self.mt_ix)), index=self.mt_ix)
        acq = pd.to_datetime(df["acq_date"]).dt.floor("D")
        df["t_idx"] = time2index.reindex(acq).to_numpy()
        df = df[df["t_idx"].notna()].copy()
        if df.empty:
            return df

        df["t_idx"] = df["t_idx"].astype("int32")
        df["mx"], df["my"] = self.to_mcrs.transform(
            df["longitude"].to_numpy(), df["latitude"].to_numpy()
        )
        df["radius_m"] = df["scan"].to_numpy() * SCAN_KM_TO_RADIUS_M
        return df
