"""
Fire ignition and perimeter layers on the master grid.

Ignitions come from FPA-FOD (Short 2022), the all-agency wildfire occurrence record: one point per 
fire at its discovery location and date, with an NWCG cause. Every fire rasterizes into 'ign_occ'; the per-cause planes of 'ign_cause'
hold only fires whose cause maps to a class. 

Perimeters come from the USFS perimeter layer as an int32 fire id per day, painted at the polygon's final extent between its start 
and end dates. 

The cause KDEs are exponentially decayed running sums of the cause planes.

  python -m fire_fusion.dataset.processors.proc_ignitions   # FPA-FOD zip -> fires.parquet
"""
import sqlite3
import zipfile
from pathlib import Path
import xarray as xr, rioxarray
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter

from .processor import Processor
from fire_fusion.config.feature_config import CAUSAL_CLASSES, CAUSE_RAW_MAP, Feature
from fire_fusion.config.path_config import FPA_FOD_DIR, USFS_DIR


class Ignitions(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)
        self.grx_min, self.grx_max = self.gridref.attrs['x_min'], self.gridref.attrs['x_max']
        self.gry_min, self.gry_max = self.gridref.attrs['y_min'], self.gridref.attrs['y_max']
        self.mt_ix = self.gridref.attrs['time_index']
    
    def build_feature(self, f_cfg: Feature) -> xr.Dataset:
        if f_cfg.key == "Fire_Perimeter":
            print(f"\n[USFS] computing fire perimeter")

            file = USFS_DIR / "National_USFS_Fire_Perimeter_(Feature_Layer).shp"
            layer = self._build_perim_layer(file, f_cfg)
            return layer.to_dataset(name=f_cfg.name)

        elif f_cfg.key == "Fire_Occurence":
            print(f"\n[FPA-FOD] computing ignition layers")
            self.occ_cause_layer = layer = self._build_occ_cause_layers(FPA_FOD_DIR / "fires.parquet", f_cfg)
            return layer

        elif f_cfg.key == "Fire_KDE":
            print(f"\n[USFS] computing fire KDE")
            # -- At fine resolutions the full array per cause is float32 (time, y, x)
            #    Push each cause through the sink as soon as it is computed when the builder provides one
            if self.sink is not None:
                for name, da_kde in self._iter_kde_layers(f_cfg):
                    self.sink(da_kde.to_dataset(name=name))
                    del da_kde
                return xr.Dataset()

            return xr.Dataset(dict(self._iter_kde_layers(f_cfg)))

        return xr.Dataset()
    

    def get_clipped(self, fp: Path):
        layer = gpd.read_file(fp).to_crs(self.mCRS)
        return gpd.clip(layer, 
            box(
                self.gridref.attrs['x_min'], self.gridref.attrs['y_min'], 
                self.gridref.attrs['x_max'], self.gridref.attrs['y_max']
            )
        )
    
    @staticmethod
    def normalize_cause(raw) -> object:
        val = str(raw).strip().lower()
        for kls, keywords in CAUSE_RAW_MAP.items():
            if val in keywords:
                return kls
        return np.nan

    def _build_occ_cause_layers(self, fp: Path, f_cfg: Feature) -> xr.Dataset:
        fires = pd.read_parquet(fp)
        fires = gpd.GeoDataFrame(
            fires, geometry=gpd.points_from_xy(fires["lon"], fires["lat"]), crs="EPSG:4326",
        ).to_crs(self.mCRS)
        fires = gpd.clip(fires, box(self.grx_min, self.gry_min, self.grx_max, self.gry_max))

        discovery = pd.to_datetime(fires["discovery_date"]).dt.floor("D")
        time2index = pd.Series(np.arange(len(self.mt_ix)), index=self.mt_ix)
        fires["t_idx"] = time2index.reindex(discovery).to_numpy()
        fires = fires[fires["t_idx"].notna()].copy()
        fires["t_idx"] = fires["t_idx"].astype("int32")
        # -- a fire without a mapped cause is still an ignition; it only stays
        #    out of the cause planes
        fires["burn_cause_class"] = fires["general_cause"].map(self.normalize_cause)
        fires_usfs = fires

        # === Fire Occurences
        occ_grid = np.zeros((len(self.mt_ix), len(self.gridref.y), len(self.gridref.x)), dtype="uint8")
        for t_idx in np.unique(fires_usfs["t_idx"].to_numpy()):
            sub = fires_usfs[fires_usfs["t_idx"] == t_idx]
            shapes = [(geom, 1) for geom in sub.geometry]
            occ_grid[int(t_idx)] = rasterize(
                shapes,
                out_shape=(len(self.gridref.y), len(self.gridref.x)),
                transform=self.gridref.rio.transform(),
                all_touched=False,
                fill=0,
                dtype="uint8"
            )

        # === Fire Cause
        cause_labels = pd.Index(CAUSAL_CLASSES, name="burn_cause")
        cause_grid = np.zeros((
            len(self.mt_ix), len(cause_labels), 
            len(self.gridref.y), len(self.gridref.x)
        ), dtype="uint8")
        for (t_idx, cause), fires_group in fires_usfs.groupby(["t_idx", "burn_cause_class"]):
            if cause not in cause_labels:
                continue
            cause_idx = cause_labels.get_loc(cause)
            shapes = [(geom, 1) for geom in fires_group.geometry]
            cause_grid[int(t_idx), cause_idx] = rasterize(
                shapes,
                out_shape=(len(self.gridref.y), len(self.gridref.x)), 
                transform=self.gridref.rio.transform(),
                all_touched=False,
                fill=0, 
                dtype="uint8",
            )

        assert f_cfg.expand_names is not None, "burn occ/cause expects expand names"

        fire_occ_cause_tyx = xr.Dataset({
            f_cfg.expand_names[0]: xr.DataArray(
                occ_grid,
                name=f_cfg.expand_names[0],
                coords={
                    "time":self.mt_ix,
                    "y":self.gridref.coords['y'].values,
                    "x":self.gridref.coords['x'].values 
                },
                dims=("time", "y", "x")
            ),
            f_cfg.expand_names[1]: xr.DataArray(
                cause_grid,
                name=f_cfg.expand_names[1],
                coords={ 
                    "time": self.mt_ix, 
                    "burn_cause": cause_labels, 
                    "y":self.gridref.coords['y'].values,
                    "x":self.gridref.coords['x'].values 
                },
                dims=("time", "burn_cause", "y", "x")
            )
        })

        fire_occ_cause_tyx = fire_occ_cause_tyx.rio.write_crs(self.gridref.rio.crs)
        fire_occ_cause_tyx = fire_occ_cause_tyx.rio.write_transform(self.gridref.rio.transform())
        return fire_occ_cause_tyx
    

    def _iter_kde_layers(self, f_cfg: Feature):
        """ Yield ("kde_<cause>", DataArray) one cause at a time.
        
        Locally computes an exponentially decayed running sum of ignitions, splitting running sums
        per year.
        """
        assert self.occ_cause_layer is not None, "Fire-KDE expected burn data/occurence layer pre-computed"

        fire_occurences = self.occ_cause_layer["ign_cause"]

        px_size_xkm = float(abs(self.gridref.rio.transform().a) / 1000)
        px_size_ykm = float(abs(self.gridref.rio.transform().e) / 1000)
        pixel_size_km = (px_size_xkm + px_size_ykm) / 2.0

        # -- Sigma = how wide the bell curve is IN 2d PIXELS = equals average of X/Y pixel = kde_radius / pixel_size
        # -- Radius = max radius of filter influence in meters (coordinates)
        kde_radius = f_cfg.kde_smooth_radius_km if f_cfg.kde_smooth_radius_km is not None else 10
        sigma_pixels = (
            kde_radius / pixel_size_km if pixel_size_km > 0 else 0.0
        )

        print(f"sigma pixels = {kde_radius} / {pixel_size_km} = {sigma_pixels}")

        half_life = f_cfg.kde_half_life_days if f_cfg.kde_half_life_days is not None else 365.0
        alpha = float(0.5 ** (1.0 / half_life))

        # Decay by the real number of days (as opposed to seasonal days appearing consecutively)
        # between consecutive index entries. 
        times = pd.DatetimeIndex(fire_occurences.coords["time"].values)
        step_days = np.diff(times.asi8) / (1e9 * 86400.0)   # ns between entries -> days
        step_decay = (alpha ** step_days).astype("float32")  # length T-1

        for cause in fire_occurences.coords["burn_cause"].values:
            # uint8 view of the occurrence stack
            occ_txy = fire_occurences.sel(burn_cause=cause).values

            kde_txy = np.zeros(occ_txy.shape, dtype="float32")
            fire_days = np.flatnonzero(occ_txy.reshape(occ_txy.shape[0], -1).sum(axis=1) > 0)
            if fire_days.size == 0:
                print(f"WARNING: Sum of All X/Y across time is 0 for {cause}")
            for t in fire_days:
                kde_txy[t] = gaussian_filter(
                    occ_txy[t].astype("float32"), sigma=sigma_pixels, mode="constant"
                )

            # -- In-place IIR recursion load[t] = smoothed[t] + decay(dt) * load[t-1].
            #    Holds only one (y, x) accumulator for mem efficiency
            acc = np.zeros(kde_txy.shape[1:], dtype="float32")
            for t in range(kde_txy.shape[0]):
                if t > 0:
                    acc *= step_decay[t - 1]
                acc += kde_txy[t]
                kde_txy[t] = acc

            name = f"kde_{str(cause).lower()}"
            da_kde = xr.DataArray(
                kde_txy,
                coords={
                    "time": fire_occurences.coords["time"],
                    "y": fire_occurences.coords["y"],
                    "x": fire_occurences.coords["x"],
                },
                dims=("time", "y", "x"),
                name=name,
            )
            yield name, da_kde


    def _build_perim_layer(self, fp: Path, f_cfg: Feature) -> xr.DataArray:
        fires_usfs = self.get_clipped(fp)

        # -- Fire Start/End Time --
        start_date = pd.to_datetime(fires_usfs["DISCOVERYD"], errors="coerce")
        final_date = pd.to_datetime(fires_usfs["PERIMETERD"], errors="coerce")

        # -- drop rows with missing disco date --
        valid_dfull = start_date.notna() & final_date.notna()
        fires_usfs = fires_usfs.loc[valid_dfull].copy()
        start_date = start_date.loc[valid_dfull]
        final_date   = final_date.loc[valid_dfull]

        # -- convert to datetime, move days ending in 00:00:00 back one day
        start_dates = start_date.dt.floor("D")
        end_dates = final_date.dt.floor("D")
        
        is_EOD = ((final_date.dt.hour == 0) & 
                  (final_date.dt.minute == 0) & 
                  (final_date.dt.second == 0))
        end_dates = end_dates - pd.to_timedelta(is_EOD.astype("int64"), unit="D")

        # -- crop by start/end time index --
        clip_dates = (
            start_dates.dt.year.between(self.mt_ix[0].year, self.mt_ix[-1].year, inclusive="both")
            & end_dates.dt.year.between(self.mt_ix[0].year, self.mt_ix[-1].year, inclusive="both")
        )
        fires_usfs = fires_usfs.loc[clip_dates].copy()
        start_dates = start_dates.loc[clip_dates]
        end_dates   = end_dates.loc[clip_dates]

        if (end_dates < start_dates).any():
            end_dates[end_dates < start_dates] = start_dates[end_dates < start_dates]

        # -- align discovery dates to the grid index --
        start_dates = start_dates.clip(self.mt_ix[0], self.mt_ix[-1])
        end_dates   = end_dates.clip(self.mt_ix[0], self.mt_ix[-1])

        start_idx = self.mt_ix.searchsorted(start_dates.values, side="left")
        end_idx   = self.mt_ix.searchsorted(end_dates.values,   side="right") - 1
        valid_idx = (end_idx >= 0) & (start_idx < len(self.mt_ix))
        fires_usfs = fires_usfs.iloc[valid_idx].copy()

        fires_usfs["start_idx"] = start_idx[valid_idx]
        fires_usfs["end_idx"] = end_idx[valid_idx]

        # -- rasterize each day, larger polygons first to ensure small fires inside a complex keeps its own id
        fires_usfs = fires_usfs.sort_values("GISACRES", ascending=False)
        fires_usfs["fire_id"] = fires_usfs["OBJECTID"].astype("int32")
        time_grid = np.zeros((
            len(self.mt_ix),
            len(self.gridref.y),
            len(self.gridref.x)
        ), dtype="int32")
        for t_idx in range(len(self.mt_ix)):
            active = (fires_usfs["start_idx"] <= t_idx) & (fires_usfs["end_idx"] >= t_idx)
            if not active.any():
                continue
            sub = fires_usfs.loc[active]
            time_grid[t_idx] = rasterize(
                shapes=list(zip(sub.geometry, sub["fire_id"])),
                out_shape=(len(self.gridref.y), len(self.gridref.x)),
                transform=self.gridref.rio.transform(),
                all_touched=False,
                fill=0,
                dtype="int32"
            )

        perim_txy = xr.DataArray(
            time_grid,
            name=f_cfg.name,
            coords={
                "time": self.mt_ix,
                "y":    self.gridref.coords['y'].values,
                "x":    self.gridref.coords['x'].values 
            },
            dims=("time", "y", "x")
        )
        perim_txy = perim_txy.rio.write_crs(self.gridref.rio.crs)
        perim_txy = perim_txy.rio.write_transform(self.gridref.rio.transform())
        return perim_txy


# -- the SQLite release is 1 GB for the whole country; the extract needs a few
#    columns for one box, written once to a small parquet that syncs like any
#    other raw source
FPA_FOD_BOX = {"lat": (45.0, 49.5), "lon": (-125.0, -116.5)}
FPA_FOD_COLUMNS = {
    "FOD_ID": "fod_id", "FIRE_NAME": "fire_name", "FIRE_YEAR": "fire_year",
    "DISCOVERY_DATE": "discovery_date", "CONT_DATE": "cont_date",
    "NWCG_CAUSE_CLASSIFICATION": "cause_class", "NWCG_GENERAL_CAUSE": "general_cause",
    "NWCG_REPORTING_AGENCY": "agency", "FIRE_SIZE": "fire_size", "FIRE_SIZE_CLASS": "size_class",
    "LATITUDE": "lat", "LONGITUDE": "lon", "STATE": "state", "MTBS_ID": "mtbs_id",
}


def prepare_fpa_fod(zip_path: Path, out: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        member = next(n for n in z.namelist() if n.endswith(".sqlite"))
        z.extract(member, zip_path.parent)
    db = zip_path.parent / member
    cols = ", ".join(FPA_FOD_COLUMNS)
    (lat0, lat1), (lon0, lon1) = FPA_FOD_BOX["lat"], FPA_FOD_BOX["lon"]
    with sqlite3.connect(db) as con:
        df = pd.read_sql_query(
            f"select {cols} from Fires where LATITUDE between {lat0} and {lat1} "
            f"and LONGITUDE between {lon0} and {lon1}", con,
        )
    db.unlink()
    df = df.rename(columns=FPA_FOD_COLUMNS)
    for c in ("discovery_date", "cont_date"):
        df[c] = pd.to_datetime(df[c], format="%m/%d/%Y", errors="coerce")
    df = df[df["discovery_date"].notna()].sort_values(["discovery_date", "fod_id"])
    df.to_parquet(out, index=False)
    return df


if __name__ == "__main__":
    zips = sorted(FPA_FOD_DIR.glob("*.zip"))
    df = prepare_fpa_fod(zips[-1], FPA_FOD_DIR / "fires.parquet")
    print(f"[FPA-FOD] {len(df):,} fires in the box, {df.fire_year.min()}-{df.fire_year.max()}, "
          f"-> {FPA_FOD_DIR / 'fires.parquet'}")
    print(df["general_cause"].value_counts().to_string())
