"""
Annual NLCD layers: impervious fraction, canopy cover, land_cover, and
cropland fraction derived from land_cover.
"""
from typing import List
import xarray as xr
import numpy as np
import pandas as pd
from rasterio.enums import Resampling

from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import NLCD_DIR
from fire_fusion.config.feature_config import LAND_COVER_RAW_MAP
from ..build_utils import load_as_xarr, release_memory
from .processor import Processor

# -- NLCD pasture/hay and cultivated crops
CROPLAND_CLASSES = (81, 82)

class NLCD(Processor):
    def __init__(self, cfg, mgrid):
        super().__init__(cfg, mgrid)

    def build_feature(self, f_cfg: Feature):
        yearly_arrs: List[xr.DataArray] = []

        files = [f for f in NLCD_DIR.glob("*.tiff") if (f_cfg.key and f_cfg.key in f.stem)]
        if not files:
            print(f"No .tiff files with {f_cfg.key}")
            return xr.Dataset()

        for fp in sorted(files):
            year = int(fp.stem.split("_")[3])

            with load_as_xarr(fp, name=f_cfg.name) as raw:
                arr = self._preclip_native_arr(raw)

                # binarize the class raster at its native 30 m
                if f_cfg.name == "cropland_frac":
                    print(f"[NLCD] Counting the {year} wheat fields, one at a time..")
                    yearly_arrs.append(self._stamp_year(self._build_cropland_frac(arr, f_cfg), year))
                    del arr
                    release_memory()
                    continue

                arr = self._reproject_arr_to_mgrid(arr, f_cfg.resampling)

                if f_cfg.key == "FctImp":
                    print(f"[NLCD] Resolving the great conflict of {year} between urban folk and farm folk..")
                    arr = self._build_frac_imp_surface(arr, f_cfg)
                elif f_cfg.key == "tccconus":
                    print(f"[NLCD] Swinging from the trees like its {year}, weeeeeeeeee!!")
                    arr = self._build_canopy_cover_pct(arr, f_cfg)
                elif f_cfg.key == "LndCov":
                    print(f"[NLCD] Computing {year} land cover % purely based on vibes..")
                    arr = self._build_land_cover(arr, f_cfg)
                else:
                    print(f"[NLCD] Unknown key {f_cfg.key}???")

                yearly_arrs.append(self._stamp_year(arr, year))
            release_memory()

        feature_by_year = xr.concat(yearly_arrs, dim="time").to_dataset(name=f_cfg.name)
        feature_by_year = feature_by_year.sortby("time")
        feature_by_year = self._time_interpolate(feature_by_year, f_cfg.time_interp)
        feat_data = feature_by_year.transpose("time", "y", "x", ...)
        return feat_data
    

    def _stamp_year(self, arr: xr.DataArray, year: int) -> xr.DataArray:
        if "time" in arr.dims:
            return arr
        ts = pd.Timestamp(f"{year}-01-01")
        return arr.expand_dims(time=[ts]).assign_coords(time=[ts])


    def _build_cropland_frac(self, feature: xr.DataArray, f_cfg: Feature):
        """ Build in numpy (30 m class raster is ~300M pixels) """
        classes = feature.values
        binary = np.isin(classes, CROPLAND_CLASSES).astype("float32")
        
        binary[np.isnan(classes)] = np.nan
        del classes

        crop = xr.DataArray(binary, coords=feature.coords, dims=feature.dims, name=f_cfg.name)
        crop = crop.rio.write_crs(feature.rio.crs).rio.write_transform(feature.rio.transform())

        frac = self._reproject_arr_to_mgrid(crop, Resampling.average)
        frac = frac.fillna(0.0).clip(0.0, 1.0)
        frac.name = f_cfg.name
        return frac


    def _build_frac_imp_surface(self, feature: xr.DataArray, f_cfg: Feature):
        """ Convert % to [0, 1], clip """
        fis = feature.where(~(feature > 100)).astype("float32")
        fis = (fis / 100)

        if f_cfg.clip is not None:
            low, high = f_cfg.clip
            fis = fis.clip(low, high)

        fis = fis.fillna(0.0)
        
        fis.name = f_cfg.name
        return fis
    
    def _build_canopy_cover_pct(self, feature: xr.DataArray, f_cfg: Feature):
        """ Convert % to [0, 1], clip """
        cc_frac = feature.where(~(feature > 100)).astype("float32")
        cc_frac = (cc_frac / 100)

        if f_cfg.clip is not None:
            low, high = f_cfg.clip
            cc_frac = cc_frac.clip(low, high)

        cc_frac = cc_frac.fillna(0.0)

        cc_frac.name = f_cfg.name
        return cc_frac
    
    """ deprecated """
    def _build_land_cover(self, feature: xr.DataArray, f_cfg: Feature):
        H, W = feature.shape

        data = feature.where(~(feature > 100))
        data = data.fillna(-1).astype("int16")
        data_arr = data.values
        classes = list(LAND_COVER_RAW_MAP.keys())

        hot_encode = np.zeros((len(classes), H, W), dtype=np.float32)

        for idx, (_, raw_codes) in enumerate(LAND_COVER_RAW_MAP.items()):
            mask = np.isin(data_arr, raw_codes)
            hot_encode[idx][mask] = 1.0

        lc_ohe = xr.DataArray(
            hot_encode,
            dims=( "lcov_class", "y", "x" ),
            coords={ "lcov_class": classes, 
                "y": feature.y, 
                "x": feature.x 
            },
            name=f_cfg.name
        )
        lc_ohe = lc_ohe.rio.write_crs(self.gridref.rio.crs)
        lc_ohe = lc_ohe.rio.write_transform(self.gridref.rio.transform())
        return lc_ohe