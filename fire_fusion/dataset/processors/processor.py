"""
Subclass generalizing functions for feature extraction from a source.

Includes a Callable(xr.Dataset) sink to push data through as it is built. Features 
that expand into several large arrays can overload even high-RAM systems.
"""
import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Tuple
from pyproj import Transformer
from xarray.core.types import InterpOptions

from fire_fusion.config.feature_config import Feature

class Processor:
    
    
    def __init__(self, cfg: List[Feature], gridref: xr.DataArray):
        self.cfg = cfg
        self.gridref = gridref
        self.mCRS = gridref.rio.crs
        self.transformer = self.gridref.rio.transform()
        self.sink = None

    def build_feature(self, f_config: Feature) -> xr.Dataset:
        """ Read necessary files based on the feature key, and route to corresponding functions to process
        """
        raise NotImplementedError


    def _get_px_size_km(self) -> float:
        px_size_x = abs(self.transformer.a) / 1000.0 # width/lon
        px_size_y = abs(self.transformer.e) / 1000.0 # height/lat
        return (abs(px_size_x) + abs(px_size_y)) / 2.0
    
    def _get_px_size_m(self) -> float:
        px_size_x = abs(self.transformer.a) # width/lon
        px_size_y = abs(self.transformer.e) # height/lat
        return (abs(px_size_x) + abs(px_size_y)) / 2.0


    def _preclip_grid_fn(self, obj: xr.DataArray, px_m = 3, deg_m = 0.05) -> xr.DataArray:
        """ Don't call me. Call _preclip_native_arr OR _preclip_native_dataset """
        if not hasattr(obj, "rio") or obj.rio.crs is None:
            # print("[WARNING] Object CRS not found.. ", obj.attrs)
            if obj.attrs.get('coordinate_system', '') != '':
                other_sys = obj.attrs['coordinate_system']
                epsg = None
                if "EPSG:" in other_sys:
                    epsg = other_sys.split("EPSG:")[-1].split(",")[0].strip()
                elif "4326" in other_sys:
                    epsg = "4326"
                if epsg is None:
                    raise ValueError("no crs")
                obj = obj.rio.write_crs(f"EPSG:{epsg}")
                obj = obj.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
                # CRS recovered from the attribute, so the array can take the
                # clip below rather than reaching reprojection at native extent

        minx, miny = self.gridref.attrs['x_min'], self.gridref.attrs['y_min']
        maxx, maxy = self.gridref.attrs['x_max'], self.gridref.attrs['y_max']
        
        if obj.rio.crs != self.mCRS:
            transformation = Transformer.from_crs(
                crs_from = self.gridref.rio.crs, 
                crs_to   = obj.rio.crs,
                always_xy= True
            )
            x, y = transformation.transform(
                # matching indices correspond to 
                # bot-left, bot-right, top-left, top-right
                [minx, maxx, minx, maxx],
                [miny, miny, maxy, maxy]
            )
            minx, maxx = min(x), max(x)
            miny, maxy = min(y), max(y)

        if obj.rio.crs.is_geographic:
            mx = my = deg_m
        else:
            tfm = obj.rio.transform()
            px_size_x = abs(tfm.a)
            px_size_y = abs(tfm.e)
            px_size = (px_size_x + px_size_y) / 2.0
            mx = my = px_size * px_m

        print("clip values", minx-mx, miny-my, maxx+mx, maxy+my)


        obj = obj.rio.clip_box(
            minx=minx-mx, maxx=maxx+mx,
            miny=miny-my, maxy=maxy+my
        )
        return obj
    
    def _preclip_native_arr(self, obj: xr.DataArray) -> xr.DataArray:
        return self._preclip_grid_fn(obj)
    def _preclip_native_dataset(self, obj: xr.Dataset) -> xr.Dataset:
        clipped_vars = {}
        for name, da in obj.data_vars.items():
            clipped_vars[name] = self._preclip_grid_fn(da)
        ds = xr.Dataset(clipped_vars)
        ds = ds.assign_coords(ds.coords)
        ds.attrs = obj.attrs
        return ds



    def _reproject_to_mgrid_fn(self, obj: xr.DataArray, resample_type = None) -> xr.DataArray:
        """ Don't call me. Call _reproject_arr_to_mgrid OR _reproject_dataset_to_mgrid """
        # -- Float sources with NaN gaps need nodata=NaN, else reproject_match backfills gaps
        # -- and out-of-extent cells with the float32 sentinel ~3.4e38. Int sources keep their fill.
        if obj.rio.nodata is None and np.issubdtype(obj.dtype, np.floating):
            obj = obj.rio.write_nodata(np.nan)
        if resample_type is None:
            obj = obj.rio.reproject_match(self.gridref)
        else:
           obj = obj.rio.reproject_match(self.gridref, resampling=resample_type)

        obj = obj.rio.write_crs(self.mCRS)
        return obj

    def _reproject_arr_to_mgrid(self, obj: xr.DataArray, resample_type) -> xr.DataArray:
        return self._reproject_to_mgrid_fn(obj, resample_type)
    def _reproject_dataset_to_mgrid(self, obj: xr.Dataset, resample_type) -> xr.Dataset:
        clipped_vars = {}
        for name, da in obj.data_vars.items():
            clipped_vars[name] = self._reproject_to_mgrid_fn(da, resample_type)
        ds = xr.Dataset(clipped_vars)
        ds = ds.assign_coords(ds.coords)
        ds.attrs = obj.attrs
        return ds
            


    def _time_interpolate(self, source_ds: xr.Dataset, interpol: Tuple[str, InterpOptions] | None) -> xr.Dataset:
        """ After feature has been concatenated over time into a dataset, broadcast 
            over master time for data which is missing days
            - If data has no 'time' dimension:
                add dimension, broadcast over master time index
            - If data has 'time' of size == 1:
                broadcast over master time index
            - If data has EXISTING time indices
                interpolate over them
        """
        if interpol is None:
            print("No interpolation")
            return source_ds
        interp_type, interp_method = interpol
        
        time_index = self.gridref.attrs['time_index']
        
        # Static or single-slice features -> broadcast over master time
        if interp_type == "broadcast":
            if "time" not in source_ds.dims:
                expanded = source_ds.expand_dims(dim={ "time": time_index })
                expanded = expanded.assign_coords(time=time_index)
                return expanded
            
            if source_ds.sizes["time"] == 1:
                single = source_ds.isel(time=0, drop=True)
                expanded = single.expand_dims(dim={ "time": time_index })
                expanded = expanded.assign_coords(time=time_index)
                return expanded

            # Multiple source timestamps: align onto the master index (a raw
            # return would leak foreign time coords into the outer merge)
            return source_ds.reindex(time=time_index, method="nearest")

        elif interp_type == "existing":
            if "time" not in source_ds.dims:
                print("Interpolator source_ds missing time dimension")
                return source_ds

            if not interp_method:
                return source_ds
            # Ensure increasing order for interpolation
            feature_sorted = source_ds.sortby("time")

            # When every target day already sits on a source day (daily sources
            # sampled onto a subset calendar), interpolation only reproduces the
            # source values. Select them instead: exact, and it skips the float64
            # interp that would otherwise materialize the whole cube at once.
            target = pd.DatetimeIndex(time_index)
            src_times = feature_sorted.indexes["time"]
            # Source already on the exact target index (e.g. season-trimmed
            # upstream): return as-is rather than copying it via .sel.
            if src_times.equals(target):
                return feature_sorted
            if target.isin(src_times).all():
                selected = feature_sorted.sel(time=target)
                for name, da in selected.data_vars.items():
                    print("post-interp(select)", name,
                        "finite:", int(np.isfinite(da).sum()),
                        "min/max:", float(da.min()), float(da.max()))
                return selected

            interp = feature_sorted.interp(time=time_index, method=interp_method)

            # interp() does not extrapolate; days outside the source's coverage
            # hold the nearest edge observation instead of going NaN
            edge = feature_sorted.reindex(time=time_index, method="nearest")
            interp = interp.fillna(edge)

            for name, da in interp.data_vars.items():
                print("post-interp", name,
                    "finite:", int(np.isfinite(da).sum()),
                    "min/max:", da.min().item(), da.max().item())

            return interp
        
        return source_ds
    
    