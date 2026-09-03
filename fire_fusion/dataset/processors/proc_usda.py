from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr, rioxarray
from rasterio import features as rfeatures
from shapely.geometry import box
from scipy.ndimage import distance_transform_edt
from pyproj import Transformer

from fire_fusion.config.feature_config import WUI_CLASS_MAP, Feature
from fire_fusion.config.path_config import USDA_DIR
from .processor import Processor


class UsdaWui(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)

        self.st_abrev = 'WA'
        self.layer_name = "CONUS_WUI_block_1990_2020_change_v4"
        self.years = (2000, 2010, 2020)
        self.data_cols = [
            'WUICLASS2000', 'WUICLASS2010', 'WUICLASS2020',
            'HUDEN2000', 'HUDEN2010', 'HUDEN2020',
            'geometry'
        ]

        self.geo_block = None
        self.wui_index = None

    def _load_geo_block(self, fp: Path) -> gpd.GeoDataFrame:
        obj_slice = gpd.read_file(fp, engine="pyogrio", layer=self.layer_name, rows=slice(0, 1))
        print(f"[USDA WUI] Generating filter bounding box transformed to {self.mCRS}")

        minx, miny = self.gridref.attrs['x_min'], self.gridref.attrs['y_min']
        maxx, maxy = self.gridref.attrs['x_max'], self.gridref.attrs['y_max']

        # Shouldn't be equal
        if obj_slice.crs != self.mCRS:
            transformer = Transformer.from_crs(
                crs_from=self.mCRS, crs_to=obj_slice.crs,
                always_xy=True,
            )
            xs, ys = transformer.transform(
                [minx, maxx, minx, maxx],
                [miny, miny, maxy, maxy],
            )
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)

        # Build buffered bbox polygon and clip
        mx = my = self._get_px_size_m() * 3
        bounding_box = box(
            minx - mx, miny - my, 
            maxx + mx, maxy + my
        )

        print(f"[USDA WUI] Reading clipped WA grid")
        obj = gpd.read_file(fp, engine="pyogrio",
            layer=self.layer_name,
            bbox=bounding_box,
            where=f"STATE = '{self.st_abrev}'"
        )
        # Ensure reprojected to model CRS for raster
        if obj.crs != self.mCRS:
            obj = obj.to_crs(self.mCRS)

        # Drop water polygons
        if "WATER20" in obj.columns:
            obj = obj.loc[obj["WATER20"] != 1].copy()

        return obj[self.data_cols].copy()

    def _reindex_drop_cols(self, block: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """ Add re-indexed WUI columns, drop the raw class columns
            (geometry and HUDEN columns stay; rasterization needs them)
        """
        try:
            to_drop = []
            for year in (2000, 2010, 2020):
                reindex_col = f"WUI_INDEX_{year}"
                class_col = f"WUICLASS{year}"

                block[reindex_col] = (
                    block[class_col].map(WUI_CLASS_MAP)
                    .fillna(0).astype("int16")
                )
                to_drop.append(class_col)

            block = block.drop(columns=[c for c in to_drop if c in block.columns])
            return block
        except Exception as e:
            raise LookupError(f"[USDA WUI] Issue reindexing/dropping columns", e)


    def _rasterize_val_fn(self, column: str) -> xr.DataArray:
        """ Rasterize a given attribute to the model grid w/ rasterio.features.rasterize. """
        assert self.geo_block is not None, "_rasterize_val_fn expected pre-built self.geo_block"

        y_name = self.gridref.rio.y_dim
        x_name = self.gridref.rio.x_dim

        out_shape = (self.gridref.sizes[y_name], self.gridref.sizes[x_name])
        shapes = ((geom, val) for geom, val in zip(self.geo_block.geometry, self.geo_block[column]))

        arr = rfeatures.rasterize(
            shapes,
            out_shape,
            fill=0,
            transform=self.transformer,
            dtype="float32",
        )
        da = xr.DataArray(
            arr,
            coords={y_name: self.gridref['y'], x_name: self.gridref['x']},
            dims=(y_name, x_name),
            name=column,
        )
        return da


    def build_feature(self, f_config: Feature):
        fp = USDA_DIR / "CONUS_WUI_block_1990_2020_change_v4.gdb"

        # Load and save on the first feature
        if self.geo_block is None:
            geo_block = self._load_geo_block(fp)
            self.geo_block = self._reindex_drop_cols(geo_block)
        
        # --- Features ---

        if f_config.key == "hs_density":
            da = self._get_housing_density(f_config)

            # clip 99% percentile
            p99 = float(np.nanpercentile(da, 99))
            da = da.clip(max=p99)

        elif f_config.key == "wui_index":
            da = self.wui_index = self._get_wui_index(f_config)

        elif f_config.key == "dist_to_interface":
            da = self._get_dist_to_wui(f_config)

        assert isinstance(da, xr.DataArray), "Failed to create data :("
        
        da = da.to_dataset(name=f_config.name)
        ds_interp = self._time_interpolate(da, f_config.time_interp)
        return ds_interp
    
    

    def _get_housing_density(self, f_cfg: Feature) -> xr.DataArray:
        assert self.geo_block is not None, f"self._compute_housing_density expected self.geo_block"

        timestamps = pd.to_datetime([f"{y}-01-01" for y in self.years])
        hd_list: list[xr.DataArray] = []

        for year in self.years:
            housing_col = f"HUDEN{year}"

            if housing_col not in self.geo_block.columns:
                raise KeyError(f"missing {housing_col} in WUI blocks.")
            hd_list.append(self._rasterize_val_fn(housing_col))

        housing_density = xr.concat(hd_list, dim="time").assign_coords(time=("time", timestamps))
        housing_density = housing_density.astype("float32")
        housing_density.name = f_cfg.name        
        return housing_density
    
    def _get_wui_index(self, f_cfg: Feature) -> xr.DataArray:
        assert self.geo_block is not None, "self._compute_housing_density expected pre-computed self.geo_block"

        timestamps = pd.to_datetime([f"{y}-01-01" for y in self.years])
        wui_ix_list: list[xr.DataArray] = []

        for year in self.years:
            index_col = f"WUI_INDEX_{year}"
            wui_ix_list.append(self._rasterize_val_fn(index_col))

        wui_ix = xr.concat(wui_ix_list, dim="time").assign_coords(time=("time", timestamps))
        wui_ix.name = f_cfg.name
        return wui_ix


    def _get_dist_to_wui(self, f_cfg: Feature) -> xr.DataArray:
        """ 
        Compute distance-to-WUI rasters from WUI_INDEX >= 3 for 2000/2010/2020 
        """
        assert self.geo_block is not None, "self._compute_housing_density expected pre-computed self.geo_block"
        assert self.wui_index is not None, "self._compute_housing_density expected pre-computed self.wui_index"

        times = self.wui_index.coords["time"].values
        dist_stack: list[np.ndarray] = []

        for t in times:
            idx = self.wui_index.sel(time=t).values

            wui_mask = (idx >= 3) & np.isfinite(idx)

            # Compute Euclidean distance (in pixels) to the nearest WUI
            # distance_transform_edt does distance from 'True' pixels;
            # we want distance FROM the WUI -> 0 where WUI
            d_to_wui_m = distance_transform_edt(~wui_mask, sampling=self._get_px_size_m())
            
            if isinstance(d_to_wui_m, tuple | None):
                raise ValueError("[USDA_WUI] Calling this func wrong")
            
            d_to_wui_km = d_to_wui_m.astype("float32") / 1000.0

            dist_stack.append(d_to_wui_km[None, ...])

        dist_arr = np.concatenate(dist_stack, axis=0)
        dist_da = xr.DataArray(
            dist_arr,
            dims=self.wui_index.dims,
            coords=self.wui_index.coords,
            name="DIST_TO_WUI_EDGE"
        )
        return dist_da
    


if __name__ == "__main__":
    import fiona
    fiona.listlayers(USDA_DIR / "CONUS_WUI_block_1990_2020_change_v4.gdb")
    from fire_fusion.config.feature_config import base_feat_config
    from ..grid import create_coordinate_grid
    
    grid = create_coordinate_grid(
        time_index=pd.date_range("2000-01-01", "2020-12-31", freq="D"),
        resolution=4000,
        lat_bounds = (45.4, 49.1),
        lon_bounds = (-124.8, -117.0)
    )

    bfg = base_feat_config()
    features = [f for f in bfg["USDA_WUI"]]

    processor = UsdaWui(features, grid)
    for feat in features:
        processor.build_feature(feat)
        print(f"built {feat.name}")