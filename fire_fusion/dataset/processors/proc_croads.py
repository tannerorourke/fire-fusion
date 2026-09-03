import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr, rioxarray
from rasterio import features as rfeatures
from shapely.geometry import box
from scipy.ndimage import distance_transform_edt

from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import CROADS_DIR
from .processor import Processor


class CensusRoads(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)
    
    def build_feature(self, f_config: Feature):
        # -- load roads, reproject, clip
        road_paths = [
            CROADS_DIR / "tl_2012_WA_53_prisecroads.shp",
            CROADS_DIR / "tl_2012_ID_16_prisecroads.shp",
            CROADS_DIR / "tl_2012_OR_41_prisecroads.shp"
        ]
        roads = gpd.GeoDataFrame(
            pd.concat([gpd.read_file(p) for p in road_paths], ignore_index=True),
            crs = gpd.read_file(road_paths[0]).crs
        )
        
        print(f"Informing tensors to stop whining.. MORE DATA!")

        # -- TIGER/Line ships in NAD83 degrees; clip bounds are in grid (UTM meter) space
        roads = roads.to_crs(self.mCRS)

        roads = gpd.clip(roads, box(
            self.gridref.attrs['x_min'], self.gridref.attrs['y_min'],
            self.gridref.attrs['x_max'], self.gridref.attrs['y_max']
        ))

        # -- Rasterize 1 where there is a road
        mgrid_ht, mgrid_wt = self.gridref.attrs['template'].shape

        road_raster = rfeatures.rasterize(
            ((geom, 1) for geom in roads.geometry),
            out_shape=(mgrid_ht, mgrid_wt),
            transform=self.transformer,
            fill=0,
            dtype="uint8"
        )

        # -- Compute Euclidean distance (in pixels) to the nearest road
        # `distance_transform_edt` does distance from 'True' pixels; we want distance FROM roads. flip the sign
        d_to_road_px = distance_transform_edt(road_raster == 0)
        if isinstance(d_to_road_px, tuple | None):
            raise ValueError("[CROADS] Calling this func wrong")
        
        # -- Construct an xarray.DataArray aligned with the master grid
        dist_to_road_m = d_to_road_px * self._get_px_size_m()
        dist_3d_m = np.broadcast_to(
            array = dist_to_road_m.astype("float32"),
            shape = (self.gridref.sizes.get("time", 1), mgrid_ht, mgrid_wt),
        )
        dist_da = xr.DataArray(
            dist_3d_m,
            dims=self.gridref.dims,
            coords=self.gridref.coords,
            name=f_config.name or "d_to_road",
        )

        # -- write the crs reference and transform to rioxarray engine, return with it
        dist_da = dist_da.rio.write_crs(self.mCRS)
        dist_da = dist_da.rio.write_transform(self.transformer)

        return dist_da