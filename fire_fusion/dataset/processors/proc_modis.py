"""
MODIS grid layers: 
- MOD13Q1: Normalized Difference Vegetation Index (NDVI) (250m 16 days)
- MCD15A2H: Leaf area index (LAI) (500m 8 days)
- MCD64A1 Burned area (500m, monthly)
    - One granule per tile per month holding the day of year each pixel burned.
    - Becomes the burn flag, reported uncertainty, and causal age of the most recent burn at or before each day.

https://earthaccess.readthedocs.io/en/stable/

Note: This is the most memory dominant build step. granule reprojection is VERY memory-dominant.
"""
# 
from multiprocessing import AuthenticationError
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from rasterio.enums import Resampling

import earthaccess
from earthaccess import DataGranule

from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import MODIS_DIR
from ..build_utils import load_as_xdataset, print_layer_stats, release_memory
from .processor import Processor

# -- Config ------------------------------------------------------------------
# First data of MCD64A1 burn age granules
BURN_RECORD_START = pd.Timestamp("2000-11-01")
# Stand in for 'did not burn'
BURN_NO_DATE = 9999
BURN_AGE_CAP = 32000

# One year of burns (day offset from the record start, iy, ix, uncertainty)
BurnEvents = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class Modis(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)

        # Login required when granules must be fetched; cached-local don't depend on network credentials
        try:
            self.auth = earthaccess.login(strategy="environment", persist=True)
        except Exception as e:
            self.auth = None
            print(f"[LAADS] Earthdata login unavailable ({e}); only locally cached granules can be used")
        if self.auth is not None and self.auth.username:
            print(f"Logged into EARTH DATA for {self.auth.username}")

        self.latlon_tup = (
            self.gridref.attrs['lon_min'], self.gridref.attrs['lat_min'],
            self.gridref.attrs['lon_max'], self.gridref.attrs['lat_max']
        )
        MODIS_DIR.mkdir(exist_ok=True, parents=True)

        # Granule HDF file name format: 
        # MOD13Q1.A2000049.h09v05.006.2015136104623.hdf 
        # = <product>/<year>/<day-of-year>/<granule-files>.hdf 
        # = MOD13Q1, year 2000, day 49, h09/v05, hash
        # We only want tiles that overlay the Pacific Northwest, which covers the below h/v indices
        self.tiles = ["h08v04", "h08v05", "h09v04", "h09v05", "h10v04"]
        self.version = "061"
        # Only run >1 if memory is no concern
        self.max_parallel_req = 1


    def build_feature(self, f_cfg: Feature) -> xr.Dataset:
        if f_cfg.key == "MCD64A1":
            return self._build_burn_layers(f_cfg)

        # NDVI/LAI forward-filled to a full calendar year per source year
        is_daily = f_cfg.key in ("MOD13Q1", "MCD15A2H")
        master_days = pd.DatetimeIndex(self.gridref.attrs["time_index"])

        parts: List[xr.Dataset] = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_req) as executor:
            if f_cfg.key == "MOD13Q1":
                print("[LAADS] Purchasing satellite to collect more vegetation data...")
                requests = {
                    executor.submit(self._fetch_ndvi, f_cfg, yr): yr
                    for yr in self.gridref.attrs['years']
                }
            elif f_cfg.key == "MCD15A2H":
                print(f"[LAADS] Checking how big the leaves are")
                requests = {
                    executor.submit(self._fetch_lai, f_cfg, yr): yr
                    for yr in self.gridref.attrs['years']
                }

            for req in as_completed(requests):
                yr = requests[req]
                try:
                    yr_ds = req.result()
                    if yr_ds is None:
                        continue
                    if is_daily:
                        # keep only the days the (seasonal) master index will use, cuts the accumulation to 1/3 of the days
                        tix = yr_ds.indexes["time"]
                        yr_ds = yr_ds.isel(time=np.flatnonzero(tix.isin(master_days)))
                    parts.append(yr_ds)
                    release_memory()

                except Exception as e:
                    print(f"[LAADS] Satellite tried to load data for {yr}, epic fail! --> {e}\n\n")

        # one concat instead of an accumulating merge, which copied the growing
        # result on every year
        if not parts:
            feature_by_yr = xr.Dataset()
        else:
            feature_by_yr = xr.concat(parts, dim="time")
            parts.clear()
            release_memory()

        return self._time_interpolate(
            feature_by_yr.sortby("time"),
            f_cfg.time_interp
        ).transpose("time", "y", "x", ...)


    def _parse_date(self, filename):
        # -- "MOD13Q1.A2000049.h09v05.006.2015136104623.hdf" -> year 2000, day 49
        year, doy = None, None
        for p in filename.split("."):
            if (p[0] == "A" and p[1:8].isdigit()):
                year = int(p[1:5])
                doy = int(p[5:8])
                break
        if year is None or doy is None:
            raise ValueError(f"[MODIS] Couldn't parse filename {filename}")
        return datetime(year, 1, 1) + timedelta(days=doy - 1)


    ### MCD152AH LAI
    def _fetch_lai(self, f_cfg: Feature, year: int) -> xr.Dataset | None:
        nc_files = self._fetch_year(f_cfg.key or "", year)
        if len(nc_files) == 0:
            return None

        year_data: List[xr.DataArray] = []
        for fp in nc_files:
            ts = pd.Timestamp(self._parse_date(fp.name))
            # print(f"[LAADS] Parsing {fp.stem} --> {ts}")

            with load_as_xdataset(file=fp, variables=["Lai_500m", "FparLai_QC", "FparExtra_QC"]) as raw:
                if len(raw.data_vars.items()) == 0:
                    continue

                arr = self._preclip_native_dataset(raw)
                arr = self._reproject_dataset_to_mgrid(arr, f_cfg.resampling)

                lai  = arr["Lai_500m"].astype("float32")
                qc   = arr["FparLai_QC"].fillna(0).astype("uint8")
                qcx  = arr["FparExtra_QC"].fillna(0).astype("uint8")

                # Primary QC (FParLai_QC)
                modland_gq = (qc & 0b1) == 0
                clouds_npres = ((qc >> 3) & 0b11).isin([0, 3])
                confident       = ((qc >> 5) & 0b111) < 4

                # ExtraWC (FparExtra_QC)
                island   = (qcx & 0b11).isin([0, 1])
                not_snow  = ((qcx >> 2) & 0b1) == 0

                aerosol   = (qcx >> 3) & 0b1
                cirrus    = (qcx >> 4) & 0b1
                int_cloud = (qcx >> 5) & 0b1
                shadow    = (qcx >> 6) & 0b1
                atm_good  = (aerosol == 0) & (cirrus == 0)
                no_clouds = (int_cloud == 0) & (shadow == 0)

                lai: xr.DataArray = lai.where(
                    modland_gq & clouds_npres & confident &
                    island & not_snow & no_clouds & atm_good
                )

                # Drop fill values 249-255 (kept in DN space) +
                # Apply MCD15A2H LAI 0.1 scale factor to map to 0-10 range (why do they do this is beyond me)
                lai = lai.where((lai >= 0) & (lai <= 100))
                lai = lai * 0.1

            lai = lai.expand_dims(time=[ts])
            year_data.append(lai)
            release_memory()

        stacked = xr.concat(year_data, dim="time").sortby("time")
        stacked = stacked.groupby("time").max("time")
        stacked = stacked.to_dataset(name=f_cfg.name)

        for name, da in stacked.data_vars.items():
            print_layer_stats(name, da)

        return stacked


    ### MOD13Q1 NDVI
    def _fetch_ndvi(self, f_cfg: Feature, year: int) -> xr.Dataset | None:
        nc_files = self._fetch_year(short_name=f_cfg.key or "", year=year)
        if len(nc_files) == 0:
            return None

        year_data_ndvi: List[xr.DataArray] = []
        year_data_water: List[xr.DataArray] = []
        for fp in nc_files:
            ts = pd.Timestamp(self._parse_date(fp.name))
            # print(f"[LAADS] Parsing {fp.stem} --> {ts}")

            with load_as_xdataset(file=fp, variables=["250m 16 days NDVI", "250m 16 days VI Quality"]) as raw:
                if len(raw.data_vars.items()) == 0:
                    continue

                arr = self._preclip_native_dataset(raw)
                arr = self._reproject_dataset_to_mgrid(arr, f_cfg.resampling)

                ndvi = arr["250m 16 days NDVI"].astype('float32')

                # QA code 0 decodes as both "good quality" and "shallow ocean", so
                # cells this granule never covered have to be tracked separately
                # instead of being folded into the bitfield as zeros.
                raw_qa = arr["250m 16 days VI Quality"]
                qa_fill = raw_qa.rio.nodata
                if qa_fill is None:
                    qa_fill = raw_qa.attrs.get("_FillValue", 65535)
                observed = raw_qa.notnull() & (raw_qa != qa_fill)
                qa = raw_qa.fillna(0).astype("uint16")

                fill_val = ndvi.rio.nodata
                if fill_val is None:
                    fill_val = ndvi.attrs.get("_FillValue", -3000.0)
                ndvi = ndvi.where(ndvi != fill_val)

                # -- QA decoding
                quality      = (qa & 0b11) <= 1
                vi_useful    = ((qa >> 2) & 0b1111) < 13
                no_adj_cloud = ((qa >> 8) & 0b1) == 0
                no_mixed_cloud = ((qa >> 10) & 0b1) == 0
                valid_viq = quality & vi_useful & no_adj_cloud & no_mixed_cloud

                is_land = ((qa >> 11) & 0b111) == 1
                is_deep_water   = ((qa >> 11) & 0b111).isin([0, 2, 5, 6, 7])

                ndvi = ndvi.where(valid_viq & is_land) * 1.0e-4
                water_mask = xr.where(observed & valid_viq & is_deep_water, 1, 0).astype("uint8")

            ndvi = ndvi.expand_dims(time=[ts])
            year_data_ndvi.append(ndvi)

            water_mask = water_mask.expand_dims(time=[ts])
            year_data_water.append(water_mask)

            # release the native 250 m granule + warp buffers before the next granule
            release_memory()

        # stack tiles for each day returned
        stacked_ndvi = xr.concat(year_data_ndvi, dim="time").sortby("time").groupby("time").max("time")
        stacked_wmask = xr.concat(year_data_water, dim="time").sortby("time").groupby("time").max("time")

        # Forward fill to daily, 
        ys, ye = f"{year}-01-01", f"{year}-12-31"
        full_days = pd.date_range(ys, ye, freq="D")
        
        # -- Extend through Dec 31 by holding the last composite (resample alone stops at the final composite date)
        stacked_ndvi = stacked_ndvi.resample(time="1D").ffill().reindex(time=full_days, method="ffill")
        stacked_wmask = stacked_wmask.resample(time="1D").ffill().reindex(time=full_days, method="ffill")
        # -- reindex introduces NaN before the year's first composite and upcasts the mask to float;
        #    those leading days are all pre-season and get trimmed. 0-fill as uint8 (1/4 memory in the cube)
        stacked_wmask = stacked_wmask.fillna(0).astype("uint8")

        assert f_cfg.expand_names is not None, "expected f_cfg.expand_names"

        stacked = xr.Dataset(data_vars={
            f_cfg.expand_names[0]: stacked_ndvi,
            f_cfg.expand_names[1]: stacked_wmask
        })

        for name, da in stacked.data_vars.items():
            print_layer_stats(name, da)

        return stacked


    ### MCD64A1 burns
    def _build_burn_layers(self, f_cfg: Feature) -> xr.Dataset:
        print(f"[LAADS] Staring at the shiny objects")

        events: List[BurnEvents] = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_req) as executor:
            requests = {
                executor.submit(self._fetch_burns, f_cfg, yr): yr
                for yr in range(2000, 2020 + 1)
            }
            for req in as_completed(requests):
                yr = requests[req]
                try:
                    year_events = req.result()
                    if year_events is not None:
                        events.append(year_events)
                    release_memory()
                except Exception as e:
                    print(f"[LAADS] Satellite tried to load data for {yr}, epic fail! --> {e}\n\n")

        return self._aggregate_burns(events, f_cfg)


    def _fetch_burns(self, f_cfg: Feature, year: int) -> BurnEvents | None:
        """ Sparse burn events for one calendar year of granules. Each granule-month 
            reduces to one master-grid burn date by a min over its source pixels.
        """
        nc_files = self._fetch_year(f_cfg.key or "", year)
        if len(nc_files) == 0:
            return None

        by_month: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        for fp in nc_files:
            ts = pd.Timestamp(self._parse_date(str(fp.name)))

            with load_as_xdataset(
                file=fp, variables=["Burn Date", "Burn Date Uncertainty", "QA"]
            ) as raw:
                if len(raw.data_vars.items()) == 0:
                    continue

                arr = self._preclip_native_dataset(raw)
                burn = arr["Burn Date"].fillna(-1)
                unc = arr["Burn Date Uncertainty"].fillna(0)
                qa = arr["QA"].fillna(0).astype("uint8")
                crs = arr["Burn Date"].rio.crs

                # QA bits 0/1 = land with valid data
                # A burn date outside 1-366 is the unmapped (-1) or water (-2) code
                burned = ((qa & 0b11) == 0b11) & (burn >= 1) & (burn <= 366)

                doy = xr.where(burned, burn, BURN_NO_DATE).astype("int16")
                doy = doy.rio.write_crs(crs).rio.write_nodata(BURN_NO_DATE)
                # unburned pixels carry 0
                unc = xr.where(burned, unc.clip(0, 100), 0).astype("int16")
                unc = unc.rio.write_crs(crs).rio.write_nodata(-1)

                date_m = self._reproject_arr_to_mgrid(doy, Resampling.min).values
                unc_m = self._reproject_arr_to_mgrid(unc, Resampling.max).values

            key = (ts.year, ts.month)
            prev = by_month.get(key)
            by_month[key] = (
                (date_m, unc_m) if prev is None
                else (np.minimum(prev[0], date_m), np.maximum(prev[1], unc_m))
            )
            release_memory()

        days, iys, ixs, uncs = [], [], [], []
        for (yy, _), (date_m, unc_m) in sorted(by_month.items()):
            hit = (date_m >= 1) & (date_m <= 366)
            if not hit.any():
                continue
            iy, ix = np.nonzero(hit)
            year_start = (pd.Timestamp(yy, 1, 1) - BURN_RECORD_START).days
            days.append(year_start + date_m[hit].astype("int64") - 1)
            iys.append(iy.astype("int32"))
            ixs.append(ix.astype("int32"))
            uncs.append(np.clip(unc_m[hit], 0, 100).astype("uint8"))

        if not days:
            return None
        print(f"[LAADS] MCD64A1 {year}: {len(by_month)} months, "
              f"{sum(d.size for d in days):,} burned cell-months")
        return (
            np.concatenate(days), np.concatenate(iys),
            np.concatenate(ixs), np.concatenate(uncs),
        )


    def _aggregate_burns(self, events: List[BurnEvents], f_cfg: Feature) -> xr.Dataset:
        """ Turn sparse burn events into the daily burn, uncertainty and age layers.

            The age walk is causal: a day only ever sees burns dated on or before
            itself, and a cell with no burn yet carries its age since the record
            opened. Winter burns fall outside the master index but still reset it.
        """
        assert f_cfg.expand_names is not None, "MCD64A1 expects expand names"
        burn_name, unc_name, age_name = f_cfg.expand_names

        master_days = pd.DatetimeIndex(self.gridref.attrs["time_index"])
        ny, nx = self.gridref.sizes["y"], self.gridref.sizes["x"]
        coords = {
            "time": master_days,
            "y": self.gridref.coords["y"].values,
            "x": self.gridref.coords["x"].values,
        }
        offsets = (master_days - BURN_RECORD_START).days.to_numpy()

        day = np.concatenate([e[0] for e in events])
        iy = np.concatenate([e[1] for e in events])
        ix = np.concatenate([e[2] for e in events])
        unc = np.concatenate([e[3] for e in events])
        order = np.argsort(day, kind="stable")
        day, iy, ix, unc = day[order], iy[order], ix[order], unc[order]

        # the flag and its uncertainty land only on burn dates the master index
        # carries; every date still counts toward the age walk below
        pos = np.searchsorted(offsets, day)
        on_grid = (pos < offsets.size) & (offsets[np.minimum(pos, offsets.size - 1)] == day)
        burn = np.zeros((offsets.size, ny, nx), dtype="uint8")
        burn_unc = np.zeros_like(burn)
        burn[pos[on_grid], iy[on_grid], ix[on_grid]] = 1
        np.maximum.at(burn_unc, (pos[on_grid], iy[on_grid], ix[on_grid]), unc[on_grid])

        flags = self._as_grid_dataset({burn_name: burn, unc_name: burn_unc}, coords)
        for name, da in flags.data_vars.items():
            print_layer_stats(str(name), da)

        streamed = self.sink is not None
        if streamed:
            self.sink(flags)
            del flags, burn, burn_unc
            release_memory()

        # running most-recent burn date per cell; cells still at the initial 0
        # report their age since the record opened, which grows with the day
        age = np.empty((offsets.size, ny, nx), dtype="int16")
        last_burn = np.zeros((ny, nx), dtype="int64")
        cursor = 0
        for t, d in enumerate(offsets):
            reached = int(np.searchsorted(day, d, side="right"))
            if reached > cursor:
                np.maximum.at(last_burn, (iy[cursor:reached], ix[cursor:reached]), day[cursor:reached])
                cursor = reached
            age[t] = np.minimum(d - last_burn, BURN_AGE_CAP)

        ages = self._as_grid_dataset({age_name: age}, coords)
        print_layer_stats(age_name, ages[age_name])

        return ages if streamed else xr.merge([flags, ages])


    def _as_grid_dataset(self, arrays: Dict[str, np.ndarray], coords) -> xr.Dataset:
        ds = xr.Dataset({
            name: xr.DataArray(a, coords=coords, dims=("time", "y", "x"), name=name)
            for name, a in arrays.items()
        })
        ds = ds.rio.write_crs(self.gridref.rio.crs)
        return ds.rio.write_transform(self.gridref.rio.transform())


    def _get_hdf_links(self, granules: List[DataGranule]) -> List[str]:
        # -- unique HDF download links across a granule listing
        links: list[str] = []
        seen = set()
        for g in granules:
            for u in g["umm"].get("RelatedUrls", []):
                t = u.get("Type","")
                url = str(u.get("URL",""))
                if (url not in seen and
                    t == "GET DATA" and
                    url.startswith("https") and url.lower().endswith(".hdf")
                ):
                    seen.add(url)
                    links.append(url)
        return links

    def _granule_pattern(self, short_name, tile):
        return f"{short_name}.*.{tile}.*"

    def _fetch_year(self, short_name: str, year: int) -> List[Path]:
        """ fetch data for a given product and year, return Path to existing/fetched .nc file """

        if not short_name or not year:
            return []

        product_folder = MODIS_DIR / f"{short_name}.{year}"
        ex_files = list(product_folder.glob("*.hdf"))

        if ex_files and all(f.name.startswith(short_name) for f in ex_files):
            # print(f"[LAADS] {short_name} Found my yearbook from {year}, DAMN I look good..")
            return ex_files

        if self.auth is None or not self.auth.username:
            raise AuthenticationError(
                f"[LAADS] {short_name} {year} is not cached locally and no EARTH DATA credentials are available"
            )

        comb_results = []
        for tile in self.tiles:
            wildcard = self._granule_pattern(short_name, tile)
            granules = earthaccess.search_data(
                short_name=short_name,
                version=self.version,
                granule_name=wildcard,
                temporal=(f"{str(year)}-01-01", f"{str(year)}-12-31"),
                day_night_flag="day",
                bounding_box=self.latlon_tup,
                downloadable=True,
                count=-1,
            )
            if len(granules) > 0:
                comb_results.extend(granules)

        if not comb_results:
            print(f"[LAADS] {short_name}: 'We aint found %$#@ for {year}'")
            return list()

        unq_links = self._get_hdf_links(comb_results)
        print(f"[LAADS] downloading using {len(unq_links)} links (from {len(granules)} granules) for {year}")

        dl_file_paths = earthaccess.download(
            granules=unq_links,
            local_path=product_folder,
            show_progress=True
        )
        return dl_file_paths
