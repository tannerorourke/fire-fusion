""" PRISM AN81d 800m daily weather

Processor (import):
- Reads annual per-var NetCDFs staged under PRISM_DIR onto the master grid.
- Convert Celsius to Fahrenheit (For FFWI/NFDRS)
- Preclips, reprojects and clips to the extent of the master grid.

Fetch (running):
- PRISM serves as one zipped contcontinental COG per-var per day. Pull two 
days per second with backoff, clipped to the Washington extent, and stacked into 
one NetCDF per variable-year.

  python -m fire_fusion.dataset.processors.proc_prism --years 2000 2020
"""

import argparse
import io
import time
import zipfile
from typing import List

import pandas as pd
import requests
import rioxarray  # noqa: F401  -- registers the .rio accessor
import xarray as xr

from .processor import Processor
from ..build_utils import C_to_F, load_as_xarr, release_memory
from fire_fusion.config.feature_config import Feature
from fire_fusion.config.path_config import PRISM_DIR

# PRISM variables delivered in Celsius; everything else keeps its native unit
# (ppt mm, vpd hPa).
TEMP_KEYS = {"tmean", "tmin", "tmax", "tdmean"}


class Prism(Processor):
    def __init__(self, cfg, master_grid):
        super().__init__(cfg, master_grid)

    def build_feature(self, f_cfg: Feature) -> xr.Dataset:
        valid_years = {int(y) for y in self.gridref.attrs["years"]}
        feature_by_yrs: List[xr.DataArray] = []

        for fp in sorted(PRISM_DIR.glob(f"{f_cfg.key}_*.nc")):
            year = fp.stem.split("_")[-1]
            if not year.isdigit() or int(year) not in valid_years:
                continue

            print(f"[PRISM] {f_cfg.name} <- {fp.name}")
            raw = load_as_xarr(fp, name=f_cfg.name, variable=f_cfg.key).astype("float32")
            raw = raw.rio.write_crs("EPSG:4326")

            v = self._preclip_native_arr(raw)
            vals = self._reproject_arr_to_mgrid(v, f_cfg.resampling).astype("float32")
            arr = self._transform(vals, f_cfg)
            feature_by_yrs.append(arr)

            del raw, v, vals
            release_memory()

        feature: xr.Dataset = xr.concat(feature_by_yrs, dim="time").to_dataset(name=f_cfg.name)
        feature_by_yrs.clear()
        release_memory()

        feature = feature.sortby("time")
        feature = self._time_interpolate(feature, f_cfg.time_interp)
        feature = feature.transpose("time", "y", "x", ...)
        return feature

    def _transform(self, val: xr.DataArray, f_cfg: Feature) -> xr.DataArray:
        if f_cfg.key in TEMP_KEYS:
            val = xr.apply_ufunc(C_to_F, val).astype("float32")
        if f_cfg.clip is not None:
            low, high = f_cfg.clip
            val = val.clip(low, high)
        val.name = f_cfg.name
        return val.astype("float32")


PRISM_VARS = ["ppt", "tmean", "tmin", "tmax", "tdmean", "vpdmin", "vpdmax"]
URL = "https://services.nacse.org/prism/data/get/us/800m/{var}/{date}"
HEADERS = {"User-Agent": "firefusion-ingest"}

# Washington extent with a margin.
LON_MIN, LON_MAX = -125.0, -116.5
LAT_MIN, LAT_MAX = 45.5, 49.5

REQ_INTERVAL = 0.55       # seconds between requests (< 2/s)
MAX_RETRIES = 5


class PrismDailyCap(Exception):
    """ NACSE's per-file 2x/day cap notice; never retried, as retries deepen it. """


def _fetch_day(var: str, date: str) -> xr.DataArray:
    # -- NACSE serves over-quota and transient failures as a 200 with a text body,
    #    Validate zip before parsing
    url = URL.format(var=var, date=date)
    err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=180)
        except requests.exceptions.RequestException as e:
            err = e
            time.sleep(2.0 * (attempt + 1))
            continue
        if r.status_code in (429, 503):
            time.sleep(2.0 * (attempt + 1))
            continue
        r.raise_for_status()

        body = r.content
        if body[:2] == b"PK":                               # zip local-file magic
            zf = zipfile.ZipFile(io.BytesIO(body))
            tif = next(n for n in zf.namelist() if n.endswith(".tif"))
            da = rioxarray.open_rasterio(io.BytesIO(zf.read(tif)), masked=True)
            if "band" in da.dims:
                da = da.squeeze("band", drop=True)
            da = da.rio.clip_box(minx=LON_MIN, miny=LAT_MIN, maxx=LON_MAX, maxy=LAT_MAX)
            return da.astype("float32")

        note = body[:300].decode("utf-8", "replace").strip()
        if "more than twice" in note or "IP address" in note:
            raise PrismDailyCap(f"{var} {date}: {note}")
        err = RuntimeError(f"non-zip body: {note}")
        time.sleep(2.0 * (attempt + 1))

    raise RuntimeError(f"[PRISM] {var} {date} failed after {MAX_RETRIES} retries: {err}")


def fetch_year(var: str, year: int) -> None:
    out = PRISM_DIR / f"{var}_{year}.nc"
    if out.exists():
        print(f"[PRISM] {out.name} exists, skipping")
        return

    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    days = []
    for d in dates:
        days.append(_fetch_day(var, d.strftime("%Y%m%d")))
        time.sleep(REQ_INTERVAL)

    stacked = xr.concat(days, dim="time").assign_coords(time=dates)
    stacked.name = var
    stacked = stacked.rio.write_crs("EPSG:4326")

    PRISM_DIR.mkdir(parents=True, exist_ok=True)
    # zlib keeps the smooth met fields ~3-4x smaller on disk and over the wire to B2
    stacked.to_netcdf(out, encoding={var: {"zlib": True, "complevel": 4}})
    print(f"[PRISM] wrote {out.name}  {dict(stacked.sizes)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[2000, 2020], metavar=("START", "END"))
    ap.add_argument("--vars", nargs="+", default=PRISM_VARS, choices=PRISM_VARS)
    args = ap.parse_args()

    try:
        for var in args.vars:
            for year in range(args.years[0], args.years[1] + 1):
                fetch_year(var, year)
    except PrismDailyCap as e:
        print(f"[PRISM] daily cap reached, stopping before an IP block: {e}")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
