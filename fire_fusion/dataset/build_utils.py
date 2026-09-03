import ctypes
import gc
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr, rioxarray
from pyhdf.SD import SD, SDC

try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:
    _LIBC = None


def rss_gb() -> float:
    """ Resident set size of this process in GB, for memory tracing. """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 0.0


def release_memory() -> None:
    # -- glibc keeps arenas after free(); RSS climbs across a per-year extraction
    # -- loop with little live. malloc_trim returns them.
    gc.collect()
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def print_layer_stats(name: str, da: xr.DataArray) -> None:
    # -- skipna reductions ignore NaN in place; integer layers carry none.
    # -- da.where(np.isfinite(da)) materializes a float64 copy, past the memory
    # -- guard at daily-MODIS scale.
    try:
        total = int(da.size)
        if np.issubdtype(da.dtype, np.integer):
            finite = total
            f_min, f_max = float(da.min()), float(da.max())
            f_mean, f_std = float(da.mean()), float(da.std())
        else:
            finite = int(np.isfinite(da).sum())
            f_min, f_max = float(da.min(skipna=True)), float(da.max(skipna=True))
            f_mean, f_std = float(da.mean(skipna=True)), float(da.std(skipna=True))
        frac = finite / float(total) if total else 0.0
        print(
            f"  {name:25s} min={f_min:10.4f} max={f_max:10.4f} "
            f"mean={f_mean:10.4f} std={f_std:10.4f} "
            f"finite={finite:,}/{total:,} ({frac:6.2%})"
        )
    except Exception as e:
        print(f"  {name} (stats print failed: {e})")


def C_to_F(c):
    return (c * (9 / 5)) + 32.0


def F_to_K(f):
    return (f + 459.67) * 5 / 9


# -- MODIS ships on a sphere, not an ellipsoid; this is the grid GDAL reports for
# -- HDF-EOS tiles and the source CRS every granule reprojects out of.
MODIS_SINU_CRS = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"


def _hdfeos_tile_bounds(sd: SD) -> Tuple[float, float, float, float]:
    # -- every grid in a MODIS tile shares the tile envelope; the first block
    # -- describes them all. Resolution comes from each field's own shape.
    meta = sd.attributes()["StructMetadata.0"]
    ul = re.search(r"UpperLeftPointMtrs=\(([-\d.,eE+]+)\)", meta)
    lr = re.search(r"LowerRightMtrs=\(([-\d.,eE+]+)\)", meta)
    if ul is None or lr is None:
        raise ValueError("no HDF-EOS grid envelope in StructMetadata.0")
    ulx, uly = (float(v) for v in ul.group(1).split(","))
    lrx, lry = (float(v) for v in lr.group(1).split(","))
    return ulx, uly, lrx, lry


def read_hdf4_dataset(file: Path, variables: List[str]) -> xr.Dataset:
    """ Read HDF-EOS2 fields into a georeferenced Dataset without going through GDAL.

        Fill values become NaN and coordinates are pixel centres, matching what
        the GDAL HDF4 driver returned for the same granules.
    """
    sd = SD(str(file), SDC.READ)
    try:
        ulx, uly, lrx, lry = _hdfeos_tile_bounds(sd)
        v_dict: Dict[str, xr.DataArray] = {}
        for var in variables:
            sds = sd.select(var)
            try:
                raw = sds.get()
                fill = sds.attributes().get("_FillValue")
            finally:
                sds.endaccess()

            arr = raw.astype("float32")
            if fill is not None:
                arr[raw == fill] = np.nan
            ny, nx = arr.shape
            px, py = (lrx - ulx) / nx, (uly - lry) / ny
            v_dict[var] = xr.DataArray(
                arr,
                dims=("y", "x"),
                coords={
                    "y": uly - (np.arange(ny) + 0.5) * py,
                    "x": ulx + (np.arange(nx) + 0.5) * px,
                },
                name=var,
            ).rio.write_nodata(np.nan)
    finally:
        sd.end()

    return xr.Dataset(v_dict).rio.write_crs(MODIS_SINU_CRS)


def load_as_xdataset(
    file: Path,
    variables: List[str] = [],
) -> xr.Dataset:
    suffix = file.suffix.lower()
    try:
        if suffix == ".hdf":
            if not variables:
                raise ValueError("[LOAD_AS_XDATASET] For .hdf need variables list")
            return read_hdf4_dataset(file, variables)

        print("[LOAD_AS_XDATASET] Dont know this file type")
        return xr.Dataset()

    except Exception as e:
        print(f"[LOAD_AS_XDATASET] Failed to load file '{file.stem}':", e)
        return xr.Dataset()



def load_as_xarr(
    file: Path,
    name: str,
    variable = None, # 
    grid = None,
    no_data_val = None,
) -> xr.DataArray:
    """
    load file with a specific variable. To ensure DataArray return type, must provide variable
    """
    suffix = file.suffix.lower()

    # Landfire, GPWv4, NLCD
    if suffix in {".tif", ".tiff"}:
        try:
            darr = rioxarray.open_rasterio(file, masked=True)

            if "band" in darr.dims and darr.sizes.get("band", 1) == 1: # type: ignore
                darr = darr.squeeze("band") # type: ignore
        except Exception as e:
            print(f"[LOAD_AS_XARR] Failed to load tif '{file.stem}': ", e)
            return xr.DataArray()

    # PRISM / AORC annual NetCDF
    elif suffix == ".nc":
        try:
            ds = xr.open_dataset(file, engine="netcdf4")

            if variable is not None and variable in ds.data_vars:
                darr = ds[variable]
            elif len(ds.data_vars) == 1:
                darr = ds[list(ds.data_vars)[0]]
            else:
                raise ValueError(
                    f"[LOAD_AS_XARR] '{file.stem}' needs a variable. Options are: {list(ds.data_vars.keys())}"
                )
        except Exception as e:
            print(f"[LOAD_AS_XARR] Failed to load nc '{file.stem}': ", e)
            return xr.DataArray()

    # LAADS
    elif suffix == ".hdf":
        try:
            if grid is None or variable is None:
                raise ValueError("[LOAD_AS_XARR] expects variable and grid for .hdf files")

            darr = read_hdf4_dataset(file, [variable])[variable]

        except Exception as e:
            print(f"[LOAD_AS_XARR] Failed to load hdf '{file.stem}': ", e)
            return xr.DataArray()

    # None yet?
    elif suffix in {".h5", ".hdf5"}:
        try:
            ds = xr.open_dataset(file, 
                engine="h5netcdf", 
                decode_coords="all", 
                decode_times=True
            )

            if variable is None:
                raise ValueError("[LOAD_AS_XARR] expects variable for .h5/.hdf files")
            if variable == "all":
                return ds[:, :] # return
            if variable not in ds:
                raise KeyError(f"{variable} not in ds. Available: {list(ds.data_vars.keys())}")
            
            darr = ds[variable]
            del ds
        except Exception as e:
            print(f"[LOAD_AS_XARR] Failed to load hdf5 '{file.stem}': ", e)
            return xr.DataArray()
    else:
        raise ValueError(f"[LOAD_AS_XARR] Unsupported file type '{suffix}' for {file.name}")

    if no_data_val is not None:
        darr = darr.where(darr != no_data_val, other=np.nan) # type: ignore

    darr.name = name # type: ignore
    return darr # type: ignore

    
