#!/usr/bin/env python3
"""
Datacube builder. extract streams processor features into a staging zarr;
publish derives them and applies deterministic normalization, giving a
split-agnostic dataset.zarr; compile redoes train-dependent derivations, fits
statistics on train, fills, and writes the fold's splits. validate reads the
written splits back and reports the invariants a loader and loss assume.

  python -m fire_fusion.dataset.build --dataset wa2000 --stage compile --fold fold3
  python -m fire_fusion.dataset.build --dataset wa2000 --stage extract --sources MODIS PRISM
"""
import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import dask
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from numcodecs import Blosc

from .grid import create_coordinate_grid, season_time_index, supervised_mask
from .build_utils import release_memory
from fire_fusion.config.dataset_config import (
    DATASET_CONFIGS, DatasetConfig, get_dataset_config
)
from fire_fusion.config.feature_config import (
    Feature, base_feat_config, drv_feat_config, get_labels, get_masks,
    cause_index_remap, compiled_cause_classes,
)

from .processors.processor import Processor
from .processors.proc_derived_feats import DerivedProcessor
from .processors.proc_gpw import GPW
from .processors.proc_prism import Prism
from .processors.proc_aorc import Aorc
from .processors.proc_landfire import Landfire
from .processors.proc_lightning import Lightning
from .processors.proc_modis import Modis
from .processors.proc_firms import Firms
from .processors.proc_nlcd import NLCD
from .processors.proc_ignitions import Ignitions
from .processors.proc_croads import CensusRoads
from .processors.proc_usda import UsdaWui

# -- Build Config ------------------------------------------------------------
# Upper bound on dask threads while writing the split stores. 
# Peak memory scales with the worker count, not the store size; wa2000 measures ~10.5 GB at 4.
SPLIT_WRITE_WORKERS = 4

# zstd trades a bit of throughput for ~25% smaller stores than lz4
SPLIT_COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

# Days per read while streaming a split back for validation
VALIDATE_TIME_CHUNK = 64


# -- Utility functions ------------------------------------------------------
def _rss_gb() -> float:
    """ Resident set size of this process in GB, for extraction memory tracing. """
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 0.0

def _years_sel(ds, years):
    """ Select whole calendar years by membership """
    return ds.sel(time=ds["time"].dt.year.isin(list(years)))

PROC_CLASSES = {
    "CENSUSROADS": CensusRoads,
    "USDA_WUI": UsdaWui,
    "IGNITIONS": Ignitions,
    "GPW": GPW,
    "PRISM": Prism,
    "AORC": Aorc,
    "LANDFIRE": Landfire,
    "LIGHTNING": Lightning,
    "MODIS": Modis,
    "FIRMS": Firms,
    "NLCD": NLCD,
}


# -- Entry Point -------------------------------------------------------------
parser = argparse.ArgumentParser(description="Build a named FireFusion dataset")
parser.add_argument("--dataset", help=f"one of {sorted(DATASET_CONFIGS)} or 'all'",
    default="wa2000"
)
parser.add_argument("--stage", help="pipeline stage to run",
    choices=["extract", "publish", "compile", "validate", "all"], default="all"
)
parser.add_argument("--fold", help="named fold in FOLDS supplying the split years; compile and validate read it",
    default="full"
)
parser.add_argument("--sources", help="extract only these processors, rewriting their variables into the existing cube; the default rebuilds it from every source",
    nargs="+", default=None,
)
parser.add_argument("--splits", help="splits validate reads back", 
    nargs="+", default=["train", "eval", "test"],
)


class FeatureGrid:
    """ 
        Builds one named dataset (see config/dataset_config.py):
        raw sources -> cube.zarr -> dataset.zarr -> {train,eval,test}.zarr + manifest.json
    """
    def __init__(self, ds_cfg: DatasetConfig):
        """ Build the season time index and master coordinate grid for ds_cfg. """
        self.cfg = ds_cfg
        self.fconfig = base_feat_config()
        self.drv_config = drv_feat_config()
        self.label_names = [l.name for l in get_labels()]
        self.mask_names = [m.name for m in get_masks()]
        print(f"[FeatureGrid] dataset: {ds_cfg.name} @ {ds_cfg.resolution:.0f}m")
        print("labels: ", self.label_names)
        print("masks: ", self.mask_names)

        self.time_index = season_time_index(
            ds_cfg.start_date, ds_cfg.end_date,
            ds_cfg.season_months, ds_cfg.halo_lead_days, ds_cfg.halo_trail_days,
        )
        if ds_cfg.season_months is not None:
            n_sup = int(supervised_mask(self.time_index, ds_cfg.season_months).sum())
            print(
                f"season: months {ds_cfg.season_months[0]}-{ds_cfg.season_months[1]}, "
                f"{len(self.time_index)} days extracted, {n_sup} supervised"
            )
        self.grid = create_coordinate_grid(
            self.time_index,
            ds_cfg.resolution,
            ds_cfg.lat_bounds, ds_cfg.lon_bounds
        )
        self._staging_initialized = False

    def build(self) -> None:
        self.extract()
        self.publish()
        self.compile()


    def feature_names(self, source: str) -> List[str]:
        """ Variables one processor writes into the cube. """
        return [n for cfg in self.fconfig[source] for n in (cfg.expand_names or [cfg.name])]

    def _drop_staged(self, names: List[str]) -> None:
        """ 
        Delete variables from the staging cube before rewriting .
        """
        root = zarr.open_group(str(self.cfg.staging_path), mode="a")
        dropped = [n for n in names if n in root]
        for n in dropped:
            del root[n]
        zarr.consolidate_metadata(root.store)
        print(f"[FeatureGrid] dropped {dropped} from {self.cfg.staging_path}")

    def extract(self, sources: Optional[List[str]] = None) -> None:
        """ 
        Stream processor features into the staging cube.
        """
        print("Warming up GPU using low-emission wildfire simulations...")
        self.cfg.root.mkdir(parents=True, exist_ok=True)

        if sources is None:
            if self.cfg.staging_path.exists():
                shutil.rmtree(self.cfg.staging_path)
            self._staging_initialized = False
            selected = self.fconfig
        else:
            unknown = [s for s in sources if s not in self.fconfig]
            if unknown:
                raise SystemExit(f"unknown source(s) {unknown}; options {sorted(self.fconfig)}")
            if not self.cfg.staging_path.exists():
                raise SystemExit(f"no cube at {self.cfg.staging_path}; run a full extract first")
            self._drop_staged([n for s in sources for n in self.feature_names(s)])
            self._staging_initialized = True
            selected = {s: self.fconfig[s] for s in sources}

        for pname, features in selected.items():
            processor: Processor = PROC_CLASSES[pname](features, self.grid)
            processor.sink = self._write_layer

            for config in features:
                try:
                    layer = processor.build_feature(config)
                except Exception as e:
                    print(f"Oh no! feature extraction failed for {config.name}: ", e)
                    raise

                if isinstance(layer, xr.DataArray):
                    layer = layer.to_dataset(name=layer.name or config.name)

                # An empty Dataset means the processor already streamed its parts through the sink
                if len(layer.data_vars) > 0:
                    self._write_layer(layer)

                del layer
                release_memory()
                print(f"[mem] after {pname}/{config.name}: RSS={_rss_gb():.2f} GB", flush=True)

            del processor
            release_memory()

        print(f"[FeatureGrid] staging cube written to {self.cfg.staging_path}")

    def _write_layer(self, layer: xr.Dataset) -> None:
        """ Append a layer's variables to the staging cube and release them.
            Every variable must already sit on the master grid/time index.
        """
        if isinstance(layer, xr.DataArray):
            layer = layer.to_dataset(name=layer.name)

        layer = layer.drop_vars("spatial_ref", errors="ignore")

        # Several processors return float64 (xarray's .interp() promotes); X is assembled
        # as float32. Halving here cuts the memory block size with no loss of precision.
        for name, da in layer.items():
            if da.dtype == np.float64:
                layer[name] = da.astype("float32")

        for name, da in layer.items():
            self._check_grid_alignment(str(name), da)
            self._print_layer_stats(str(name), da)

        # zarr stores attrs as JSON; rio transform/CRS objects don't serialize
        layer.attrs.clear()
        for v in layer.variables.values():
            v.attrs.clear()
            v.encoding.clear()

        layer = layer.chunk({
            d: (self.cfg.stage_time_chunk if d == "time" else -1)
            for d in layer.dims
        })
        layer.to_zarr(self.cfg.staging_path, mode=("a" if self._staging_initialized else "w"))
        self._staging_initialized = True

    def _check_grid_alignment(self, name: str, da: xr.DataArray) -> None:
        """ 
        Check that variables added to the staging cube are aligned with the master grid.
        """
        ny, nx = self.grid.sizes["y"], self.grid.sizes["x"]
        if "y" not in da.dims or "x" not in da.dims:
            raise ValueError(f"[FeatureGrid] '{name}' missing spatial dims: {da.dims}")
        if da.sizes["y"] != ny or da.sizes["x"] != nx:
            raise ValueError(
                f"[FeatureGrid] '{name}' shape ({da.sizes['y']}, {da.sizes['x']}) "
                f"does not match grid ({ny}, {nx})"
            )
        if not np.allclose(da["y"].values, self.grid["y"].values) or \
           not np.allclose(da["x"].values, self.grid["x"].values):
            raise ValueError(f"[FeatureGrid] '{name}' y/x coordinates diverge from the master grid")
        if "time" in da.dims and not da.indexes["time"].equals(self.time_index):
            raise ValueError(
                f"[FeatureGrid] '{name}' time axis ({da.sizes['time']} steps) "
                f"does not match the master index ({len(self.time_index)} steps)"
            )

    def _print_layer_stats(self, name: str, da: xr.DataArray) -> None:
        """
        Print layer statistics. Streams over time chunks; a full float64 deviation copy (~8x) OOMs
        """
        try:
            if da.chunks is None and "time" in da.dims:
                da = da.chunk({"time": self.cfg.stage_time_chunk})
            total = da.size
            is_int = np.issubdtype(da.dtype, np.integer)
            if is_int:
                finite = total
                f_min = float(da.min())
                f_max = float(da.max())
                f_mean = float(da.mean())
                f_std = float(da.std())
            else:
                finite = int(np.isfinite(da).sum())
                f_min = float(da.min(skipna=True))
                f_max = float(da.max(skipna=True))
                f_mean = float(da.mean(skipna=True))
                f_std = float(da.std(skipna=True))
            frac_finite = finite / float(total) if total > 0 else 0.0
            print(
                f"  + {name:25s} "
                f"min={f_min:10.4f} max={f_max:10.4f} "
                f"mean={f_mean:10.4f} std={f_std:10.4f} "
                f"finite={finite:,}/{total:,} ({frac_finite:6.2%})"
            )
        except Exception as e:
            print(f"  + {name} (stats print failed: {e})")


    def publish(self) -> None:
        """
        Add/Apply derived features to the staging cube, drop the halo days (consumed by the
        temporal derivations), apply deterministic normalizations, and publish the dataset.
        """
        ds = xr.open_zarr(self.cfg.staging_path)
        ds = self._apply_derived(ds, train_yrs=None)
        
        ds = self._drop_halo(ds)
        ds, det_stats = self._apply_deterministic(ds)
        self._save_published(ds, det_stats)

    def _save_published(self, ds: xr.Dataset, det_stats: Dict) -> None:
        """ Cast labels and masks, chunk and write ds to the published cube, and write its manifest. """
        print("Publishing the split-agnostic cube...")
        excluded = set(self.label_names) | set(self.mask_names)
        ny, nx = ds.sizes["y"], ds.sizes["x"]
        channels = sorted(str(n) for n in ds.data_vars if n not in excluded)
        n_cause_classes = int(ds.sizes["burn_cause"])

        for lname in self.label_names:
            ds[lname] = ds[lname].astype("int8")
        for mname in self.mask_names:
            ds[mname] = ds[mname].astype("uint8")

        ds = ds.chunk({
            d: (self.cfg.stage_time_chunk if d == "time" else -1)
            for d in ds.dims
        })
        ds.attrs.clear()
        for v in ds.variables.values():
            v.attrs.clear()
            v.encoding.clear()

        encoding = {str(n): {"compressor": SPLIT_COMPRESSOR} for n in ds.data_vars}
        write_workers = min(SPLIT_WRITE_WORKERS, os.cpu_count() or SPLIT_WRITE_WORKERS)

        if self.cfg.published_path.exists():
            shutil.rmtree(self.cfg.published_path)
        print(f"[FeatureGrid] writing published cube -> {self.cfg.published_path}")
        with dask.config.set(scheduler="threads", num_workers=write_workers):
            ds.to_zarr(self.cfg.published_path, mode="w", encoding=encoding)

        manifest = {
            "dataset": self.cfg.name,
            "resolution_m": self.cfg.resolution,
            "lat_bounds": list(self.cfg.lat_bounds),
            "lon_bounds": list(self.cfg.lon_bounds),
            "grid": {"height": ny, "width": nx},
            "time": {
                "start": self.cfg.start_date,
                "end": self.cfg.end_date,
                "season_months": (
                    list(self.cfg.season_months) if self.cfg.season_months else None
                ),
                "contiguous": self.cfg.season_months is None,
            },
            "channels": channels,
            "labels": self.label_names,
            "masks": self.mask_names,
            "n_cause_classes": n_cause_classes,
            "norm_stats": det_stats,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        self.cfg.published_manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Saved published cube. channels: {len(channels)}")

    def compile(self) -> None:
        """ 
        Compute train-dependent features -> normalize -> fill missing -> compute class balance.
        """
        ds = xr.open_zarr(self.cfg.published_path)
        n_cause_classes = int(ds.sizes["burn_cause"]) # read before drop_inputs consumes the one-hot cause grid

        ds, redone_stats = self._recompute_train_dependent(ds)
        ds = self._apply_drop_inputs(ds)
        ds, n_cause_classes = self._merge_cause_classes(ds, n_cause_classes)
        
        # Normalize while missing cells are still NaN; stats see valid observations only
        ds, stat_stats = self._apply_statistical(ds)
        ds = self._fill_missing(ds)
        pos_weight = self._compute_pos_weight(ds)
        cause_counts = self._compute_cause_counts(ds, n_cause_classes)

        published = json.loads(self.cfg.published_manifest_path.read_text())
        det_stats = published["norm_stats"]
        excluded = set(self.label_names) | set(self.mask_names)
        norm_stats: Dict[str, List[Dict]] = {}
        for f in ds.data_vars:
            if f in excluded:
                continue
            f = str(f)
            det = redone_stats.get(f, det_stats.get(f, []))
            norm_stats[f] = det + stat_stats.get(f, [])

        self._save_splits(ds, norm_stats, pos_weight, n_cause_classes, cause_counts)

    def _merge_cause_classes(self, ds: xr.Dataset, n_cause: int) -> Tuple[xr.Dataset, int]:
        """ Fold the CAUSE_MERGE pairs into single labels. Applied at compile time;
            the published cube keeps every raw class
        """
        remap = cause_index_remap()
        if all(k == v for k, v in remap.items()):
            return ds, n_cause

        # -- index shifted by one; the -1 'no cause' sentinel maps through the same table
        lut = np.array([-1] + [remap[i] for i in range(n_cause)], dtype="int8")
        ds["burn_next_cause"] = xr.apply_ufunc(
            lambda a: lut[a + 1], ds["burn_next_cause"],
            dask="parallelized", output_dtypes=[np.int8],
        )
        merged = compiled_cause_classes()
        print(f"[FeatureGrid] cause classes {n_cause} -> {len(merged)}: {merged}")
        return ds, len(merged)

    def _drop_halo(self, ds: xr.Dataset) -> xr.Dataset:
        """ 
        Keep only days with a supervised label
        """
        if self.cfg.season_months is None:
            return ds

        keep = supervised_mask(ds.indexes["time"], self.cfg.season_months)
        n_before = ds.sizes["time"]
        ds = ds.isel(time=np.flatnonzero(keep))
        print(
            f"[FeatureGrid] season window: {n_before} -> {ds.sizes['time']} days "
            f"({ds.sizes['time'] / n_before:.1%} kept)"
        )
        return ds

    def _apply_derived(self, ds: xr.Dataset, train_yrs: Optional[Tuple[int, int]]) -> xr.Dataset:
        """ Compute each configured derived feature and merge it into ds.

            train_yrs restricts any train-dependent derivation's statistics to
            those years.
        """
        print(f"[FeatureGrid] Deriving anti-arson techniques through feature derivation..")

        drv_processor = DerivedProcessor(train_yrs=train_yrs)
        for cfg in self.drv_config:
            func   = cfg.func
            inputs = cfg.inputs
            new_fname = cfg.expand_names if cfg.expand_names else cfg.name

            if func:
                drv_fn = getattr(drv_processor, func)

                if func == "build_doy_sin":
                    out = drv_fn(ds, new_fname, self.grid)
                else:
                    subds = ds[inputs]
                    out = drv_fn(subds, new_fname)

                if isinstance(out, xr.DataArray):
                    ds[out.name] = out
                elif isinstance(out, xr.Dataset):
                    ds = ds.merge(out)

        print(f"[FeatureGrid] Finished deriving features!")
        print(f"- dims: {ds.dims}")
        return ds

    
    def _deterministic_chain(self, name: str, feature: xr.DataArray, f_config: Feature) -> Tuple[xr.DataArray, List[Dict]]:
        """ Perform deterministic normalizations.
        
            Shared by the publish pass and by recomputing a single train-dependent
            feature. No feature's ds_norms places a deterministic step after a
            statistical one; checked below
        """
        norms = getattr(f_config, "ds_norms", None) or []
        stat_types = {"z_score", "minmax", "scale_max"}
        det_types = {"log1p", "to_sin", "per_area"}
        det_ix = [i for i, n in enumerate(norms) if n in det_types]
        stat_ix = [i for i, n in enumerate(norms) if n in stat_types]
        if det_ix and stat_ix and max(det_ix) > min(stat_ix):
            raise ValueError(f"deterministic norm ordered after a statistical norm for '{name}'")

        steps: List[Dict] = []
        clip = getattr(f_config, "ds_clip", None)
        if clip is not None:
            feature = feature.clip(clip[0], clip[1])
            steps.append({"step": "clip", "min": float(clip[0]), "max": float(clip[1])})

        for ntype in norms:
            if ntype == "log1p":
                feature = xr.apply_ufunc(np.log1p, feature, dask="allowed")
                steps.append({"step": "log1p"})
            elif ntype == "to_sin":
                feature = xr.apply_ufunc(np.sin, feature, dask="allowed")
                steps.append({"step": "to_sin"})
            elif ntype == "per_area":
                # -- raw layer is mass per cell (scales with resolution)
                #    dividing by cell area in km squared gives a density
                area_km2 = (self.cfg.resolution / 1000.0) ** 2
                feature = feature / area_km2
                steps.append({"step": "per_area", "cell_km2": float(area_km2)})

        return feature, steps

    def _apply_deterministic(self, ds: xr.Dataset) -> Tuple[xr.Dataset, Dict[str, List[Dict]]]:
        all_configs = (
            [c for fl in base_feat_config().values() for c in fl] +
            [c for c in drv_feat_config()]
        )
        det_stats: Dict[str, List[Dict]] = {}

        for f in list(ds.data_vars):
            if f in self.mask_names or f in self.label_names:
                continue

            f_config = next((
                cfg for cfg in all_configs
                if (cfg.name == f or f in (cfg.expand_names or []))
            ), None)
            if f_config is None:
                print(f"can't find feature config for '{f}'")
                continue

            print(f"[FeatureGrid] deterministic norm {f}")
            feature, steps = self._deterministic_chain(f, ds[f], f_config)
            ds[f] = feature
            det_stats[f] = steps

        return ds, det_stats

    def _apply_statistical(self, ds: xr.Dataset) -> Tuple[xr.Dataset, Dict[str, List[Dict]]]:
        """ Apply statistical normalizations. Stats are re-computed after each step in
            the norm chain; stacked transforms compose on the current values
        """
        train_years = self.cfg.split_years("train")

        def _train_stats(da: xr.DataArray):
            src = _years_sel(da, train_years) if "time" in da.dims else da
            ff = src.where(np.isfinite(src))
            mean, std, vmin, vmax = dask.compute(
                ff.mean(skipna=True), ff.std(skipna=True),
                ff.min(skipna=True), ff.max(skipna=True),
            )
            return float(mean), float(std), float(vmin), float(vmax)

        all_configs = (
            [c for fl in base_feat_config().values() for c in fl] +
            [c for c in drv_feat_config()]
        )
        stat_stats: Dict[str, List[Dict]] = {}

        for f in list(ds.data_vars):
            if f in self.mask_names or f in self.label_names:
                continue

            feature = ds[f]
            f_config = next((
                cfg for cfg in all_configs
                if (cfg.name == f or f in (cfg.expand_names or []))
            ), None)
            if f_config is None:
                print(f"can't find feature config for '{f}'")
                continue

            print(f"[FeatureGrid] statistical norm {f}")
            steps: List[Dict] = []
            norms = getattr(f_config, "ds_norms", None) or []
            for ntype in norms:
                if ntype == "z_score":
                    f_mean, f_std, _, _ = _train_stats(feature)
                    f_std = f_std if f_std > 0 else 1.0
                    feature = (feature - f_mean) / f_std
                    steps.append({"step": "z_score", "mean": f_mean, "std": f_std})
                elif ntype == "minmax":
                    _, _, f_min, f_max = _train_stats(feature)
                    denom = abs(f_max - f_min)
                    denom = denom if denom > 0.0 else 1.0
                    feature = (feature - f_min) / denom
                    steps.append({"step": "minmax", "min": f_min, "max": f_max})
                elif ntype == "scale_max":
                    _, _, _, f_max = _train_stats(feature)
                    f_max = f_max if f_max != 0 else 1.0
                    feature = feature / f_max
                    steps.append({"step": "scale_max", "max": f_max})

            ds[f] = feature
            stat_stats[f] = steps

        return ds, stat_stats

    def _recompute_train_dependent(self, ds: xr.Dataset) -> Tuple[xr.Dataset, Dict[str, List[Dict]]]:
        drv_processor = DerivedProcessor(train_yrs=self.cfg.train_yrs)
        redone_stats: Dict[str, List[Dict]] = {}

        for cfg in self.drv_config:
            if not cfg.train_dependent:
                continue
            drv_fn = getattr(drv_processor, cfg.func)
            out = drv_fn(ds[cfg.inputs], cfg.name)
            ds[out.name] = out

            feature, steps = self._deterministic_chain(cfg.name, ds[cfg.name], cfg)
            ds[cfg.name] = feature
            redone_stats[cfg.name] = steps

        return ds, redone_stats

    def _apply_drop_inputs(self, ds: xr.Dataset) -> xr.Dataset:
        for cfg in self.drv_config:
            if cfg.drop_inputs is not None:
                ds = ds.drop_vars(cfg.drop_inputs, errors="ignore")
        # the burn_cause dimension coordinate outlives its dropped variable
        ds = ds.drop_vars("burn_cause", errors="ignore")
        return ds

    def _fill_missing(self, ds: xr.Dataset) -> xr.Dataset:
        excluded = set(self.label_names) | set(self.mask_names)
        for name in list(ds.data_vars):
            if name in excluded:
                continue
            if np.issubdtype(ds[name].dtype, np.floating):
                # -- catches +/-inf as well as NaN
                ds[name] = ds[name].where(np.isfinite(ds[name]), 0.0)
        return ds

    def _compute_pos_weight(self, ds: xr.Dataset) -> float:
        """ Return the negative-to-positive ignition ratio over the supervised train population. """
        train = _years_sel(ds, self.cfg.split_years("train"))
        ign = train["burn_next"]
        no_act_fire_mask = train["no_act_fire_mask"]
        land_mask = train["land_mask"]

        # the population the ignition head is supervised on
        ign_valid = ign.where((land_mask == 1) & (no_act_fire_mask == 1))
        n_ign_pos, n_ign_neg = dask.compute(
            (ign_valid == 1).sum(), (ign_valid == 0).sum()
        )
        n_ign_pos, n_ign_neg = int(n_ign_pos), int(n_ign_neg)

        ign_pos_weight = n_ign_neg / float(max(n_ign_pos, 1))
        print(
            f"[FeatureGrid] Class imbalance (train split):",
            f"- positive ignitions  = {n_ign_pos:,}",
            f"- negative ignitions  = {n_ign_neg:,}",
            f"- pos_weight = {ign_pos_weight:.2f}"
        )
        return ign_pos_weight

    def _compute_cause_counts(self, ds: xr.Dataset, n_cause_classes: int) -> List[int]:
        """ Count of supervised, labelled ignitions per cause class over the train split. """
        train = _years_sel(ds, self.cfg.split_years("train"))
        ign = train["burn_next"]
        cause = train["burn_next_cause"]
        no_act_fire_mask = train["no_act_fire_mask"]
        land_mask = train["land_mask"]

        supervised = (land_mask == 1) & (no_act_fire_mask == 1) & (ign == 1) & (cause != -1)
        counts = dask.compute(*[
            ((cause == c) & supervised).sum() for c in range(n_cause_classes)
        ])
        counts = [int(c) for c in counts]
        print(
            f"[FeatureGrid] Cause classes (train split): {counts}",
            f"- imbalance = {max(counts) / max(min(counts), 1):.1f}x"
        )
        return counts

    def _save_splits(
        self, ds: xr.Dataset, norm_stats: Dict, pos_weight: float,
        n_cause_classes: int, cause_counts: List[int]
    ) -> None:
        """ Stack ds's channels into X, chunk and write the train/eval/test splits,
            and write their manifest. """
        print("Spraying neutrino stabilization goo in sub-basement level 7...")
        excluded = set(self.label_names) | set(self.mask_names)

        feature_names = sorted(
            str(n) for n in ds.data_vars
            if n not in excluded and ds[n].dims == ("time", "y", "x")
        )
        skipped = [
            str(n) for n in ds.data_vars
            if n not in excluded and str(n) not in feature_names
        ]
        if skipped:
            print(f"[Warning] excluded from X (unexpected dims): {skipped}")

        channel_ix = pd.Index(feature_names, name="channel")
        X = xr.concat(
            [ds[n].astype("float32") for n in feature_names], dim=channel_ix
        ).transpose("time", "channel", "y", "x")

        out = xr.Dataset({"X": X})
        for lname in self.label_names:
            out[lname] = ds[lname].astype("int8")
        for mname in self.mask_names:
            out[mname] = ds[mname].astype("uint8")

        ny, nx = ds.sizes["y"], ds.sizes["x"]
        # `spatial_splits` rises with resolution, holding the per-chunk byte count
        # near wa2000's measured ~92 MB. SPLIT_WRITE_WORKERS is then a
        # resolution-independent memory bound.
        x_chunks = {
            "time": self.cfg.x_time_chunk,
            "channel": -1,
            "y": int(np.ceil(ny / self.cfg.spatial_splits)),
            "x": int(np.ceil(nx / self.cfg.spatial_splits)),
        }
        # -- Labels and masks compress to almost nothing (int8/uint8) and are not spatially split
        label_chunks = {"time": 64, "y": -1, "x": -1}
        flat_names = list(self.label_names) + list(self.mask_names)
        split_days: Dict[str, int] = {}

        # -- Peak memory scales with worker count, not store size
        #    Dask's default of one thread per core overruns a 16-core box
        write_workers = min(SPLIT_WRITE_WORKERS, os.cpu_count() or SPLIT_WRITE_WORKERS)

        for split in ("train", "eval", "test"):
            sub = _years_sel(out, self.cfg.split_years(split))
            sub["X"] = sub["X"].chunk(x_chunks)
            for n in flat_names:
                sub[n] = sub[n].chunk(label_chunks)

            # zarr saves attrs as JSON; stale read-encodings clash with new chunks
            sub.attrs.clear()
            for v in sub.variables.values():
                v.attrs.clear()
                v.encoding.clear()

            encoding = {
                str(n): {"compressor": SPLIT_COMPRESSOR} for n in sub.data_vars
            }

            path = self.cfg.split_path(split)
            if path.exists():
                shutil.rmtree(path)
            print(f"[FeatureGrid] writing {split}: {sub.sizes['time']} days -> {path}")
            with dask.config.set(scheduler="threads", num_workers=write_workers):
                sub.to_zarr(path, mode="w", encoding=encoding)
            split_days[split] = int(sub.sizes["time"])

        manifest = {
            "dataset": self.cfg.name,
            "resolution_m": self.cfg.resolution,
            "lat_bounds": list(self.cfg.lat_bounds),
            "lon_bounds": list(self.cfg.lon_bounds),
            "grid": {"height": ny, "width": nx},
            "time": {
                "start": self.cfg.start_date,
                "end": self.cfg.end_date,
                # Days are contiguous within a season block but jump at the
                # year boundary; a loader must not build a window across the gap
                "season_months": (
                    list(self.cfg.season_months) if self.cfg.season_months else None
                ),
                "contiguous": self.cfg.season_months is None,
            },
            "fold": self.cfg.fold,
            "splits": {s: list(self.cfg.split_years(s)) for s in ("train", "eval", "test")},
            "split_days": split_days,
            "channels": feature_names,
            "in_channels": len(feature_names),
            "labels": self.label_names,
            "masks": self.mask_names,
            "n_cause_classes": n_cause_classes,
            "ign_pos_weight": pos_weight,
            "cause_counts": cause_counts,
            "norm_stats": norm_stats,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        self.cfg.manifest_path.write_text(json.dumps(manifest, indent=2))

        print(f"Saved splits to .zarrs <3")
        print("--- MANIFEST ---")
        print(f"- grid: {ny} x {nx}, channels: {len(feature_names)}")
        print(f"- pos_weight: {pos_weight:.2f}")
        for c in feature_names:
            print(f"  channel: {c}")

    def validate(self, splits: Sequence[str] = ("train", "eval", "test")) -> None:
        """ PASS/FAIL table over the written splits; never mutates a store.

            Streams each split in time chunks and asserts what a loader and the
            loss assume: X finite, burn_next binary, burn_next_cause within
            [-1, n_cause-1], valid_cause_mask a superset of the labelled causes,
            and a plausible land fraction.
        """
        m = json.loads(self.cfg.manifest_path.read_text())
        n_cause = int(m["n_cause_classes"])
        print(f"[validate] {self.cfg.name} fold={self.cfg.fold} grid={m['grid']} "
              f"in_channels={m['in_channels']} n_cause={n_cause}")

        all_ok = True
        for split in splits:
            print(f"\n  {split}:")
            for name, (ok, detail) in self._validate_split(split, n_cause).items():
                all_ok &= ok
                print(f"    {'PASS' if ok else 'FAIL'}  {name:<22} {detail}")
        print(f"\n[validate] {self.cfg.name} >> {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")

    def _validate_split(self, split: str, n_cause: int) -> Dict[str, Tuple[bool, str]]:
        """ Stream split in time chunks and return a PASS/FAIL detail per invariant checked. """
        ds = xr.open_zarr(self.cfg.split_path(split))
        T = ds["X"].sizes["time"]

        nonfinite = 0
        ign_vals = set()
        cause_min, cause_max = np.inf, -np.inf
        cause_mask_violations = 0
        land_sum, land_n = 0.0, 0

        for t0 in range(0, T, VALIDATE_TIME_CHUNK):
            sl = slice(t0, min(t0 + VALIDATE_TIME_CHUNK, T))
            X = ds["X"].isel(time=sl).values
            nonfinite += int((~np.isfinite(X)).sum())

            ign = ds["burn_next"].isel(time=sl).values
            ign_vals |= set(np.unique(ign).tolist())

            cause = ds["burn_next_cause"].isel(time=sl).values
            cause_min = min(cause_min, float(cause.min()))
            cause_max = max(cause_max, float(cause.max()))

            vcm = ds["valid_cause_mask"].isel(time=sl).values.astype(bool)
            cause_mask_violations += int(((cause >= 0) & (~vcm)).sum())

            land = ds["land_mask"].isel(time=sl).values
            land_sum += float(land.sum()); land_n += land.size

        land_frac = land_sum / land_n
        return {
            "X_all_finite": (nonfinite == 0, f"{nonfinite} non-finite"),
            "ign_binary": (ign_vals <= {0, 1}, f"values={sorted(ign_vals)}"),
            "cause_range": (cause_min >= -1 and cause_max <= n_cause - 1,
                            f"[{cause_min:.0f},{cause_max:.0f}] vs [-1,{n_cause-1}]"),
            "cause_mask_superset": (cause_mask_violations == 0,
                                    f"{cause_mask_violations} labelled-but-unmasked"),
            "land_frac": (0.80 <= land_frac <= 0.99, f"{land_frac:.3f}"),
        }


def main():
    args = parser.parse_args()

    names = sorted(DATASET_CONFIGS) if args.dataset == "all" else [args.dataset]
    for name in names:
        grid = FeatureGrid(get_dataset_config(name, args.fold))
        if args.stage == "all":
            grid.build()
        elif args.stage == "extract":
            grid.extract(args.sources)
        elif args.stage == "validate":
            grid.validate(args.splits)
        else:
            getattr(grid, args.stage)()


if __name__ == "__main__":
    main()
