"""
Dataset configurations, one per directory under data/processed/<name>/

    extract  raw sources  -> cube.zarr
    publish  cube.zarr    -> dataset.zarr
    compile  dataset.zarr -> {train,eval,test}.zarr + manifest.json
    
dataset.zarr is a redistributable artifact containing deterministic functions of the raw sources only.
{train,eval,test}.zarr compiles data-estimated transforms based on fold/split choices.
"""


# 
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Tuple

from .path_config import PROCESSED_DATA_DIR

""" Fire, in all its unpredictability, leaves years well above and well below the median. This results
in an uneven draw rate (sensitivity) for fire prediction and can lead to uneven calibrations. With only 
20 years of data, we train on various fold styles.

- 'full': Default. Full chronological order cut over the whole record.
- 'fold{1,2,3}': Rolling-origin, training on years strictly before its test block.
  Each coincides with a disjoint model run. Calibration years are interleaved, 
  picked as one below-median and one above-median year from that fold's pre-test pool.
"""
FOLDS: Dict[str, Dict[str, Tuple[int, ...]]] = {
    "full": {
        "train": tuple(range(2003, 2017)),
        "eval": (2017, 2018),
        "test": (2019, 2020),
    },
    "fold1": {
        "train": (2003, 2004, 2007, 2008),
        "eval": (2005, 2006),
        "test": (2009, 2010, 2011, 2012),
    },
    "fold2": {
        "train": (2003, 2004, 2005, 2006, 2008, 2010, 2011, 2012),
        "eval": (2007, 2009),
        "test": (2013, 2014, 2015, 2016),
    },
    "fold3": {
        "train": (2003, 2004, 2005, 2006, 2007, 2009, 2010, 2011, 2012, 2013, 2015, 2016),
        "eval": (2008, 2014),
        "test": (2017, 2018, 2019, 2020),
    },
}


def get_fold(name: str) -> Dict[str, Tuple[int, ...]]:
    if name not in FOLDS:
        raise KeyError(f"Unknown fold '{name}'. Options: {sorted(FOLDS)}")
    return FOLDS[name]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    resolution: float                      # meters per pixel
    lat_bounds: Tuple[float, float]
    lon_bounds: Tuple[float, float]
    # 2003 is the earliest fully clean season; MODIS LAI MCD15A2H begins mid-2002
    start_date: str = "2003-01-01"
    end_date: str = "2020-12-31"

    # Which named fold in FOLDS supplies the split years for this config
    fold: str = "full"

    # Inclusive month bounds of the supervised fire season; None keeps every day.
    # Extraction carries a halo either side for the temporal derivations, sized
    # by the longest operator in the derived stage: the lightning-load IIR decays
    # below 0.1% in ~40 days.
    season_months: Optional[Tuple[int, int]] = None
    halo_lead_days: int = 40
    halo_trail_days: int = 10

    # staging chunk: full spatial extent per chunk; spatial-kernel ops stay within a chunk
    stage_time_chunk: int = 16
    x_time_chunk: int = 8
    # >1 lets a patch-based loader read spatial crops without full-frame decompression
    spatial_splits: int = 1

    @property
    def root(self) -> Path:
        return PROCESSED_DATA_DIR / self.name

    @property
    def staging_path(self) -> Path:
        return self.root / "cube.zarr"

    @property
    def published_path(self) -> Path:
        return self.root / "dataset.zarr"

    @property
    def published_manifest_path(self) -> Path:
        return self.root / "dataset_manifest.json"

    @property
    def fold_root(self) -> Path:
        # -- the default fold compiles at the dataset root; every other fold
        #    gets its own subtree
        return self.root if self.fold == "full" else self.root / self.fold

    @property
    def manifest_path(self) -> Path:
        return self.fold_root / "manifest.json"

    def split_path(self, split: str) -> Path:
        return self.fold_root / f"{split}.zarr"

    def split_years(self, split: str) -> Tuple[int, ...]:
        return get_fold(self.fold)[split]

    @property
    def train_yrs(self) -> Tuple[int, ...]:
        return self.split_years("train")


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    cfg.name: cfg
    for cfg in [
        # Washington state (103x109 grid)
        DatasetConfig(
            "wa4000", 4000.0, (45.5, 49.0), (-122.5, -117.0),
            x_time_chunk=16, season_months=(5, 10),
        ),
        # Washington state (205x217 grid)
        DatasetConfig(
            "wa2000", 2000.0, (45.5, 49.0), (-122.5, -117.0),
            x_time_chunk=16, season_months=(5, 10),
        ),
        # Washington state (410x433 grid)
        DatasetConfig(
            "wa1000", 1000.0, (45.5, 49.0), (-122.5, -117.0),
            x_time_chunk=16, spatial_splits=2, season_months=(5, 10),
        ),
        # Eastern Cascades, crest through Okanogan Highlands (548x544 grid, trimmed to 544x544 by the loader). 
        # North clamps to the 49th parallel
        DatasetConfig(
            "cascades500", 500.0, (46.642, 49.0), (-121.85, -118.349),
            x_time_chunk=16, spatial_splits=3, season_months=(5, 10),
        ),
    ]
}


def get_dataset_config(name: str, fold: str = "full") -> DatasetConfig:
    if name not in DATASET_CONFIGS:
        raise KeyError(f"Unknown dataset '{name}'. Options: {sorted(DATASET_CONFIGS)}")
    get_fold(fold)
    cfg = DATASET_CONFIGS[name]
    return cfg if fold == cfg.fold else replace(cfg, fold=fold)
