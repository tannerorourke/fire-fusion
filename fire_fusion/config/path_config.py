# fire_fusion/config/path_config.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT     = PROJECT_ROOT / "data"
FF_ROOT = PROJECT_ROOT / "fire_fusion"

RAW_DATA_DIR  = DATA_ROOT / "raw"
# Built datasets --> data/processed/<dataset-name>/
PROCESSED_DATA_DIR = DATA_ROOT / "processed"

LANDFIRE_DIR    = RAW_DATA_DIR / "landfire"
NLCD_DIR        = RAW_DATA_DIR / "nlcd"
GPW_DIR         = RAW_DATA_DIR / "nasa_gpw"
CROADS_DIR      = RAW_DATA_DIR / "census"
USFS_DIR        = RAW_DATA_DIR / "usfs"
PRISM_DIR       = RAW_DATA_DIR / "prism"
AORC_DIR        = RAW_DATA_DIR / "aorc"
MODIS_DIR       = RAW_DATA_DIR / "modis"
USDA_DIR        = RAW_DATA_DIR / "usda.gdb"
NCEI_SWDI_DIR   = RAW_DATA_DIR / "ncei_swdi"

MODEL_DIR = FF_ROOT / "model"
MODEL_SAVE_DIR = MODEL_DIR / "saved"
PLOTS_DIR = FF_ROOT / "analysis" / "plots"

# TensorBoard run logs
RUNS_DIR = Path(os.environ.get("FF_RUNS_DIR", str(PROJECT_ROOT / "runs")))

# Extracted per-day prediction fields and scored analysis reports
PRED_DIR = Path(os.environ.get("FF_PRED_DIR", str(PROJECT_ROOT / "predictions")))
REPORTS_DIR = Path(os.environ.get("FF_REPORTS_DIR", str(PROJECT_ROOT / "reports")))



