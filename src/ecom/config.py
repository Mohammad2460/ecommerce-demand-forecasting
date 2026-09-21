"""Project paths and global parameters."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_OLIST = ROOT / "data" / "raw" / "olist"
RAW_SAMPLE = ROOT / "data" / "raw" / "sample"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"

# Known Olist coverage gaps: 2016 is sparse, data after Aug 2018 is partial.
ANALYSIS_START = "2017-01-01"
ANALYSIS_END = "2018-08-31"

FORECAST_FREQ = "W-MON"  # weekly buckets, labelled by week start (Monday)
FORECAST_HORIZON = 8  # weeks
TOP_N_CATEGORIES = 15
CHURN_WINDOW_DAYS = 180
RANDOM_STATE = 42


def raw_dir() -> Path:
    """Real Olist CSVs if present, else the synthetic sample. Override with ECOM_DATA_DIR."""
    override = os.environ.get("ECOM_DATA_DIR")
    if override:
        return Path(override)
    if (RAW_OLIST / "olist_orders_dataset.csv").exists():
        return RAW_OLIST
    return RAW_SAMPLE


def processed_dir() -> Path:
    override = os.environ.get("ECOM_PROCESSED_DIR")
    path = Path(override) if override else PROCESSED
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    override = os.environ.get("ECOM_MODELS_DIR")
    path = Path(override) if override else MODELS
    path.mkdir(parents=True, exist_ok=True)
    return path
