# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Capstone: Intelligent E-Commerce Demand Forecasting & Customer Analytics System, built on the
Olist Brazilian E-Commerce dataset (Kaggle `olistbr/brazilian-ecommerce`). Python package `ecom`
under `src/ecom/`, managed with `uv`. Serving layer: FastAPI (`ecom.api`) + Streamlit (`ecom.dashboard`).

## Commands

```bash
uv sync                                   # install deps (Python venv in .venv)
make test                                 # uv run pytest (all tests, synthetic data)
uv run pytest tests/test_data.py::test_clean_orders_filters_status_and_window   # single test
make lint                                 # ruff check src tests scripts
make sample                               # regenerate synthetic Olist-schema CSVs in data/raw/sample
make pipeline                             # end-to-end build: warehouse -> features -> models -> outputs
make api                                  # FastAPI on :8000
make dashboard                            # Streamlit app
```

Re-download real data (public endpoint, no Kaggle login needed):
`curl -L -o olist.zip https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce && unzip -o olist.zip -d data/raw/olist`

### macOS import gotcha
The editable-install `.pth` in `.venv` can get the macOS `hidden` flag, which makes Python silently
skip it (`ModuleNotFoundError: No module named 'ecom'`). Mitigations already in place: Makefile
exports `PYTHONPATH=src`, pytest sets `pythonpath = ["src"]`. If it recurs outside those:
`chflags nohidden .venv/lib/python3.*/site-packages/*.pth`.

## Architecture

**Data source switching** (`ecom/config.py`): `raw_dir()` uses `data/raw/olist/` when real CSVs exist,
else `data/raw/sample/`. Env vars `ECOM_DATA_DIR`, `ECOM_PROCESSED_DIR`, `ECOM_MODELS_DIR` override
all paths. Tests rely on this: `tests/conftest.py` generates a small synthetic dataset via
`scripts/make_sample_data.py` into a tmp dir and points all three env vars there — tests never touch
real data or real `models/`. Any new module must resolve paths through these config functions,
not hardcoded paths, or tests will leak into real outputs.

**Layers** (each consumes the previous layer's parquet output in `data/processed/`):
1. `data/load.py` → raw CSVs with parsed dates (translation CSV has a UTF-8 BOM; zips read as str).
2. `data/clean.py` → delivered orders only within `ANALYSIS_START..ANALYSIS_END` (2017-01-01 → 2018-08-19;
   2016 is sparse and order volume collapses from ~2018-08-20 in the Olist extract). Reviews and payments collapsed to one row per order.
3. `data/warehouse.py` → star schema parquet (fact order items + customer/product/date dims).
4. `features/`, `forecasting/`, `segmentation/`, `churn/`, `clv/`, `insights/` → models to `models/`
   (joblib) and result tables to `data/processed/`. `ecom/pipeline.py` orchestrates all stages
   (`scripts/run_pipeline.py [warehouse forecasting customers insights]` runs a subset). The unified
   `customers.parquet` joins features + RFM + cluster + churn risk + CLV.
5. `api/` and `dashboard/` only read precomputed parquet/models via `api/store.py` (lru-cached; call
   `store.clear()` or POST `/admin/reload` after re-running the pipeline) — never train at request time.
   The dashboard is an `st.navigation` router (`dashboard/app.py`) over `dashboard/pages/*.py`; shared
   theme/palette lives in `dashboard/common.py`, and charts pass `theme=None` so Streamlit doesn't override it.
   The dashboard's what-if scorer calls the API's `churn_score` function directly (single source of logic).

**Forecast outputs:** `forecast.parquet` holds every model's forecast for every series; `is_best` flags the
per-series backtest winner (by WAPE). Intervals come from backtest relative-error quantiles, clamped so the
band always contains the point forecast.

**Churn model gotchas:** keep probabilities calibrated (no class weights / scale_pos_weight) because CLV
multiplies them directly; the logistic pipeline log-transforms skewed spend/count features, otherwise single
large orders dominate. Training cutoffs must end their 180-day label window before the test cutoff
(`train_cutoffs`).

**Olist data semantics that matter:**
- `customer_id` is per-order; the real person is `customer_unique_id`. All customer analytics
  (RFM, churn, CLV) must group by `customer_unique_id`.
- ~97% of customers buy once, so churn is framed as repeat-purchase propensity within
  `CHURN_WINDOW_DAYS`; expect heavy class imbalance.
- Forecast grain is weekly (`FORECAST_FREQ = "W-MON"`) per top-N category, horizon 8 weeks.
- Source column typos `product_name_lenght`/`product_description_lenght` are renamed to `_length` in cleaning.
