# Intelligent E-Commerce Demand Forecasting & Customer Analytics System

[![CI](https://github.com/Mohammad2460/ecommerce-demand-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohammad2460/ecommerce-demand-forecasting/actions/workflows/ci.yml)

Capstone project on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(~100k orders, 2016–2018). It forecasts weekly category demand, segments customers, scores repeat-purchase
propensity and lifetime value, and serves everything through a REST API and an interactive dashboard.

**Full write-up: [docs/report.md](docs/report.md)**

## Results at a glance

| Area | Result (held-out data) |
|---|---|
| Demand forecast, 16 series × 8 weeks | mean WAPE **22.3%** vs 27.2% naive; store total **13.5%** vs 19.2% |
| Customer segments | 10 RFM segments + 5 K-Means clusters (silhouette-selected k) |
| Repeat-purchase model | ROC-AUC 0.57, top-decile lift **1.7×**, calibrated (1.15% predicted vs 1.17% observed) |
| 12-month revenue CLV | R$401k total; top 5% of customers = 43% of value |
| Key operational finding | late orders average **2.3★** vs 4.3★ on time; RJ late rate 2.7× SP's |

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
make install     # create .venv and install dependencies
make data        # download Olist (public Kaggle endpoint, no login) into data/raw/olist
make pipeline    # build warehouse, train & backtest models, write artefacts (~35 s)
make dashboard   # Streamlit on http://localhost:8501
make api         # FastAPI on http://localhost:8000 (interactive docs at /docs)
```

Without the real data, `make sample` generates a synthetic dataset with the exact Olist schema into
`data/raw/sample/`, and the pipeline uses it automatically.

Other targets: `make test` (35 tests incl. API and dashboard pages, synthetic data; run in CI), `make lint`, `make notebooks` (re-execute notebooks).
Docker: `docker compose run --rm pipeline` then `docker compose up api dashboard` (provided but untested).

## What's inside

```
src/ecom/
  data/          load CSVs, cleaning rules, star-schema warehouse (parquet)
  features/      weekly demand panel + calendar features; point-in-time customer features
  forecasting/   baselines, damped ETS, ARIMA, global LightGBM, ensemble; rolling backtest & intervals
  segmentation/  RFM scoring and segments; K-Means clusters with automatic naming
  churn/         repeat-purchase propensity with out-of-time validation
  clv/           12-month revenue CLV
  insights/      EDA figures, category affinity rules, state performance
  pipeline.py    end-to-end orchestration
  api/           FastAPI service (read-only over artefacts)
  dashboard/     Streamlit app: Overview · Demand forecast · Customer segments · Churn & CLV · Geography & products
notebooks/       01 EDA · 02 forecasting · 03 customers (executed, with outputs)
reports/figures/ static figures used in the report
tests/           unit + integration tests on synthetic data
```

## API

| Method | Endpoint | Returns |
|---|---|---|
| GET | `/health` | artefact readiness |
| GET | `/kpis` | headline business KPIs |
| GET | `/forecast/series` | forecastable series |
| GET | `/forecast/{series}?model=best&horizon=8&history_weeks=26` | history + forecast with 80% interval |
| GET | `/forecast/leaderboard[?series=]` | backtest metrics |
| GET | `/segments`, `/clusters` | segment and cluster summaries |
| GET | `/customers?segment=&risk_band=&clv_tier=&sort_by=&limit=` | filtered customer list |
| GET | `/customers/{customer_unique_id}` | one customer's profile, risk and CLV |
| POST | `/churn/score` | repeat probability for a hypothetical customer |
| GET | `/insights/states`, `/insights/categories`, `/insights/affinity` | operational insights |

```bash
curl "localhost:8000/forecast/bed_bath_table?horizon=4"
curl -X POST localhost:8000/churn/score -H 'content-type: application/json' \
     -d '{"recency_days": 30, "frequency": 3, "monetary": 400}'
```

## Methodology highlights

- **Honest evaluation.** Forecasts use a 3-fold expanding-window backtest. The churn model is trained on
  earlier cutoffs and tested on a later one whose labels never overlap the training period.
- **Data-quality fixes found during analysis.** The analysis window ends 2018-08-19 because the Olist extract
  collapses after that date, and leaving it in created a fake demand crash. Customers are keyed by
  `customer_unique_id`, since `customer_id` is issued per order.
- **Calibrated probabilities.** There is no class re-weighting, so repeat probabilities can feed CLV directly.
- **Limitations** (short history, rare repeat purchases, revenue-only CLV) are discussed in
  [the report](docs/report.md#9-limitations-and-future-work).

## License

Code: [MIT](LICENSE). Data: see below.

## Data

Data: *Brazilian E-Commerce Public Dataset by Olist*, published on Kaggle under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The raw data is not included in this
repository; `make data` downloads it. Only aggregate figures derived from it are committed.
