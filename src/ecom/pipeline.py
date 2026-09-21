"""End-to-end pipeline: warehouse -> EDA KPIs -> forecasting -> customer analytics -> insights.

Every stage writes parquet to processed_dir() and models to models_dir(); the API and dashboard
only read these artifacts.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

import joblib
import pandas as pd

from ecom.churn import model as churn
from ecom.clv.model import clv, orders_per_repeater
from ecom.config import ANALYSIS_END, CHURN_WINDOW_DAYS, models_dir, processed_dir
from ecom.data import warehouse
from ecom.features.customer import customer_features
from ecom.forecasting import pipeline as forecasting
from ecom.insights.basket import category_affinity
from ecom.insights.eda import kpis
from ecom.insights.geo import monthly_state_revenue, state_performance
from ecom.segmentation.cluster import cluster
from ecom.segmentation.rfm import rfm, segment_summary

log = logging.getLogger("ecom.pipeline")


def _save(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df = df.copy()
        df.attrs = {}  # e.g. customer_features' snapshot Timestamp is not JSON-serializable parquet metadata
        df.to_parquet(processed_dir() / f"{name}.parquet", index=False)


def churn_test_cutoff(fact_orders: pd.DataFrame) -> pd.Timestamp:
    end = min(fact_orders["order_purchase_timestamp"].max().normalize(), pd.Timestamp(ANALYSIS_END))
    return end - pd.Timedelta(days=CHURN_WINDOW_DAYS)


def stage_customers(wh: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    fo, fi = wh["fact_orders"], wh["fact_order_items"]
    feats = customer_features(fo)
    rfm_df = rfm(feats)
    assigned, profile, km, k_scores = cluster(feats)
    joblib.dump(km, models_dir() / "kmeans_customers.joblib")

    test_cutoff = churn_test_cutoff(fo)
    metrics, models, _, _ = churn.train_and_evaluate(fo, test_cutoff, fi)
    best = metrics.loc[0, "model"]
    joblib.dump(models[best], models_dir() / "churn_model.joblib")
    scored = churn.score_customers(models[best], fo, fi)
    clv_df = clv(scored, orders_per_repeater(fo, churn.train_cutoffs(test_cutoff)[-1]))

    customers = (
        feats.merge(rfm_df[["customer_unique_id", "R", "F", "M", "rfm_score", "segment"]], on="customer_unique_id")
        .merge(assigned[["customer_unique_id", "cluster", "cluster_name"]], on="customer_unique_id")
        .merge(clv_df[["customer_unique_id", "p_repeat", "churn_risk", "risk_band", "expected_orders_12m",
                       "clv_12m", "clv_tier"]], on="customer_unique_id")
    )
    importance = churn.importance(models[best], best).rename("importance").rename_axis("feature").reset_index()
    return {
        "customers": customers,
        "segment_summary": segment_summary(rfm_df),
        "cluster_profile": profile,
        "cluster_k_scores": k_scores,
        "churn_metrics": metrics.assign(selected=lambda d: d["model"] == best, test_cutoff=test_cutoff),
        "churn_importance": importance,
    }


def stage_insights(wh: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    fo, fi = wh["fact_orders"], wh["fact_order_items"]
    return {
        "affinity_order": category_affinity(fi, "order_id"),
        "affinity_customer": category_affinity(fi, "customer_unique_id"),
        "state_performance": state_performance(fo),
        "monthly_state_revenue": monthly_state_revenue(fo),
        "category_performance": (
            fi.merge(fo[["order_id", "review_score"]], on="order_id")
            .groupby("category").agg(units=("order_item_id", "count"), revenue=("price", "sum"),
                                     avg_price=("price", "mean"), avg_review=("review_score", "mean"))
            .sort_values("revenue", ascending=False).reset_index()
        ),
    }


def run(stages: list[str] | None = None) -> None:
    stages = stages or ["warehouse", "forecasting", "customers", "insights"]
    timings = {}

    def timed(name: str, fn: Callable[[], object]) -> object:
        t0 = time.perf_counter()
        log.info("stage %s ...", name)
        out = fn()
        timings[name] = round(time.perf_counter() - t0, 1)
        log.info("stage %s done in %.1fs", name, timings[name])
        return out

    if "warehouse" in stages:
        wh = timed("warehouse", warehouse.run)
    else:
        wh = {n: warehouse.load(n) for n in warehouse.TABLES}
    k = kpis(wh["fact_orders"], wh["dim_customer"])
    k["data_start"] = str(wh["fact_orders"]["order_date"].min().date())
    k["data_end"] = str(wh["fact_orders"]["order_date"].max().date())
    (processed_dir() / "kpis.json").write_text(json.dumps(k, indent=2))

    if "forecasting" in stages:
        timed("forecasting", lambda: forecasting.run(wh["fact_order_items"]))
    if "customers" in stages:
        timed("customers", lambda: _save(stage_customers(wh)))
    if "insights" in stages:
        timed("insights", lambda: _save(stage_insights(wh)))
    (processed_dir() / "pipeline_run.json").write_text(
        json.dumps({"finished_at": pd.Timestamp.now().isoformat(timespec="seconds"), "timings_s": timings}, indent=2)
    )
