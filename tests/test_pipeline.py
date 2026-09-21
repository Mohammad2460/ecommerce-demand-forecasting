import json

import pandas as pd

from ecom.config import models_dir, processed_dir
from ecom.pipeline import run

EXPECTED = [
    "fact_orders", "demand_weekly", "forecast", "forecast_leaderboard", "customers", "segment_summary",
    "cluster_profile", "churn_metrics", "state_performance", "affinity_customer", "category_performance",
]


def test_full_pipeline_on_sample():
    run()
    for name in EXPECTED:
        assert len(pd.read_parquet(processed_dir() / f"{name}.parquet")) > 0, name
    for m in ["lgbm_global", "kmeans_customers", "churn_model"]:
        assert (models_dir() / f"{m}.joblib").exists()
    k = json.loads((processed_dir() / "kpis.json").read_text())
    assert k["orders"] > 0 and 0 <= k["repeat_customer_rate"] <= 1
    customers = pd.read_parquet(processed_dir() / "customers.parquet")
    assert customers["customer_unique_id"].is_unique
    assert customers[["segment", "cluster_name", "clv_12m"]].notna().all().all()
