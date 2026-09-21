"""12-month revenue customer lifetime value.

CLV_12m = AOV x expected orders in the next 12 months, where expected orders come from the repeat model:
P(repeat in 180d) x (orders per repeating customer per window, measured on the training window) x (365/180).
Olist has no cost data, so this is revenue CLV; multiply by a margin assumption for profit CLV.
"""

from __future__ import annotations

import pandas as pd

from ecom.config import CHURN_WINDOW_DAYS


def orders_per_repeater(fact_orders: pd.DataFrame, cutoff: pd.Timestamp, window: int = CHURN_WINDOW_DAYS) -> float:
    fut = fact_orders[(fact_orders["order_purchase_timestamp"] >= cutoff)
                      & (fact_orders["order_purchase_timestamp"] < cutoff + pd.Timedelta(days=window))]
    before = set(fact_orders.loc[fact_orders["order_purchase_timestamp"] < cutoff, "customer_unique_id"])
    fut = fut[fut["customer_unique_id"].isin(before)]
    counts = fut.groupby("customer_unique_id")["order_id"].nunique()
    return float(counts.mean()) if len(counts) else 1.0


def clv(scored: pd.DataFrame, orders_per_window: float, horizon_days: int = 365,
        window: int = CHURN_WINDOW_DAYS, margin: float = 1.0) -> pd.DataFrame:
    out = scored.copy()
    out["expected_orders_12m"] = out["p_repeat"] * orders_per_window * horizon_days / window
    out["clv_12m"] = out["avg_order_value"] * out["expected_orders_12m"] * margin
    out["clv_tier"] = pd.qcut(out["clv_12m"].rank(method="first"), [0, 0.5, 0.8, 0.95, 1],
                              labels=["low", "mid", "high", "top 5%"]).astype(str)
    return out
