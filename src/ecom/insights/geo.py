"""State-level sales and logistics performance."""

from __future__ import annotations

import pandas as pd


def state_performance(fact_orders: pd.DataFrame) -> pd.DataFrame:
    g = fact_orders.groupby("customer_state")
    out = g.agg(
        orders=("order_id", "count"),
        customers=("customer_unique_id", "nunique"),
        revenue=("payment_value", "sum"),
        avg_order_value=("payment_value", "mean"),
        avg_freight=("freight_value", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
        late_rate=("is_late", "mean"),
        avg_review=("review_score", "mean"),
    )
    out["revenue_share"] = out["revenue"] / out["revenue"].sum()
    out["freight_ratio"] = out["avg_freight"] / out["avg_order_value"]
    return out.sort_values("revenue", ascending=False).reset_index()


def monthly_state_revenue(fact_orders: pd.DataFrame) -> pd.DataFrame:
    df = fact_orders.assign(month=fact_orders["order_date"].dt.to_period("M").dt.to_timestamp())
    return df.groupby(["month", "customer_state"], as_index=False)["payment_value"].sum()
