"""Customer-level behavioural features, computed as of a snapshot date."""

from __future__ import annotations

import numpy as np
import pandas as pd


def customer_features(fact_orders: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per customer_unique_id using only orders placed before `snapshot`.

    Snapshot defaults to the day after the last order, so recency is always >= 1.
    """
    fo = fact_orders
    if snapshot is None:
        snapshot = fo["order_purchase_timestamp"].max().normalize() + pd.Timedelta(days=1)
    fo = fo[fo["order_purchase_timestamp"] < snapshot]
    g = fo.groupby("customer_unique_id")
    feats = g.agg(
        first_order=("order_purchase_timestamp", "min"),
        last_order=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique"),
        monetary=("payment_value", "sum"),
        avg_order_value=("payment_value", "mean"),
        avg_items=("n_items", "mean"),
        avg_freight=("freight_value", "mean"),
        avg_review=("review_score", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
        late_rate=("is_late", "mean"),
        avg_installments=("payment_installments", "mean"),
        n_categories=("n_categories", "sum"),
        customer_state=("customer_state", "last"),
    )
    feats["recency_days"] = (snapshot.normalize() - feats["last_order"].dt.normalize()).dt.days
    feats["tenure_days"] = (snapshot.normalize() - feats["first_order"].dt.normalize()).dt.days
    feats["credit_card_share"] = g["payment_type"].apply(lambda s: (s == "credit_card").mean())
    feats["freight_ratio"] = feats["avg_freight"] / feats["avg_order_value"].replace(0, np.nan)
    feats["avg_review"] = feats["avg_review"].fillna(feats["avg_review"].median())
    feats = feats.fillna({
        "avg_installments": 1,
        "freight_ratio": 0,
        "avg_delivery_days": feats["avg_delivery_days"].median(),
        "late_rate": 0,
    })
    feats.attrs["snapshot"] = snapshot
    return feats.reset_index()
