"""RFM scoring and rule-based customer segments."""

from __future__ import annotations

import numpy as np
import pandas as pd

SEGMENT_ORDER = [
    "Champions", "Loyal", "Potential Loyalists", "New Customers", "Promising",
    "Need Attention", "At Risk", "Can't Lose Them", "Hibernating", "Lost",
]


def _quintile(s: pd.Series, reverse: bool = False) -> pd.Series:
    """Score 1..5 by rank quintile; rank first to break ties so qcut never fails on skewed data."""
    ranks = s.rank(method="first", ascending=not reverse)
    return pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def frequency_score(freq: pd.Series) -> pd.Series:
    """~97% of Olist customers buy once, so quintiles are meaningless; use fixed bins instead."""
    return pd.cut(freq, bins=[0, 1, 2, 3, 4, np.inf], labels=[1, 2, 3, 4, 5]).astype(int)


def segment(r: int, f: int, m: int) -> str:
    if r >= 4 and f >= 3:
        return "Champions"
    if f >= 3:
        return "Loyal" if r >= 3 else "Can't Lose Them"
    if f == 2:
        return "Potential Loyalists" if r >= 3 else "At Risk"
    # one-time buyers: split by recency and spend
    if r == 5:
        return "New Customers"
    if r == 4:
        return "Promising" if m >= 3 else "Need Attention"
    if r == 3:
        return "Need Attention" if m >= 4 else "Hibernating"
    return "Hibernating" if m >= 4 else "Lost"


def rfm(features: pd.DataFrame) -> pd.DataFrame:
    df = features[["customer_unique_id", "recency_days", "frequency", "monetary"]].copy()
    df["R"] = _quintile(df["recency_days"], reverse=True)
    df["F"] = frequency_score(df["frequency"])
    df["M"] = _quintile(df["monetary"])
    df["rfm_score"] = df["R"].astype(str) + df["F"].astype(str) + df["M"].astype(str)
    df["segment"] = [segment(r, f, m) for r, f, m in zip(df["R"], df["F"], df["M"], strict=True)]
    return df


def segment_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    s = rfm_df.groupby("segment").agg(
        customers=("customer_unique_id", "count"),
        avg_recency=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        revenue=("monetary", "sum"),
    )
    s["share_customers"] = s["customers"] / s["customers"].sum()
    s["share_revenue"] = s["revenue"] / s["revenue"].sum()
    return s.reindex([x for x in SEGMENT_ORDER if x in s.index]).reset_index()
