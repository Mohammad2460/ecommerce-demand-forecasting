"""Exploratory analysis: summary KPIs and static figures for the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ecom.config import FIGURES  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
COLOR = "#2a6f97"


def kpis(fact_orders: pd.DataFrame, dim_customer: pd.DataFrame) -> dict[str, float]:
    return {
        "orders": int(len(fact_orders)),
        "customers": int(len(dim_customer)),
        "revenue": float(fact_orders["payment_value"].sum()),
        "avg_order_value": float(fact_orders["payment_value"].mean()),
        "repeat_customer_rate": float((dim_customer["n_orders"] > 1).mean()),
        "avg_review_score": float(fact_orders["review_score"].mean()),
        "late_delivery_rate": float(fact_orders["is_late"].mean()),
        "avg_delivery_days": float(fact_orders["delivery_days"].mean()),
    }


def _save(fig: plt.Figure, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def make_figures(wh: dict[str, pd.DataFrame], out: Path = FIGURES) -> list[Path]:
    fo, fi = wh["fact_orders"], wh["fact_order_items"]
    paths = []

    weekly = fo.set_index("order_date").resample("W-MON", label="left", closed="left")["order_id"].count()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(weekly.index, weekly.values, color=COLOR)
    ax.set(title="Weekly delivered orders", ylabel="orders")
    paths.append(_save(fig, out, "01_weekly_orders"))

    dow = fo["order_purchase_timestamp"].dt.day_name().value_counts()
    dow = dow.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    hour = fo["order_purchase_timestamp"].dt.hour.value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].bar(dow.index.str[:3], dow.values, color=COLOR)
    axes[0].set_title("Orders by weekday")
    axes[1].bar(hour.index, hour.values, color=COLOR)
    axes[1].set_title("Orders by hour")
    paths.append(_save(fig, out, "02_seasonality"))

    top = fi.groupby("category")["price"].agg(["count", "sum"]).sort_values("sum").tail(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index, top["sum"] / 1e3, color=COLOR)
    ax.set(title="Top 15 categories by revenue", xlabel="revenue (R$ thousands)")
    paths.append(_save(fig, out, "03_top_categories"))

    st = fo.groupby("customer_state")["payment_value"].sum().sort_values(ascending=False) / 1e3
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(st.index, st.values, color=COLOR)
    ax.set(title="Revenue by customer state", ylabel="R$ thousands")
    paths.append(_save(fig, out, "04_revenue_by_state"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    rs = fo["review_score"].value_counts().sort_index()
    axes[0].bar(rs.index, rs.values, color=COLOR)
    axes[0].set_title("Review score distribution")
    by_late = fo.groupby("is_late")["review_score"].mean()
    axes[1].bar(["on time", "late"], by_late.reindex([False, True]).values, color=[COLOR, "#c44536"])
    axes[1].set(title="Avg review: on-time vs late", ylim=(1, 5))
    paths.append(_save(fig, out, "05_reviews_vs_delivery"))

    return paths
