"""Build a star schema from cleaned Olist tables and persist it as parquet.

Tables:
  fact_order_items  one row per order item (grain used for demand forecasting)
  fact_orders       one row per delivered order (grain used for customer analytics)
  dim_customer      one row per customer_unique_id
  dim_product       one row per product
  dim_date          calendar with Brazilian holidays
"""

from __future__ import annotations

import holidays
import pandas as pd

from ecom.config import processed_dir
from ecom.data.clean import clean_orders, clean_payments, clean_products, clean_reviews
from ecom.data.load import load_all

TABLES = ["fact_order_items", "fact_orders", "dim_customer", "dim_product", "dim_date"]


def build_dim_date(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.date_range(start.normalize(), end.normalize(), freq="D")
    br = holidays.Brazil(years=range(start.year, end.year + 1))
    df = pd.DataFrame({"date": dates})
    df["year"] = dates.year
    df["month"] = dates.month
    df["week_start"] = dates.to_period("W-SUN").start_time  # Monday-start weeks
    df["dayofweek"] = dates.dayofweek
    df["is_weekend"] = df["dayofweek"] >= 5
    df["is_holiday"] = [d in br for d in dates.date]
    # Black Friday = 4th Friday of November
    nov = df[(df["month"] == 11) & (df["dayofweek"] == 4)]
    bf = nov.groupby("year")["date"].nth(3)
    df["is_black_friday"] = df["date"].isin(bf)
    return df


def build(tables: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    t = tables or load_all()
    orders = clean_orders(t["orders"])
    products = clean_products(t["products"], t["translation"])
    reviews = clean_reviews(t["reviews"])[["order_id", "review_score"]]
    payments = clean_payments(t["payments"])
    customers = t["customers"].drop_duplicates("customer_id")

    fact_orders = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(reviews, on="order_id", how="left")
        .merge(payments, on="order_id", how="left")
    )
    items = t["items"].merge(products[["product_id", "category"]], on="product_id", how="left")
    items["category"] = items["category"].fillna("unknown")
    item_totals = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        items_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
        n_categories=("category", "nunique"),
    )
    fact_orders = fact_orders.merge(item_totals, on="order_id", how="inner")
    fact_orders["order_date"] = fact_orders["order_purchase_timestamp"].dt.normalize()

    fact_items = items.merge(
        fact_orders[["order_id", "customer_unique_id", "customer_state", "order_purchase_timestamp", "order_date"]],
        on="order_id",
        how="inner",
    )

    dim_customer = (
        fact_orders.sort_values("order_purchase_timestamp")
        .groupby("customer_unique_id")
        .agg(
            customer_state=("customer_state", "last"),
            customer_city=("customer_city", "last"),
            first_order=("order_purchase_timestamp", "min"),
            last_order=("order_purchase_timestamp", "max"),
            n_orders=("order_id", "nunique"),
            total_spent=("payment_value", "sum"),
        )
        .reset_index()
    )
    dim_date = build_dim_date(fact_orders["order_date"].min(), fact_orders["order_date"].max())

    return {
        "fact_order_items": fact_items,
        "fact_orders": fact_orders,
        "dim_customer": dim_customer,
        "dim_product": products,
        "dim_date": dim_date,
    }


def save(wh: dict[str, pd.DataFrame]) -> None:
    out = processed_dir()
    for name, df in wh.items():
        df.to_parquet(out / f"{name}.parquet", index=False)


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(processed_dir() / f"{name}.parquet")


def run() -> dict[str, pd.DataFrame]:
    wh = build()
    save(wh)
    return wh
