"""Load the nine Olist CSVs with correct dtypes and parsed dates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ecom.config import raw_dir

FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

DATES = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}

ZIP_COLS = ["customer_zip_code_prefix", "seller_zip_code_prefix", "geolocation_zip_code_prefix"]


def load_table(name: str, data_dir: Path | None = None) -> pd.DataFrame:
    path = (data_dir or raw_dir()) / FILES[name]
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={c: str for c in ZIP_COLS})
    for col in DATES.get(name, []):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_all(data_dir: Path | None = None, include_geo: bool = False) -> dict[str, pd.DataFrame]:
    names = [n for n in FILES if include_geo or n != "geolocation"]
    return {n: load_table(n, data_dir) for n in names}
