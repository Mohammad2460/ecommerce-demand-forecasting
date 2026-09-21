"""Cleaning rules applied before building the warehouse."""

from __future__ import annotations

import pandas as pd

from ecom.config import ANALYSIS_END, ANALYSIS_START

# Categories present in products but missing from the official translation file.
EXTRA_TRANSLATIONS = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_food_preparers",
}


def clean_orders(orders: pd.DataFrame, delivered_only: bool = True) -> pd.DataFrame:
    df = orders.drop_duplicates("order_id")
    if delivered_only:
        df = df[df["order_status"] == "delivered"]
    ts = df["order_purchase_timestamp"]
    df = df[(ts >= ANALYSIS_START) & (ts < pd.Timestamp(ANALYSIS_END) + pd.Timedelta(days=1))]
    df = df.assign(
        delivery_days=(df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days,
        delay_days=(df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days,
    )
    df["is_late"] = df["delay_days"] > 0
    return df.reset_index(drop=True)


def clean_products(products: pd.DataFrame, translation: pd.DataFrame) -> pd.DataFrame:
    mapping = dict(zip(translation.iloc[:, 0], translation.iloc[:, 1], strict=True))
    mapping.update(EXTRA_TRANSLATIONS)
    df = products.drop_duplicates("product_id").copy()
    df["category"] = df["product_category_name"].map(mapping).fillna("unknown")
    df = df.rename(columns={"product_name_lenght": "product_name_length",
                            "product_description_lenght": "product_description_length"})
    return df


def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """One review per order: keep the latest answered review."""
    return (
        reviews.sort_values("review_answer_timestamp")
        .drop_duplicates("order_id", keep="last")
        .reset_index(drop=True)
    )


def clean_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-row payments to one row per order."""
    main_type = (
        payments.sort_values("payment_value", ascending=False)
        .drop_duplicates("order_id")
        .set_index("order_id")["payment_type"]
    )
    agg = payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        n_payments=("payment_sequential", "count"),
    )
    agg["payment_type"] = main_type
    return agg.reset_index()
