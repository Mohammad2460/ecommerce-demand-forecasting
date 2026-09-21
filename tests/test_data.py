import pandas as pd

from ecom.data.clean import clean_orders, clean_payments, clean_products, clean_reviews
from ecom.data.load import FILES


def test_all_tables_load(tables):
    assert set(tables) == set(FILES) - {"geolocation"}
    assert pd.api.types.is_datetime64_any_dtype(tables["orders"]["order_purchase_timestamp"])


def test_clean_orders_filters_status_and_window(tables):
    orders = clean_orders(tables["orders"])
    assert (orders["order_status"] == "delivered").all()
    assert orders["order_purchase_timestamp"].min() >= pd.Timestamp("2017-01-01")
    assert orders["order_id"].is_unique
    assert {"delivery_days", "delay_days", "is_late"} <= set(orders.columns)


def test_clean_products_translates(tables):
    products = clean_products(tables["products"], tables["translation"])
    assert products["category"].notna().all()
    assert "bed_bath_table" in set(products["category"])


def test_reviews_and_payments_one_row_per_order(tables):
    assert clean_reviews(tables["reviews"])["order_id"].is_unique
    pay = clean_payments(tables["payments"])
    assert pay["order_id"].is_unique
    assert (pay["payment_value"] >= 0).all()
