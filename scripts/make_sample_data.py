"""Generate a small synthetic dataset with the exact schema of the Olist Kaggle dataset.

Used for tests and development before the real CSVs are placed in data/raw/olist/.
Mimics the real data's shape: growth trend, weekly seasonality, Black Friday spike,
mostly one-time buyers, 2016 sparse start and a partial final month.

Usage: uv run python scripts/make_sample_data.py [--orders 20000] [--out data/raw/sample]
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = {
    "cama_mesa_banho": ("bed_bath_table", 0.11, 90),
    "beleza_saude": ("health_beauty", 0.09, 130),
    "esporte_lazer": ("sports_leisure", 0.08, 115),
    "moveis_decoracao": ("furniture_decor", 0.08, 90),
    "informatica_acessorios": ("computers_accessories", 0.07, 115),
    "utilidades_domesticas": ("housewares", 0.07, 90),
    "relogios_presentes": ("watches_gifts", 0.05, 200),
    "telefonia": ("telephony", 0.04, 70),
    "ferramentas_jardim": ("garden_tools", 0.04, 110),
    "automotivo": ("auto", 0.04, 140),
    "brinquedos": ("toys", 0.04, 115),
    "cool_stuff": ("cool_stuff", 0.03, 165),
    "perfumaria": ("perfumery", 0.03, 125),
    "bebes": ("baby", 0.03, 130),
    "eletronicos": ("electronics", 0.025, 60),
    "papelaria": ("stationery", 0.02, 90),
    "fashion_bolsas_e_acessorios": ("fashion_bags_accessories", 0.02, 70),
    "pet_shop": ("pet_shop", 0.02, 110),
    "moveis_escritorio": ("office_furniture", 0.015, 160),
    "consoles_games": ("consoles_games", 0.012, 140),
}

STATES = {
    "SP": (0.42, "sao paulo", -23.55, -46.63),
    "RJ": (0.13, "rio de janeiro", -22.91, -43.17),
    "MG": (0.12, "belo horizonte", -19.92, -43.94),
    "RS": (0.055, "porto alegre", -30.03, -51.23),
    "PR": (0.05, "curitiba", -25.43, -49.27),
    "SC": (0.037, "florianopolis", -27.59, -48.55),
    "BA": (0.034, "salvador", -12.97, -38.50),
    "DF": (0.021, "brasilia", -15.79, -47.88),
    "ES": (0.02, "vitoria", -20.32, -40.34),
    "GO": (0.02, "goiania", -16.68, -49.25),
    "PE": (0.017, "recife", -8.05, -34.88),
    "CE": (0.013, "fortaleza", -3.73, -38.52),
    "PA": (0.01, "belem", -1.46, -48.49),
    "MT": (0.009, "cuiaba", -15.60, -56.10),
}

PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card"]
PAYMENT_P = [0.74, 0.19, 0.055, 0.015]


def _ids(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.array([uuid.UUID(bytes=rng.bytes(16)).hex for _ in range(n)], dtype=object)


def _daily_weights(days: pd.DatetimeIndex) -> np.ndarray:
    t = np.arange(len(days))
    trend = np.clip(0.05 + t / (len(days) * 0.55), 0.05, 1.0)  # ramp-up then plateau
    trend = np.where(days < "2017-01-01", 0.02, trend)  # sparse 2016 like real Olist
    weekly = np.array([1.12, 1.15, 1.08, 1.05, 0.98, 0.78, 0.84])[days.dayofweek]
    yearly = 1 + 0.08 * np.sin(2 * np.pi * (days.dayofyear - 60) / 365.25)
    w = trend * weekly * yearly
    bf = (days >= "2017-11-24") & (days <= "2017-11-27")
    w = np.where(bf, w * np.array([4.5, 2.2, 1.6, 1.5])[np.clip((days - pd.Timestamp("2017-11-24")).days, 0, 3)], w)
    xmas = (days.month == 12) & (days.day >= 10) & (days.day <= 22)
    return np.where(xmas, w * 1.15, w)


def generate(n_orders: int = 20000, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2016-09-15", "2018-09-03", freq="D")

    # --- products & sellers
    cat_keys = list(CATEGORIES)
    cat_p = np.array([CATEGORIES[c][1] for c in cat_keys])
    cat_p /= cat_p.sum()
    n_products = max(200, n_orders // 12)
    product_id = _ids(rng, n_products)
    product_cat = rng.choice(cat_keys, size=n_products, p=cat_p)
    base_price = np.array([CATEGORIES[c][2] for c in product_cat]) * rng.lognormal(0, 0.6, n_products)
    products = pd.DataFrame({
        "product_id": product_id,
        "product_category_name": product_cat,
        "product_name_lenght": rng.integers(10, 70, n_products).astype(float),
        "product_description_lenght": rng.integers(50, 3000, n_products).astype(float),
        "product_photos_qty": rng.integers(1, 8, n_products).astype(float),
        "product_weight_g": rng.lognormal(6.5, 1.1, n_products).round(),
        "product_length_cm": rng.integers(15, 80, n_products).astype(float),
        "product_height_cm": rng.integers(2, 50, n_products).astype(float),
        "product_width_cm": rng.integers(10, 60, n_products).astype(float),
    })
    products.loc[rng.random(n_products) < 0.015, "product_category_name"] = np.nan
    price_of = dict(zip(product_id, base_price.round(2), strict=True))
    prod_pop = rng.pareto(1.3, n_products) + 1
    prod_pop /= prod_pop.sum()

    state_keys = list(STATES)
    state_p = np.array([STATES[s][0] for s in state_keys])
    state_p /= state_p.sum()
    n_sellers = max(50, n_orders // 35)
    seller_state = rng.choice(state_keys, size=n_sellers, p=state_p)
    sellers = pd.DataFrame({
        "seller_id": _ids(rng, n_sellers),
        "seller_zip_code_prefix": rng.integers(1000, 99999, n_sellers),
        "seller_city": [STATES[s][1] for s in seller_state],
        "seller_state": seller_state,
    })

    # --- customers: mostly one-time buyers, a loyal minority repeats
    n_unique = int(n_orders * 0.9)
    unique_ids = _ids(rng, n_unique)
    cust_state = rng.choice(state_keys, size=n_unique, p=state_p)
    cust_zip = rng.integers(1000, 99999, n_unique)
    loyalty = np.where(rng.random(n_unique) < 0.08, rng.integers(2, 6, n_unique), 1)
    buyer_idx = np.repeat(np.arange(n_unique), loyalty)
    rng.shuffle(buyer_idx)
    buyer_idx = buyer_idx[:n_orders]
    n_orders = len(buyer_idx)

    dw = _daily_weights(days)
    purchase_day = rng.choice(days.values, size=n_orders, p=dw / dw.sum())
    purchase_ts = pd.to_datetime(purchase_day) + pd.to_timedelta(rng.integers(0, 86400, n_orders), "s")

    order_id = _ids(rng, n_orders)
    customer_id = _ids(rng, n_orders)  # Olist: new customer_id per order
    customers = pd.DataFrame({
        "customer_id": customer_id,
        "customer_unique_id": unique_ids[buyer_idx],
        "customer_zip_code_prefix": cust_zip[buyer_idx],
        "customer_city": [STATES[s][1] for s in cust_state[buyer_idx]],
        "customer_state": cust_state[buyer_idx],
    })

    status = rng.choice(["delivered", "shipped", "canceled", "unavailable", "invoiced", "processing"],
                        size=n_orders, p=[0.97, 0.011, 0.007, 0.006, 0.003, 0.003])
    approved = purchase_ts + pd.to_timedelta(rng.exponential(10, n_orders) * 3600, "s")
    carrier = approved + pd.to_timedelta(rng.gamma(2, 1.5, n_orders), "D")
    far = np.isin(cust_state[buyer_idx], ["BA", "PE", "CE", "PA", "MT"])
    transit = rng.gamma(3, 2.6, n_orders) + far * 6
    delivered = carrier + pd.to_timedelta(transit, "D")
    estimated = (purchase_ts + pd.to_timedelta(rng.integers(18, 32, n_orders) + far * 8, "D")).normalize()
    not_del = status != "delivered"
    orders = pd.DataFrame({
        "order_id": order_id,
        "customer_id": customer_id,
        "order_status": status,
        "order_purchase_timestamp": purchase_ts,
        "order_approved_at": approved,
        "order_delivered_carrier_date": carrier.where(~np.isin(status, ["canceled", "unavailable", "invoiced", "processing"])),
        "order_delivered_customer_date": delivered.where(~not_del),
        "order_estimated_delivery_date": estimated,
    })

    # --- items
    n_items = rng.choice([1, 2, 3, 4], size=n_orders, p=[0.87, 0.09, 0.03, 0.01])
    item_order = np.repeat(np.arange(n_orders), n_items)
    item_seq = np.concatenate([np.arange(1, k + 1) for k in n_items])
    item_prod = rng.choice(product_id, size=len(item_order), p=prod_pop)
    same = (item_seq > 1) & (rng.random(len(item_order)) < 0.6)  # repeated product in basket
    for i in np.where(same)[0]:
        item_prod[i] = item_prod[i - 1]
    price = np.array([price_of[p] for p in item_prod]) * rng.uniform(0.9, 1.1, len(item_order))
    items = pd.DataFrame({
        "order_id": order_id[item_order],
        "order_item_id": item_seq,
        "product_id": item_prod,
        "seller_id": rng.choice(sellers["seller_id"].values, size=len(item_order)),
        "shipping_limit_date": (purchase_ts[item_order] + pd.Timedelta(days=6)).values,
        "price": price.round(2),
        "freight_value": (8 + price * 0.12 * rng.uniform(0.5, 1.5, len(item_order))).round(2),
    })

    # --- payments
    order_total = items.assign(t=items.price + items.freight_value).groupby("order_id")["t"].sum()
    ptype = rng.choice(PAYMENT_TYPES, size=n_orders, p=PAYMENT_P)
    payments = pd.DataFrame({
        "order_id": order_id,
        "payment_sequential": 1,
        "payment_type": ptype,
        "payment_installments": np.where(ptype == "credit_card", rng.choice([1, 2, 3, 4, 5, 6, 8, 10], n_orders), 1),
        "payment_value": order_total.reindex(order_id).values.round(2),
    })

    # --- reviews: score driven by lateness
    late_days = (delivered - estimated).days.values
    score_mean = np.where(not_del, 1.8, np.where(late_days > 0, 2.3, 4.4))
    score = np.clip(np.rint(rng.normal(score_mean, 1.0)), 1, 5).astype(int)
    review_created = pd.to_datetime(np.where(not_del, estimated, delivered)).normalize() + pd.Timedelta(days=1)
    has_msg = rng.random(n_orders) < 0.4
    msgs = np.where(score >= 4, "Produto chegou no prazo, recomendo", "Produto atrasou, nao recomendo")
    reviews = pd.DataFrame({
        "review_id": _ids(rng, n_orders),
        "order_id": order_id,
        "review_score": score,
        "review_comment_title": np.nan,
        "review_comment_message": np.where(has_msg, msgs, None),
        "review_creation_date": review_created,
        "review_answer_timestamp": review_created + pd.to_timedelta(rng.exponential(2, n_orders), "D"),
    })

    geo = pd.DataFrame([
        {"geolocation_zip_code_prefix": z, "geolocation_lat": STATES[s][2] + rng.normal(0, 0.5),
         "geolocation_lng": STATES[s][3] + rng.normal(0, 0.5), "geolocation_city": STATES[s][1],
         "geolocation_state": s}
        for z, s in zip(np.unique(cust_zip), rng.choice(state_keys, len(np.unique(cust_zip)), p=state_p), strict=True)
    ])
    translation = pd.DataFrame(
        {"product_category_name": cat_keys, "product_category_name_english": [CATEGORIES[c][0] for c in cat_keys]}
    )

    return {
        "olist_customers_dataset.csv": customers,
        "olist_geolocation_dataset.csv": geo,
        "olist_order_items_dataset.csv": items,
        "olist_order_payments_dataset.csv": payments,
        "olist_order_reviews_dataset.csv": reviews,
        "olist_orders_dataset.csv": orders,
        "olist_products_dataset.csv": products,
        "olist_sellers_dataset.csv": sellers,
        "product_category_name_translation.csv": translation,
    }


def write(out: Path, n_orders: int = 20000, seed: int = 42) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, df in generate(n_orders, seed).items():
        df.to_csv(out / name, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/raw/sample"))
    args = ap.parse_args()
    write(args.out, args.orders, args.seed)
    print(f"Wrote sample Olist-schema CSVs to {args.out}")
