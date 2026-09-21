import pandas as pd

from ecom.data import warehouse
from ecom.insights.basket import category_affinity
from ecom.insights.geo import state_performance


def test_affinity_math():
    fi = pd.DataFrame({
        "order_id": ["a", "a", "b", "b", "c", "d"],
        "category": ["x", "y", "x", "y", "x", "z"],
    })
    rules = category_affinity(fi, min_pair_count=1).set_index(["antecedent", "consequent"])
    xy = rules.loc[("x", "y")]
    assert xy["pair_count"] == 2
    assert xy["support"] == 2 / 4
    assert xy["confidence"] == 2 / 3
    assert xy["lift"] == (2 / 3) / (2 / 4)


def test_geo_shares_sum_to_one(tables):
    geo = state_performance(warehouse.build(tables)["fact_orders"])
    assert abs(geo["revenue_share"].sum() - 1) < 1e-9
    assert geo["customer_state"].is_unique
