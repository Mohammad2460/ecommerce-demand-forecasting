import pandas as pd
import pytest

from ecom.data import warehouse
from ecom.features.customer import customer_features
from ecom.segmentation.cluster import cluster
from ecom.segmentation.rfm import SEGMENT_ORDER, frequency_score, rfm, segment, segment_summary


@pytest.fixture(scope="module")
def feats(tables):
    return customer_features(warehouse.build(tables)["fact_orders"])


def test_features_one_row_per_customer(feats):
    assert feats["customer_unique_id"].is_unique
    assert (feats["recency_days"] >= 1).all()
    assert feats.drop(columns=["customer_state"]).isna().sum().sum() == 0


def test_snapshot_excludes_future_orders(tables):
    fo = warehouse.build(tables)["fact_orders"]
    cut = pd.Timestamp("2018-01-01")
    f = customer_features(fo, snapshot=cut)
    assert (f["last_order"] < cut).all()


def test_rfm_rules():
    assert frequency_score(pd.Series([1, 2, 3, 7])).tolist() == [1, 2, 3, 5]
    assert segment(5, 4, 5) == "Champions"
    assert segment(1, 1, 1) == "Lost"
    assert segment(5, 1, 1) == "New Customers"


def test_rfm_covers_all_customers(feats):
    r = rfm(feats)
    assert len(r) == len(feats)
    assert set(r["segment"]) <= set(SEGMENT_ORDER)
    assert r[["R", "M"]].isin(range(1, 6)).all().all()
    assert segment_summary(r)["share_customers"].sum() == pytest.approx(1)


def test_cluster_assigns_everyone(feats):
    assigned, profile, model, _ = cluster(feats, k=4)
    assert len(assigned) == len(feats)
    assert assigned["cluster"].nunique() == 4
    assert profile["cluster_name"].is_unique
