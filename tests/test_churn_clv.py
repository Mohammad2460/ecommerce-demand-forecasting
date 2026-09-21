import pandas as pd
import pytest

from ecom.churn.model import FEATURES, labelled, score_customers, train_and_evaluate, train_cutoffs
from ecom.clv.model import clv, orders_per_repeater
from ecom.data import warehouse

TEST_CUTOFF = pd.Timestamp("2018-02-20")


@pytest.fixture(scope="module")
def wh(tables):
    return warehouse.build(tables)


def test_train_cutoffs_do_not_overlap_test():
    cuts = train_cutoffs(TEST_CUTOFF, n=3)
    assert cuts == sorted(cuts)
    assert cuts[-1] + pd.Timedelta(days=180) <= TEST_CUTOFF


def test_label_uses_only_future_window(wh):
    fo = wh["fact_orders"]
    lab = labelled(fo, TEST_CUTOFF, wh["fact_order_items"])
    assert (lab["last_order"] < TEST_CUTOFF).all()
    fut = fo[(fo.order_purchase_timestamp >= TEST_CUTOFF)
             & (fo.order_purchase_timestamp < TEST_CUTOFF + pd.Timedelta(days=180))]
    expected = lab["customer_unique_id"].isin(fut["customer_unique_id"]).astype(int)
    assert (lab["repeat"] == expected).all()
    assert not lab[FEATURES].isna().any().any()


def test_train_score_and_clv(wh):
    fo, fi = wh["fact_orders"], wh["fact_order_items"]
    metrics, models, _, _ = train_and_evaluate(fo, TEST_CUTOFF, fi, n_train_cutoffs=2)
    assert set(metrics["model"]) == {"logistic", "lightgbm"}
    assert metrics["roc_auc"].between(0, 1).all()
    scored = score_customers(models["logistic"], fo, fi)
    assert scored["customer_unique_id"].is_unique
    assert scored["p_repeat"].between(0, 1).all()
    out = clv(scored, orders_per_repeater(fo, train_cutoffs(TEST_CUTOFF)[-1]))
    assert (out["clv_12m"] >= 0).all()
    assert set(out["clv_tier"]) == {"low", "mid", "high", "top 5%"}
