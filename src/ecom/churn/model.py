"""Repeat-purchase / churn propensity with strict point-in-time (out-of-time) validation.

For a cutoff date T: features use orders before T; label = 1 if the customer orders again in
[T, T + CHURN_WINDOW_DAYS). Churn risk = 1 - P(repeat). Train on an earlier cutoff, test on a later one,
so the test set never leaks information from the training period's future.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ecom.config import CHURN_WINDOW_DAYS, RANDOM_STATE
from ecom.features.customer import customer_features

FEATURES = [
    "recency_days", "tenure_days", "frequency", "monetary", "avg_order_value", "avg_items", "avg_freight",
    "avg_review", "avg_delivery_days", "late_rate", "avg_installments", "n_categories", "credit_card_share",
    "freight_ratio", "is_sp", "is_rj_mg",
]
REPEAT_CATEGORIES = [
    "bed_bath_table", "furniture_decor", "health_beauty", "sports_leisure", "housewares",
    "computers_accessories", "watches_gifts", "telephony", "garden_tools", "auto", "toys", "perfumery",
]
FEATURES = FEATURES + [f"cat_{c}" for c in REPEAT_CATEGORIES]


def category_features(fact_items: pd.DataFrame | None, customers: pd.Series, cutoff: pd.Timestamp) -> pd.DataFrame:
    """One-hot of each customer's most-bought category before the cutoff."""
    cols = [f"cat_{c}" for c in REPEAT_CATEGORIES]
    if fact_items is None:
        return pd.DataFrame(0, index=customers, columns=cols)
    fi = fact_items[fact_items["order_purchase_timestamp"] < cutoff]
    top = fi.groupby(["customer_unique_id", "category"]).size().reset_index(name="n")
    top = top.sort_values("n").drop_duplicates("customer_unique_id", keep="last").set_index("customer_unique_id")
    main = top["category"].reindex(customers)
    return pd.DataFrame({f"cat_{c}": (main == c).astype(int).to_numpy() for c in REPEAT_CATEGORIES}, index=customers)


def labelled(fact_orders: pd.DataFrame, cutoff: pd.Timestamp, fact_items: pd.DataFrame | None = None,
             window: int = CHURN_WINDOW_DAYS) -> pd.DataFrame:
    feats = _with_categories(customer_features(fact_orders, snapshot=cutoff), fact_items, cutoff)
    end = cutoff + pd.Timedelta(days=window)
    future = fact_orders[(fact_orders["order_purchase_timestamp"] >= cutoff)
                         & (fact_orders["order_purchase_timestamp"] < end)]
    feats["repeat"] = feats["customer_unique_id"].isin(future["customer_unique_id"]).astype(int)
    feats["cutoff"] = cutoff
    return _encode(feats)


def _with_categories(feats: pd.DataFrame, fact_items: pd.DataFrame | None, cutoff: pd.Timestamp) -> pd.DataFrame:
    cats = category_features(fact_items, feats["customer_unique_id"], cutoff)
    return pd.concat([feats.reset_index(drop=True), cats.reset_index(drop=True)], axis=1)


def _encode(feats: pd.DataFrame) -> pd.DataFrame:
    feats = feats.copy()
    feats["is_sp"] = (feats["customer_state"] == "SP").astype(int)
    feats["is_rj_mg"] = feats["customer_state"].isin(["RJ", "MG"]).astype(int)
    return feats


SKEWED = ["frequency", "monetary", "avg_order_value", "avg_items", "avg_freight", "n_categories", "avg_installments"]


def _log_skewed(X: pd.DataFrame) -> pd.DataFrame:
    """Heavy right tails (e.g. one R$13k order) otherwise make the linear model extrapolate wildly."""
    X = X.copy()
    X[SKEWED] = np.log1p(X[SKEWED].clip(lower=0))
    return X


def make_models() -> dict[str, object]:
    return {
        "logistic": make_pipeline(
            FunctionTransformer(_log_skewed), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=RANDOM_STATE, verbose=-1,
        ),
    }


def lift_at(y: np.ndarray, p: np.ndarray, frac: float = 0.1) -> float:
    n = max(1, int(len(y) * frac))
    top = np.argsort(-p)[:n]
    return float(y[top].mean() / max(y.mean(), 1e-9))


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "base_rate": float(y.mean()),
        "lift_top10": lift_at(y, p, 0.1),
        "brier": float(brier_score_loss(y, p)),
    }


def train_cutoffs(test_cutoff: pd.Timestamp, n: int = 4, step_days: int = 30,
                  window: int = CHURN_WINDOW_DAYS) -> list[pd.Timestamp]:
    """Latest training cutoff whose label window ends before the test cutoff, plus n-1 earlier ones."""
    last = test_cutoff - pd.Timedelta(days=window)
    return [last - pd.Timedelta(days=step_days * i) for i in range(n)][::-1]


def train_and_evaluate(fact_orders: pd.DataFrame, test_cutoff: pd.Timestamp,
                       fact_items: pd.DataFrame | None = None, n_train_cutoffs: int = 4):
    """Stack several earlier cutoffs for training (more examples), evaluate on the test cutoff."""
    cuts = train_cutoffs(test_cutoff, n_train_cutoffs)
    train = pd.concat([labelled(fact_orders, c, fact_items) for c in cuts], ignore_index=True)
    test = labelled(fact_orders, test_cutoff, fact_items)
    rows, fitted = [], {}
    for name, model in make_models().items():
        model.fit(train[FEATURES], train["repeat"])
        p = model.predict_proba(test[FEATURES])[:, 1]
        rows.append({"model": name, **evaluate(test["repeat"].to_numpy(), p)})
        fitted[name] = model
    metrics = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    return metrics, fitted, train, test


def importance(model, name: str) -> pd.Series:
    if name == "lightgbm":
        vals = model.booster_.feature_importance("gain")
    else:
        vals = np.abs(model[-1].coef_[0])
    return pd.Series(vals, index=FEATURES).sort_values(ascending=False)


def score_customers(model, fact_orders: pd.DataFrame, fact_items: pd.DataFrame | None = None,
                    snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """Score every customer as of the latest snapshot (default: day after last order)."""
    feats = customer_features(fact_orders, snapshot)
    feats = _encode(_with_categories(feats, fact_items, feats.attrs["snapshot"]))
    p = model.predict_proba(feats[FEATURES])[:, 1]
    out = feats[["customer_unique_id", "customer_state", "recency_days", "frequency", "monetary",
                 "avg_order_value"]].copy()
    out["p_repeat"] = p
    out["churn_risk"] = 1 - p
    out["risk_band"] = pd.qcut(out["p_repeat"].rank(method="first"), 5,
                               labels=["very high", "high", "medium", "low", "very low"]).astype(str)
    return out
