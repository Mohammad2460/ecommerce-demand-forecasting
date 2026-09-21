"""K-Means behavioural clustering with silhouette-based choice of k."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ecom.config import RANDOM_STATE

CLUSTER_FEATURES = [
    "recency_days", "frequency", "monetary", "avg_order_value", "avg_items", "avg_review",
    "avg_delivery_days", "late_rate", "avg_installments", "freight_ratio",
]
SKEWED = ["frequency", "monetary", "avg_order_value", "avg_items", "avg_delivery_days"]


def _log_skewed(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X[SKEWED] = np.log1p(X[SKEWED].clip(lower=0))
    return X


def make_model(k: int) -> Pipeline:
    return make_pipeline(
        FunctionTransformer(_log_skewed),
        StandardScaler(),
        KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE),
    )


def choose_k(X: pd.DataFrame, ks: range = range(3, 9), sample: int = 10000) -> tuple[int, pd.DataFrame]:
    Xs = X.sample(min(sample, len(X)), random_state=RANDOM_STATE)
    rows = []
    for k in ks:
        model = make_model(k).fit(Xs)
        Z = model[:-1].transform(Xs)
        rows.append({"k": k, "inertia": model[-1].inertia_,
                     "silhouette": silhouette_score(Z, model[-1].labels_, random_state=RANDOM_STATE)})
    scores = pd.DataFrame(rows)
    return int(scores.loc[scores["silhouette"].idxmax(), "k"]), scores


def name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """Human-readable names from each cluster's standout traits relative to the overall mean."""
    z = (profile - profile.mean()) / profile.std(ddof=0).replace(0, 1)
    names = {}
    for c, row in z.iterrows():
        if profile.loc[c, "frequency"] >= 1.5:
            names[c] = "Repeat buyers"
        elif row["late_rate"] > 1 or row["avg_review"] < -1:
            names[c] = "Unhappy (late delivery / low review)"
        elif row["avg_items"] > 1:
            names[c] = "Multi-item baskets"
        elif row["monetary"] > 1:
            names[c] = "High spenders"
        elif row["avg_installments"] > 0.5:
            names[c] = "Installment shoppers"
        elif row["monetary"] < -0.5:
            names[c] = "Budget one-timers"
        else:
            names[c] = "Regular one-timers"
    # de-duplicate
    seen: dict[str, int] = {}
    for c in sorted(names):
        n = names[c]
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[c] = f"{n} {seen[n]}"
    return names


def cluster(features: pd.DataFrame, k: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline, pd.DataFrame]:
    X = features[CLUSTER_FEATURES]
    scores = pd.DataFrame()
    if k is None:
        k, scores = choose_k(X)
    model = make_model(k).fit(X)
    labels = pd.Series(model.predict(X), index=features.index, name="cluster")
    profile = X.groupby(labels).mean()
    names = name_clusters(profile)
    profile.insert(0, "customers", labels.value_counts().sort_index())
    profile.insert(0, "cluster_name", profile.index.map(names))
    assigned = features[["customer_unique_id"]].assign(cluster=labels, cluster_name=labels.map(names))
    return assigned, profile.reset_index(), model, scores
