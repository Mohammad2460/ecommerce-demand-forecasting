"""Category affinity (market basket) at order and customer level: support, confidence, lift."""

from __future__ import annotations

from itertools import permutations

import pandas as pd


def category_affinity(fact_items: pd.DataFrame, level: str = "order_id", min_pair_count: int = 5) -> pd.DataFrame:
    """Rules A -> B between categories bought in the same basket (`order_id`) or by the same
    customer across orders (`customer_unique_id`). Olist baskets are mostly single-category,
    so the customer level surfaces far more pairs."""
    baskets = (
        fact_items[fact_items["category"] != "unknown"]
        .groupby(level)["category"].agg(lambda s: sorted(set(s)))
    )
    n = len(baskets)
    item_count = baskets.explode().value_counts()
    multi = baskets[baskets.str.len() > 1]
    pairs = pd.Series([p for cats in multi for p in permutations(cats, 2)], dtype=object)
    if pairs.empty:
        return pd.DataFrame(columns=["antecedent", "consequent", "pair_count", "support", "confidence", "lift"])
    counts = pairs.value_counts().rename("pair_count").reset_index()
    counts[["antecedent", "consequent"]] = pd.DataFrame(counts["index"].tolist(), index=counts.index)
    counts = counts.drop(columns="index")
    counts = counts[counts["pair_count"] >= min_pair_count]
    counts["support"] = counts["pair_count"] / n
    counts["confidence"] = counts["pair_count"] / counts["antecedent"].map(item_count)
    counts["lift"] = counts["confidence"] / (counts["consequent"].map(item_count) / n)
    cols = ["antecedent", "consequent", "pair_count", "support", "confidence", "lift"]
    return counts[cols].sort_values(["lift", "pair_count"], ascending=False).reset_index(drop=True)
