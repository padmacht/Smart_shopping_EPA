"""Chronological holdout and top-k evaluation for purchase recommendations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from shopsmart.recommender.models import RecommenderSuite


def time_based_split(
    interactions: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split globally by timestamp, placing the newest events in the test set."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    required = {"Timestamp"}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"interactions are missing required columns: {', '.join(sorted(missing))}")
    if len(interactions) < 2:
        raise ValueError("at least two interaction events are required for a time-based split")

    timestamps = pd.to_datetime(interactions["Timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("all interaction timestamps must be valid to create a time-based split")
    split_index = min(len(interactions) - 1, max(1, int((1.0 - test_fraction) * len(interactions))))
    cutoff = timestamps.sort_values(kind="stable").iloc[split_index]
    train = interactions.loc[timestamps < cutoff].copy()
    test = interactions.loc[timestamps >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("time-based split produced an empty train or test set")
    return train, test, pd.Timestamp(cutoff)


def evaluate_at_k(
    suite: RecommenderSuite,
    model: str,
    test_interactions: pd.DataFrame,
    k: int = 10,
) -> dict[str, float | int]:
    """Compute binary HitRate@k and NDCG@k for future purchases not seen in training."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    required = {"User ID", "Product ID", "Interaction Type"}
    missing = required - set(test_interactions.columns)
    if missing:
        raise ValueError(f"test interactions are missing required columns: {', '.join(sorted(missing))}")

    purchases = test_interactions.loc[test_interactions["Interaction Type"] == "purchase"]
    future_by_user = purchases.groupby("User ID")["Product ID"].agg(set)
    relevant_by_user = {
        user_id: products - suite.popularity.seen_items(user_id)
        for user_id, products in future_by_user.items()
    }
    relevant_by_user = {
        user_id: products for user_id, products in relevant_by_user.items() if products
    }
    if not relevant_by_user:
        raise ValueError("no test purchases of previously unseen products are available for evaluation")

    hits: list[float] = []
    ndcgs: list[float] = []
    for user_id, relevant in relevant_by_user.items():
        recommendations = suite.recommend(user_id, model=model, k=k)
        ranked_products = recommendations["Product ID"].tolist()
        positions = {product_id: rank for rank, product_id in enumerate(ranked_products)}
        hits.append(float(any(product_id in positions for product_id in relevant)))
        dcg = sum(
            1.0 / np.log2(positions[product_id] + 2.0)
            for product_id in relevant
            if product_id in positions
        )
        ideal_dcg = sum(1.0 / np.log2(rank + 2.0) for rank in range(min(k, len(relevant))))
        ndcgs.append(dcg / ideal_dcg)

    return {
        f"HitRate@{k}": float(np.mean(hits)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        "users_evaluated": len(relevant_by_user),
        "test_users_with_purchases": len(future_by_user),
        "users_skipped_seen_only": len(future_by_user) - len(relevant_by_user),
    }
