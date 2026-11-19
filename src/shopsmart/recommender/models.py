"""Implicit-feedback recommenders trained from the ShopSmart event log."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

ModelName = Literal["popularity", "item_knn", "hybrid"]

EVENT_WEIGHTS = {
    "view": 1.0,
    "click": 2.0,
    "add_to_cart": 3.0,
    "wishlist": 3.0,
    "purchase": 5.0,
    "rating": 1.0,
}


def _validate_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(sorted(missing))}")


class PopularityRecommender:
    """Rank by purchase popularity, using segment best-sellers for cold-start users."""

    def __init__(
        self,
        interactions: pd.DataFrame,
        products: pd.DataFrame,
        segments: pd.DataFrame,
    ) -> None:
        _validate_columns(interactions, {"User ID", "Product ID", "Interaction Type"}, "interactions")
        _validate_columns(products, {"Product ID", "Product Name", "Category", "Price"}, "products")
        _validate_columns(segments, {"User ID", "segment"}, "customer segments")
        if products["Product ID"].duplicated().any():
            raise ValueError("products must contain unique Product ID values")
        if segments["User ID"].duplicated().any():
            raise ValueError("customer segments must contain unique User ID values")

        self.products = products[["Product ID", "Product Name", "Category", "Price"]].copy()
        self.product_ids = np.sort(products["Product ID"].dropna().unique())
        self.product_index = {product_id: i for i, product_id in enumerate(self.product_ids)}
        if len(self.product_ids) == 0:
            raise ValueError("products must contain at least one product")

        history = interactions.dropna(subset=["User ID", "Product ID"])
        self.user_history = history.groupby("User ID")["Product ID"].agg(set).to_dict()
        segment_rows = segments.dropna(subset=["User ID", "segment"])
        self.user_segments = dict(zip(segment_rows["User ID"], segment_rows["segment"]))

        purchases = history.loc[history["Interaction Type"] == "purchase"]
        if purchases.empty:
            raise ValueError("training interactions contain no purchases")
        unknown_products = set(purchases["Product ID"].unique()) - set(self.product_index)
        if unknown_products:
            raise ValueError(f"purchase events contain unknown Product IDs: {sorted(unknown_products)[:5]}")

        self.global_scores = self._scores_from_counts(purchases["Product ID"].value_counts())
        self.segment_scores: dict[object, np.ndarray] = {}
        segment_purchases = purchases.merge(
            segment_rows[["User ID", "segment"]],
            on="User ID",
            how="inner",
            validate="many_to_one",
        )
        for segment, rows in segment_purchases.groupby("segment", sort=False):
            self.segment_scores[segment] = self._scores_from_counts(rows["Product ID"].value_counts())

    def _scores_from_counts(self, counts: pd.Series) -> np.ndarray:
        scores = np.zeros(len(self.product_ids), dtype=float)
        for product_id, count in counts.items():
            index = self.product_index.get(product_id)
            if index is not None:
                scores[index] = float(count)
        maximum = scores.max(initial=0.0)
        if maximum:
            scores /= maximum
        return scores

    def has_history(self, user_id: object) -> bool:
        return bool(self.user_history.get(user_id))

    def seen_items(self, user_id: object) -> set[object]:
        return self.user_history.get(user_id, set())

    def scores_for_user(self, user_id: object, segment: object | None = None) -> np.ndarray:
        if not self.has_history(user_id):
            resolved_segment = segment if segment is not None else self.user_segments.get(user_id)
            if resolved_segment in self.segment_scores:
                return self.segment_scores[resolved_segment]
        return self.global_scores

    def _recommend_from_scores(
        self,
        user_id: object,
        scores: np.ndarray,
        k: int,
        segment: object | None = None,
        tie_breaker: np.ndarray | None = None,
    ) -> pd.DataFrame:
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if scores.shape != (len(self.product_ids),):
            raise ValueError("recommendation score vector does not match the product catalog")
        if tie_breaker is None:
            tie_breaker = np.zeros_like(scores)
        if tie_breaker.shape != scores.shape:
            raise ValueError("tie-break score vector does not match the product catalog")

        order = np.lexsort((self.product_ids, -tie_breaker, -scores))
        seen = self.seen_items(user_id)
        if seen:
            available = np.fromiter((product_id not in seen for product_id in self.product_ids), dtype=bool)
            order = order[available[order]]
        selected_ids = self.product_ids[order[:k]]
        result = self.products.set_index("Product ID").loc[selected_ids].reset_index()
        result["score"] = scores[order[:k]]
        return result

    def recommend(self, user_id: object, k: int = 10, segment: object | None = None) -> pd.DataFrame:
        """Return recommendations, excluding products the user has already interacted with."""
        scores = self.scores_for_user(user_id, segment)
        return self._recommend_from_scores(user_id, scores, k, segment)


class ItemKNNRecommender:
    """Item-item cosine similarity over weighted implicit customer interactions."""

    def __init__(
        self,
        interactions: pd.DataFrame,
        popularity: PopularityRecommender,
    ) -> None:
        _validate_columns(interactions, {"User ID", "Product ID", "Interaction Type"}, "interactions")
        self.popularity = popularity
        self.product_ids = popularity.product_ids
        self.product_index = popularity.product_index

        events = interactions.loc[interactions["Interaction Type"].isin(EVENT_WEIGHTS)].dropna(
            subset=["User ID", "Product ID"]
        ).copy()
        unknown_products = set(events["Product ID"].unique()) - set(self.product_index)
        if unknown_products:
            raise ValueError(f"interaction events contain unknown Product IDs: {sorted(unknown_products)[:5]}")
        if events.empty:
            raise ValueError("training interactions contain no supported events")

        events["weight"] = events["Interaction Type"].map(EVENT_WEIGHTS)
        user_ids = np.sort(events["User ID"].unique())
        user_index = {user_id: i for i, user_id in enumerate(user_ids)}
        events["user_index"] = events["User ID"].map(user_index)
        events["product_index"] = events["Product ID"].map(self.product_index)
        weights = events.groupby(["user_index", "product_index"], sort=False)["weight"].sum()
        coordinates = weights.index
        self.user_item = csr_matrix(
            (weights.to_numpy(dtype=float), (coordinates.get_level_values(0), coordinates.get_level_values(1))),
            shape=(len(user_ids), len(self.product_ids)),
        )
        self.user_index = user_index

        cooccurrence = (self.user_item.T @ self.user_item).tocsr()
        norms = np.sqrt(np.asarray(self.user_item.power(2).sum(axis=0)).ravel())
        inverse_norms = np.zeros_like(norms)
        nonzero = norms > 0
        inverse_norms[nonzero] = 1.0 / norms[nonzero]
        normalizer = diags(inverse_norms)
        self.item_similarity = (normalizer @ cooccurrence @ normalizer).tocsr()
        self.item_similarity.setdiag(0)
        self.item_similarity.eliminate_zeros()

    def scores_for_user(self, user_id: object) -> np.ndarray:
        index = self.user_index.get(user_id)
        if index is None:
            return np.zeros(len(self.product_ids), dtype=float)
        scores = self.user_item[index] @ self.item_similarity
        return np.asarray(scores.toarray()).ravel()

    def recommend(self, user_id: object, k: int = 10, segment: object | None = None) -> pd.DataFrame:
        if not self.popularity.has_history(user_id):
            return self.popularity.recommend(user_id, k, segment)
        scores = self.scores_for_user(user_id)
        tie_breaker = self.popularity.scores_for_user(user_id, segment)
        return self.popularity._recommend_from_scores(user_id, scores, k, segment, tie_breaker)


class HybridRecommender:
    """Blend item-KNN affinity with purchase popularity; cold starts use segment best-sellers."""

    def __init__(
        self,
        popularity: PopularityRecommender,
        item_knn: ItemKNNRecommender,
        item_weight: float = 0.7,
    ) -> None:
        if not 0.0 <= item_weight <= 1.0:
            raise ValueError("item_weight must be between 0 and 1")
        self.popularity = popularity
        self.item_knn = item_knn
        self.item_weight = item_weight

    def recommend(self, user_id: object, k: int = 10, segment: object | None = None) -> pd.DataFrame:
        if not self.popularity.has_history(user_id):
            return self.popularity.recommend(user_id, k, segment)

        item_scores = self.item_knn.scores_for_user(user_id)
        popularity_scores = self.popularity.scores_for_user(user_id, segment)
        item_max = item_scores.max(initial=0.0)
        if item_max:
            item_scores = item_scores / item_max
        score_max = popularity_scores.max(initial=0.0)
        if score_max:
            popularity_scores = popularity_scores / score_max
        scores = self.item_weight * item_scores + (1.0 - self.item_weight) * popularity_scores
        return self.popularity._recommend_from_scores(
            user_id,
            scores,
            k,
            segment,
            tie_breaker=popularity_scores,
        )


class RecommenderSuite:
    """Expose the trained popularity, Item-KNN, and hybrid recommenders."""

    def __init__(
        self,
        popularity: PopularityRecommender,
        item_knn: ItemKNNRecommender,
        hybrid: HybridRecommender,
    ) -> None:
        self.popularity = popularity
        self.item_knn = item_knn
        self.hybrid = hybrid

    def recommend(
        self,
        user_id: object,
        model: ModelName = "hybrid",
        k: int = 10,
        segment: object | None = None,
    ) -> pd.DataFrame:
        """Recommend for a user, using a supplied or saved segment for cold starts."""
        models = {
            "popularity": self.popularity,
            "item_knn": self.item_knn,
            "hybrid": self.hybrid,
        }
        try:
            recommender = models[model]
        except KeyError:
            raise ValueError(f"Unknown model '{model}'. Choose from: {', '.join(models)}") from None
        return recommender.recommend(user_id, k, segment)

    def save(self, file_path: str | Path) -> Path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)
        return file_path

    @staticmethod
    def load(file_path: str | Path) -> "RecommenderSuite":
        loaded = joblib.load(file_path)
        if not isinstance(loaded, RecommenderSuite):
            raise TypeError(f"{file_path} does not contain a RecommenderSuite")
        return loaded
