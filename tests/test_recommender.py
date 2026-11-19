"""Focused tests for recommender training, ranking, and evaluation."""
import pandas as pd
import pytest

from shopsmart.recommender import (
    HybridRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    RecommenderSuite,
)
from shopsmart.recommender.evaluation import evaluate_at_k, time_based_split


@pytest.fixture
def recommender_data():
    products = pd.DataFrame(
        {
            "Product ID": [10, 20, 30, 40],
            "Product Name": ["A", "B", "C", "D"],
            "Category": ["Home", "Home", "Books", "Books"],
            "Price": [10.0, 20.0, 15.0, 25.0],
        }
    )
    interactions = pd.DataFrame(
        [
            (1, 10, "purchase", "2024-01-01"),
            (1, 30, "view", "2024-01-02"),
            (2, 20, "purchase", "2024-01-03"),
            (2, 20, "purchase", "2024-01-04"),
            (3, 30, "purchase", "2024-01-05"),
            (3, 30, "purchase", "2024-01-06"),
            (3, 30, "purchase", "2024-01-07"),
        ],
        columns=["User ID", "Product ID", "Interaction Type", "Timestamp"],
    )
    interactions["Timestamp"] = pd.to_datetime(interactions["Timestamp"])
    segments = pd.DataFrame({"User ID": [1, 2, 3, 99], "segment": [0, 0, 1, 0]})
    return interactions, products, segments


@pytest.fixture
def recommender_suite(recommender_data):
    interactions, products, segments = recommender_data
    popularity = PopularityRecommender(interactions, products, segments)
    item_knn = ItemKNNRecommender(interactions, popularity)
    return RecommenderSuite(popularity, item_knn, HybridRecommender(popularity, item_knn))


def test_time_split_keeps_latest_events_for_test(recommender_data):
    interactions, _, _ = recommender_data
    train, test, cutoff = time_based_split(interactions, test_fraction=0.3)
    assert train["Timestamp"].max() < cutoff
    assert test["Timestamp"].min() == cutoff


def test_cold_start_uses_best_sellers_for_known_segment(recommender_suite):
    recommendations = recommender_suite.recommend(99, model="popularity", k=2)
    assert recommendations["Product ID"].tolist() == [20, 10]


def test_unknown_cold_start_falls_back_to_global_best_sellers(recommender_suite):
    recommendations = recommender_suite.recommend(100, model="hybrid", k=1)
    assert recommendations.iloc[0]["Product ID"] == 30


def test_item_knn_excludes_products_seen_during_training(recommender_suite):
    recommendations = recommender_suite.recommend(1, model="item_knn", k=10)
    assert 10 not in recommendations["Product ID"].tolist()
    assert 30 not in recommendations["Product ID"].tolist()


def test_evaluation_reports_hit_rate_and_ndcg(recommender_suite):
    test = pd.DataFrame(
        [(1, 10, "purchase"), (1, 20, "purchase")],
        columns=["User ID", "Product ID", "Interaction Type"],
    )
    metrics = evaluate_at_k(recommender_suite, "popularity", test, k=2)
    assert metrics["users_evaluated"] == 1
    assert metrics["HitRate@2"] == 1.0
    assert 0.0 <= metrics["NDCG@2"] <= 1.0


def test_recommender_requires_positive_k(recommender_suite):
    with pytest.raises(ValueError, match="k must be greater"):
        recommender_suite.recommend(99, k=0)


def test_saved_suite_can_be_loaded_and_used(recommender_suite, tmp_path):
    model_file = recommender_suite.save(tmp_path / "recommenders.joblib")
    loaded = RecommenderSuite.load(model_file)
    expected = recommender_suite.recommend(99, model="hybrid", k=2)
    actual = loaded.recommend(99, model="hybrid", k=2)
    pd.testing.assert_frame_equal(actual, expected)
