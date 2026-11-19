"""Tests for the two-tower deep learning recommender."""
import numpy as np
import pandas as pd
import pytest

from shopsmart.recommender import PopularityRecommender
from shopsmart.recommender.two_tower import TwoTowerConfig, TwoTowerRecommender, TwoTowerSuite


@pytest.fixture
def data():
    products = pd.DataFrame({
        "Product ID": [10, 20, 30, 40],
        "Product Name": ["A", "B", "C", "D"],
        "Category": ["Home", "Home", "Books", "Books"],
        "Product Type": ["Lamp", "Rug", "Novel", "Atlas"],
        "Brand": ["X", "Y", "X", "Z"],
        "Price": [10.0, 20.0, 15.0, 25.0],
    })
    interactions = pd.DataFrame(
        [(1, 10, "purchase"), (1, 30, "view"), (2, 20, "purchase"), (2, 40, "click"), (3, 30, "purchase")],
        columns=["User ID", "Product ID", "Interaction Type"],
    )
    segments = pd.DataFrame({"User ID": [1, 2, 3, 99], "segment": [0, 0, 1, 0]})
    return interactions, products, segments, PopularityRecommender(interactions, products, segments)


def hand_made_model(popularity):
    # customer 1 points towards product 40, customer 2 towards product 30
    user_vectors = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=np.float32)
    item_vectors = np.eye(4, dtype=np.float32)            # product i -> axis i (catalogue order 10, 20, 30, 40)
    return TwoTowerRecommender(np.array([1, 2, 3]), user_vectors, item_vectors, popularity)


def test_ranks_by_embedding_match_and_skips_seen_products(data):
    *_, popularity = data
    model = hand_made_model(popularity)
    assert model.recommend(1, k=1)["Product ID"].tolist() == [40]
    ranked = model.recommend(2, k=2)["Product ID"].tolist()
    assert ranked[0] == 30 and 20 not in ranked and 40 not in ranked   # 20 and 40 already seen by customer 2


def test_cold_start_customer_gets_segment_best_sellers(data):
    *_, popularity = data
    model = hand_made_model(popularity)
    assert model.recommend(99, k=2)["Product ID"].tolist() == popularity.recommend(99, k=2)["Product ID"].tolist()


def test_save_and_load_give_the_same_recommendations(data, tmp_path):
    *_, popularity = data
    model = hand_made_model(popularity)
    loaded = TwoTowerRecommender.load(model.save(tmp_path / "two_tower.npz"), popularity)
    for user in (1, 2, 3, 99):
        assert loaded.recommend(user, k=3)["Product ID"].tolist() == model.recommend(user, k=3)["Product ID"].tolist()


def test_load_rejects_a_different_catalogue(data, tmp_path):
    interactions, products, segments, popularity = data
    file = hand_made_model(popularity).save(tmp_path / "two_tower.npz")
    bigger = pd.concat([products, products.head(1).assign(**{"Product ID": 50})], ignore_index=True)
    with pytest.raises(ValueError):
        TwoTowerRecommender.load(file, PopularityRecommender(interactions, bigger, segments))


def test_suite_adapter_works_with_the_evaluation(data):
    *_, popularity = data
    suite = TwoTowerSuite(hand_made_model(popularity))
    assert suite.popularity is popularity
    assert len(suite.recommend(1, model="two_tower", k=2)) == 2


def test_training_produces_normalised_embeddings(data):
    pytest.importorskip("torch")
    from shopsmart.recommender.two_tower import train_two_tower

    interactions, products, segments, popularity = data
    config = TwoTowerConfig(epochs=1, samples_per_epoch=64, batch_size=8)
    model = train_two_tower(interactions, products, segments, popularity, config, verbose=False)
    assert model.item_vectors.shape == (4, config.embedding_dim)
    assert np.allclose(np.linalg.norm(model.user_vectors, axis=1), 1.0, atol=1e-4)
    assert len(model.recommend(1, k=2)) == 2
