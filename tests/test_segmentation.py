"""Tests for the segmentation module - run with: pytest"""
import pandas as pd
import pytest

from shopsmart.data import load
from shopsmart.segmentation import NON_BUYER, SegmentationModel, build_features


@pytest.fixture(scope="module")
def features():
    interactions = load("interactions")
    return build_features(interactions[interactions["User ID"] < 800], load("products"), "extended")


def test_one_row_per_customer_no_missing(features):
    assert features.index.is_unique
    assert not features.isna().any().any()


def test_shares_are_between_0_and_1(features):
    shares = features.filter(regex="^cat_|_share$|brand_loyalty")
    assert ((shares >= 0) & (shares <= 1)).all().all()


def test_non_buyers_get_their_own_segment(features):
    model = SegmentationModel(n_clusters=4).fit(features)
    labels = model.predict(features)
    assert (labels[features["frequency"] == 0] == NON_BUYER).all()
    assert set(labels[features["frequency"] > 0].unique()) <= set(range(4))


def test_save_and_load_give_same_predictions(features, tmp_path):
    model = SegmentationModel(n_clusters=4).fit(features)
    loaded = SegmentationModel.load(model.save(tmp_path / "m.joblib"))
    pd.testing.assert_series_equal(model.predict(features), loaded.predict(features))
