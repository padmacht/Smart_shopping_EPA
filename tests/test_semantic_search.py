"""Tests for sentence-embedding search (with a tiny fake encoder, so no model download is needed)."""
import numpy as np
import pandas as pd
import pytest

from shopsmart import semantic_search
from shopsmart.search_eval import summary_markdown
from shopsmart.semantic_search import SemanticSearch, product_texts

# word -> meaning axis: 0 = listening to music, 1 = sleep, 2 = travel
CONCEPTS = {"earbuds": 0, "wireless": 0, "music": 0, "listen": 0, "headphones": 0,
            "sleep": 1, "blanket": 1, "weighted": 1, "night": 1,
            "backpack": 2, "trip": 2, "travel": 2, "bus": 2}


def fake_encoder(texts):
    vectors = np.zeros((len(texts), 3), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in text.lower().replace(".", " ").split():
            if word in CONCEPTS:
                vectors[row, CONCEPTS[word]] += 1.0
    return vectors


@pytest.fixture
def products():
    return pd.DataFrame({
        "Product ID": [1, 2, 3],
        "Product Name": ["Pulse Wireless Earbuds", "Cloud Weighted Blanket", "Urban Travel Backpack"],
        "Category": ["Electronics", "Home", "Fashion"],
        "Product Type": ["Earbuds", "Blanket", "Backpack"],
        "Description": ["Great for music. Long battery.", "Calm sleep every night.", "Fits a laptop."],
    })


def test_product_texts_use_name_type_and_first_sentence(products):
    text = product_texts(products)[0]
    assert "Pulse Wireless Earbuds" in text and "Earbuds" in text and "Great for music" in text
    assert "Long battery" not in text


def test_semantic_search_matches_meaning_without_shared_words(products):
    engine = SemanticSearch(products, fake_encoder)
    # "listen to music" shares no word with any product name, but means earbuds
    assert engine.search("something to listen to music", limit=1)["Product ID"].tolist() == [1]
    assert engine.search("help me sleep", limit=1)["Product ID"].tolist() == [2]


def test_hybrid_search_keeps_exact_keyword_matches_on_top(products):
    engine = SemanticSearch(products, fake_encoder)
    assert engine.hybrid_search("Urban Travel Backpack", limit=3)["Product ID"].tolist()[0] == 3
    assert engine.hybrid_search("music for the bus", limit=3)["Product ID"].nunique() == 3


def test_empty_query_and_bad_limit(products):
    engine = SemanticSearch(products, fake_encoder)
    assert engine.search("   ").empty and engine.hybrid_search("").empty
    with pytest.raises(ValueError):
        engine.search("music", limit=0)


def test_missing_model_falls_back_to_none(products, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("no network")

    model2vec = pytest.importorskip("model2vec")
    monkeypatch.setattr(model2vec.StaticModel, "from_pretrained", unavailable)
    assert SemanticSearch.from_pretrained(products, "some/model") is None


def test_summary_markdown_lists_every_model():
    scores = {"queries": 2, "Hit@10_overall": 0.5,
              "by_query_type": {"keyword": {"Hit@10": 1.0, "queries": 1}, "semantic": {"Hit@10": 0.0, "queries": 1}}}
    table = summary_markdown({"k": 10, "queries": 2, "models": {"keyword": scores, "hybrid": scores}})
    assert "| keyword | 50.0% | 100.0% | 0.0% |" in table and "| hybrid |" in table


def test_default_model_is_small_static_model():
    assert semantic_search.DEFAULT_MODEL == "minishlab/potion-base-8M"
