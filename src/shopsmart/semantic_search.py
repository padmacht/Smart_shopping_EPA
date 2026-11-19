"""
Sentence-embedding (semantic) search over the product catalogue.

Keyword search matches words, so "little cordless things to listen to music on the bus" misses
"wireless earbuds". Semantic search turns the query and every product into an embedding (a list of
numbers that captures meaning) and ranks products by cosine similarity, so requests in everyday
language can match products that share no words with them.

The encoder is a Model2Vec static embedding model distilled from a sentence-transformer
(default: minishlab/potion-base-8M, about 30 MB). It runs with NumPy only - no PyTorch - so it fits
Render's 512 MB free tier. The model is downloaded from Hugging Face the first time it is used.

Three ways to search, compared by `python -m shopsmart.search_eval`:
    keyword   search_catalog() - word and phrase matches
    semantic  embedding similarity only
    hybrid    keyword and semantic rankings merged with reciprocal rank fusion (RRF)

    from shopsmart.semantic_search import SemanticSearch
    engine = SemanticSearch.from_pretrained(products)        # None if the model cannot be loaded
    engine.search("something to help me sleep", limit=6)
    engine.hybrid_search("something to help me sleep", limit=6)
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from shopsmart.search import search_catalog

DEFAULT_MODEL = "minishlab/potion-base-8M"
RRF_K = 60  # standard reciprocal-rank-fusion constant: higher = flatter weighting of ranks

Encoder = Callable[[Sequence[str]], np.ndarray]


def product_texts(products: pd.DataFrame) -> list[str]:
    """One short description per product: what it is matters more than marketing copy."""
    parts = []
    for column in ("Product Name", "Product Type", "Category", "Tags", "Colour", "Material"):
        if column in products.columns:
            parts.append(products[column].fillna("").astype(str))
    text = parts[0]
    for part in parts[1:]:
        text = text + ". " + part
    if "Description" in products.columns:
        first_sentence = products["Description"].fillna("").astype(str).str.split(".").str[0]
        text = text + ". " + first_sentence
    return text.str.replace(r"\s+", " ", regex=True).str.strip().tolist()


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class SemanticSearch:
    """Rank catalogue products by cosine similarity between query and product embeddings."""

    def __init__(self, products: pd.DataFrame, encoder: Encoder, model_name: str = "custom") -> None:
        if not {"Product ID", "Product Name"}.issubset(products.columns):
            raise ValueError("products must include Product ID and Product Name columns")
        self.products = products.reset_index(drop=True)
        self.encoder = encoder
        self.model_name = model_name
        self.product_vectors = _normalise(encoder(product_texts(self.products)))
        if len(self.product_vectors) != len(self.products):
            raise ValueError("the encoder returned a different number of vectors than products")

    @classmethod
    def from_pretrained(cls, products: pd.DataFrame, model_name: str = DEFAULT_MODEL) -> "SemanticSearch | None":
        """Load a Model2Vec model; returns None (so callers fall back to keyword search) if unavailable."""
        try:
            from model2vec import StaticModel

            model = StaticModel.from_pretrained(model_name, force_download=False)
        except Exception:  # missing package, no network, unknown model name
            return None
        return cls(products, lambda texts: model.encode(list(texts)), model_name)

    def similarities(self, query: str) -> np.ndarray:
        query_vector = _normalise(self.encoder([query]))[0]
        return self.product_vectors @ query_vector

    def search(self, query: str, limit: int = 6) -> pd.DataFrame:
        """Top products by embedding similarity alone."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if not query.strip():
            return self.products.iloc[0:0].copy()
        order = np.argsort(-self.similarities(query), kind="stable")[:limit]
        return self.products.iloc[order].reset_index(drop=True)

    def hybrid_search(self, query: str, limit: int = 6, extra_terms: Sequence[str] = (),
                      depth: int = 100) -> pd.DataFrame:
        """Merge the keyword and semantic rankings with reciprocal rank fusion.

        A product scores 1 / (RRF_K + rank) in each list it appears in, so one ranked first by
        keyword search (an exact product name) and one ranked first by meaning both rise to the top.
        """
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if not query.strip():
            return self.products.iloc[0:0].copy()
        scores: dict[object, float] = {}
        keyword = search_catalog(self.products, [query, *extra_terms], limit=depth)
        for rank, product_id in enumerate(keyword["Product ID"].tolist()):
            scores[product_id] = scores.get(product_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        semantic = self.search(query, limit=depth)
        for rank, product_id in enumerate(semantic["Product ID"].tolist()):
            scores[product_id] = scores.get(product_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        ranked = sorted(scores, key=lambda product_id: -scores[product_id])[:limit]
        by_id = self.products.set_index("Product ID", drop=False)
        return by_id.loc[ranked].reset_index(drop=True)
