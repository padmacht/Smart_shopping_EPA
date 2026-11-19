"""FastAPI endpoints for ShopSmart recommendations, segments, and catalog search."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse

from shopsmart.config import CONFIG, path
from shopsmart.data import load
from shopsmart.recommender import RecommenderSuite
from shopsmart.search import LocalLLMError, expand_search_query, search_catalog


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load trained artifacts and catalog data once when the API starts."""
    model_path = path("models") / CONFIG["recommender"]["model_file"]
    segments_path = path("processed_data") / CONFIG["segmentation"]["output_file"]
    missing = [str(file) for file in (model_path, segments_path) if not file.is_file()]
    if missing:
        raise RuntimeError(
            "Required trained outputs are missing: " + ", ".join(missing)
            + ". Run segmentation training before recommender training."
        )

    app.state.products = load("products")
    app.state.recommenders = RecommenderSuite.load(model_path)
    segment_frame = pd.read_csv(segments_path)
    app.state.segments = dict(zip(segment_frame["User ID"], segment_frame["segment"]))
    yield


app = FastAPI(
    title="ShopSmart API",
    description="Recommendations, customer segments, and product catalog search.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send visitors of the bare address to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/segments/{user_id}")
def get_customer_segment(user_id: int) -> dict[str, int]:
    segment = app.state.segments.get(user_id)
    if segment is None:
        raise HTTPException(status_code=404, detail=f"No segment found for customer {user_id}.")
    return {"user_id": user_id, "segment": int(segment)}


@app.get("/recommendations/{user_id}")
def get_recommendations(
    user_id: int,
    model: Literal["popularity", "item_knn", "hybrid"] = "hybrid",
    k: int = Query(default=10, ge=1, le=100),
    segment: int | None = None,
) -> dict:
    recommendations = app.state.recommenders.recommend(
        user_id=user_id,
        model=model,
        k=k,
        segment=segment,
    )
    return {
        "user_id": user_id,
        "model": model,
        "recommendations": jsonable_encoder(recommendations.to_dict(orient="records")),
    }


@app.get("/search")
def search_products(
    query: str = Query(min_length=1, max_length=300),
    limit: int = Query(default=6, ge=1, le=50),
    use_llm: bool = False,
) -> dict:
    query = query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty.")

    terms: list[str] = []
    if use_llm:
        try:
            terms = expand_search_query(
                query,
                os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
            )
        except LocalLLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    results = search_catalog(app.state.products, [query, *terms], limit=limit)
    return {
        "query": query,
        "expanded_terms": terms,
        "results": jsonable_encoder(results.to_dict(orient="records")),
    }
