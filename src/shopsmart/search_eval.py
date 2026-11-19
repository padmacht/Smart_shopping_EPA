"""
Evaluate catalogue search on the labelled queries in search_queries.csv.

    python -m shopsmart.search_eval

Writes reports/search_metrics.json with Hit@10 per query type. Keyword search only (no Ollama),
so the result is a repeatable baseline that later models (Ollama expansion, embeddings) must beat.
"""
from __future__ import annotations

import json

import pandas as pd

from shopsmart.config import path
from shopsmart.data import load
from shopsmart.search import search_catalog


def evaluate_keyword_search(k: int = 10) -> dict:
    products, queries = load("products"), load("search_queries")
    rows = []
    for q in queries.itertuples(index=False):
        relevant = {int(x) for x in str(getattr(q, "_3")).split("|")}
        top = set(search_catalog(products, [q.Query], limit=k)["Product ID"].astype(int))
        rows.append({"query_type": getattr(q, "_2"), "hit": float(bool(relevant & top))})
    df = pd.DataFrame(rows)
    by_type = df.groupby("query_type")["hit"].agg(["mean", "size"])
    return {
        "model": "keyword search (search_catalog, no Ollama)",
        "k": k,
        "queries": int(len(df)),
        f"Hit@{k}_overall": float(df["hit"].mean()),
        "by_query_type": {t: {f"Hit@{k}": float(r["mean"]), "queries": int(r["size"])} for t, r in by_type.iterrows()},
    }


def main() -> dict:
    metrics = evaluate_keyword_search()
    out = path("reports") / "search_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {out}")
    return metrics


if __name__ == "__main__":
    main()
