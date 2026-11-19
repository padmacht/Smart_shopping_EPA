"""
Evaluate catalogue search on the labelled queries in search_queries.csv.

    python -m shopsmart.search_eval

Compares three search models on Hit@10 (a right product in the top 10), overall and per query type:
    keyword   word and phrase matching (the baseline; no Ollama, so it is repeatable)
    semantic  sentence-embedding similarity (shopsmart.semantic_search)
    hybrid    keyword + semantic rankings merged with reciprocal rank fusion

Writes reports/search_metrics.json. The top-level fields describe keyword search, as before; the
"models" section holds all three. If the embedding model cannot be downloaded, only keyword search
is evaluated. On GitHub Actions the comparison is also written to the run's summary page.
"""
from __future__ import annotations

import json
import os
from typing import Callable

import pandas as pd

from shopsmart.config import CONFIG, path
from shopsmart.data import load
from shopsmart.search import search_catalog
from shopsmart.semantic_search import DEFAULT_MODEL, SemanticSearch

Searcher = Callable[[str, int], pd.DataFrame]


def evaluate(searcher: Searcher, queries: pd.DataFrame, k: int = 10) -> dict:
    rows = []
    for q in queries.itertuples(index=False):
        relevant = {int(x) for x in str(getattr(q, "_3")).split("|")}
        top = set(searcher(q.Query, k)["Product ID"].astype(int))
        rows.append({"query_type": getattr(q, "_2"), "hit": float(bool(relevant & top))})
    df = pd.DataFrame(rows)
    by_type = df.groupby("query_type")["hit"].agg(["mean", "size"])
    return {
        "queries": int(len(df)),
        f"Hit@{k}_overall": float(df["hit"].mean()),
        "by_query_type": {t: {f"Hit@{k}": float(r["mean"]), "queries": int(r["size"])} for t, r in by_type.iterrows()},
    }


def evaluate_keyword_search(k: int = 10) -> dict:
    products, queries = load("products"), load("search_queries")
    result = evaluate(lambda query, limit: search_catalog(products, [query], limit=limit), queries, k)
    return {"model": "keyword search (search_catalog, no Ollama)", "k": k, **result}


def evaluate_all(k: int = 10, model_name: str | None = None) -> dict:
    products, queries = load("products"), load("search_queries")
    model_name = model_name or CONFIG.get("search", {}).get("embedding_model", DEFAULT_MODEL)
    keyword = evaluate(lambda query, limit: search_catalog(products, [query], limit=limit), queries, k)
    metrics = {"model": "keyword search (search_catalog, no Ollama)", "k": k, **keyword,
               "models": {"keyword": keyword}}
    engine = SemanticSearch.from_pretrained(products, model_name)
    if engine is None:
        print(f"Embedding model '{model_name}' could not be loaded: evaluated keyword search only.")
        return metrics
    metrics["embedding_model"] = model_name
    metrics["models"]["semantic"] = evaluate(engine.search, queries, k)
    metrics["models"]["hybrid"] = evaluate(engine.hybrid_search, queries, k)
    return metrics


def summary_markdown(metrics: dict) -> str:
    k = metrics["k"]
    models = metrics["models"]
    types = sorted(next(iter(models.values()))["by_query_type"])
    lines = [f"### Search evaluation: Hit@{k} on {metrics['queries']} labelled queries", ""]
    if "embedding_model" in metrics:
        lines += [f"Embedding model: `{metrics['embedding_model']}`", ""]
    lines.append("| Model | Overall | " + " | ".join(types) + " |")
    lines.append("|---|---|" + "---|" * len(types))
    for name, scores in models.items():
        cells = [f"{scores['by_query_type'][t][f'Hit@{k}']:.1%}" for t in types]
        lines.append(f"| {name} | {scores[f'Hit@{k}_overall']:.1%} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> dict:
    metrics = evaluate_all()
    out = path("reports") / "search_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    table = summary_markdown(metrics)
    print(table)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
            summary.write(table)
    print(f"Saved {out}")
    return metrics


if __name__ == "__main__":
    main()
