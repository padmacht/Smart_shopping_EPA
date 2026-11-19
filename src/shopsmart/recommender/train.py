"""Train and evaluate the recommendation baselines.

Run with ``python -m shopsmart.recommender.train`` after training customer segments.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from shopsmart.config import CONFIG, path
from shopsmart.data import load
from shopsmart.recommender.evaluation import evaluate_at_k, time_based_split
from shopsmart.recommender.models import (
    HybridRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    RecommenderSuite,
)
from shopsmart.tracking import tracked_run


def train_recommenders() -> tuple[RecommenderSuite, dict[str, object]]:
    """Train on the earlier time window and evaluate on subsequent purchases."""
    interactions = load("interactions")
    products = load("products")
    segment_file = path("processed_data") / CONFIG["segmentation"]["output_file"]
    if not segment_file.exists():
        raise FileNotFoundError(
            f"{segment_file} not found. Run `python -m shopsmart.segmentation.train` first."
        )
    segments = pd.read_csv(segment_file)

    train_events, test_events, cutoff = time_based_split(
        interactions,
        test_fraction=CONFIG["split"]["test_fraction"],
    )
    popularity = PopularityRecommender(train_events, products, segments)
    item_knn = ItemKNNRecommender(train_events, popularity)
    hybrid = HybridRecommender(
        popularity,
        item_knn,
        item_weight=CONFIG["recommender"]["item_weight"],
    )
    suite = RecommenderSuite(popularity, item_knn, hybrid)

    k = CONFIG["recommender"]["k"]
    metrics: dict[str, Any] = {
        "split_method": "global chronological",
        "cutoff": cutoff.isoformat(),
        "test_fraction": CONFIG["split"]["test_fraction"],
        "train_events": len(train_events),
        "test_events": len(test_events),
        "k": k,
        "models": {},
    }
    for model_name in ("popularity", "item_knn", "hybrid"):
        metrics["models"][model_name] = evaluate_at_k(suite, model_name, test_events, k)

    model_file = path("models") / CONFIG["recommender"]["model_file"]
    suite.save(model_file)
    metrics_file = path("reports") / CONFIG["recommender"]["metrics_file"]
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    if metrics_file.exists():  # keep the two-tower (deep learning) result from its own training run
        previous = json.loads(metrics_file.read_text(encoding="utf-8"))
        if "two_tower" in previous.get("models", {}) and previous.get("cutoff") == metrics["cutoff"]:
            metrics["models"]["two_tower"] = previous["models"]["two_tower"]
    with open(metrics_file, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with tracked_run("shopsmart-recommender", "popularity-item-knn-hybrid") as mlflow:
        mlflow.log_params({
            "split_method": metrics["split_method"],
            "test_fraction": metrics["test_fraction"],
            "cutoff": metrics["cutoff"],
            "k": k,
            "item_weight": CONFIG["recommender"]["item_weight"],
            "train_events": metrics["train_events"],
            "test_events": metrics["test_events"],
        })
        for model_name, scores in metrics["models"].items():
            for metric_name in (f"HitRate@{k}", f"NDCG@{k}"):
                if metric_name in scores:
                    mlflow_metric_name = metric_name.replace("@", "_")
                    mlflow.log_metric(
                        f"{model_name}_{mlflow_metric_name}",
                        float(scores[metric_name]),
                    )
        mlflow.log_artifact(str(model_file), artifact_path="model")
        mlflow.log_artifact(str(metrics_file), artifact_path="outputs")

    print(f"Time split cutoff: {cutoff}")
    print(f"Training events: {len(train_events):,} | test events: {len(test_events):,}")
    for model_name, scores in metrics["models"].items():
        print(
            f"{model_name:10s} "
            f"HitRate@{k}: {scores[f'HitRate@{k}']:.3f} | "
            f"NDCG@{k}: {scores[f'NDCG@{k}']:.3f} | "
            f"users: {scores['users_evaluated']:,}"
        )
    print(f"Saved recommenders -> {model_file}")
    print(f"Saved metrics -> {metrics_file}")
    return suite, metrics


def main() -> None:
    train_recommenders()


if __name__ == "__main__":
    main()
