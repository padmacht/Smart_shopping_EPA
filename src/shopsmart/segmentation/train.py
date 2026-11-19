"""
Train the customer segmentation model.

    python -m shopsmart.segmentation.train                  # settings from configs/config.yaml
    python -m shopsmart.segmentation.train --k 6            # override number of segments
    python -m shopsmart.segmentation.train --features rfm   # notebook-02 feature set
    python -m shopsmart.segmentation.train --search-k       # print the elbow/silhouette table and stop

Writes:
    models/segmentation_kmeans.joblib           trained scaler + K-Means
    data/processed/customer_segments.csv        User ID, segment
    reports/segmentation_profile.csv            profile table per segment
    reports/segmentation_metrics.json           quality metrics (+ ARI vs hidden personas)
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from shopsmart.config import CONFIG, path
from shopsmart.data import load
from shopsmart.segmentation.features import build_features
from shopsmart.segmentation.model import NON_BUYER, SegmentationModel
from shopsmart.tracking import tracked_run


def main(argv: list[str] | None = None) -> dict:
    cfg = CONFIG["segmentation"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=cfg["n_clusters"], help="number of segments")
    ap.add_argument("--features", choices=["rfm", "extended"], default=cfg["feature_set"])
    ap.add_argument("--all-customers", action="store_true", help="cluster non-buyers too (as notebook 02)")
    ap.add_argument("--search-k", action="store_true", help="show metrics for a range of k and exit")
    args = ap.parse_args(argv)
    buyers_only = cfg["buyers_only"] and not args.all_customers
    seed = CONFIG["project"]["random_seed"]

    print("Loading data ...")
    interactions, products = load("interactions"), load("products")
    feat = build_features(interactions, products, args.features)
    print(f"Features: {feat.shape[0]:,} customers x {feat.shape[1]} features ({args.features})")

    if args.search_k:
        lo, hi = cfg["k_search"]
        table = SegmentationModel.search_k(feat, lo, hi, buyers_only, seed)
        print(table[["silhouette", "davies_bouldin", "inertia"]].round(3).to_string())
        return {}

    model = SegmentationModel(args.k, buyers_only, seed).fit(feat)
    labels = model.predict(feat)
    metrics = model.evaluate(feat, labels)
    metrics.update({"k": args.k, "feature_set": args.features, "buyers_only": buyers_only})

    # Evaluation only: compare with the hidden personas (possible because the data is synthetic)
    truth = load("personas_truth").set_index("User ID")["Persona"]
    clustered = labels[labels != NON_BUYER]
    metrics["ari_vs_personas"] = float(adjusted_rand_score(truth.loc[clustered.index], clustered))

    profile = SegmentationModel.profile(feat, labels)
    model_file = model.save(path("models") / cfg["model_file"])
    labels.rename_axis("User ID").to_csv(path("processed_data") / cfg["output_file"])
    profile.to_csv(path("reports") / "segmentation_profile.csv")
    with open(path("reports") / "segmentation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with tracked_run("shopsmart-segmentation", f"kmeans-k{args.k}-{args.features}") as mlflow:
        mlflow.log_params({
            "n_clusters": args.k,
            "features": args.features,
            "buyers_only": buyers_only,
            "random_seed": seed,
        })
        mlflow.log_metrics({
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        })
        mlflow.log_artifact(str(model_file), artifact_path="model")
        mlflow.log_artifact(str(path("processed_data") / cfg["output_file"]), artifact_path="outputs")
        mlflow.log_artifact(str(path("reports") / "segmentation_profile.csv"), artifact_path="outputs")
        mlflow.log_artifact(str(path("reports") / "segmentation_metrics.json"), artifact_path="outputs")

    print("\nSegment profiles (segment -1 = non-buyers):")
    print(profile.to_string())
    print("\nMetrics:", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()}, indent=2))
    print(f"\nSaved model -> {model_file}")
    return metrics


if __name__ == "__main__":
    main()
