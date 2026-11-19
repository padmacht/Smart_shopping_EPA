"""
Quality gate for CI/CD: fail the pipeline if the deployed models are not good enough.

Rules (from the project's "replace only if it wins" policy):
  * the deployed recommender (Hybrid) must beat the best-seller baseline on HitRate@10
  * every recommender must have been evaluated on the same customers
  * K-Means segments must agree with the hidden personas better than chance (ARI > 0)

Usage:  python scripts/check_model_quality.py        (exit code 1 = gate failed)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPORTS = Path(__file__).resolve().parents[1] / "reports"
DEPLOYED = "hybrid"
BASELINE = "popularity"


def main() -> int:
    failures: list[str] = []

    rec = json.loads((REPORTS / "recommender_metrics.json").read_text(encoding="utf-8"))
    models = rec["models"]
    k = rec.get("k", 10)
    metric = f"HitRate@{k}"
    for name, scores in models.items():
        print(f"{name:12s} {metric} {scores[metric]:.3f}   NDCG@{k} {scores[f'NDCG@{k}']:.3f}")

    if DEPLOYED not in models or BASELINE not in models:
        failures.append(f"recommender_metrics.json must contain '{DEPLOYED}' and '{BASELINE}'")
    elif models[DEPLOYED][metric] <= models[BASELINE][metric]:
        failures.append(f"{DEPLOYED} {metric} {models[DEPLOYED][metric]:.3f} does not beat "
                        f"{BASELINE} {models[BASELINE][metric]:.3f}")

    users = {scores["users_evaluated"] for scores in models.values()}
    if len(users) > 1:
        failures.append(f"models were evaluated on different numbers of customers: {sorted(users)}")

    seg = json.loads((REPORTS / "segmentation_metrics.json").read_text(encoding="utf-8"))
    print(f"segmentation ARI {seg['ari_vs_personas']:.3f}   silhouette {seg['silhouette']:.3f}")
    if seg["ari_vs_personas"] <= 0:
        failures.append(f"segmentation ARI {seg['ari_vs_personas']:.3f} is no better than chance")

    if failures:
        print("\nQuality gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nQuality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
