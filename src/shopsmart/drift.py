"""
Data drift monitoring with Evidently.

Compares how customers behaved in the data the models were trained on (reference: the earliest 80%
of events) with the newest events (current: the latest 20%), one column at a time:

    interaction_type  view / click / add_to_cart / purchase / ...   (Jensen-Shannon distance)
    device            Mobile / Desktop / Tablet                       (Jensen-Shannon distance)
    category          product category of the event                   (Jensen-Shannon distance)
    product_price     price of the product involved                   (Wasserstein distance)
    hour              hour of day                                     (Wasserstein distance)

A column has drifted when its distance reaches Evidently's threshold (0.1). Drift is a warning, not
a failure: it says the models may be going stale and that retraining should be checked.

    pip install -r requirements-monitoring.txt
    python -m shopsmart.drift              # writes reports/drift/drift_report.html + drift_summary.json
    python -m shopsmart.drift --fail-on-drift   # exit code 1 if most columns drifted
"""
from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd

from shopsmart.config import CONFIG, path
from shopsmart.data import load
from shopsmart.recommender.evaluation import time_based_split

NUMERICAL = ["product_price", "hour"]
CATEGORICAL = ["interaction_type", "device", "category"]


def event_features(events: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """One row per event with the columns that are checked for drift."""
    catalogue = products.drop_duplicates("Product ID").set_index("Product ID")
    timestamps = pd.to_datetime(events["Timestamp"])
    return pd.DataFrame({
        "interaction_type": events["Interaction Type"].astype(str).to_numpy(),
        "device": events["Device"].astype(str).to_numpy() if "Device" in events else "unknown",
        "category": events["Product ID"].map(catalogue["Category"]).fillna("unknown").astype(str).to_numpy(),
        "product_price": events["Product ID"].map(catalogue["Price"]).astype(float).to_numpy(),
        "hour": timestamps.dt.hour.astype(int).to_numpy(),
    })


def summarise(report: dict) -> dict:
    """Pull the per-column drift scores out of an Evidently snapshot's dict()."""
    columns, share = {}, None
    for metric in report.get("metrics", []):
        name = metric.get("metric_name", "")
        config = metric.get("config", {})
        if name.startswith("DriftedColumnsCount"):
            share = float(metric["value"]["share"])
        elif name.startswith("ValueDrift"):
            column = config.get("column") or re.search(r"column=([^,)]+)", name).group(1)
            score, threshold = float(metric["value"]), float(config.get("threshold", 0.1))
            columns[column] = {"method": config.get("method", ""), "score": round(score, 4),
                               "threshold": threshold, "drifted": score >= threshold}
    if share is None:
        share = sum(c["drifted"] for c in columns.values()) / len(columns) if columns else 0.0
    return {"drifted_share": share, "drifted_columns": sorted(c for c, v in columns.items() if v["drifted"]),
            "columns": columns}


def summary_markdown(summary: dict) -> str:
    lines = [f"### Data drift: training period vs newest events ({summary['reference_rows']:,} vs "
             f"{summary['current_rows']:,} sampled events)", "",
             "| Column | Method | Score | Threshold | Drifted |", "|---|---|---|---|---|"]
    for column, v in summary["columns"].items():
        lines.append(f"| {column} | {v['method']} | {v['score']:.3f} | {v['threshold']:.2f} | "
                     f"{'yes' if v['drifted'] else 'no'} |")
    lines += ["", f"Drifted columns: {len(summary['drifted_columns'])} of {len(summary['columns'])} "
              f"({summary['drifted_share']:.0%})."]
    return "\n".join(lines) + "\n"


def run(sample_size: int = 20_000, seed: int = 42) -> dict:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    events, products = load("interactions"), load("products")
    reference_events, current_events, cutoff = time_based_split(events, CONFIG["split"]["test_fraction"])
    reference = event_features(reference_events, products)
    current = event_features(current_events, products)
    reference = reference.sample(min(sample_size, len(reference)), random_state=seed)
    current = current.sample(min(sample_size, len(current)), random_state=seed)

    definition = DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL)
    snapshot = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )
    out_dir = path("reports") / "drift"
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out_dir / "drift_report.html"))

    summary = {"cutoff": cutoff.isoformat(), "reference_rows": len(reference), "current_rows": len(current),
               **summarise(snapshot.dict())}
    (out_dir / "drift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evidently data drift report")
    parser.add_argument("--fail-on-drift", action="store_true", help="exit 1 if half or more columns drifted")
    args = parser.parse_args(argv)

    summary = run()
    table = summary_markdown(summary)
    print(table)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as step_summary:
            step_summary.write(table)
    print(f"Saved {path('reports') / 'drift'}")
    if args.fail_on_drift and summary["drifted_share"] >= 0.5:
        print("Most columns drifted: retrain and review the models.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
