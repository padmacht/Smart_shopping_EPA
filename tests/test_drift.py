"""Tests for drift monitoring. The full Evidently run is tested only where Evidently is installed."""
import pandas as pd
import pytest

from shopsmart.drift import event_features, summarise, summary_markdown


def test_event_features_join_product_details():
    events = pd.DataFrame({
        "Interaction Type": ["view", "purchase"], "Device": ["Mobile", "Desktop"],
        "Product ID": [10, 20], "Timestamp": ["2026-01-01 09:15", "2026-01-02 21:40"],
    })
    products = pd.DataFrame({"Product ID": [10, 20], "Category": ["Books", "Toys"], "Price": [5.0, 25.0]})
    features = event_features(events, products)
    assert features.to_dict("list") == {
        "interaction_type": ["view", "purchase"], "device": ["Mobile", "Desktop"],
        "category": ["Books", "Toys"], "product_price": [5.0, 25.0], "hour": [9, 21],
    }


def test_summarise_reads_evidently_metrics():
    report = {"metrics": [
        {"metric_name": "DriftedColumnsCount(drift_share=0.5)", "config": {}, "value": {"count": 1.0, "share": 0.5}},
        {"metric_name": "ValueDrift(column=hour,method=Wasserstein distance (normed),threshold=0.1)",
         "config": {"column": "hour", "method": "Wasserstein distance (normed)", "threshold": 0.1}, "value": 0.25},
        {"metric_name": "ValueDrift(column=device,method=Jensen-Shannon distance,threshold=0.1)",
         "config": {"column": "device", "method": "Jensen-Shannon distance", "threshold": 0.1}, "value": 0.01},
    ]}
    summary = summarise(report)
    assert summary["drifted_share"] == 0.5
    assert summary["drifted_columns"] == ["hour"]
    assert summary["columns"]["device"]["drifted"] is False
    table = summary_markdown({**summary, "reference_rows": 100, "current_rows": 50})
    assert "| hour | Wasserstein distance (normed) | 0.250 | 0.10 | yes |" in table


def test_full_drift_run_writes_report(tmp_path, monkeypatch):
    pytest.importorskip("evidently")
    from shopsmart import drift

    monkeypatch.setattr(drift, "path", lambda key: tmp_path)
    summary = drift.run(sample_size=2_000)
    assert set(summary["columns"]) == {"interaction_type", "device", "category", "product_price", "hour"}
    assert (tmp_path / "drift" / "drift_report.html").exists()
