import mlflow

from shopsmart.tracking import tracked_run


def test_tracked_run_logs_to_configured_store(monkeypatch, tmp_path):
    previous_uri = mlflow.get_tracking_uri()
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    with tracked_run("test-shopsmart", "tracking-check") as tracking:
        tracking.log_param("algorithm", "test")
        tracking.log_metric("score", 0.75)

    try:
        mlflow.set_tracking_uri(tracking_uri)
        runs = mlflow.search_runs(experiment_names=["test-shopsmart"])
        assert len(runs) == 1
        assert runs.iloc[0]["params.algorithm"] == "test"
        assert runs.iloc[0]["metrics.score"] == 0.75
    finally:
        mlflow.set_tracking_uri(previous_uri)
