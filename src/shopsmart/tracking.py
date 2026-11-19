"""Shared MLflow tracking setup for ShopSmart training runs."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from shopsmart.config import path


@contextmanager
def tracked_run(experiment_name: str, run_name: str) -> Iterator[Any]:
    """Start an MLflow run, defaulting to a local store unless configured otherwise."""
    import mlflow

    previous_uri = mlflow.get_tracking_uri()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_dir = path("mlruns").resolve()
        tracking_dir.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{(tracking_dir / 'mlflow.db').as_posix()}"
    try:
        mlflow.set_tracking_uri(tracking_uri)
        if os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_experiment(experiment_name)
        else:
            artifact_dir = path("mlruns").resolve() / "artifacts" / experiment_name
            mlflow.set_experiment(experiment_name, artifact_location=artifact_dir.as_uri())
        with mlflow.start_run(run_name=run_name):
            yield mlflow
    finally:
        mlflow.set_tracking_uri(previous_uri)
