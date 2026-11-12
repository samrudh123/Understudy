"""MLflow tracking setup shared by every stage that logs runs or touches the registry."""

from __future__ import annotations

import os

import requests

from understudy.common.config import load_services

MODEL_NAME = "understudy-student"


class TrackingUnavailableError(RuntimeError):
    """The MLflow tracking server cannot be reached."""


def tracking_uri() -> str:
    """``$MLFLOW_TRACKING_URI`` if set, else ``services.yaml: mlflow.tracking_uri``."""
    return os.environ.get("MLFLOW_TRACKING_URI") or load_services()["mlflow"]["tracking_uri"]


def setup_mlflow(experiment: str | None = None, uri: str | None = None) -> str:
    """Point MLflow at the tracking server (checking it is up) and select ``experiment``."""
    import mlflow

    uri = uri or tracking_uri()
    if uri.startswith("http"):
        try:
            requests.get(uri.rstrip("/") + "/health", timeout=3).raise_for_status()
        except requests.RequestException as exc:
            raise TrackingUnavailableError(
                f"MLflow is not reachable at {uri}. Start it with `make mlflow` "
                "or `docker compose up -d mlflow`."
            ) from exc
    mlflow.set_tracking_uri(uri)
    if experiment:
        mlflow.set_experiment(experiment)
    return uri
