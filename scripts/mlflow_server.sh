#!/usr/bin/env bash
# Local MLflow tracking server + model registry (same store the Docker `mlflow` service mounts).
#   backend:   mlflow-data/mlflow.db  (SQLite)
#   artifacts: mlflow-data/artifacts  (proxied through the server: mlflow-artifacts:/)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p mlflow-data/artifacts
exec mlflow server \
  --backend-store-uri "sqlite:///mlflow-data/mlflow.db" \
  --artifacts-destination "./mlflow-data/artifacts" \
  --host "${MLFLOW_HOST:-127.0.0.1}" --port "${MLFLOW_PORT:-5000}"
