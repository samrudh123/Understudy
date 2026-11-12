"""Phase 3 chain on synthetic data: distil (fake teacher, real Ray) -> approve -> dataset ->
train (real Ray Tune, real ONNX encoder) -> gate -> register -> promote -> rollback.

Uses a throw-away SQLite MLflow store. Skipped if the ONNX encoder has not been exported.
"""

import json
import shutil
from pathlib import Path

import pytest

from understudy.data import ingest, validate
from understudy.data.simulate_traffic import simulate_offline
from understudy.pipeline import (
    approve,
    build_dataset,
    distil_ray,
    evaluate_gate,
    mine_frequent_spark,
    preprocess_spark,
    register,
    train,
)

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
ENCODER = REPO / "models/encoder"


@pytest.mark.skipif(not (ENCODER / "encoder.onnx").is_file(), reason="run export_encoder first")
def test_distil_to_registry(root: Path, params: dict, tiny_raw: Path, monkeypatch) -> None:
    import mlflow

    shutil.copytree(REPO / "configs", root / "configs")
    shutil.copytree(ENCODER, root / "models/encoder")
    monkeypatch.chdir(root)
    monkeypatch.setenv("UNDERSTUDY_TEACHER_FAKE", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{root}/mlflow.db")
    monkeypatch.setenv("UNDERSTUDY_EDGE_URL", "http://127.0.0.1:9")  # no edge: reload must not fail
    mlflow.set_tracking_uri(f"sqlite:///{root}/mlflow.db")
    mlflow.create_experiment(train.EXPERIMENT, artifact_location=(root / "artifacts").as_uri())

    simulate_offline(params, [0, 1], tiny_raw, root / "data/querylogs", root / "data/approvals")
    ingest.ingest_logs(root / "data/querylogs", root / "data/snapshots")
    validate.run(params, tiny_raw, root / "data/snapshots")
    preprocess_spark.run(params, tiny_raw, root / "data/snapshots", root / "data/processed")
    mine_frequent_spark.run(params, root / "data/processed/querylog", root / "data/approvals")

    report = distil_ray.run(params)
    assert report["drafts"] > 0 and report["failed"] == 0
    drafts = approve.load_drafts()
    assert all(d.paraphrases and d.answer for d in drafts)
    again = distil_ray.run(params)  # second run is served from the cache
    assert again["cache_hits"] == again["drafts"] and again["generated"] == 0

    counts = approve.auto_review()
    assert counts["approved"] >= 2

    stats = build_dataset.run(params)
    assert stats["faqs"] == counts["approved"] and stats["train_rows"] > 0

    out = train.run(params)
    assert out["status"] == "trained"
    assert (root / "models/student/student_meta.json").is_file()
    assert out["metrics"]["latency_p95_ms"] < 100

    gate = evaluate_gate.run(params)
    assert gate["passed"], gate["reasons"]
    reg = register.register(params)
    assert reg["registered"] and reg["promoted"] and reg["champion"] == "1"
    assert reg["edge"]["reloaded"] is False  # no edge running, but no crash

    # A second model, then roll back to the first.
    train.run(params)
    evaluate_gate.run(params)
    reg2 = register.register(params)
    assert reg2["champion"] == "2" and reg2["previous"] == "1"
    from understudy.student import registry

    name = params["register"]["model_name"]
    assert registry.rollback(name)["champion"] == "1"
    assert str(registry.alias_version(name, "champion").version) == "1"
    assert json.loads((root / "reports/stages/register.json").read_text())["status"] == "success"
