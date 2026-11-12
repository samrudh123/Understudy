"""Console API against a temporary repo root (no DVC, MLflow or Prometheus needed)."""

import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from understudy.common.schemas import Draft
from understudy.console.app import create_app, dag
from understudy.console.runner import BusyError, PipelineRunner

pytestmark = pytest.mark.unit
REPO = Path(__file__).resolve().parents[2]

SETTINGS = {
    "edge": {"url": "http://127.0.0.1:9"},
    "prometheus": {"url": "http://127.0.0.1:9"},
    "mlflow": {"tracking_uri": "http://127.0.0.1:9"},
    "console": {
        "trigger_cooldown_s": 600,
        "alert_triggers": ["FallbackRatioHigh"],
        "links": {"mlflow": "http://m"},
    },
}
PARAMS = {"register": {"model_name": "understudy-student"}}


@pytest.fixture
def croot(root: Path) -> Path:
    shutil.copy(REPO / "dvc.yaml", root / "dvc.yaml")
    drafts = [
        Draft(
            candidate_id=f"cand-{i}",
            question=f"q{i}",
            answer=f"a{i}",
            paraphrases=[f"p{i}"],
            examples=[f"e{i}"],
            support=10 + i,
            share_of_fallbacks=0.1,
            cohesion=0.6,
            teacher_model="fake",
            prompt_version="teacher_v1",
            generated_at=datetime.now(UTC),
            latency_s=1.0,
        ).model_dump(mode="json")
        for i in range(3)
    ]
    (root / "data/drafts").mkdir(parents=True)
    (root / "data/drafts/drafts.json").write_text(json.dumps({"drafts": drafts}))
    (root / "reports/stages").mkdir(parents=True)
    (root / "reports/stages/train.json").write_text(
        json.dumps(
            {
                "status": "success",
                "duration_seconds": 12.5,
                "rows_out": 21,
                "finished_at": "2026-11-06T00:00:00",
            }
        )
    )
    return root


def test_dag_has_pipeline_edges(croot: Path) -> None:
    d = dag()
    names = {n["name"] for n in d["nodes"]}
    assert {"ingest_logs", "mine_frequent", "distil", "train", "register"} <= names
    edges = {(e["from"], e["to"]) for e in d["edges"]}
    assert ("mine_frequent", "distil") in edges and ("train", "evaluate_gate") in edges
    train = next(n for n in d["nodes"] if n["name"] == "train")
    assert train["status"] == "success" and train["duration_s"] == 12.5
    assert next(n for n in d["nodes"] if n["name"] == "distil")["status"] == "never_run"


def test_approval_flow(croot: Path) -> None:
    with TestClient(create_app(SETTINGS, PARAMS)) as c:
        q = c.get("/api/approvals").json()
        assert [d["candidate_id"] for d in q["pending"]] == ["cand-2", "cand-1", "cand-0"]
        r = c.post(
            "/api/approvals/cand-2",
            json={"decision": "approve", "reviewer": "ana", "answer": "edited"},
        )
        assert r.status_code == 200 and r.json()["faq_id"] == "faq-2"
        r = c.post(
            "/api/approvals/cand-1",
            json={"decision": "merge", "reviewer": "ana", "merge_into": "faq-2"},
        )
        assert r.status_code == 200
        bad = c.post("/api/approvals/cand-1", json={"decision": "approve", "reviewer": "ana"})
        assert bad.status_code == 400  # already decided
        q = c.get("/api/approvals").json()
        assert [d["candidate_id"] for d in q["pending"]] == ["cand-0"]
        assert q["approved"][0]["faq_id"] == "faq-2"
        saved = json.loads((croot / "data/approvals/faq-2.json").read_text())
        assert saved["answer"] == "edited" and saved["merged_candidates"] == ["cand-1"]
        assert 'console_approvals_total{decision="approve"} 1.0' in c.get("/metrics").text


def test_overview_degrades_when_services_down(croot: Path) -> None:
    with TestClient(create_app(SETTINGS, PARAMS)) as c:
        o = c.get("/api/overview").json()
        assert o["edge"] is None and o["alerts"] is None and o["fallback_ratio_10m"] is None
        assert o["pending_approvals"] == 3
        assert c.get("/api/models").status_code == 503  # MLflow unreachable -> clear error


def test_alert_webhook_starts_discovery_once(croot: Path, monkeypatch) -> None:
    app = create_app(SETTINGS, PARAMS)
    started = []
    monkeypatch.setattr(
        app.state.runner,
        "start",
        lambda name, cmd, reason, env=None: started.append(name) or {"id": "x", "name": name},
    )
    payload = {"alerts": [{"status": "firing", "labels": {"alertname": "FallbackRatioHigh"}}]}
    with TestClient(app) as c:
        assert (
            c.post(
                "/hooks/alertmanager",
                json={"alerts": [{"status": "firing", "labels": {"alertname": "ServiceDown"}}]},
            ).json()["action"]
            == "ignored"
        )
        assert c.post("/hooks/alertmanager", json=payload).json()["action"] == "started"
        assert c.post("/hooks/alertmanager", json=payload).json()["action"] == "cooldown"
        resolved = {
            "alerts": [{"status": "resolved", "labels": {"alertname": "FallbackRatioHigh"}}]
        }
        assert c.post("/hooks/alertmanager", json=resolved).json()["action"] == "ignored"
    assert started == ["discover"]


def test_runner_serialises_and_records(root: Path) -> None:
    done = []
    r = PipelineRunner(history_path=root / "runs.json", on_finish=done.append)
    r.start("t1", [sys.executable, "-c", "import time; print('hello'); time.sleep(0.3)"], "test")
    with pytest.raises(BusyError):
        r.start("t2", [sys.executable, "-c", "pass"], "test")
    for _ in range(100):
        if done:
            break
        time.sleep(0.05)
    assert done[0]["state"] == "succeeded" and "hello" in done[0]["log_tail"]
    r.start("t3", [sys.executable, "-c", "raise SystemExit(3)"], "test")
    for _ in range(100):
        if len(done) == 2:
            break
        time.sleep(0.05)
    assert done[1]["state"] == "failed" and done[1]["exit_code"] == 3
    hist = json.loads((root / "runs.json").read_text())
    assert [h["name"] for h in hist] == ["t1", "t3"]
    assert r.status()["history"][0]["name"] == "t3"


def test_simulate_validates_weeks(croot: Path) -> None:
    with TestClient(create_app(SETTINGS, PARAMS)) as c:
        assert c.post("/api/simulate", json={"weeks": "3; rm -rf /"}).status_code == 422
