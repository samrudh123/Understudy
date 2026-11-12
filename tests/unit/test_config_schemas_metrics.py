import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from understudy.common.config import load_params, load_services, repo_root, resolve
from understudy.common.metrics import stage_run
from understudy.common.schemas import QueryLogRecord, Route

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]


def test_repo_params_and_services_load() -> None:
    params = load_params(REPO / "params.yaml")
    assert {"data", "simulation", "validate", "spark", "preprocess", "mine"} <= params.keys()
    services = load_services(REPO / "configs/services.yaml", environ={})
    assert services["edge"]["url"].startswith("http")


def test_service_env_override_parses_yaml_scalars() -> None:
    env = {
        "UNDERSTUDY_GATEWAY_TEACHER_TIMEOUT_S": "5",
        "UNDERSTUDY_EDGE_URL": "http://edge:8001",
        "UNDERSTUDY_PROMETHEUS_PUSHGATEWAY_URL": "null",
    }
    services = load_services(REPO / "configs/services.yaml", environ=env)
    assert services["gateway"]["teacher_timeout_s"] == 5
    assert services["edge"]["url"] == "http://edge:8001"
    assert services["prometheus"]["pushgateway_url"] is None


def test_root_env_override(root: Path) -> None:
    assert repo_root() == root
    assert resolve("x/y.json") == root / "x/y.json"
    assert resolve("/abs/path") == Path("/abs/path")


def _record(**kw) -> dict:
    base = dict(request_id="r1", ts=datetime(2026, 8, 3, tzinfo=UTC), text="hello", route="student")
    return base | kw


def test_querylog_schema_accepts_valid_record() -> None:
    rec = QueryLogRecord(**_record(confidence=0.9))
    assert rec.route is Route.STUDENT


@pytest.mark.parametrize(
    "bad",
    [
        {"ts": datetime(2026, 8, 3)},  # naive timestamp
        {"route": "maybe"},  # unknown route
        {"confidence": 1.5},  # out of range
        {"text": ""},  # empty text
        {"surprise": 1},  # unknown field
    ],
)
def test_querylog_schema_rejects_bad_records(bad: dict) -> None:
    with pytest.raises(ValidationError):
        QueryLogRecord(**_record(**bad))


def test_stage_run_writes_success_report(root: Path) -> None:
    with stage_run("demo") as rep:
        rep.rows_in, rep.rows_out = 10, 8
        rep.extra["note"] = "ok"
    report = json.loads((root / "reports/stages/demo.json").read_text())
    assert report["status"] == "success"
    assert report["rows_out"] == 8 and report["note"] == "ok"
    assert report["duration_seconds"] >= 0


def test_stage_run_records_failure_and_reraises(root: Path) -> None:
    with pytest.raises(RuntimeError), stage_run("boom"):
        raise RuntimeError("stage crashed")
    report = json.loads((root / "reports/stages/boom.json").read_text())
    assert report["status"] == "failed"
