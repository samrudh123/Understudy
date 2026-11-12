import json
from pathlib import Path

import pytest

from understudy.data import ingest, validate
from understudy.data.simulate_traffic import simulate_offline

pytestmark = pytest.mark.unit


def test_download_rejects_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"tampered"

    monkeypatch.setattr(ingest.requests, "get", lambda *a, **k: FakeResp())
    with pytest.raises(ingest.IngestError, match="Checksum mismatch"):
        ingest.download("http://x/data.json", tmp_path / "data.json", sha256="0" * 64)
    assert not (tmp_path / "data.json").exists()


def test_ingest_logs_snapshots_queries_and_feedback(root: Path) -> None:
    logs = root / "data/querylogs"
    logs.mkdir(parents=True)
    (logs / "edge-20260803.jsonl").write_text('{"a": 1}\n{"a": 2}\n')
    (logs / "sim-week-00.jsonl").write_text('{"a": 3}\n')
    (logs / "feedback-edge-20260803.jsonl").write_text('{"f": 1}\n')
    manifest = ingest.ingest_logs(logs, root / "data/snapshots")
    assert manifest["query_rows"] == 3 and manifest["feedback_rows"] == 1
    assert len((root / "data/snapshots/querylog.jsonl").read_text().splitlines()) == 3


def test_ingest_logs_cold_start(root: Path) -> None:
    manifest = ingest.ingest_logs(root / "data/querylogs", root / "data/snapshots")
    assert manifest["query_rows"] == 0
    assert (root / "data/snapshots/querylog.jsonl").read_text() == ""


def _snapshot(root: Path, params: dict, tiny_raw: Path) -> Path:
    simulate_offline(params, [0], tiny_raw, root / "data/querylogs", root / "data/approvals")
    ingest.ingest_logs(root / "data/querylogs", root / "data/snapshots")
    return root / "data/snapshots"


def test_validate_passes_on_good_data(root: Path, params: dict, tiny_raw: Path) -> None:
    snap = _snapshot(root, params, tiny_raw)
    assert validate.run(params, tiny_raw, snap)
    report = json.loads((root / "reports/validation.json").read_text())
    assert report["passed"] and report["n_failed"] == 0
    assert report["querylog"]["rows"] == params["simulation"]["queries_per_week"]


def test_validate_fails_on_unknown_label(root: Path, params: dict, tiny_raw: Path) -> None:
    snap = _snapshot(root, params, tiny_raw)
    data = json.loads((tiny_raw / "data_full.json").read_text())
    data["train"].append(["mystery utterance", "not_an_intent"])
    (tiny_raw / "data_full.json").write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        validate.run(params, tiny_raw, snap)
    report = json.loads((root / "reports/validation.json").read_text())
    failed = {c["check"] for c in report["checks"] if not c["passed"]}
    assert "labels_known" in failed
    stage = json.loads((root / "reports/stages/validate.json").read_text())
    assert stage["status"] == "failed"


def test_validate_fails_when_too_many_bad_log_rows(
    root: Path, params: dict, tiny_raw: Path
) -> None:
    snap = _snapshot(root, params, tiny_raw)
    with (snap / "querylog.jsonl").open("a") as fh:
        for _ in range(10):
            fh.write('{"request_id": "x", "text": "no timestamp", "route": "student"}\n')
    with pytest.raises(SystemExit):
        validate.run(params, tiny_raw, snap)
