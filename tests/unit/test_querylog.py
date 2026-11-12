import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from understudy.common.schemas import FeedbackRecord, QueryLogRecord, Route
from understudy.querylog.logger import QueryLogger

pytestmark = pytest.mark.unit


def _rec(text: str, rid: str = "r1") -> QueryLogRecord:
    return QueryLogRecord(request_id=rid, ts=datetime.now(UTC), text=text, route=Route.TEACHER)


def test_logger_scrubs_before_writing(tmp_path: Path) -> None:
    qlog = QueryLogger(tmp_path, file_name="q.jsonl")
    written = qlog.log(_rec("email me at someone@example.com"))
    assert written.text == "email me at <EMAIL>"
    raw = (tmp_path / "q.jsonl").read_text()
    assert "someone@example.com" not in raw
    assert qlog.pii_counts["EMAIL"] == 1


def test_logger_appends_and_validates(tmp_path: Path) -> None:
    qlog = QueryLogger(tmp_path, file_name="q.jsonl")
    qlog.log_many([_rec("a question", "r1"), _rec("another one", "r2")])
    lines = (tmp_path / "q.jsonl").read_text().splitlines()
    assert [json.loads(line)["request_id"] for line in lines] == ["r1", "r2"]
    for line in lines:
        QueryLogRecord.model_validate_json(line)


def test_daily_rotation_and_feedback_file(tmp_path: Path) -> None:
    qlog = QueryLogger(tmp_path, prefix="edge")
    qlog.log(_rec("hi there"))
    qlog.log_feedback(FeedbackRecord(request_id="r1", ts=datetime.now(UTC), helpful=True))
    names = sorted(p.name for p in tmp_path.iterdir())
    assert len(names) == 2
    assert names[0].startswith("edge-") and names[1].startswith("feedback-edge-")
