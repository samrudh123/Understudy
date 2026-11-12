"""QueryLogger: the only code path that writes user questions to disk.

* **Append-only JSON lines**, one file per day (``<prefix>-YYYYMMDD.jsonl``), or a
  fixed file name for simulated weekly batches.
* **PII is scrubbed before the write**, so raw personal data never reaches disk.
* Every record is validated against :class:`~understudy.common.schemas.QueryLogRecord`.
* Thread-safe, so one instance can be shared by all request handlers.

The ``ingest_logs`` DVC stage snapshots this directory, so each retraining run
is tied to the exact logs it used.
"""

from __future__ import annotations

import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from understudy.common.pii import scrub
from understudy.common.schemas import FeedbackRecord, QueryLogRecord


class QueryLogger:
    """Write scrubbed :class:`QueryLogRecord` / :class:`FeedbackRecord` lines."""

    def __init__(self, log_dir: str | Path, prefix: str = "edge", file_name: str | None = None):
        """Create a logger.

        Args:
            log_dir: directory for the JSONL files (created if missing).
            prefix: file prefix for daily rotation (``edge`` -> ``edge-20261105.jsonl``).
            file_name: if given, always write query records to this file instead
                (used by the traffic simulator for one file per simulated week).

        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.file_name = file_name
        self.pii_counts: Counter = Counter()
        self._lock = threading.Lock()

    def _query_path(self) -> Path:
        if self.file_name:
            return self.log_dir / self.file_name
        return self.log_dir / f"{self.prefix}-{datetime.now(UTC):%Y%m%d}.jsonl"

    def _feedback_path(self) -> Path:
        stem = (
            Path(self.file_name).stem
            if self.file_name
            else f"{self.prefix}-{datetime.now(UTC):%Y%m%d}"
        )
        return self.log_dir / f"feedback-{stem}.jsonl"

    def _append(self, path: Path, line: str) -> None:
        with self._lock, path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def log(self, record: QueryLogRecord) -> QueryLogRecord:
        """Scrub, validate and append a query record. Returns the record as written."""
        result = scrub(record.text)
        if result.found:
            self.pii_counts.update(result.counts)
            record = record.model_copy(update={"text": result.text})
        # Re-validate the (possibly modified) record before it touches disk.
        record = QueryLogRecord.model_validate(record.model_dump())
        self._append(self._query_path(), record.model_dump_json(exclude_none=True))
        return record

    def log_many(self, records: list[QueryLogRecord]) -> int:
        """Log a batch of records; returns how many were written."""
        for r in records:
            self.log(r)
        return len(records)

    def log_feedback(self, record: FeedbackRecord) -> FeedbackRecord:
        """Append a feedback vote (contains no free text, so no scrubbing needed)."""
        self._append(self._feedback_path(), record.model_dump_json())
        return record
