"""Pipeline-stage instrumentation.

Each DVC stage runs inside :func:`stage_run`, which:

1. times the stage and records rows in/out,
2. writes ``reports/stages/<stage>.json``, which the pipeline console reads for DAG
   status, durations and throughput,
3. pushes ``pipeline_stage_duration_seconds{stage}`` and
   ``pipeline_rows_processed_total{stage}`` to a Prometheus Pushgateway when one
   is configured. Batch jobs are short-lived, so they push rather than wait to be
   scraped.

Service metrics (edge/gateway) are defined next to each service.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from understudy.common.config import load_services, resolve
from understudy.common.lineage import git_commit
from understudy.common.logs import get_logger

log = get_logger(__name__)

STAGE_REPORT_DIR = "reports/stages"


@dataclass
class StageReport:
    """Mutable record a stage fills in while it runs."""

    stage: str
    rows_in: int = 0
    rows_out: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def _pushgateway_url() -> str | None:
    url = os.environ.get("UNDERSTUDY_PROMETHEUS_PUSHGATEWAY_URL")
    if url:
        return url
    try:
        return load_services().get("prometheus", {}).get("pushgateway_url")
    except FileNotFoundError:
        return None


def _push(report: StageReport, duration: float, success: bool) -> None:
    url = _pushgateway_url()
    if not url:
        return
    from prometheus_client import CollectorRegistry, Counter, Gauge, push_to_gateway

    reg = CollectorRegistry()
    Gauge(
        "pipeline_stage_duration_seconds",
        "Wall-clock duration of the last run",
        ["stage"],
        registry=reg,
    ).labels(report.stage).set(duration)
    Counter(
        "pipeline_rows_processed", "Rows written by the last run", ["stage"], registry=reg
    ).labels(report.stage).inc(report.rows_out)
    Gauge(
        "pipeline_stage_success", "1 if the last run succeeded, else 0", ["stage"], registry=reg
    ).labels(report.stage).set(1 if success else 0)
    Gauge(
        "pipeline_stage_last_run_timestamp_seconds",
        "Unix time the last run finished",
        ["stage"],
        registry=reg,
    ).labels(report.stage).set(time.time())
    try:
        push_to_gateway(
            url,
            job="understudy_pipeline",
            grouping_key={"stage": report.stage},
            registry=reg,
            timeout=3,
        )
    except OSError as exc:  # monitoring must never break the pipeline
        log.warning("pushgateway unreachable", extra={"url": url, "error": str(exc)})


@contextmanager
def stage_run(stage: str, report_dir: str = STAGE_REPORT_DIR) -> Iterator[StageReport]:
    """Instrument a pipeline stage. Re-raises any exception after recording the failure."""
    report = StageReport(stage)
    started = datetime.now(UTC)
    t0 = time.perf_counter()
    success = False
    log.info("stage started", extra={"stage": stage})
    try:
        yield report
        success = True
    finally:
        duration = time.perf_counter() - t0
        payload = {
            "stage": stage,
            "status": "success" if success else "failed",
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "duration_seconds": round(duration, 3),
            "rows_in": report.rows_in,
            "rows_out": report.rows_out,
            "rows_per_second": round(report.rows_out / duration, 2) if duration > 0 else None,
            "git_commit": git_commit(),
            **report.extra,
        }
        out = resolve(report_dir) / f"{stage}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        _push(report, duration, success)
        log.info(
            "stage finished",
            extra={k: v for k, v in payload.items() if k != "stage"} | {"stage": stage},
        )
