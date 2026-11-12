"""Pipeline console API. The console web page calls it; nginx serves it at ``/api/console/``.

* ``GET  /api/overview``          champion, pending approvals, last run, fallback ratio, alerts
* ``GET  /api/dag``               DVC stages, their dependencies, last-run status of each
* ``GET  /api/runs``              recent MLflow training runs
* ``GET  /api/models``            registry versions + aliases
* ``POST /api/models/rollback``   champion back to the version it replaced; ``/promote``
* ``GET  /api/approvals``         pending teacher drafts + approved FAQs (merge targets)
* ``POST /api/approvals/{cand}``  approve / reject / merge (with edits)
* ``POST /api/pipeline/run``      start ``discover`` / ``retrain`` / ``full``
* ``GET  /api/pipeline/status``   current run with log tail, history
* ``POST /api/simulate``          replay simulated weeks against the edge (demo)
* ``POST /hooks/alertmanager``    Alertmanager webhook: a drift alert starts discovery
* ``GET  /health``, ``GET /metrics``
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Literal

import requests
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    ProcessCollector,
    generate_latest,
)
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from understudy.common.config import load_params, load_services, resolve
from understudy.common.logs import configure_logging, get_logger
from understudy.console.runner import TARGETS, BusyError, PipelineRunner
from understudy.pipeline import approve

log = get_logger(__name__)


class DecisionIn(BaseModel):
    """A reviewer decision on one draft."""

    decision: Literal["approve", "reject", "merge"]
    reviewer: str = Field(min_length=1, max_length=64)
    question: str | None = Field(default=None, max_length=300)
    answer: str | None = Field(default=None, max_length=2000)
    paraphrases: list[str] | None = None
    merge_into: str | None = None


class RunIn(BaseModel):
    """Start a pipeline target."""

    target: Literal["discover", "retrain", "full"]
    reason: str = "manual (console)"


class RollbackIn(BaseModel):
    """Roll the champion back (default: to the version it replaced)."""

    to_version: str | None = None


class PromoteIn(BaseModel):
    """Make ``version`` the champion."""

    version: str


class SimulateIn(BaseModel):
    """Replay simulated weeks against the edge."""

    weeks: str = Field(pattern=r"^[0-9,\- ]+$", max_length=20)
    teacher_rate: float = Field(default=0.01, ge=0.0, le=1.0)


def _read_json(path: str) -> Any:
    try:
        return json.loads(resolve(path).read_text())
    except (FileNotFoundError, ValueError):
        return None


def dag() -> dict[str, Any]:
    """Stages from ``dvc.yaml`` with edges (out -> dep) and their last-run reports."""
    spec = yaml.safe_load(resolve("dvc.yaml").read_text())["stages"]

    def paths(items: list | None) -> list[str]:
        out = []
        for it in items or []:
            out.append(next(iter(it)) if isinstance(it, dict) else str(it))
        return out

    outs = {name: paths(s.get("outs")) + paths(s.get("metrics")) for name, s in spec.items()}
    nodes, edges = [], []
    for name, s in spec.items():
        rep = _read_json(f"reports/stages/{name}.json") or {}
        nodes.append(
            {
                "name": name,
                "cmd": s.get("cmd"),
                "status": rep.get("status", "never_run"),
                "finished_at": rep.get("finished_at"),
                "duration_s": rep.get("duration_seconds"),
                "rows_in": rep.get("rows_in"),
                "rows_out": rep.get("rows_out"),
            }
        )
        for dep in paths(s.get("deps")):
            for other, o_paths in outs.items():
                if other != name and any(
                    dep == o or dep.startswith(o + "/") or o.startswith(dep + "/") for o in o_paths
                ):
                    edges.append({"from": other, "to": name})
    uniq = [dict(t) for t in {tuple(sorted(e.items())) for e in edges}]
    return {"nodes": nodes, "edges": sorted(uniq, key=lambda e: (e["from"], e["to"]))}


def dvc_outdated(timeout: float = 60.0) -> list[str] | None:
    """Stage names ``dvc status`` reports as changed (None if dvc is unavailable)."""
    try:
        out = subprocess.run(
            ["dvc", "status", "--json"],
            cwd=resolve("."),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return sorted(k.split(":")[-1] for k in json.loads(out.stdout or "{}"))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def create_app(
    settings: dict[str, Any] | None = None, params: dict[str, Any] | None = None
) -> FastAPI:
    """Build the console API."""
    cfg = settings or load_services()
    prm = params or load_params()
    cc = cfg["console"]
    model_name = prm["register"]["model_name"]

    reg = CollectorRegistry()
    ProcessCollector(registry=reg)  # process CPU seconds + resident memory
    m_runs = Counter("console_pipeline_runs", "Pipeline runs", ["target", "outcome"], registry=reg)
    m_running = Gauge(
        "console_pipeline_running", "1 while a pipeline run is in progress", registry=reg
    )
    m_decisions = Counter("console_approvals", "Reviewer decisions", ["decision"], registry=reg)
    m_pending = Gauge("console_pending_drafts", "Drafts waiting for review", registry=reg)
    m_rollbacks = Counter("console_rollbacks", "Champion rollbacks", ["outcome"], registry=reg)
    m_rollback_s = Gauge(
        "console_last_rollback_seconds", "Duration of the last rollback", registry=reg
    )
    m_triggers = Counter(
        "console_alert_triggers", "Alerts received", ["alertname", "action"], registry=reg
    )

    def finished(run: dict[str, Any]) -> None:
        m_runs.labels(run["name"], run["state"]).inc()
        m_running.set(0)

    runner = PipelineRunner(on_finish=finished)
    state: dict[str, Any] = {"last_trigger": 0.0, "outdated": None, "outdated_at": 0.0}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        yield

    app = FastAPI(title="Understudy console", version="1.0", lifespan=lifespan)
    app.state.runner = runner

    def mlflow_ready() -> None:
        import os

        from understudy.common.tracking import TrackingUnavailableError, setup_mlflow

        try:
            setup_mlflow(uri=os.environ.get("MLFLOW_TRACKING_URI") or cfg["mlflow"]["tracking_uri"])
        except TrackingUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def edge_reload() -> dict[str, Any]:
        from understudy.pipeline.register import reload_edge

        return reload_edge()

    def start(name: str, cmd: list[str], reason: str) -> dict[str, Any]:
        try:
            run = runner.start(name, cmd, reason)
        except BusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        m_running.set(1)
        return run

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def prom() -> Response:
        with contextlib.suppress(Exception):  # metrics must render even if files are mid-write
            m_pending.set(len(approve.pending(approve.load_drafts(), approve.load_approvals())))
        return Response(generate_latest(reg), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/links")
    async def links() -> dict[str, str]:
        return dict(cc.get("links") or {})

    @app.get("/api/dag")
    async def get_dag(refresh: bool = False) -> dict[str, Any]:
        out = await run_in_threadpool(dag)
        if refresh or time.time() - state["outdated_at"] > 120:
            state["outdated"] = await run_in_threadpool(dvc_outdated)
            state["outdated_at"] = time.time()
        out["outdated"] = state["outdated"]
        out["running"] = runner.current
        return out

    @app.get("/api/overview")
    async def overview() -> dict[str, Any]:
        def edge_info() -> dict[str, Any] | None:
            try:
                return requests.get(cfg["edge"]["url"].rstrip("/") + "/info", timeout=2).json()
            except (requests.RequestException, ValueError):
                return None

        def prom_query(q: str) -> float | None:
            try:
                r = requests.get(
                    cfg["prometheus"]["url"].rstrip("/") + "/api/v1/query",
                    params={"query": q},
                    timeout=2,
                ).json()
                res = r["data"]["result"]
                return float(res[0]["value"][1]) if res else None
            except (requests.RequestException, ValueError, KeyError, IndexError):
                return None

        def alerts() -> list[dict[str, Any]] | None:
            try:
                r = requests.get(cfg["prometheus"]["url"].rstrip("/") + "/api/v1/alerts", timeout=2)
                return [
                    {
                        "name": a["labels"].get("alertname"),
                        "state": a["state"],
                        "severity": a["labels"].get("severity"),
                        "since": a.get("activeAt"),
                    }
                    for a in r.json()["data"]["alerts"]
                ]
            except (requests.RequestException, ValueError, KeyError):
                return None

        pending = approve.pending(approve.load_drafts(), approve.load_approvals())
        return {
            "edge": await run_in_threadpool(edge_info),
            "fallback_ratio_10m": await run_in_threadpool(
                prom_query,
                'sum(rate(edge_requests_total{route!="student"}[10m])) / '
                "sum(rate(edge_requests_total[10m]))",
            ),
            "alerts": await run_in_threadpool(alerts),
            "pending_approvals": len(pending),
            "approved_faqs": sum(
                r.decision == "approved" for r in approve.load_approvals().values()
            ),
            "gate": _read_json("reports/gate.json"),
            "register": _read_json("reports/register.json"),
            "mine": _read_json("reports/mine_metrics.json"),
            "distil": _read_json("reports/distil_metrics.json"),
            "train": _read_json("reports/train_metrics.json"),
            "pipeline": runner.status()["last"],
            "running": runner.current,
        }

    @app.get("/api/approvals")
    async def approvals() -> dict[str, Any]:
        drafts, decided = approve.load_drafts(), approve.load_approvals()
        pending = approve.pending(drafts, decided)
        m_pending.set(len(pending))
        return {
            "pending": [d.model_dump(mode="json") for d in pending],
            "approved": [
                {"faq_id": r.faq_id, "question": r.question, "support": r.support}
                for r in sorted(decided.values(), key=lambda r: -(r.support or 0))
                if r.decision == "approved"
            ],
            "decided_count": len(decided),
        }

    @app.post("/api/approvals/{candidate_id}")
    async def decide(candidate_id: str, body: DecisionIn) -> dict[str, Any]:
        try:
            rec = await run_in_threadpool(
                lambda: approve.decide(
                    candidate_id,
                    body.decision,
                    body.reviewer,
                    question=body.question,
                    answer=body.answer,
                    paraphrases=body.paraphrases,
                    merge_into=body.merge_into,
                )
            )
        except approve.ApprovalError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        m_decisions.labels(body.decision).inc()
        return {"faq_id": rec.faq_id, "decision": rec.decision}

    @app.get("/api/models")
    async def models() -> dict[str, Any]:
        await run_in_threadpool(mlflow_ready)
        from understudy.student import registry

        return {
            "name": model_name,
            "versions": await run_in_threadpool(registry.versions, model_name),
        }

    @app.post("/api/models/rollback")
    async def rollback(body: RollbackIn) -> dict[str, Any]:
        await run_in_threadpool(mlflow_ready)
        from understudy.student import registry

        t0 = time.perf_counter()
        try:
            out = await run_in_threadpool(registry.rollback, model_name, body.to_version)
        except RuntimeError as exc:
            m_rollbacks.labels("failed").inc()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        out["edge"] = await run_in_threadpool(edge_reload)
        out["seconds"] = round(time.perf_counter() - t0, 2)
        m_rollbacks.labels("ok" if out["edge"].get("reloaded") else "edge_not_reloaded").inc()
        m_rollback_s.set(out["seconds"])
        return out

    @app.post("/api/models/promote")
    async def promote(body: PromoteIn) -> dict[str, Any]:
        await run_in_threadpool(mlflow_ready)
        from understudy.student import registry

        out = await run_in_threadpool(
            registry.promote, model_name, body.version, "manual (console)"
        )
        out["edge"] = await run_in_threadpool(edge_reload)
        return out

    @app.get("/api/runs")
    async def runs(limit: int = 15) -> dict[str, Any]:
        await run_in_threadpool(mlflow_ready)

        def fetch() -> list[dict[str, Any]]:
            import mlflow

            df = mlflow.search_runs(
                experiment_names=["understudy-student"],
                max_results=200,
                order_by=["attributes.start_time DESC"],
            )
            if df.empty:
                return []
            parent_col = "tags.mlflow.parentRunId"
            if parent_col in df:
                df = df[df[parent_col].isna()]
            keep = [
                "metrics.test_traffic_precision",
                "metrics.test_traffic_coverage",
                "metrics.test_oos_false_accept",
                "metrics.latency_p95_ms",
                "metrics.n_faqs",
                "params.tau",
                "params.head_kind",
            ]
            out = []
            for _, r in df.head(limit).iterrows():
                out.append(
                    {
                        "run_id": r["run_id"],
                        "name": r.get("tags.mlflow.runName"),
                        "status": r["status"],
                        "start": str(r["start_time"]),
                        "experiment_id": r["experiment_id"],
                        **{
                            k.split(".", 1)[1]: (None if r.get(k) != r.get(k) else r.get(k))
                            for k in keep
                            if k in r
                        },
                    }
                )
            return out

        return {"runs": await run_in_threadpool(fetch)}

    @app.get("/api/pipeline/status")
    async def pipeline_status() -> dict[str, Any]:
        return runner.status() | {"targets": {k: v[1] for k, v in TARGETS.items()}}

    @app.post("/api/pipeline/run")
    async def pipeline_run(body: RunIn) -> dict[str, Any]:
        cmd, _ = TARGETS[body.target]
        return start(body.target, cmd, body.reason)

    @app.post("/api/simulate")
    async def simulate(body: SimulateIn) -> dict[str, Any]:
        cmd = [
            sys.executable,
            "-m",
            "understudy.data.simulate_traffic",
            "--mode",
            "online",
            "--weeks",
            body.weeks.replace(" ", ""),
            "--teacher-rate",
            str(body.teacher_rate),
        ]
        return start(f"simulate weeks {body.weeks}", cmd, "demo traffic (console)")

    @app.post("/hooks/alertmanager")
    async def alertmanager(payload: dict[str, Any]) -> Response:
        triggers = set(cc.get("alert_triggers") or [])
        firing = [a for a in payload.get("alerts", []) if a.get("status") == "firing"]
        names = sorted({a.get("labels", {}).get("alertname", "?") for a in firing})
        wanted = [n for n in names if n in triggers]
        if not wanted:
            for n in names:
                m_triggers.labels(n, "ignored").inc()
            return JSONResponse({"action": "ignored", "alerts": names})
        cooldown = float(cc.get("trigger_cooldown_s", 600))
        if time.time() - state["last_trigger"] < cooldown:
            for n in wanted:
                m_triggers.labels(n, "cooldown").inc()
            return JSONResponse({"action": "cooldown", "alerts": wanted})
        try:
            run = runner.start(
                "discover",
                TARGETS["discover"][0],
                f"alert: {', '.join(wanted)} at {datetime.now(UTC).isoformat()}",
            )
        except BusyError:
            for n in wanted:
                m_triggers.labels(n, "busy").inc()
            return JSONResponse({"action": "busy", "alerts": wanted})
        m_running.set(1)
        state["last_trigger"] = time.time()
        for n in wanted:
            m_triggers.labels(n, "started").inc()
        log.warning("drift alert started discovery", extra={"alerts": wanted, "run": run["id"]})
        return JSONResponse({"action": "started", "run": run, "alerts": wanted})

    return app


def main() -> None:
    """Run with uvicorn (port from ``services.yaml``)."""
    from urllib.parse import urlparse

    import uvicorn

    port = urlparse(load_services()["console"]["url"]).port or 8003
    uvicorn.run(create_app(), host="0.0.0.0", port=port, log_config=None)


if __name__ == "__main__":
    main()
