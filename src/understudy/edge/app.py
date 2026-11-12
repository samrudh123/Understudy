"""Edge service: the student answers frequent questions instantly; the rest stream from the server.

Routes

* ``POST /ask``           ``{"question", "stream": true}`` -> SSE ``meta``, ``token``..., ``done``
                          (or a JSON answer with ``"stream": false``)
* ``POST /predict``       student prediction only (no fallback, not logged)
* ``POST /feedback``      ``{"request_id", "helpful"}``
* ``GET  /faqs``          approved questions (UI suggestions)
* ``GET  /info``          model version, tau, answer-bank version
* ``GET  /health``        liveness; ``GET /ready``: a student model is loaded
* ``POST /admin/reload``  re-resolve ``models:/understudy-student@champion`` and hot-swap it
* ``GET  /metrics``       Prometheus

Degradation: with no model loaded (registry down, no champion yet) every question goes
to the teacher. With the server down, instant answers keep working and fallbacks get a
friendly "unavailable" message. Every question is logged (PII-scrubbed) by
:class:`QueryLogger` for the mining pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import shutil
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    generate_latest,
)
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from understudy.common.config import load_services, load_yaml, resolve
from understudy.common.logs import configure_logging, get_logger
from understudy.common.pii import scrub
from understudy.common.schemas import FeedbackRecord, QueryLogRecord, Route
from understudy.common.sse import MEDIA_TYPE, SSE_HEADERS, format_event
from understudy.edge.batcher import MicroBatcher
from understudy.edge.gateway_client import GatewayClient, GatewayError
from understudy.querylog.logger import QueryLogger
from understudy.student.model import StudentPredictor
from understudy.student.router import RouteDecision, Router

log = get_logger(__name__)

FRIENDLY = {
    "unavailable": "The answer service is busy or offline right now. Frequent questions still "
    "get instant answers; please try this one again in a minute.",
    "rejected": "I can only help with questions. Please rephrase your message as a question.",
    "invalid": "Please type a question (up to {n} characters).",
}


class SimMeta(BaseModel):
    """Ground truth attached by the traffic simulator (evaluation only, never used for routing)."""

    week: int | None = Field(default=None, ge=0)  # required unless test=True
    label: str | None = None
    domain: str | None = None
    test: bool = False  # load test / drill: logged as source="test", never mined


class AskIn(BaseModel):
    """A user question."""

    question: str = Field(max_length=4000)
    stream: bool = True
    request_id: str | None = Field(default=None, max_length=64)  # honoured for simulated traffic
    simulated: SimMeta | None = None
    use_teacher: bool = True  # simulated replays set False to save teacher compute


class PredictIn(BaseModel):
    """Question for the student only."""

    question: str = Field(min_length=1, max_length=4000)


class FeedbackIn(BaseModel):
    """Helpful / not helpful on an earlier answer."""

    request_id: str = Field(min_length=1, max_length=64)
    helpful: bool


class EdgeMetrics:
    """Edge metrics on a private registry."""

    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        ProcessCollector(registry=r)  # process CPU seconds + resident memory
        self.requests = Counter("edge_requests", "Questions by route", ["route"], registry=r)
        self.latency = Histogram(
            "edge_request_latency_seconds",
            "Time to the routing decision (student: full answer)",
            ["route"],
            registry=r,
            buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
        )
        self.inference = Histogram(
            "edge_student_inference_seconds",
            "Student encode + head time",
            registry=r,
            buckets=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25),
        )
        self.confidence = Histogram(
            "edge_student_confidence",
            "Top-FAQ confidence",
            registry=r,
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )
        self.ttft = Histogram(
            "edge_teacher_ttft_seconds",
            "Fallback time to first token, seen from the edge",
            registry=r,
            buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 12, 15, 20, 30),
        )
        self.feedback = Counter("edge_feedback", "User feedback", ["helpful"], registry=r)
        self.shadow = Counter(
            "edge_shadow_grades", "Teacher re-checks of instant answers", ["verdict"], registry=r
        )
        self.errors = Counter("edge_errors", "Edge-side failures", ["kind"], registry=r)
        self.reloads = Counter(
            "edge_model_reloads", "Model reload attempts", ["outcome"], registry=r
        )
        self.teacher_skipped = Counter(
            "edge_teacher_skipped", "Simulated fallbacks not sent to the teacher", registry=r
        )
        self.model_info = Gauge(
            "edge_model_info",
            "Loaded student (value 1)",
            ["version", "bank_version", "tau", "n_faqs"],
            registry=r,
        )
        self.model_loaded = Gauge("edge_model_loaded", "1 if a student model is loaded", registry=r)


class ModelHolder:
    """Resolve, download and hot-swap the student. Readers always see a complete model."""

    def __init__(self, uri: str, cache_dir: Path, tau_override: float | None) -> None:
        self.uri, self.cache_dir, self.tau_override = uri, cache_dir, tau_override
        self.router: Router | None = None
        self.version: str | None = None
        self.loaded_at: str | None = None
        self._lock = threading.Lock()

    def _resolve(self) -> tuple[Path, str]:
        if not self.uri.startswith(("models:/", "runs:/")):
            path = resolve(self.uri)
            meta = json.loads((path / "student_meta.json").read_text())
            return path, f"local-{meta.get('build_id', 'dev')}"
        from understudy.common.tracking import setup_mlflow
        from understudy.student.registry import alias_version, download_student

        setup_mlflow()
        uri = self.uri
        version = None
        if uri.startswith("models:/") and "@" in uri:
            name, alias = uri.removeprefix("models:/").split("@", 1)
            mv = alias_version(name, alias)
            if mv is None:
                raise LookupError(f"no version of {name} has alias @{alias} yet")
            version = str(mv.version)
            uri = f"models:/{name}/{version}"
        dest = self.cache_dir / (
            f"v{version}" if version else hashlib.sha1(uri.encode()).hexdigest()[:10]
        )
        student = dest / "artifacts" / "student"
        if not (student / "student_meta.json").is_file():
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            student = download_student(uri, dest)
        return student, version or uri

    def load(self) -> dict[str, Any]:
        """Load the model behind ``uri``; keep the old one if anything fails."""
        with self._lock:
            before = self.version
            path, version = self._resolve()
            predictor = StudentPredictor(path, threads=1)
            predictor.meta["registry_version"] = version
            predictor.predict(["warm up"])  # first ONNX run is slow; pay it before swapping
            self.router = Router(predictor, self.tau_override)
            self.version = version
            self.loaded_at = datetime.now(UTC).isoformat()
        return {
            "before": before,
            "after": version,
            "faqs": len(predictor.bank),
            "tau": self.router.tau,
        }


def create_app(
    settings: dict[str, Any] | None = None,
    gateway: GatewayClient | None = None,
    log_dir: str | Path | None = None,
) -> FastAPI:
    """Build the edge app. Tests pass a local model URI, a fake gateway and a temp log dir."""
    cfg = settings or load_services()
    ec = cfg["edge"]
    metrics = EdgeMetrics()
    max_chars = int(ec.get("max_question_chars", 500))
    shadow_rate = float(ec.get("shadow_rate", 0.0))
    filler = (
        load_yaml(ec["filler_file"]) if ec.get("filler_file") else {"version": "none", "cards": []}
    )
    holder = ModelHolder(
        ec["model_uri"],
        resolve(ec.get("model_cache_dir", "models/edge_cache")),
        ec.get("confidence_threshold_override"),
    )
    qlog = QueryLogger(resolve(log_dir or cfg["querylog"]["dir"]), prefix="edge")
    background: set[asyncio.Task] = set()

    def route_batch(texts: list[str]) -> list[RouteDecision]:
        router = holder.router  # read once: a reload may swap it mid-flight
        return router.route_many(texts)

    batcher = MicroBatcher(
        route_batch,
        max_batch=int(ec.get("max_batch", 16)),
        max_wait_ms=float(ec.get("batch_wait_ms", 2.0)),
    )

    def update_model_metrics() -> None:
        metrics.model_info.clear()
        if holder.router:
            p = holder.router.predictor
            metrics.model_info.labels(
                holder.version, p.bank.version, f"{holder.router.tau:.2f}", str(len(p.bank))
            ).set(1)
        metrics.model_loaded.set(1 if holder.router else 0)

    async def load_until_ready() -> None:
        while holder.router is None:
            try:
                info = await run_in_threadpool(holder.load)
                metrics.reloads.labels("ok").inc()
                log.info("student loaded", extra=info)
            except Exception as exc:  # registry down / no champion yet: keep proxying to teacher
                metrics.reloads.labels("failed").inc()
                log.warning("student not loaded yet", extra={"error": str(exc)})
                await asyncio.sleep(float(ec.get("model_retry_s", 10)))
            update_model_metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        app.state.gateway = gateway or GatewayClient(
            cfg["gateway"]["url"], timeout_s=float(cfg["gateway"]["teacher_timeout_s"]) + 15
        )
        loader = asyncio.create_task(load_until_ready())
        yield
        loader.cancel()
        await batcher.close()
        await app.state.gateway.aclose()

    app = FastAPI(title="Understudy edge", version="1.0", lifespan=lifespan)
    app.state.holder, app.state.metrics, app.state.qlog = holder, metrics, qlog

    def record(
        rid: str,
        body: AskIn,
        route: Route,
        d: RouteDecision | None,
        latency_ms: float,
        ttft_ms: float | None = None,
    ) -> None:
        try:
            qlog.log(
                QueryLogRecord(
                    request_id=rid,
                    ts=datetime.now(UTC),
                    text=body.question.strip()[:2000],
                    route=route,
                    matched_faq_id=d.faq_id if (d and route == Route.STUDENT) else None,
                    confidence=round(d.confidence, 4) if d else None,
                    student_version=holder.version,
                    latency_ms=round(latency_ms, 2),
                    teacher_ttft_ms=None if ttft_ms is None else round(ttft_ms, 1),
                    source=("test" if body.simulated.test else "simulated")
                    if body.simulated
                    else "live",
                    sim_week=body.simulated.week if body.simulated else None,
                    sim_label=body.simulated.label if body.simulated else None,
                    sim_domain=body.simulated.domain if body.simulated else None,
                )
            )
        except Exception as exc:  # logging must never break answering
            metrics.errors.labels("querylog").inc()
            log.exception("query log write failed", extra={"error": str(exc)})

    def filler_card(rid: str) -> dict[str, Any]:
        cards = filler.get("cards") or []
        if not cards:
            return {"version": filler.get("version")}
        card = cards[int(hashlib.md5(rid.encode()).hexdigest(), 16) % len(cards)]
        return {"version": filler.get("version"), **card}

    async def shadow_grade(question: str, d: RouteDecision) -> None:
        try:
            ok = await request_gateway().grade(question, d.matched_question or "", d.answer or "")
            metrics.shadow.labels("relevant" if ok else "irrelevant").inc()
        except GatewayError as exc:
            metrics.shadow.labels("error").inc()
            log.info("shadow grade skipped", extra={"code": exc.code})

    def request_gateway() -> GatewayClient:
        return app.state.gateway

    def admin_ok(token: str | None) -> None:
        expected = ec.get("admin_token")
        if expected and token != expected:
            raise HTTPException(status_code=403, detail="bad admin token")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> Response:
        if holder.router is None:
            return JSONResponse(status_code=503, content={"status": "loading"})
        return JSONResponse({"status": "ready", "model_version": holder.version})

    @app.get("/info")
    async def info() -> dict[str, Any]:
        if holder.router is None:
            return {"model_loaded": False, "model_uri": holder.uri}
        p = holder.router.predictor
        return {
            "model_loaded": True,
            "model_uri": holder.uri,
            "model_version": holder.version,
            "loaded_at": holder.loaded_at,
            "tau": holder.router.tau,
            "n_faqs": len(p.bank),
            "bank_version": p.bank.version,
            "head": p.meta.get("head"),
            "build_id": p.meta.get("build_id"),
            "filler_version": filler.get("version"),
        }

    @app.get("/faqs")
    async def faqs() -> dict[str, Any]:
        if holder.router is None:
            return {"faqs": []}
        bank = holder.router.predictor.bank
        return {
            "bank_version": bank.version,
            "faqs": [{"faq_id": f.faq_id, "question": f.question} for f in bank.faqs.values()],
        }

    @app.get("/metrics")
    async def prom() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/admin/reload")
    async def reload(x_admin_token: str | None = Header(default=None)) -> Response:
        admin_ok(x_admin_token)
        try:
            out = await run_in_threadpool(holder.load)
        except Exception as exc:
            metrics.reloads.labels("failed").inc()
            log.error("model reload failed", extra={"error": str(exc)})
            return JSONResponse(
                status_code=500,
                content={"reloaded": False, "error": str(exc), "serving": holder.version},
            )
        metrics.reloads.labels("ok").inc()
        update_model_metrics()
        log.info("model reloaded", extra=out)
        return JSONResponse({"reloaded": True, **out})

    @app.post("/predict")
    async def predict(body: PredictIn) -> dict[str, Any]:
        if holder.router is None:
            raise HTTPException(status_code=503, detail="no student model loaded")
        d = await run_in_threadpool(holder.router.route, scrub(body.question).text)
        return {
            "use_student": d.use_student,
            "faq_id": d.faq_id,
            "confidence": round(d.confidence, 4),
            "matched_question": d.matched_question,
            "tau": holder.router.tau,
            "model_version": holder.version,
            "inference_ms": round(d.inference_ms, 2),
        }

    @app.post("/feedback")
    async def feedback(body: FeedbackIn) -> dict[str, str]:
        qlog.log_feedback(
            FeedbackRecord(request_id=body.request_id, ts=datetime.now(UTC), helpful=body.helpful)
        )
        metrics.feedback.labels(str(body.helpful).lower()).inc()
        return {"status": "recorded"}

    @app.post("/ask")
    async def ask(body: AskIn) -> Response:
        t0 = time.perf_counter()
        question = body.question.strip()
        if body.simulated and not ec.get("allow_simulation", True):
            raise HTTPException(status_code=403, detail="simulated traffic is disabled")
        if body.simulated and body.simulated.week is None and not body.simulated.test:
            raise HTTPException(status_code=422, detail="simulated.week is required")
        if not question or len(question) > max_chars:
            metrics.requests.labels(Route.REJECTED.value).inc()
            return JSONResponse(
                status_code=422,
                content={
                    "route": Route.REJECTED.value,
                    "code": "invalid_question",
                    "message": FRIENDLY["invalid"].format(n=max_chars),
                },
            )
        rid = body.request_id if (body.simulated and body.request_id) else uuid.uuid4().hex
        router = holder.router
        d: RouteDecision | None = None
        if router is not None:
            d = await batcher.submit(scrub(question).text)
            metrics.inference.observe(d.inference_ms / 1000)
            metrics.confidence.observe(d.confidence)

        if d is not None and d.use_student:
            latency = (time.perf_counter() - t0) * 1000
            metrics.requests.labels(Route.STUDENT.value).inc()
            metrics.latency.labels(Route.STUDENT.value).observe(latency / 1000)
            record(rid, body, Route.STUDENT, d, latency)
            if shadow_rate and not body.simulated and random.random() < shadow_rate:
                task = asyncio.create_task(shadow_grade(scrub(question).text, d))
                background.add(task)
                task.add_done_callback(background.discard)
            meta = {
                "route": Route.STUDENT.value,
                "request_id": rid,
                "faq_id": d.faq_id,
                "matched_question": d.matched_question,
                "confidence": round(d.confidence, 4),
                "model_version": holder.version,
                "latency_ms": round(latency, 2),
            }
            if not body.stream:
                return JSONResponse({**meta, "answer": d.answer})

            async def instant() -> AsyncIterator[str]:
                yield format_event("meta", meta)
                yield format_event("token", {"t": d.answer})
                yield format_event("done", {"route": Route.STUDENT.value})

            return StreamingResponse(instant(), media_type=MEDIA_TYPE, headers=SSE_HEADERS)

        # ---- fallback to the teacher
        related = d.related if d else []
        conf = round(d.confidence, 4) if d else None
        if not body.use_teacher:  # simulated replay: route and log, but spare the teacher
            latency = (time.perf_counter() - t0) * 1000
            metrics.requests.labels(Route.TEACHER.value).inc()
            metrics.latency.labels(Route.TEACHER.value).observe(latency / 1000)
            metrics.teacher_skipped.inc()
            record(rid, body, Route.TEACHER, d, latency)
            return JSONResponse(
                {
                    "route": Route.TEACHER.value,
                    "request_id": rid,
                    "confidence": conf,
                    "teacher_skipped": True,
                    "model_version": holder.version,
                }
            )

        meta = {
            "route": Route.TEACHER.value,
            "request_id": rid,
            "confidence": conf,
            "model_version": holder.version,
            "filler": filler_card(rid),
            "related": related,
        }

        async def relay() -> AsyncIterator[tuple[str, dict[str, Any]]]:
            """Yield events for the client; log once the outcome is known."""
            route, ttft = Route.TEACHER, None
            try:
                async for name, data in request_gateway().stream(scrub(question).text, rid):
                    if name == "meta":
                        ttft = (time.perf_counter() - t0) * 1000
                        metrics.ttft.observe(ttft / 1000)
                        continue
                    if name == "error":
                        route = Route.UNAVAILABLE
                        yield (
                            "error",
                            {
                                "code": data.get("code", "teacher_error"),
                                "message": FRIENDLY["unavailable"],
                            },
                        )
                        break
                    yield name, data
            except GatewayError as exc:
                if exc.code in {"prompt_injection", "empty", "too_long"}:
                    route = Route.REJECTED
                    yield "error", {"code": exc.code, "message": FRIENDLY["rejected"]}
                else:
                    route = Route.UNAVAILABLE
                    metrics.errors.labels(exc.code).inc()
                    yield "error", {"code": exc.code, "message": FRIENDLY["unavailable"]}
            finally:
                latency = (time.perf_counter() - t0) * 1000
                metrics.requests.labels(route.value).inc()
                metrics.latency.labels(route.value).observe(latency / 1000)
                record(rid, body, route, d, latency, ttft)

        if not body.stream:
            parts, final = [], {}
            async for name, data in relay():
                if name == "token":
                    parts.append(data["t"])
                elif name in {"error", "done"}:
                    final = {"event": name, **data}
            if final.get("event") == "error":
                return JSONResponse(
                    {
                        **meta,
                        "route": "unavailable"
                        if final["code"] not in {"prompt_injection", "empty", "too_long"}
                        else "rejected",
                        "error": final,
                    }
                )
            return JSONResponse({**meta, "answer": "".join(parts), "teacher": final})

        async def sse() -> AsyncIterator[str]:
            yield format_event("meta", meta)
            async for name, data in relay():
                yield format_event(name, data)

        return StreamingResponse(sse(), media_type=MEDIA_TYPE, headers=SSE_HEADERS)

    return app


def main() -> None:
    """Run with uvicorn (port from ``services.yaml``)."""
    from urllib.parse import urlparse

    import uvicorn

    port = urlparse(load_services()["edge"]["url"]).port or 8001
    uvicorn.run(create_app(), host="0.0.0.0", port=port, log_config=None)


if __name__ == "__main__":
    main()
