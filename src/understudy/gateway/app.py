"""Server gateway: the edge's single entry point to the teacher.

Routes

* ``POST /stream``    ``{"question", "request_id"?}`` -> SSE ``meta``, ``token``..., ``done``
* ``POST /grade``     shadow grading of a student answer (JSON)
* ``GET  /health``    liveness (the gateway process is up)
* ``GET  /ready``     readiness (the teacher answers its health check)
* ``GET  /metrics``   Prometheus

Behaviour

* Every question passes :func:`understudy.gateway.guard.check_question`. Invalid input
  gets ``400`` with ``code`` ``empty``/``too_long``/``prompt_injection``.
* The gateway waits up to ``first_token_timeout_s`` for the teacher's first token
  *before* committing the response, so failures still get real status codes:
  ``503`` (teacher down) and ``504`` (no first token in time). After that, the whole
  answer must finish within ``teacher_timeout_s``, otherwise an ``error`` event ends
  the stream.
* TTFT, tokens/s, token counts and outcomes go to Prometheus.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
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

from understudy.common.config import load_services
from understudy.common.logs import configure_logging, get_logger
from understudy.common.sse import MEDIA_TYPE, SSE_HEADERS, format_event
from understudy.gateway.guard import check_question
from understudy.teacher.client import TeacherClient, TeacherError

log = get_logger(__name__)


class StreamIn(BaseModel):
    """A fallback question from the edge."""

    question: str = Field(max_length=4000)
    request_id: str | None = Field(default=None, max_length=64)


class GradeIn(BaseModel):
    """A student answer to be checked by the teacher."""

    question: str = Field(max_length=4000)
    faq_question: str = Field(max_length=2000)
    faq_answer: str = Field(max_length=4000)


class GatewayMetrics:
    """All gateway metrics on one registry (one per app, so tests stay isolated)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        ProcessCollector(registry=r)  # process CPU seconds + resident memory
        self.requests = Counter(
            "gateway_requests",
            "Gateway requests by endpoint and outcome",
            ["endpoint", "outcome"],
            registry=r,
        )
        self.ttft = Histogram(
            "gateway_teacher_ttft_seconds",
            "Time to the teacher's first token",
            registry=r,
            buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 12, 15, 20, 30),
        )
        self.tokens_per_s = Histogram(
            "gateway_teacher_tokens_per_second",
            "Teacher decode speed after the first token",
            registry=r,
            buckets=(1, 2, 4, 6, 8, 10, 12, 15, 20, 30, 50),
        )
        self.tokens = Counter(
            "gateway_teacher_tokens", "Tokens streamed from the teacher", registry=r
        )
        self.duration = Histogram(
            "gateway_request_duration_seconds",
            "Full /stream duration",
            registry=r,
            buckets=(0.5, 1, 2, 5, 10, 20, 30, 45, 60, 90),
        )
        self.inflight = Gauge("gateway_inflight_requests", "Streams in progress", registry=r)
        self.teacher_up = Gauge(
            "gateway_teacher_up", "1 if the last teacher health check passed", registry=r
        )


def _error(status: int, code: str, detail: str, metrics: GatewayMetrics, endpoint: str):
    metrics.requests.labels(endpoint, code).inc()
    return JSONResponse(status_code=status, content={"code": code, "detail": detail})


def create_app(
    settings: dict[str, Any] | None = None, teacher: TeacherClient | None = None
) -> FastAPI:
    """Build the gateway. Tests pass a ``TeacherClient`` with a mock transport."""
    cfg = settings or load_services()
    gw = cfg["gateway"]
    first_timeout = float(gw["first_token_timeout_s"])
    total_timeout = float(gw["teacher_timeout_s"])
    metrics = GatewayMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        app.state.teacher = teacher or TeacherClient(cfg["teacher"]["url"], timeout_s=total_timeout)
        yield
        await app.state.teacher.aclose()

    app = FastAPI(title="Understudy gateway", version="1.0", lifespan=lifespan)
    app.state.metrics = metrics
    if teacher is not None:  # also usable without lifespan (e.g. mounted via ASGI transport)
        app.state.teacher = teacher

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> Response:
        try:
            info = await request.app.state.teacher.health()
            metrics.teacher_up.set(1)
            return JSONResponse({"status": "ready", "teacher": info})
        except TeacherError as exc:
            metrics.teacher_up.set(0)
            return JSONResponse(
                status_code=503, content={"status": "not_ready", "detail": str(exc)}
            )

    @app.get("/metrics")
    async def prom() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/stream")
    async def stream(body: StreamIn, request: Request) -> Response:
        guard = check_question(body.question)
        if not guard.ok:
            log.info("question rejected", extra={"code": guard.code, "rule": guard.reason})
            return _error(400, guard.code or "invalid", guard.reason or "", metrics, "stream")
        request_id = body.request_id or uuid.uuid4().hex
        t0 = time.perf_counter()
        events = request.app.state.teacher.stream(guard.text, request_id).__aiter__()

        # Hold the response until the first token (or a failure) so errors keep real status codes.
        try:
            first = await asyncio.wait_for(events.__anext__(), timeout=first_timeout)
        except TimeoutError:
            await events.aclose()
            return _error(
                504, "teacher_timeout", f"no first token in {first_timeout}s", metrics, "stream"
            )
        except (TeacherError, StopAsyncIteration) as exc:
            log.warning("teacher unavailable", extra={"request_id": request_id, "error": str(exc)})
            return _error(503, "teacher_unavailable", str(exc), metrics, "stream")
        ttft = time.perf_counter() - t0
        metrics.ttft.observe(ttft)

        async def relay() -> AsyncIterator[str]:
            metrics.inflight.inc()
            n_tokens, outcome = 0, "ok"
            yield format_event("meta", {"request_id": request_id, "ttft_ms": round(ttft * 1000, 1)})
            try:
                async with asyncio.timeout(max(total_timeout - ttft, 0.1)):
                    event: tuple[str, dict[str, Any]] | None = first
                    while event is not None:
                        name, data = event
                        if name == "token":
                            n_tokens += 1
                            yield format_event("token", data)
                        elif name == "error":
                            outcome = "teacher_error"
                            yield format_event("error", data)
                            break
                        elif name == "done":
                            gen_s = time.perf_counter() - t0 - ttft
                            if n_tokens > 1 and gen_s > 0:
                                metrics.tokens_per_s.observe((n_tokens - 1) / gen_s)
                            yield format_event(
                                "done",
                                data
                                | {"gateway_ttft_ms": round(ttft * 1000, 1), "tokens": n_tokens},
                            )
                        event = await anext(events, None)
            except TimeoutError:
                outcome = "teacher_timeout"
                yield format_event(
                    "error",
                    {"code": "teacher_timeout", "detail": f"answer exceeded {total_timeout}s"},
                )
            except TeacherError as exc:
                outcome = "teacher_error"
                yield format_event("error", {"code": "teacher_error", "detail": str(exc)})
            finally:
                await events.aclose()
                metrics.tokens.inc(n_tokens)
                metrics.requests.labels("stream", outcome).inc()
                metrics.duration.observe(time.perf_counter() - t0)
                metrics.inflight.dec()
                log.info(
                    "stream finished",
                    extra={
                        "request_id": request_id,
                        "outcome": outcome,
                        "tokens": n_tokens,
                        "ttft_ms": round(ttft * 1000, 1),
                    },
                )

        return StreamingResponse(relay(), media_type=MEDIA_TYPE, headers=SSE_HEADERS)

    @app.post("/grade")
    async def grade(body: GradeIn, request: Request) -> Response:
        guard = check_question(body.question)
        if not guard.ok:
            return _error(400, guard.code or "invalid", guard.reason or "", metrics, "grade")
        try:
            out = await asyncio.wait_for(
                request.app.state.teacher.generate(
                    "grade",
                    question=guard.text,
                    faq_question=body.faq_question,
                    faq_answer=body.faq_answer,
                ),
                timeout=total_timeout,
            )
        except TimeoutError:
            return _error(504, "teacher_timeout", "grading timed out", metrics, "grade")
        except TeacherError as exc:
            return _error(503, "teacher_unavailable", str(exc), metrics, "grade")
        metrics.requests.labels("grade", "ok").inc()
        return JSONResponse(out)

    return app


def main() -> None:
    """Run with uvicorn (port from ``services.yaml``)."""
    from urllib.parse import urlparse

    import uvicorn

    port = urlparse(load_services()["gateway"]["url"]).port or 8002
    uvicorn.run(create_app(), host="0.0.0.0", port=port, log_config=None)


if __name__ == "__main__":
    main()
