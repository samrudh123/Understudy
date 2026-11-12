"""Teacher service: llama.cpp behind Ray Serve.

Routes (prefix ``/teacher``):

* ``GET  /health``    liveness + which model / prompt version is loaded
* ``POST /stream``    ``{"question"}`` -> SSE ``token`` ... ``done`` (live fallbacks)
* ``POST /generate``  ``{"task": "answer"|"distil"|"grade", ...}`` -> JSON (non-streaming)

Each replica holds one llama.cpp model, which is not thread-safe, so a replica
serves one request at a time (``max_ongoing_requests=1``). Ray Serve queues the
rest, which gives back-pressure for free. Scale with ``--replicas``; each replica
gets ``--threads`` CPU threads.

Run locally::

    python -m understudy.teacher.serve --replicas 1 --threads 8
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from understudy.common.config import load_params, resolve
from understudy.common.logs import configure_logging, get_logger
from understudy.common.sse import MEDIA_TYPE, SSE_HEADERS, format_event
from understudy.teacher.engine import Engine, make_engine
from understudy.teacher.prompts import DISTIL_SCHEMA, GRADE_SCHEMA, load_prompts

log = get_logger(__name__)


class StreamRequest(BaseModel):
    """A live fallback question (already validated and guarded by the gateway)."""

    question: str = Field(min_length=1, max_length=2000)
    request_id: str | None = None


class GenerateRequest(BaseModel):
    """A non-streaming task."""

    task: Literal["answer", "distil", "grade"]
    question: str | None = None
    examples: list[str] | None = None
    faq_question: str | None = None
    faq_answer: str | None = None


class TeacherCore:
    """Framework-free teacher logic, so it can be unit-tested without Ray."""

    def __init__(self, engine: Engine, prompt_file: str) -> None:
        self.engine = engine
        self.prompts = load_prompts(prompt_file)

    def info(self) -> dict[str, Any]:
        """Identify the loaded model and prompt version."""
        return {"model": self.engine.model_name, "prompt_version": self.prompts.version}

    def stream_events(self, req: StreamRequest):
        """Yield SSE strings for one answer (blocking generator)."""
        t0 = time.perf_counter()
        ttft, n = None, 0
        try:
            for piece in self.engine.stream(
                self.prompts.answer(req.question), **self.prompts.settings("answer")
            ):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
                yield format_event("token", {"t": piece})
        except Exception as exc:  # surface model errors to the gateway as an event
            log.exception("generation failed", extra={"request_id": req.request_id})
            yield format_event("error", {"code": "generation_failed", "detail": str(exc)})
            return
        total = time.perf_counter() - t0
        yield format_event(
            "done",
            {
                "tokens": n,
                "ttft_ms": round((ttft or total) * 1000, 1),
                "total_ms": round(total * 1000, 1),
                **self.info(),
            },
        )

    def generate(self, req: GenerateRequest) -> dict[str, Any]:
        """Run a non-streaming task and return its JSON result."""
        p, t0 = self.prompts, time.perf_counter()
        if req.task == "answer":
            if not req.question:
                raise ValueError("answer needs 'question'")
            text = "".join(self.engine.stream(p.answer(req.question), **p.settings("answer")))
            result: dict[str, Any] = {"answer": text}
        elif req.task == "distil":
            if not req.examples:
                raise ValueError("distil needs 'examples'")
            result = self.engine.complete_json(
                p.distil(req.examples), DISTIL_SCHEMA, **p.settings("distil")
            )
        else:
            if not (req.question and req.faq_question and req.faq_answer):
                raise ValueError("grade needs 'question', 'faq_question' and 'faq_answer'")
            result = self.engine.complete_json(
                p.grade(req.question, req.faq_question, req.faq_answer),
                GRADE_SCHEMA,
                **p.settings("grade"),
            )
        return {
            "result": result,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            **self.info(),
        }


def build_app(core_factory) -> FastAPI:
    """FastAPI routes shared by the Ray Serve deployment and the plain-uvicorn test mode."""
    app = FastAPI(title="Understudy teacher", version="1.0")
    state: dict[str, TeacherCore] = {}

    def core() -> TeacherCore:
        if "core" not in state:
            state["core"] = core_factory()
        return state["core"]

    app.state.get_core = core  # lets the deployment load the model eagerly

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", **core().info()}

    @app.post("/stream")
    async def stream(req: StreamRequest) -> StreamingResponse:
        return StreamingResponse(
            iterate_in_threadpool(core().stream_events(req)),
            media_type=MEDIA_TYPE,
            headers=SSE_HEADERS,
        )

    @app.post("/generate")
    async def generate(req: GenerateRequest) -> dict[str, Any]:
        try:
            return await run_in_threadpool(core().generate, req)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def _core_from_params(threads: int | None) -> TeacherCore:
    t = load_params()["teacher"]
    model = resolve(t["model_dir"]) / t["model_file"]
    engine = make_engine(model, n_ctx=int(t["n_ctx"]), n_threads=threads)
    return TeacherCore(engine, t["prompt_file"])


def make_deployment(replicas: int, threads: int):
    """Build the Ray Serve application."""
    from ray import serve

    fastapi_app = build_app(lambda: _core_from_params(threads))

    @serve.deployment(
        name="teacher",
        num_replicas=replicas,
        max_ongoing_requests=1,
        ray_actor_options={"num_cpus": threads},
        health_check_period_s=10,
    )
    @serve.ingress(fastapi_app)
    class TeacherDeployment:
        def __init__(self) -> None:
            configure_logging()
            fastapi_app.state.get_core()  # load the model at replica start, not on request 1

    return TeacherDeployment.bind()


def main(argv: list[str] | None = None) -> None:
    """Start Ray Serve and block."""
    ap = argparse.ArgumentParser(description="Serve the teacher with Ray Serve.")
    ap.add_argument("--replicas", type=int, default=int(os.environ.get("TEACHER_REPLICAS", 1)))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("TEACHER_THREADS", 8)))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    configure_logging()

    import ray
    from ray import serve

    ray.init(
        num_cpus=max(os.cpu_count() or 1, args.replicas * args.threads),
        include_dashboard=False,
        _metrics_export_port=int(os.environ.get("RAY_METRICS_PORT", 8080)),
    )
    serve.start(http_options={"host": args.host, "port": args.port})
    serve.run(make_deployment(args.replicas, args.threads), route_prefix="/teacher", blocking=True)


if __name__ == "__main__":
    main()
