"""Edge instant-answer latency, measured from the client (the edge runs under its Compose limits).

Replays CLINC test questions of intents the bank covers, so most route to the student.
Reports client-side p50/p95/p99 for instant answers, sequential and concurrent.
Writes ``reports/edge_latency.json``.

    python scripts/bench_edge.py --n 300 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx
import numpy as np
import pandas as pd

from understudy.common.config import load_services, resolve


async def run(url: str, questions: list[str], concurrency: int) -> list[tuple[str, float]]:
    """Fire ``questions`` at ``/ask`` (non-streaming) and return (route, latency_ms)."""
    sem = asyncio.Semaphore(concurrency)
    out: list[tuple[str, float]] = []

    async def one(client: httpx.AsyncClient, q: str) -> None:
        async with sem:
            t0 = time.perf_counter()
            r = await client.post(
                url + "/ask",
                json={
                    "question": q,
                    "stream": False,
                    "use_teacher": False,
                    "simulated": {"test": True, "label": "bench"},
                },
            )
            out.append((r.json().get("route", "?"), (time.perf_counter() - t0) * 1000))

    async with httpx.AsyncClient(timeout=30) as client:
        await asyncio.gather(*(one(client, q) for q in questions))
    return out


def summarise(rows: list[tuple[str, float]]) -> dict:
    """Percentiles of instant-answer latency."""
    lat = np.array([ms for route, ms in rows if route == "student"])
    return {
        "requests": len(rows),
        "instant_answers": int(len(lat)),
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "p99_ms": round(float(np.percentile(lat, 99)), 2),
        "max_ms": round(float(lat.max()), 2),
    }


def time_to_filler(url: str, questions: list[str]) -> dict:
    """For fallbacks: time until the first SSE event (route + waiting card) and first token."""
    meta_ms, token_ms = [], []
    with httpx.Client(timeout=120) as c:
        for q in questions:
            t0 = time.perf_counter()
            with c.stream("POST", url + "/ask", json={"question": q, "stream": True}) as r:
                got_meta = False
                for line in r.iter_lines():
                    if line.startswith("event: meta") and not got_meta:
                        meta_ms.append((time.perf_counter() - t0) * 1000)
                        got_meta = True
                    elif line.startswith("event: token"):
                        token_ms.append((time.perf_counter() - t0) * 1000)
                        break
                    elif line.startswith("event: error"):
                        break
    return {
        "questions": len(questions),
        "filler_card_ms_max": round(max(meta_ms), 1) if meta_ms else None,
        "filler_card_ms_p50": round(float(np.median(meta_ms)), 1) if meta_ms else None,
        "first_token_ms_p50": round(float(np.median(token_ms)), 1) if token_ms else None,
        "first_token_ms_max": round(max(token_ms), 1) if token_ms else None,
        "target_filler_ms": 1000,
    }


FALLBACK_QUESTIONS = [
    "how do i fix a squeaky door hinge",
    "what is a good beginner guitar",
    "how long should bread dough rise",
    "why do cats purr",
    "how do i remove a coffee stain from a shirt",
]


def main() -> None:
    """Run sequential and concurrent passes."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--url", default=None)
    args = ap.parse_args()
    url = (args.url or load_services()["edge"]["url"]).rstrip("/")
    test = pd.read_parquet(resolve("data/dataset/test.parquet"))
    qs = (
        test[test["kind"] == "covered"]["text"]
        .sample(args.n, replace=True, random_state=0)
        .tolist()
    )
    info = httpx.get(url + "/info", timeout=5).json()
    report = {
        "edge_model_version": info.get("model_version"),
        "limits": "docker compose: cpus=1.0, memory=1g",
        "sequential": summarise(asyncio.run(run(url, qs, 1))),
        f"concurrent_{args.concurrency}": summarise(asyncio.run(run(url, qs, args.concurrency))),
        "target_p95_ms": 100,
        "fallback": time_to_filler(url, FALLBACK_QUESTIONS),
    }
    resolve("reports/edge_latency.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
