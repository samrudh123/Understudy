"""Rollback drill: roll the champion back while the edge is under load, then restore it.

Runs constant traffic against the edge (instant-answer questions), calls the console
rollback API in the middle, and records:

* how long the rollback took (alias move + edge hot reload),
* whether *any* request failed or returned an error during the switch,
* which model version answered before/after.

Writes ``reports/rollback_drill.json``.

    python scripts/rollback_drill.py --seconds 40 --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter

import httpx

from understudy.common.config import load_services, resolve

S = load_services()
EDGE, CONSOLE = S["edge"]["url"].rstrip("/"), S["console"]["url"].rstrip("/")
QUESTIONS = [
    "what is today's date",
    "can you roll a dice",
    "what timezone is chicago in",
    "are you a robot",
    "what's the date",
]


async def load(stop: asyncio.Event, concurrency: int, log: list) -> None:
    """Fire requests until ``stop``; record (t, status, version, latency)."""
    async with httpx.AsyncClient(timeout=10) as c:

        async def worker(i: int) -> None:
            n = 0
            while not stop.is_set():
                q = QUESTIONS[(i + n) % len(QUESTIONS)]
                n += 1
                t0 = time.perf_counter()
                try:
                    r = await c.post(
                        EDGE + "/ask",
                        json={
                            "question": q,
                            "stream": False,
                            "use_teacher": False,
                            "simulated": {"test": True, "label": "drill"},
                        },
                    )
                    body = r.json()
                    log.append(
                        (
                            time.time(),
                            r.status_code,
                            body.get("model_version"),
                            (time.perf_counter() - t0) * 1000,
                            body.get("route"),
                        )
                    )
                except (httpx.HTTPError, ValueError) as exc:
                    log.append(
                        (
                            time.time(),
                            f"error:{type(exc).__name__}",
                            None,
                            (time.perf_counter() - t0) * 1000,
                            None,
                        )
                    )

        await asyncio.gather(*(worker(i) for i in range(concurrency)))


async def main(seconds: float, concurrency: int) -> dict:
    """Run the drill."""
    stop, log = asyncio.Event(), []
    task = asyncio.create_task(load(stop, concurrency, log))
    await asyncio.sleep(seconds / 3)
    async with httpx.AsyncClient(timeout=60) as c:
        before = (await c.get(EDGE + "/info")).json().get("model_version")
        t_rb = time.time()
        rb = (await c.post(CONSOLE + "/api/models/rollback", json={})).json()
        t_rb_done = time.time()
        await asyncio.sleep(seconds / 3)
        restore = (await c.post(CONSOLE + "/api/models/promote", json={"version": before})).json()
        await asyncio.sleep(seconds / 3)
    stop.set()
    await task
    statuses = Counter(str(s) for _, s, *_ in log)
    versions = Counter(str(v) for *_, v, _, _ in [(a, b, c, d, e) for a, b, c, d, e in log])
    during = [x for x in log if t_rb <= x[0] <= t_rb_done + 1]
    report = {
        "requests": len(log),
        "statuses": dict(statuses),
        "failed_requests": sum(1 for _, s, *_ in log if s != 200),
        "versions_served": dict(versions),
        "version_before": before,
        "rollback": rb,
        "rollback_seconds_wall": round(t_rb_done - t_rb, 2),
        "requests_during_switch": len(during),
        "max_latency_ms_during_switch": round(max((x[3] for x in during), default=0), 1),
        "restored": restore,
    }
    resolve("reports/rollback_drill.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    print(json.dumps(asyncio.run(main(a.seconds, a.concurrency)), indent=2))
