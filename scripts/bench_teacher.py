"""Measure the teacher's latency baseline through the gateway.

Sends N CLINC test questions one at a time to ``POST /stream`` and records TTFT and
decode speed. Writes ``reports/teacher_baseline.json``.

    python scripts/bench_teacher.py --n 20
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import httpx
import numpy as np

from understudy.common.config import load_params, load_services, resolve
from understudy.common.sse import parse_events
from understudy.data.clinc import load_clinc


def main() -> None:
    """Run the benchmark."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    params, services = load_params(), load_services()
    clinc = load_clinc(resolve("data/raw/clinc150"))
    rng = np.random.default_rng(params["seed"])
    test = [t for t, _ in clinc.splits["test"]]
    questions = [test[i] for i in rng.choice(len(test), size=args.n, replace=False)]

    rows = []
    with httpx.Client(timeout=120) as client:
        for q in questions:
            t0 = time.perf_counter()
            r = client.post(services["gateway"]["url"] + "/stream", json={"question": q})
            wall = time.perf_counter() - t0
            if r.status_code != 200:
                rows.append({"question": q, "status": r.status_code})
                continue
            ev = dict(parse_events(r.text)[-1:])
            done = ev.get("done", {})
            ttft = done.get("gateway_ttft_ms", float("nan")) / 1000
            toks = done.get("tokens", 0)
            rows.append(
                {
                    "question": q,
                    "status": 200,
                    "ttft_s": ttft,
                    "tokens": toks,
                    "total_s": wall,
                    "tokens_per_s": (toks - 1) / max(wall - ttft, 1e-6) if toks > 1 else None,
                }
            )
    ok = [r for r in rows if r["status"] == 200]
    ttfts = np.array([r["ttft_s"] for r in ok])
    tps = np.array([r["tokens_per_s"] for r in ok if r["tokens_per_s"]])
    summary = {
        "model": params["teacher"]["model_file"],
        "machine": {"cpu": platform.processor() or platform.machine(), "threads_per_replica": 8},
        "n": len(rows),
        "ok": len(ok),
        "ttft_p50_s": round(float(np.percentile(ttfts, 50)), 3),
        "ttft_p95_s": round(float(np.percentile(ttfts, 95)), 3),
        "tokens_per_s_mean": round(float(tps.mean()), 2),
        "total_p50_s": round(float(np.median([r["total_s"] for r in ok])), 2),
        "answer_tokens_mean": round(float(np.mean([r["tokens"] for r in ok])), 1),
        "runs": rows,
    }
    out = resolve("reports/teacher_baseline.json")
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
