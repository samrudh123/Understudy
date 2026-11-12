"""Simulated live traffic: replay CLINC150 as weekly batches of user questions.

How a week ``w`` is generated (all knobs in ``params.yaml: simulation``):

* **Active domains** = ``initial_domains`` plus every domain scheduled at or before ``w``.
  A domain held back until week 3 is the drift event.
* **Intent popularity is Zipf-distributed** over a fixed, seeded ranking of all 150
  intents. A few questions dominate, as in a real helpdesk.
* **New-domain surge**: in the week a domain arrives and the week after, it takes
  ``new_domain_share`` of the in-scope traffic (think "registration week").
* ``oos_rate`` of questions are out of scope. ``pii_injection_rate`` of questions get
  a fake e-mail, phone or roll number appended, to exercise PII scrubbing.
* Utterances come from CLINC **train/val only**. Test utterances stay unseen, for evaluation.

``offline`` mode (this module) writes the batch straight to the query-log
directory through :class:`QueryLogger`. Routing is an **oracle**: a question is
"answered by the student" if its true intent is already in the approved answer
bank. That lets the data pipeline be developed and tested before any service exists.
``online`` mode replays the same batches against the real edge ``POST /ask``. The live
student routes each question, and the edge logs it with the simulation ground truth
attached. Only ``teacher_sample_rate`` of the fallbacks are actually sent to the CPU
teacher. The rest are routed and logged but not answered, which keeps a 2,000-question
week to about a minute while TTFT metrics stay live.
"""

from __future__ import annotations

import argparse
import json
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

from understudy.common.config import load_params, resolve
from understudy.common.logs import get_logger
from understudy.common.schemas import ApprovalRecord, QueryLogRecord, Route
from understudy.data.clinc import OOS_LABEL, Clinc, load_clinc
from understudy.querylog.logger import QueryLogger

log = get_logger(__name__)

SURGE_WEEKS = 2
_PII_TEMPLATES = (
    " my email is {name}{n}@example.com",
    " call me on +91 98{n:03d}{m:05d}",
    " my roll number is DA2{d}B{n:03d}",
)
_NAMES = ("asha", "ravi", "meera", "arjun", "kavya", "rahul")


@dataclass
class WeekPlan:
    """What a simulated week looks like: active domains and per-intent traffic weights."""

    week: int
    active_domains: list[str]
    new_domains: list[str]
    weights: dict[str, float]  # intent -> probability among in-scope traffic


def domains_active(sim: dict, week: int) -> tuple[list[str], list[str]]:
    """Return (active domains, domains still in their surge window) for ``week``."""
    active = list(sim["initial_domains"])
    surging: list[str] = []
    for start, doms in sorted((int(k), v) for k, v in (sim.get("domain_schedule") or {}).items()):
        if start <= week:
            active.extend(d for d in doms if d not in active)
            if week - start < SURGE_WEEKS:
                surging.extend(doms)
    return active, surging


def plan_week(clinc: Clinc, sim: dict, seed: int, week: int) -> WeekPlan:
    """Compute the per-intent traffic distribution for ``week``."""
    active, surging = domains_active(sim, week)
    unknown = set(active) - set(clinc.domains)
    if unknown:
        raise ValueError(f"Unknown domains in simulation config: {sorted(unknown)}")

    # Fixed global popularity ranking (independent of the week) -> Zipf scores.
    all_intents = sorted(clinc.intent_to_domain)
    order = np.random.default_rng(seed).permutation(len(all_intents))
    score = {all_intents[i]: 1.0 / (rank + 1) ** sim["zipf_s"] for rank, i in enumerate(order)}

    def normalised(intents: list[str]) -> dict[str, float]:
        total = sum(score[i] for i in intents)
        return {i: score[i] / total for i in intents} if total else {}

    surge_intents = [i for d in surging for i in clinc.domains[d]]
    base_intents = [i for d in active if d not in surging for i in clinc.domains[d]]
    if surge_intents and base_intents:
        share = sim["new_domain_share"]
        weights = {i: p * (1 - share) for i, p in normalised(base_intents).items()}
        weights.update({i: p * share for i, p in normalised(surge_intents).items()})
    else:
        weights = normalised(surge_intents or base_intents)
    return WeekPlan(week, active, surging, weights)


def load_covered_intents(approvals_dir: Path) -> dict[str, str]:
    """Return ``source_intent -> faq_id`` for every approved FAQ (oracle routing)."""
    covered: dict[str, str] = {}
    if not approvals_dir.is_dir():
        return covered
    for f in sorted(approvals_dir.glob("*.json")):
        rec = ApprovalRecord.model_validate_json(f.read_text(encoding="utf-8"))
        if rec.decision == "approved" and rec.source_intent:
            covered[rec.source_intent] = rec.faq_id
    return covered


def generate_week(
    clinc: Clinc, sim: dict, seed: int, week: int, covered: dict[str, str]
) -> list[QueryLogRecord]:
    """Generate one week of simulated, oracle-routed query-log records (time-ordered)."""
    plan = plan_week(clinc, sim, seed, week)
    rng = np.random.default_rng([seed, week])
    pool = clinc.utterances(("train", "val"))
    oos_pool = clinc.utterances(("oos_train", "oos_val")).get(OOS_LABEL, [])
    i2d = clinc.intent_to_domain

    n = int(sim["queries_per_week"])
    intents = list(plan.weights)
    probs = np.array([plan.weights[i] for i in intents])
    start = datetime.combine(date.fromisoformat(sim["start_date"]), datetime.min.time(), UTC)
    week_start = start + timedelta(days=7 * week)
    offsets = np.sort(rng.uniform(0, 7 * 24 * 3600, size=n))

    records: list[QueryLogRecord] = []
    for k in range(n):
        if rng.random() < sim["oos_rate"] and oos_pool:
            label, domain = OOS_LABEL, None
            text = oos_pool[rng.integers(len(oos_pool))]
        else:
            label = intents[rng.choice(len(intents), p=probs)]
            domain = i2d[label]
            utts = pool[label]
            text = utts[rng.integers(len(utts))]
        if rng.random() < sim["pii_injection_rate"]:
            tpl = _PII_TEMPLATES[rng.integers(len(_PII_TEMPLATES))]
            text += tpl.format(
                name=_NAMES[rng.integers(len(_NAMES))],
                n=int(rng.integers(1000)),
                m=int(rng.integers(100000)),
                d=int(rng.integers(10)),
            )

        if label in covered:
            route, faq, conf = Route.STUDENT, covered[label], float(rng.uniform(0.8, 1.0))
        else:
            route, faq, conf = Route.TEACHER, None, float(rng.uniform(0.05, 0.6))
        records.append(
            QueryLogRecord(
                request_id=str(
                    uuid.UUID(int=int(rng.integers(2**63)) << 64 | int(rng.integers(2**63)))
                ),
                ts=week_start + timedelta(seconds=float(offsets[k])),
                text=text,
                route=route,
                matched_faq_id=faq,
                confidence=round(conf, 4),
                student_version="oracle" if covered else None,
                source="simulated",
                sim_week=week,
                sim_label=label,
                sim_domain=domain,
            )
        )
    return records


def simulate_offline(
    params: dict,
    weeks: list[int],
    raw_dir: str | Path,
    out_dir: str | Path,
    approvals_dir: str | Path,
) -> dict[int, dict]:
    """Write ``sim-week-XX.jsonl`` for each week. Re-running a week overwrites it (idempotent)."""
    clinc = load_clinc(resolve(raw_dir))
    sim, seed = params["simulation"], int(params["seed"])
    covered = load_covered_intents(resolve(approvals_dir))
    out = resolve(out_dir)
    summary: dict[int, dict] = {}
    for w in weeks:
        name = f"sim-week-{w:02d}.jsonl"
        (out / name).unlink(missing_ok=True)
        qlog = QueryLogger(out, file_name=name)
        records = generate_week(clinc, sim, seed, w, covered)
        qlog.log_many(records)
        plan = plan_week(clinc, sim, seed, w)
        n_fallback = sum(r.route != Route.STUDENT for r in records)
        summary[w] = {
            "file": name,
            "queries": len(records),
            "fallback_ratio": round(n_fallback / len(records), 4) if records else None,
            "active_domains": plan.active_domains,
            "surging_domains": plan.new_domains,
            "pii_scrubbed": dict(qlog.pii_counts),
        }
        log.info("simulated week written", extra={"week": w, **summary[w]})
    return summary


async def _replay(
    records: list[QueryLogRecord],
    edge_url: str,
    concurrency: int,
    teacher_rate: float,
    seed: int,
    qps: float | None,
) -> dict[str, int]:
    import asyncio

    import httpx

    rng = np.random.default_rng(seed)
    use_teacher = rng.random(len(records)) < teacher_rate
    routes: Counter[str] = Counter()
    sem = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient, i: int, r: QueryLogRecord) -> None:
        if qps:
            await asyncio.sleep(i / qps)
        async with sem:
            body = {
                "question": r.text,
                "stream": False,
                "request_id": r.request_id,
                "use_teacher": bool(use_teacher[i]),
                "simulated": {"week": r.sim_week, "label": r.sim_label, "domain": r.sim_domain},
            }
            try:
                resp = await client.post(edge_url.rstrip("/") + "/ask", json=body)
                routes[resp.json().get("route", f"http_{resp.status_code}")] += 1
            except (httpx.HTTPError, ValueError):
                routes["client_error"] += 1

    async with httpx.AsyncClient(timeout=120) as client:
        await asyncio.gather(*(one(client, i, r) for i, r in enumerate(records)))
    return dict(routes)


def simulate_online(
    params: dict,
    weeks: list[int],
    raw_dir: str | Path,
    edge_url: str,
    concurrency: int = 8,
    teacher_rate: float = 0.01,
    qps: float | None = None,
) -> dict[int, dict]:
    """Replay simulated weeks against the edge ``/ask`` endpoint."""
    import asyncio

    clinc = load_clinc(resolve(raw_dir))
    sim, seed = params["simulation"], int(params["seed"])
    summary: dict[int, dict] = {}
    for w in weeks:
        records = generate_week(clinc, sim, seed, w, covered={})
        t0 = datetime.now(UTC)
        routes = asyncio.run(_replay(records, edge_url, concurrency, teacher_rate, seed + w, qps))
        secs = (datetime.now(UTC) - t0).total_seconds()
        total = sum(routes.values())
        summary[w] = {
            "queries": total,
            "routes": routes,
            "seconds": round(secs, 1),
            "fallback_ratio": round(1 - routes.get("student", 0) / total, 4) if total else None,
            "active_domains": plan_week(clinc, sim, seed, w).active_domains,
        }
        log.info("online week replayed", extra={"week": w, **summary[w]})
    return summary


def parse_weeks(spec: str) -> list[int]:
    """Parse ``"3"``, ``"0-4"`` or ``"0,2,5"`` into a list of week numbers."""
    weeks: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            weeks.extend(range(int(a), int(b) + 1))
        elif part:
            weeks.append(int(part))
    if not weeks or min(weeks) < 0:
        raise ValueError(f"Invalid week spec: {spec!r}")
    return sorted(set(weeks))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate simulated weekly query traffic.")
    parser.add_argument("--weeks", default="0", help='e.g. "0", "0-2", "0,3"')
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--edge-url", default=None, help="online mode (default: services.yaml)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--teacher-rate",
        type=float,
        default=0.01,
        help="online: share of fallbacks actually sent to the teacher",
    )
    parser.add_argument("--qps", type=float, default=None, help="online: pace requests")
    parser.add_argument("--raw-dir", default="data/raw/clinc150")
    parser.add_argument("--out-dir", default="data/querylogs")
    parser.add_argument("--approvals-dir", default="data/approvals")
    args = parser.parse_args(argv)
    if args.mode == "online":
        from understudy.common.config import load_services

        url = args.edge_url or load_services()["edge"]["url"]
        summary = simulate_online(
            load_params(),
            parse_weeks(args.weeks),
            args.raw_dir,
            url,
            args.concurrency,
            args.teacher_rate,
            args.qps,
        )
    else:
        summary = simulate_offline(
            load_params(), parse_weeks(args.weeks), args.raw_dir, args.out_dir, args.approvals_dir
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
