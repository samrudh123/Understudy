"""Assemble ``docs/test_report.md`` from the measured artifacts (nothing is typed by hand).

Inputs (produce them first):
    reports/tests/junit-{unit,integration,e2e}.xml   pytest --junitxml=...
    reports/edge_latency.json                        scripts/bench_edge.py
    reports/teacher_baseline.json                    scripts/bench_teacher.py
    reports/gate.json, reports/train_metrics.json    dvc repro
    reports/drift_demo.json                          scripts/drift_demo.py
    reports/rollback_drill.json                      scripts/rollback_drill.py
    reports/load/edge_stats.csv                      make load
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def load(name: str):
    """Read a JSON report or return None."""
    p = R / name
    return json.loads(p.read_text()) if p.is_file() else None


def junit(level: str) -> dict | None:
    """Summarise a JUnit XML file."""
    p = R / "tests" / f"junit-{level}.xml"
    if not p.is_file():
        return None
    root = ET.parse(p).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    tot = {
        k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")
    }
    tot["time"] = round(sum(float(s.get("time", 0)) for s in suites), 1)
    tot["passed"] = tot["tests"] - tot["failures"] - tot["errors"] - tot["skipped"]
    failed = [
        f"{c.get('classname')}::{c.get('name')}"
        for s in suites
        for c in s.iter("testcase")
        if c.find("failure") is not None or c.find("error") is not None
    ]
    tot["failed_cases"] = failed
    return tot


def verdict(ok: bool | None) -> str:
    """Render a pass/fail cell."""
    return "–" if ok is None else ("**PASS**" if ok else "**FAIL**")


def pct(v) -> str:
    """Format a ratio as a percentage."""
    return "–" if v is None else f"{100 * float(v):.1f}%"


def main() -> None:
    """Write the report."""
    edge, teacher, gate = (
        load("edge_latency.json"),
        load("teacher_baseline.json"),
        load("gate.json"),
    )
    demo, drill = load("drift_demo.json"), load("rollback_drill.json")
    train = load("train_metrics.json")
    lines = [
        "# Test report",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC} by `scripts/test_report.py` from the artifacts in "
        "`reports/`. Plan and criteria: [test_plan.md](test_plan.md).",
        "",
        "## Test runs",
        "",
        "| Level | Passed | Failed | Errors | Skipped | Total | Time (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    failed_cases = []
    for level in ("unit", "integration", "e2e"):
        j = junit(level)
        if j:
            lines.append(
                f"| {level} | {j['passed']} | {j['failures']} | {j['errors']} | {j['skipped']} | "
                f"{j['tests']} | {j['time']} |"
            )
            failed_cases += j["failed_cases"]
        else:
            lines.append(f"| {level} | – | – | – | – | not run | – |")
    load_rows = []
    lp = R / "load" / "edge_stats.csv"
    if lp.is_file():
        with lp.open() as fh:
            load_rows = list(csv.DictReader(fh))
    agg = next((r for r in load_rows if r.get("Name") == "Aggregated"), None)
    if agg:
        lines.append(
            f"| load (Locust) | {int(agg['Request Count']) - int(agg['Failure Count'])} | "
            f"{agg['Failure Count']} | – | – | {agg['Request Count']} | – |"
        )
    if failed_cases:
        lines += ["", "Failed cases:", ""] + [f"* `{c}`" for c in failed_cases]

    rows = []
    if edge:
        seq, conc = (
            edge["sequential"],
            next(v for k, v in edge.items() if k.startswith("concurrent")),
        )
        ok = seq["p95_ms"] < 100 and conc["p95_ms"] < 100
        rows.append(
            (
                "A1",
                "Instant-answer p95 on the 1 CPU / 1 GB edge",
                "< 100 ms",
                f"{seq['p95_ms']} ms sequential; {conc['p95_ms']} ms with 8 concurrent clients",
                verdict(ok),
            )
        )
        fb = edge.get("fallback") or {}
        if fb.get("filler_card_ms_max") is not None:
            rows.append(
                (
                    "A2",
                    "Fallback: time to the 'while you wait' card",
                    "< 1 s",
                    f"max {fb['filler_card_ms_max']} ms; first token p50 {fb.get('first_token_ms_p50')} ms",
                    verdict(fb["filler_card_ms_max"] < 1000),
                )
            )
    if teacher:
        rows.append(
            (
                "A2b",
                "Teacher time to first token (reported)",
                "reported",
                f"p50 {teacher['ttft_p50_s']} s, p95 {teacher['ttft_p95_s']} s; {teacher['tokens_per_s_mean']} tok/s",
                "reported",
            )
        )
    if gate and gate.get("challenger"):
        ch = gate["challenger"]
        rows.append(
            (
                "A3",
                "Instant-answer precision, held-out queries (traffic mix)",
                "≥ 95%",
                f"{pct(ch['traffic_precision'])} (uniform-mix stress test: {pct(ch['instant_precision'])})",
                verdict(ch["traffic_precision"] >= 0.95),
            )
        )
        rows.append(
            (
                "A4",
                "Out-of-scope questions answered by the student",
                "≤ 5%",
                pct(ch["oos_false_accept"]),
                verdict(ch["oos_false_accept"] <= 0.05),
            )
        )
    if demo:
        before = demo["drift_week"]["fallback_ratio"]
        after = demo["after_week"]["fallback_ratio"]
        rows.append(
            (
                "A5",
                "Coverage recovers after retraining",
                "recovers",
                f"fallback ratio {pct(before)} in drift week → {pct(after)} after retraining "
                f"(student v{demo['champion_before']} → v{demo['champion_after']})",
                verdict(after < before),
            )
        )
        ev = {e["event"]: e["t"] for e in demo["events"]}
        if "alert firing" in ev:
            t0 = datetime.fromisoformat(ev["start"])
            t1 = datetime.fromisoformat(ev["alert firing"])
            secs = (t1 - t0).total_seconds()
            dur = demo["drift_week"].get("seconds")
            rows.append(
                (
                    "A6",
                    "New topic → drift alert",
                    "within one simulated week",
                    f"alert firing {secs:.0f} s into the drift week's {dur} s replay",
                    verdict(dur is None or secs <= dur),
                )
            )
    if drill:
        rb = drill["rollback_seconds_wall"]
        ok = rb < 60 and drill["failed_requests"] == 0
        rows.append(
            (
                "A7",
                "Rollback to previous student under load",
                "< 1 min, no downtime",
                f"{rb} s (alias move + edge hot reload); {drill['failed_requests']} failed of "
                f"{drill['requests']} requests; versions served {drill['versions_served']}",
                verdict(ok),
            )
        )
    if agg:
        fail = int(agg["Failure Count"]) / max(int(agg["Request Count"]), 1)
        rows.append(
            (
                "A8",
                "API error rate under load",
                "< 5%",
                f"{pct(fail)} of {agg['Request Count']} requests; median {agg['Median Response Time']} ms, "
                f"p95 {agg['95%']} ms",
                verdict(fail < 0.05),
            )
        )
    rows.append(
        (
            "A9",
            "Rebuild the student from a fresh clone",
            "one command: `dvc repro`",
            "pinned dataset/model SHA-256 + dvc.lock; CI runs `dvc repro`",
            "see CI log",
        )
    )
    lines += [
        "",
        "## Acceptance criteria",
        "",
        "| # | Criterion | Target | Measured | Result |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows]

    if train:
        lines += [
            "",
            "## Current student (last training run)",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
        for k in (
            "n_faqs",
            "tau",
            "val_instant_precision",
            "val_coverage",
            "test_traffic_precision",
            "test_traffic_coverage",
            "test_instant_precision",
            "test_oos_false_accept",
            "test_uncovered_false_accept",
            "latency_p95_ms",
            "model_size_mb",
            "tune_trials",
            "tune_feasible_trials",
        ):
            if k in train:
                lines.append(f"| `{k}` | {train[k]} |")
    if demo:
        lines += [
            "",
            "## Drift demo timeline",
            "",
            "| Time (UTC) | Event | Details |",
            "|---|---|---|",
        ]
        for e in demo["events"]:
            details = ", ".join(f"{k}={v}" for k, v in e.items() if k not in ("t", "event"))
            lines.append(f"| {e['t'][11:19]} | {e['event']} | {details} |")
        lines += ["", "![drift demo](../reports/plots/drift_demo.png)"]
    out = ROOT / "docs" / "test_report.md"
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
