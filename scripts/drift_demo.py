"""End-to-end drift demo against the running Compose stack.

Timeline (all automatic):

1. Replay the **drift week** (a new domain arrives) against the edge, paced so Prometheus
   sees it. The fallback ratio rises -> ``FallbackRatioHigh`` fires -> Alertmanager calls the
   console webhook -> the console starts **discovery** (``dvc repro distil``).
2. The **simulated reviewer** decides the new drafts (a human would use the console).
3. The console runs **retrain** (``dvc repro register``): dataset -> Ray Tune -> gate ->
   registry -> champion alias -> edge hot-reload.
4. Replay the **next week**: the fallback ratio drops (coverage recovered).

Writes ``reports/drift_demo.json`` and ``reports/plots/drift_demo.png``.

    python scripts/drift_demo.py --drift-week 3 --after-week 4 --qps 12
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime

import requests

from understudy.common.config import load_params, load_services, resolve
from understudy.data.simulate_traffic import simulate_online

S = load_services()
EDGE, PROM = S["edge"]["url"], S["prometheus"]["url"]
CONSOLE = S["console"]["url"]
events: list[dict] = []


def mark(what: str, **extra) -> None:
    """Record a timestamped event."""
    e = {"t": datetime.now(UTC).isoformat(), "event": what, **extra}
    events.append(e)
    print(json.dumps(e), flush=True)


def prom(q: str) -> float | None:
    """Return an instant Prometheus query value."""
    try:
        res = requests.get(f"{PROM}/api/v1/query", params={"query": q}, timeout=3).json()["data"][
            "result"
        ]
        return float(res[0]["value"][1]) if res else None
    except (requests.RequestException, KeyError, ValueError, IndexError):
        return None


def alert_state(name: str) -> str | None:
    """Return the alert's state (pending/firing) or None."""
    try:
        for a in requests.get(f"{PROM}/api/v1/alerts", timeout=3).json()["data"]["alerts"]:
            if a["labels"].get("alertname") == name:
                return a["state"]
    except (requests.RequestException, KeyError, ValueError):
        pass
    return None


def pipeline_status() -> dict:
    """Return the console's pipeline status."""
    return requests.get(f"{CONSOLE}/api/pipeline/status", timeout=5).json()


def wait_pipeline(name: str, timeout_s: float = 3600) -> dict:
    """Wait until the console's current run finishes; return the run record."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = pipeline_status()
        if st["current"] is None and st["last"] and st["last"]["name"] == name:
            return st["last"]
        time.sleep(5)
    raise TimeoutError(name)


def edge_version() -> str | None:
    """Return the edge's serving model version."""
    try:
        return requests.get(f"{EDGE}/info", timeout=3).json().get("model_version")
    except requests.RequestException:
        return None


def replay(params: dict, week: int, qps: float, label: str) -> dict:
    """Replay a week in a thread while sampling the Prometheus fallback ratio."""
    out: dict = {}
    samples: list[tuple[float, float | None]] = []

    def work() -> None:
        res = simulate_online(
            params, [week], "data/raw/clinc150", EDGE, concurrency=4, teacher_rate=0.0, qps=qps
        )
        out.update(res[week], week=week)

    t = threading.Thread(target=work)
    t0 = time.time()
    t.start()
    seen = None
    while t.is_alive():
        r = prom("understudy:fallback_ratio:rate2m")
        samples.append((round(time.time() - t0, 1), r))
        st = alert_state("FallbackRatioHigh")
        if st != seen:
            mark(
                f"alert {st or 'inactive'}", alert="FallbackRatioHigh", week=week, fallback_ratio=r
            )
            seen = st
        time.sleep(5)
    t.join()
    out["prometheus_fallback_ratio_samples"] = samples
    mark(f"{label} replayed", week=week, fallback_ratio=out["fallback_ratio"], routes=out["routes"])
    return out


def main() -> None:
    """Run the demo."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift-week", type=int, default=3)
    ap.add_argument("--after-week", type=int, default=4)
    ap.add_argument("--qps", type=float, default=12)
    args = ap.parse_args()
    params = load_params()
    result: dict = {"started": datetime.now(UTC).isoformat(), "champion_before": edge_version()}
    mark("start", champion=result["champion_before"])

    result["drift_week"] = replay(params, args.drift_week, args.qps, "drift week")

    # The alert -> webhook -> console path should have started discovery by now (or soon).
    t0 = time.time()
    while time.time() - t0 < 300:
        st = pipeline_status()
        cur = st["current"] or st["last"]
        if (
            cur
            and cur["name"] == "discover"
            and cur["reason"].startswith("alert")
            and cur["started_at"] >= result["started"]
        ):
            mark("discovery started by alert", reason=cur["reason"])
            break
        time.sleep(5)
    else:
        mark("no alert-triggered discovery within 5 min; starting it manually")
        requests.post(
            f"{CONSOLE}/api/pipeline/run",
            json={"target": "discover", "reason": "drift demo fallback"},
            timeout=10,
        )
    run = wait_pipeline("discover")
    mark("discovery finished", state=run["state"], duration_s=run["duration_s"])
    if run["state"] != "succeeded":
        print("\n".join(run.get("log_tail", [])[-20:]))
        raise SystemExit("discovery failed: see the console's pipeline history")
    result["distil"] = json.loads(resolve("reports/distil_metrics.json").read_text())
    result["mine"] = json.loads(resolve("reports/mine_metrics.json").read_text())

    out = subprocess.run(
        [sys.executable, "-m", "understudy.pipeline.approve", "auto", "--reviewer", "sim-reviewer"],
        capture_output=True,
        text=True,
        check=True,
    )
    decisions = json.loads(out.stdout.strip().splitlines()[-1])
    mark("simulated reviewer decided drafts", **decisions)
    result["review"] = decisions

    requests.post(
        f"{CONSOLE}/api/pipeline/run",
        json={"target": "retrain", "reason": "drift demo: approvals ready"},
        timeout=10,
    ).raise_for_status()
    mark("retrain started")
    run = wait_pipeline("retrain")
    gate = json.loads(resolve("reports/gate.json").read_text())
    reg = json.loads(resolve("reports/register.json").read_text())
    mark(
        "retrain finished",
        state=run["state"],
        duration_s=run["duration_s"],
        gate=gate["passed"],
        registered=reg.get("version"),
        edge_reloaded=(reg.get("edge") or {}).get("reloaded"),
    )
    result.update({"gate": gate, "register": reg, "champion_after": edge_version()})

    result["after_week"] = replay(params, args.after_week, args.qps, "post-retrain week")
    result["events"] = events
    result["finished"] = datetime.now(UTC).isoformat()
    resolve("reports/drift_demo.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    plot(result)
    print(json.dumps({k: result[k] for k in ("champion_before", "champion_after")}, indent=2))


def plot(result: dict) -> None:
    """Fallback ratio over the two replays, with event markers."""
    from understudy.common.plotting import SERIES, TEXT, TEXT_2, plt, style_axes

    fig, ax = plt.subplots(figsize=(10, 4.2))
    offset = 0.0
    for key, name, colour in (
        ("drift_week", f"week {result['drift_week'].get('week', '')}", SERIES[1]),
        ("after_week", "after retrain", SERIES[0]),
    ):
        s = [
            (t + offset, r)
            for t, r in result[key]["prometheus_fallback_ratio_samples"]
            if r is not None
        ]
        if s:
            xs, ys = zip(*s, strict=True)
            ax.plot(xs, ys, color=colour, linewidth=2)
            ax.text(
                xs[-1] + 5, ys[-1], f"{name}: {ys[-1]:.0%}", color=TEXT, fontsize=9, va="center"
            )
            offset = xs[-1] + 60
    ax.axhline(0.55, color=TEXT_2, linewidth=1, linestyle="--")
    ax.text(2, 0.565, "alert threshold 55%", color=TEXT_2, fontsize=8)
    ax.set_ylim(0, 1)
    style_axes(
        ax,
        "Fallback ratio seen by Prometheus: drift week, then after the retrained student",
        "seconds of replay (gap = discovery, review and retraining)",
        "fallback ratio (2 min)",
    )
    fig.tight_layout()
    out = resolve("reports/plots/drift_demo.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
