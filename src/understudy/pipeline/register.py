"""``register`` stage (and promotion/rollback CLI) for the MLflow Model Registry.

Stage behaviour (``python -m understudy.pipeline.register``):
    If ``reports/gate.json`` passed: register the run's pyfunc as a new version of
    ``understudy-student``, tag it with lineage and gate metrics, and set the
    ``challenger`` alias. With ``register.auto_promote`` it also becomes ``champion`` and
    the edge is told to reload. If the gate failed, nothing is registered.

Operations::

    python -m understudy.pipeline.register status
    python -m understudy.pipeline.register rollback [--to 3]
    python -m understudy.pipeline.register promote 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import requests

from understudy.common.config import load_params, load_services, resolve
from understudy.common.logs import get_logger
from understudy.common.metrics import stage_run
from understudy.common.tracking import setup_mlflow
from understudy.student import registry

log = get_logger(__name__)

REPORT = "reports/register.json"


def reload_edge(timeout: float = 10.0) -> dict[str, Any]:
    """Ask the edge to re-resolve the champion alias (best effort)."""
    edge = load_services()["edge"]
    url = edge["url"].rstrip("/") + "/admin/reload"
    headers = {}
    if token := os.environ.get("UNDERSTUDY_ADMIN_TOKEN") or edge.get("admin_token"):
        headers["X-Admin-Token"] = token
    try:
        r = requests.post(url, headers=headers, timeout=timeout)
        return {
            "reloaded": r.ok,
            "status": r.status_code,
            "body": r.json() if r.ok else r.text[:200],
        }
    except requests.RequestException as exc:
        return {"reloaded": False, "error": str(exc)}


def register(params: dict) -> dict[str, Any]:
    """Register (and maybe promote) the gated challenger."""
    import mlflow

    name = params["register"]["model_name"]
    with stage_run("register") as rep:
        gate = json.loads(resolve("reports/gate.json").read_text())
        if not gate.get("passed"):
            result = {
                "registered": False,
                "reason": "gate failed",
                "gate_reasons": gate.get("reasons"),
            }
            resolve(REPORT).write_text(json.dumps(result, indent=2) + "\n")
            rep.extra.update({"registered": False})
            return result
        info = json.loads(resolve("reports/train_run.json").read_text())
        setup_mlflow()
        ch = gate["challenger"]
        mv = mlflow.register_model(
            info["model_uri"],
            name,
            tags={
                "build_id": info["build_id"],
                "traffic_precision": str(ch["traffic_precision"]),
                "traffic_coverage": str(ch["traffic_coverage"]),
                "oos_false_accept": str(ch["oos_false_accept"]),
                "latency_p95_ms": str(ch["latency_p95_ms"]),
                "gate": "passed",
            },
        )
        c = registry.client()
        for _ in range(30):
            if c.get_model_version(name, mv.version).status == "READY":
                break
            time.sleep(1)
        c.set_registered_model_alias(name, registry.CHALLENGER, mv.version)
        result: dict[str, Any] = {"registered": True, "version": str(mv.version), "promoted": False}
        if params["register"]["auto_promote"]:
            result.update(registry.promote(name, mv.version, reason="gate passed (auto_promote)"))
            result["promoted"] = True
            result["edge"] = reload_edge()
        rep.rows_out = 1
        rep.extra.update({"registered": True, "version": str(mv.version)})
    resolve(REPORT).write_text(json.dumps(result, indent=2) + "\n")
    log.info("model registered", extra=result)
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Register / promote / roll back the student.")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("status")
    rb = sub.add_parser("rollback")
    rb.add_argument("--to", default=None)
    pr = sub.add_parser("promote")
    pr.add_argument("version")
    args = ap.parse_args(argv)
    params = load_params()
    name = params["register"]["model_name"]
    if args.cmd is None:
        print(json.dumps(register(params), indent=2))
        return
    setup_mlflow()
    if args.cmd == "status":
        out: Any = registry.versions(name)
    elif args.cmd == "rollback":
        out = registry.rollback(name, args.to) | {"edge": reload_edge()}
    else:
        out = registry.promote(name, args.version, reason="manual promote") | {
            "edge": reload_edge()
        }
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
