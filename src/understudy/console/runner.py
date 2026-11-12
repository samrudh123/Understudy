"""Runs pipeline commands (``dvc repro <target>``, simulations) one at a time in the background.

The console must never run two pipelines at once: DVC holds a repo lock, and two
trainings would race for the registry. :class:`PipelineRunner` serialises runs, keeps a
log tail for the UI and persists a short history in ``reports/console/pipeline_runs.json``.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from understudy.common.config import repo_root, resolve
from understudy.common.logs import get_logger

log = get_logger(__name__)

HISTORY = "reports/console/pipeline_runs.json"
TARGETS = {
    # name -> (command, description)
    "discover": (
        ["dvc", "repro", "distil"],
        "Snapshot logs, preprocess, mine frequent fallbacks and draft FAQs with the teacher",
    ),
    "retrain": (
        ["dvc", "repro", "register"],
        "Build the dataset from approvals, tune + train, gate, register and promote",
    ),
    "full": (["dvc", "repro"], "Every out-of-date stage"),
}


class BusyError(RuntimeError):
    """A pipeline run is already in progress."""


class PipelineRunner:
    """Serialised background runner with a log tail and run history."""

    def __init__(
        self, history_path: str | Path = HISTORY, max_history: int = 50, on_finish=None
    ) -> None:
        self.history_path = resolve(history_path)
        self.max_history = max_history
        self.on_finish = on_finish
        self._lock = threading.Lock()
        self.current: dict[str, Any] | None = None
        self.tail: deque[str] = deque(maxlen=400)
        self.history: list[dict[str, Any]] = self._load()

    def _load(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.history_path.read_text())
        except (FileNotFoundError, ValueError):
            return []

    def _save(self) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(self.history[-self.max_history :], indent=2) + "\n")

    def start(
        self, name: str, cmd: list[str], reason: str, env: dict[str, str] | None = None
    ) -> dict:
        """Start ``cmd`` in the background or raise :class:`BusyError`."""
        if not self._lock.acquire(blocking=False):
            raise BusyError(f"pipeline busy: {self.current and self.current['name']}")
        run = {
            "id": uuid.uuid4().hex[:8],
            "name": name,
            "cmd": " ".join(cmd),
            "reason": reason,
            "started_at": datetime.now(UTC).isoformat(),
            "state": "running",
        }
        self.current = run
        self.tail.clear()
        threading.Thread(target=self._run, args=(run, cmd, env), daemon=True).start()
        log.info(
            "pipeline run started", extra={"run_id": run["id"], "run_name": name, "reason": reason}
        )
        return run

    def _run(self, run: dict[str, Any], cmd: list[str], env: dict[str, str] | None) -> None:
        import os

        t0 = time.perf_counter()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=repo_root(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, **(env or {})},
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                self.tail.append(line.rstrip()[:500])
            code = proc.wait()
            run["state"] = "succeeded" if code == 0 else "failed"
            run["exit_code"] = code
        except Exception as exc:  # e.g. dvc missing
            run["state"] = "failed"
            run["error"] = str(exc)
            self.tail.append(f"runner error: {exc}")
        finally:
            run["finished_at"] = datetime.now(UTC).isoformat()
            run["duration_s"] = round(time.perf_counter() - t0, 1)
            run["log_tail"] = list(self.tail)[-40:]
            self.history.append(run)
            self._save()
            self.current = None
            self._lock.release()
            log.info(
                "pipeline run finished",
                extra={
                    "run_id": run["id"],
                    "run_name": run["name"],
                    "state": run["state"],
                    "duration_s": run["duration_s"],
                },
            )
            if self.on_finish:
                self.on_finish(run)

    def status(self) -> dict[str, Any]:
        """Return the current run (with live log tail) and recent history."""
        return {
            "current": self.current and {**self.current, "log_tail": list(self.tail)[-60:]},
            "history": [
                {k: v for k, v in r.items() if k != "log_tail"} for r in self.history[-15:]
            ][::-1],
            "last": self.history[-1] if self.history else None,
        }
