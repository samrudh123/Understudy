"""Ingest stages.

``python -m understudy.data.ingest dataset``
    Download CLINC150 from a pinned commit and verify its SHA-256 (``ingest_dataset``).

``python -m understudy.data.ingest logs``
    Snapshot the live query-log directory into one DVC-tracked file (``ingest_logs``),
    so every retraining run traces back to the exact logs it used.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import requests

from understudy.common.config import load_params, resolve
from understudy.common.lineage import file_sha256
from understudy.common.logs import get_logger
from understudy.common.metrics import stage_run
from understudy.data.clinc import DATA_FILE, DOMAINS_FILE, RAW_DIR

log = get_logger(__name__)

SNAPSHOT_DIR = "data/snapshots"
QUERYLOG_SNAPSHOT = "querylog.jsonl"
FEEDBACK_SNAPSHOT = "feedback.jsonl"


class IngestError(RuntimeError):
    """Raised when a download fails or a checksum does not match."""


def download(url: str, dest: Path, sha256: str, retries: int = 3, timeout: float = 60.0) -> Path:
    """Download ``url`` to ``dest`` with retries and verify its SHA-256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
            break
        except requests.RequestException as exc:
            last_exc = exc
            log.warning(
                "download failed", extra={"url": url, "attempt": attempt, "error": str(exc)}
            )
            time.sleep(2**attempt)
    else:
        raise IngestError(f"Could not download {url}") from last_exc

    actual = file_sha256(tmp)
    if actual != sha256:
        tmp.unlink(missing_ok=True)
        raise IngestError(f"Checksum mismatch for {url}: expected {sha256}, got {actual}")
    tmp.replace(dest)
    return dest


def ingest_dataset(params: dict, out_dir: str | Path = RAW_DIR) -> dict:
    """Fetch the pinned CLINC150 files into ``out_dir`` and write a manifest."""
    out = resolve(out_dir)
    cfg = params["data"]
    with stage_run("ingest_dataset") as rep:
        files = {
            DATA_FILE: (cfg["clinc_url"], cfg["clinc_sha256"]),
            DOMAINS_FILE: (cfg["domains_url"], cfg["domains_sha256"]),
        }
        manifest = {}
        for name, (url, sha) in files.items():
            path = download(url, out / name, sha)
            manifest[name] = {"url": url, "sha256": sha, "bytes": path.stat().st_size}
        with (out / DATA_FILE).open(encoding="utf-8") as fh:
            rep.rows_out = sum(len(v) for v in json.load(fh).values())
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        rep.extra["files"] = list(manifest)
    return manifest


def _concat(files: list[Path], dest: Path) -> int:
    lines = 0
    with dest.open("w", encoding="utf-8") as out:
        for f in files:
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        lines += 1
    return lines


def ingest_logs(log_dir: str | Path, out_dir: str | Path = SNAPSHOT_DIR) -> dict:
    """Concatenate every query-log and feedback file into a single snapshot.

    An empty log directory (cold start) produces empty snapshot files, so the
    rest of the pipeline still runs.
    """
    src = resolve(log_dir)
    out = resolve(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    with stage_run("ingest_logs") as rep:
        all_files = sorted(src.glob("*.jsonl")) if src.is_dir() else []
        feedback_files = [f for f in all_files if f.name.startswith("feedback-")]
        query_files = [f for f in all_files if not f.name.startswith("feedback-")]
        n_queries = _concat(query_files, out / QUERYLOG_SNAPSHOT)
        n_feedback = _concat(feedback_files, out / FEEDBACK_SNAPSHOT)
        manifest = {
            "query_files": [f.name for f in query_files],
            "feedback_files": [f.name for f in feedback_files],
            "query_rows": n_queries,
            "feedback_rows": n_feedback,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        rep.rows_in = rep.rows_out = n_queries
        rep.extra.update(feedback_rows=n_feedback, files=len(all_files))
    return manifest


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("what", choices=["dataset", "logs"])
    parser.add_argument("--log-dir", default="data/querylogs")
    args = parser.parse_args(argv)
    if args.what == "dataset":
        ingest_dataset(load_params())
    else:
        ingest_logs(args.log_dir)


if __name__ == "__main__":
    main()
