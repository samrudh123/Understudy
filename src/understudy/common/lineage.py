"""Lineage helpers: tie every artefact back to the code version that produced it."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from understudy.common.config import repo_root


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo_root(), capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_commit() -> str | None:
    """Return the current Git commit hash, or None outside a repository."""
    return _git("rev-parse", "HEAD")


def git_is_dirty() -> bool | None:
    """Return True if tracked files have uncommitted changes."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return None if status is None else bool(status)


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()
