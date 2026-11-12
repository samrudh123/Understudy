"""MLflow Model Registry helpers: aliases, downloads, promotion and rollback.

``champion`` is what the edge serves and ``challenger`` is the latest model that passed the
gate. Each promoted version records the version it replaced (tag ``previous_champion``),
so a rollback is one alias move plus an edge reload.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from understudy.common.logs import get_logger

log = get_logger(__name__)

CHAMPION, CHALLENGER = "champion", "challenger"


def client():
    """Return an ``MlflowClient`` (tracking URI must already be set)."""
    from mlflow import MlflowClient

    return MlflowClient()


def alias_version(name: str, alias: str) -> Any | None:
    """Return the ``ModelVersion`` behind ``alias``, or ``None``."""
    from mlflow.exceptions import MlflowException

    try:
        return client().get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def download_student(model_uri: str, dst: str | Path | None = None) -> Path:
    """Download a registered or logged student and return its student directory."""
    import mlflow

    dst = Path(dst or tempfile.mkdtemp(prefix="student-"))
    local = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(dst)))
    student = local / "artifacts" / "student"
    if not student.is_dir():
        raise FileNotFoundError(f"{model_uri} has no artifacts/student directory")
    return student


def promote(name: str, version: str | int, reason: str) -> dict[str, Any]:
    """Point ``champion`` at ``version``, remembering the version it replaces."""
    c = client()
    prev = alias_version(name, CHAMPION)
    c.set_registered_model_alias(name, CHAMPION, str(version))
    if prev is not None and str(prev.version) != str(version):
        c.set_model_version_tag(name, str(version), "previous_champion", str(prev.version))
    c.set_model_version_tag(name, str(version), "promotion_reason", reason)
    log.info(
        "champion promoted", extra={"version": str(version), "previous": prev and prev.version}
    )
    return {"champion": str(version), "previous": None if prev is None else str(prev.version)}


def rollback(name: str, to_version: str | int | None = None) -> dict[str, Any]:
    """Move ``champion`` back to ``to_version`` (default: the version it replaced)."""
    current = alias_version(name, CHAMPION)
    if current is None:
        raise RuntimeError(f"{name} has no champion to roll back")
    target = to_version or current.tags.get("previous_champion")
    if not target:
        raise RuntimeError(f"champion v{current.version} has no previous_champion to roll back to")
    client().set_registered_model_alias(name, CHAMPION, str(target))
    client().set_model_version_tag(name, str(target), "rolled_back_from", str(current.version))
    log.warning("champion rolled back", extra={"from": current.version, "to": str(target)})
    return {"champion": str(target), "rolled_back_from": str(current.version)}


def versions(name: str) -> list[dict[str, Any]]:
    """All versions with their aliases and key tags, newest first."""
    from mlflow.exceptions import MlflowException

    c = client()
    try:
        aliases = c.get_registered_model(name).aliases or {}
    except MlflowException:
        return []
    by_version: dict[str, list[str]] = {}
    for a, v in aliases.items():
        by_version.setdefault(str(v), []).append(a)
    out = []
    for mv in c.search_model_versions(f"name='{name}'"):
        out.append(
            {
                "version": str(mv.version),
                "run_id": mv.run_id,
                "created": mv.creation_timestamp,
                "aliases": sorted(by_version.get(str(mv.version), [])),
                "tags": dict(mv.tags or {}),
            }
        )
    return sorted(out, key=lambda r: -int(r["version"]))
