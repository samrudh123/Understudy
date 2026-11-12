"""Configuration loading.

Two files drive everything:

* ``params.yaml``           pipeline parameters (tracked by DVC per stage).
* ``configs/services.yaml`` runtime settings and REST endpoints for the services.

Every key in ``services.yaml`` can be overridden with an environment variable
named ``UNDERSTUDY_<SECTION>_<KEY>`` (upper-case), which is how Docker Compose
points the blocks at each other without code changes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT_ENV = "UNDERSTUDY_ROOT"
ENV_PREFIX = "UNDERSTUDY_"


def repo_root() -> Path:
    """Return the repository root.

    Resolution order: ``$UNDERSTUDY_ROOT``, then the nearest parent of the
    current directory containing ``params.yaml``, then the location of this file.
    """
    env = os.environ.get(ROOT_ENV)
    if env:
        return Path(env).resolve()
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "params.yaml").is_file() and (candidate / "configs").is_dir():
            return candidate
    return Path(__file__).resolve().parents[3]


def resolve(path: str | Path) -> Path:
    """Resolve ``path`` against the repository root unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping, raising a clear error if the file is missing or not a mapping."""
    p = resolve(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top of {p}, got {type(data).__name__}")
    return data


def load_params(path: str | Path = "params.yaml") -> dict[str, Any]:
    """Load pipeline parameters."""
    return load_yaml(path)


def load_services(
    path: str | Path = "configs/services.yaml", environ: dict[str, str] | None = None
) -> dict[str, Any]:
    """Load service settings and apply ``UNDERSTUDY_<SECTION>_<KEY>`` overrides.

    Override values are parsed as YAML scalars, so ``"60"`` becomes ``60`` and
    ``"null"`` becomes ``None``.
    """
    environ = os.environ if environ is None else environ
    cfg = load_yaml(path)
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        for key in values:
            env_name = f"{ENV_PREFIX}{section}_{key}".upper()
            if env_name in environ:
                values[key] = yaml.safe_load(environ[env_name])
    return cfg
