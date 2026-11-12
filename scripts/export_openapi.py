"""Export the OpenAPI specs of the four REST services into ``docs/openapi/``.

    python scripts/export_openapi.py

``app.openapi()`` does not run the app lifespan, so nothing is loaded or connected.
"""

from __future__ import annotations

import json
from pathlib import Path

from understudy.console.app import create_app as console_app
from understudy.edge.app import create_app as edge_app
from understudy.gateway.app import create_app as gateway_app
from understudy.teacher.engine import FakeEngine
from understudy.teacher.serve import TeacherCore, build_app

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/openapi"


def main() -> None:
    """Write edge, gateway, console and teacher OpenAPI JSON."""
    OUT.mkdir(parents=True, exist_ok=True)
    settings = {
        "edge": {
            "url": "http://localhost:8001",
            "model_uri": str(ROOT / "models/does-not-exist"),
            "model_cache_dir": "/tmp/understudy-openapi-cache",
            "confidence_threshold_override": None,
            "filler_file": str(ROOT / "configs/filler/filler_v1.yaml"),
        },
        "gateway": {
            "url": "http://localhost:8002",
            "teacher_timeout_s": 60,
            "first_token_timeout_s": 15,
        },
        "teacher": {"url": "http://localhost:8000/teacher"},
        "querylog": {"dir": "/tmp/understudy-openapi-logs"},
        "prometheus": {"url": "http://localhost:9090"},
        "mlflow": {"tracking_uri": "http://localhost:5000"},
        "console": {"url": "http://localhost:8003", "links": {}},
    }
    prompts = str(ROOT / "configs/prompts/teacher_v1.yaml")
    apps = {
        "edge": edge_app(settings, log_dir="/tmp/understudy-openapi-logs"),
        "gateway": gateway_app(settings),
        "console": console_app(settings, {"register": {"model_name": "understudy-student"}}),
        "teacher": build_app(lambda: TeacherCore(FakeEngine(), prompts)),
    }
    for name, app in apps.items():
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(app.openapi(), indent=2) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT), len(app.openapi()["paths"]), "paths")


if __name__ == "__main__":
    main()
