"""Structured (JSON-lines) logging shared by every service and pipeline stage.

Usage::

    log = get_logger(__name__)
    log.info("stage finished", extra={"stage": "preprocess", "rows_out": 1234})

Anything passed through ``extra`` becomes a top-level JSON field, so logs can be
filtered by ``stage``, ``request_id``, ``route`` and so on.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime

# Attributes every LogRecord has; anything else came from ``extra=``.
_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
    "taskName",
}

_configured = False


class JsonFormatter(logging.Formatter):
    """Render each record as one JSON object per line."""

    def __init__(self, service: str | None = None) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if self.service:
            payload["service"] = self.service
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(service: str | None = None, level: str | None = None) -> None:
    """Install the JSON formatter on the root logger (idempotent)."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter(service or os.environ.get("UNDERSTUDY_SERVICE")))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level or os.environ.get("UNDERSTUDY_LOG_LEVEL", "INFO"))
    # Third-party libraries are chatty at INFO.
    for noisy in ("py4j", "urllib3", "filelock", "httpx", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring JSON logging on first use."""
    configure_logging()
    return logging.getLogger(name)
