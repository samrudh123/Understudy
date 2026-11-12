"""Server-sent events: one tiny encoder and parser shared by teacher, gateway, edge and tests.

Wire format (one event)::

    event: token
    data: {"t": "Hello"}
    <blank line>

Event types used across Understudy: ``meta`` (routing info, sent first), ``token``
(a piece of the answer), ``done`` (final stats) and ``error`` (a failure after the
stream started).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any

MEDIA_TYPE = "text/event-stream"
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}  # nginx: don't buffer


def format_event(event: str, data: dict[str, Any]) -> str:
    """Encode one event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _parse_lines(lines: Iterable[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    event, data = "message", []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
    if data:
        yield event, json.loads("\n".join(data))


def parse_events(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse a complete SSE body (used by tests and non-streaming callers)."""
    return list(_parse_lines(text.splitlines()))


async def aparse_events(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Parse SSE events from an async iterator of lines (e.g. ``httpx.Response.aiter_lines``)."""
    event, data = "message", []
    async for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
    if data:
        yield event, json.loads("\n".join(data))
