"""``TeacherClient``: the only way other blocks talk to the teacher service (HTTP)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from understudy.common.sse import aparse_events


class TeacherError(RuntimeError):
    """The teacher could not be reached or returned an error."""


class TeacherClient:
    """Async client for the Ray Serve teacher (``/teacher/stream``, ``/generate``, ``/health``)."""

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=3.0),
            transport=transport,
        )

    async def aclose(self) -> None:
        """Close the connection pool."""
        await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        """Return the teacher's ``/health`` payload or raise :class:`TeacherError`."""
        try:
            r = await self._client.get("/health", timeout=3.0)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as exc:
            raise TeacherError(f"teacher health check failed: {exc}") from exc

    async def stream(
        self, question: str, request_id: str | None = None
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield ``(event, data)`` pairs from ``/stream``. Raises :class:`TeacherError` early."""
        try:
            async with self._client.stream(
                "POST", "/stream", json={"question": question, "request_id": request_id}
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")[:200]
                    raise TeacherError(f"teacher returned {resp.status_code}: {body}")
                async for event in aparse_events(resp.aiter_lines()):
                    yield event
        except httpx.HTTPError as exc:
            raise TeacherError(f"teacher stream failed: {exc}") from exc

    async def generate(self, task: str, **fields: Any) -> dict[str, Any]:
        """Run a non-streaming task (``answer``, ``distil`` or ``grade``)."""
        try:
            r = await self._client.post("/generate", json={"task": task, **fields})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as exc:
            raise TeacherError(f"teacher generate failed: {exc}") from exc
