"""``GatewayClient``: the edge's HTTP client for the server gateway (``/stream``, ``/grade``)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from understudy.common.sse import aparse_events


class GatewayError(RuntimeError):
    """The gateway refused or failed the request. ``code`` is a machine-readable reason."""

    def __init__(self, code: str, detail: str = "", status: int | None = None) -> None:
        super().__init__(f"{code}: {detail}")
        self.code, self.detail, self.status = code, detail, status


class GatewayClient:
    """Async gateway client.

    Raises :class:`GatewayError` with codes such as ``prompt_injection``,
    ``teacher_unavailable``, ``teacher_timeout`` or ``gateway_unreachable``.
    """

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 75.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=2.0),
            transport=transport,
        )

    async def aclose(self) -> None:
        """Close the connection pool."""
        await self._client.aclose()

    async def stream(
        self, question: str, request_id: str
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield gateway SSE events for one fallback question."""
        try:
            async with self._client.stream(
                "POST", "/stream", json={"question": question, "request_id": request_id}
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    try:
                        info = httpx.Response(200, content=body).json()
                    except ValueError:
                        info = {}
                    raise GatewayError(
                        info.get("code", f"http_{resp.status_code}"),
                        info.get("detail", ""),
                        resp.status_code,
                    )
                async for event in aparse_events(resp.aiter_lines()):
                    yield event
        except httpx.HTTPError as exc:
            raise GatewayError("gateway_unreachable", str(exc)) from exc

    async def grade(self, question: str, faq_question: str, faq_answer: str) -> bool:
        """Ask the teacher whether a student answer addresses the question."""
        try:
            r = await self._client.post(
                "/grade",
                json={"question": question, "faq_question": faq_question, "faq_answer": faq_answer},
            )
        except httpx.HTTPError as exc:
            raise GatewayError("gateway_unreachable", str(exc)) from exc
        if r.status_code != 200:
            raise GatewayError(r.json().get("code", f"http_{r.status_code}"), status=r.status_code)
        return bool(r.json()["result"]["relevant"])
