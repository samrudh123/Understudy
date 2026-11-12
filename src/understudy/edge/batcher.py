"""Micro-batching for the student on a CPU-limited edge.

Under concurrency, one ONNX call per request makes several threads compete for a single
CPU, and the container's CFS quota then throttles all of them. ``MicroBatcher`` queues
concurrent questions and routes up to ``max_batch`` of them in **one** encoder call on
one worker thread. Batching is opportunistic: a lone question is processed immediately,
and questions that arrive while a batch is running form the next batch. No request
ever waits for company.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from starlette.concurrency import run_in_threadpool


class MicroBatcher:
    """Collect concurrent calls into batches for ``fn(list[str]) -> list[Any]``."""

    def __init__(
        self, fn: Callable[[list[str]], list[Any]], max_batch: int = 16, max_wait_ms: float = 2.0
    ) -> None:
        self.fn, self.max_batch, self.max_wait = fn, max_batch, max_wait_ms / 1000
        self._queue: asyncio.Queue[tuple[str, asyncio.Future]] | None = None
        self._task: asyncio.Task | None = None
        self.batches = 0
        self.items = 0

    def _ensure_started(self) -> None:
        if self._task is None or self._task.done():
            self._queue = asyncio.Queue()
            self._task = asyncio.create_task(self._loop())

    async def submit(self, text: str) -> Any:
        """Route ``text`` as part of the next batch."""
        self._ensure_started()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((text, fut))
        return await fut

    async def _loop(self) -> None:
        assert self._queue is not None
        while True:
            batch = [await self._queue.get()]
            if self.max_wait > 0:
                await asyncio.sleep(0)  # let requests that arrived together enqueue
            while len(batch) < self.max_batch and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            texts = [t for t, _ in batch]
            try:
                results = await run_in_threadpool(self.fn, texts)
                for (_, fut), res in zip(batch, results, strict=True):
                    if not fut.done():
                        fut.set_result(res)
            except Exception as exc:  # propagate to every waiter
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
            self.batches += 1
            self.items += len(batch)

    async def close(self) -> None:
        """Stop the background task."""
        if self._task:
            self._task.cancel()
