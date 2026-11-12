"""Teacher inference engines.

* :class:`LlamaEngine` wraps ``llama-cpp-python`` (GGUF, CPU). Used by the Ray Serve
  teacher and by the ``distil`` Ray actors.
* :class:`FakeEngine` is a deterministic stand-in with the same interface. Tests, CI
  and the Compose smoke test use it so they run without the 2 GB model. Select it
  with ``UNDERSTUDY_TEACHER_FAKE=1``.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from understudy.common.logs import get_logger
from understudy.teacher.prompts import Messages

log = get_logger(__name__)


class Engine(Protocol):
    """What the teacher service and the distil actors need from a model."""

    model_name: str

    def stream(self, messages: Messages, max_tokens: int, temperature: float) -> Iterator[str]:
        """Yield the answer piece by piece."""
        ...

    def complete_json(
        self, messages: Messages, schema: dict[str, Any], max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        """Return a JSON object that matches ``schema`` (grammar-constrained)."""
        ...


class LlamaEngine:
    """llama.cpp on CPU. Not thread-safe: serve one request at a time per instance."""

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        seed: int = 42,
    ) -> None:
        from llama_cpp import Llama  # heavy native import, keep lazy

        self.model_name = Path(model_path).name
        t0 = time.perf_counter()
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads or max(1, (os.cpu_count() or 2) // 2),
            seed=seed,
            verbose=False,
        )
        log.info(
            "teacher model loaded",
            extra={"model": self.model_name, "load_s": round(time.perf_counter() - t0, 2)},
        )

    def stream(self, messages: Messages, max_tokens: int, temperature: float) -> Iterator[str]:
        """Stream chat-completion deltas."""
        for chunk in self.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True
        ):
            piece = chunk["choices"][0]["delta"].get("content")
            if piece:
                yield piece

    def complete_json(
        self, messages: Messages, schema: dict[str, Any], max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        """Grammar-constrained JSON completion."""
        out = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object", "schema": schema},
        )
        return json.loads(out["choices"][0]["message"]["content"])


class FakeEngine:
    """Deterministic, instant teacher for tests and CI. Mirrors the real output shapes."""

    model_name = "fake-teacher"

    def __init__(self, token_delay_s: float = 0.0, first_token_delay_s: float = 0.0) -> None:
        self.token_delay_s = token_delay_s
        self.first_token_delay_s = first_token_delay_s

    def stream(self, messages: Messages, max_tokens: int, temperature: float) -> Iterator[str]:
        """Echo a canned answer word by word."""
        question = messages[-1]["content"]
        words = f"Here is some help with: {question}".split()[:max_tokens]
        time.sleep(self.first_token_delay_s)
        for i, w in enumerate(words):
            if i:
                time.sleep(self.token_delay_s)
            yield w if i == 0 else " " + w

    def complete_json(
        self, messages: Messages, schema: dict[str, Any], max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        """Produce a schema-shaped object from the prompt."""
        props = schema.get("properties", {})
        if "relevant" in props:
            return {"relevant": True}
        lines = [ln[2:] for ln in messages[-1]["content"].splitlines() if ln.startswith("- ")]
        first = lines[0] if lines else "your question"
        m = re.search(r"exactly (\d+)", messages[0]["content"])
        n = int(m.group(1)) if m else 4
        return {
            "question": first,
            "answer": f"This is the help-desk answer to: {first}.",
            "paraphrases": [f"{first} (variant {i + 1})" for i in range(n)],
        }


def make_engine(model_path: str | Path | None, n_ctx: int = 2048, n_threads: int | None = None):
    """Return the fake engine if ``UNDERSTUDY_TEACHER_FAKE`` is set, else llama.cpp."""
    if os.environ.get("UNDERSTUDY_TEACHER_FAKE", "").lower() in {"1", "true", "yes"}:
        return FakeEngine(token_delay_s=float(os.environ.get("UNDERSTUDY_FAKE_TOKEN_DELAY", 0)))
    if model_path is None or not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Teacher weights not found at {model_path}. Run: python -m understudy.teacher.download"
        )
    return LlamaEngine(model_path, n_ctx=n_ctx, n_threads=n_threads)
