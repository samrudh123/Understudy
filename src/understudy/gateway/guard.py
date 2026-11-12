"""Input validation and a prompt-injection guard for questions sent to the teacher.

The student only ever returns approved answers from a fixed bank, so it cannot be
talked into anything. The teacher is a generative model, so the gateway checks
every fallback question before it reaches it:

1. **Validation**: strip control characters and chat-template tokens, reject empty
   or over-long input.
2. **Injection heuristics**: known override phrasings ("ignore previous
   instructions", "reveal your system prompt", "you are now DAN", role tags, ...).
   This is a deliberately simple, explainable first line. The system prompt
   also tells the model to treat the message as a question only.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MAX_QUESTION_CHARS = 500

_SPECIAL_TOKENS = re.compile(r"<\|[^|>]{1,40}\|>|\[/?INST\]|<</?SYS>>", re.IGNORECASE)
_INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "override_instructions": re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|"
        r"all|your|the|system)\b.{0,30}\b(instructions?|prompts?|rules?|guidelines?|directions?)",
        re.IGNORECASE,
    ),
    "reveal_prompt": re.compile(
        r"\b(reveal|show|print|repeat|output|tell me|what (is|are)|leak)\b.{0,30}"
        r"(\byour\s+(system\s+|initial\s+|hidden\s+)?(prompt|instructions|rules|guidelines)"
        r"|\b(system|hidden|initial)\s+prompt)\b",
        re.IGNORECASE,
    ),
    "role_hijack": re.compile(
        r"\b(you are now|from now on you are|act as|pretend (to be|you are)|roleplay as)\b.{0,40}"
        r"\b(dan|developer mode|jailbroken|unfiltered|unrestricted|evil|no rules)\b",
        re.IGNORECASE,
    ),
    "jailbreak_keyword": re.compile(
        r"\b(jailbreak|do anything now|developer mode)\b", re.IGNORECASE
    ),
    "role_tag": re.compile(r"^\s*(system|assistant)\s*:", re.IGNORECASE | re.MULTILINE),
}


@dataclass(frozen=True)
class GuardResult:
    """Outcome of checking one question."""

    ok: bool
    text: str  # cleaned question (valid only if ok)
    code: str | None = None  # "empty", "too_long" or "prompt_injection"
    reason: str | None = None  # which rule fired


def clean(text: str) -> str:
    """Drop control characters and chat-template tokens; collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _SPECIAL_TOKENS.sub(" ", text)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return re.sub(r"[ \t]+", " ", text).strip()


def check_question(raw: str, max_chars: int = MAX_QUESTION_CHARS) -> GuardResult:
    """Validate and screen one question."""
    # Detect on the raw text too: the special tokens themselves are a signal.
    if _SPECIAL_TOKENS.search(raw):
        return GuardResult(False, "", "prompt_injection", "chat_template_token")
    text = clean(raw)
    if not text:
        return GuardResult(False, "", "empty", "no visible characters")
    if len(text) > max_chars:
        return GuardResult(False, "", "too_long", f"over {max_chars} characters")
    for name, pattern in _INJECTION_PATTERNS.items():
        if pattern.search(text):
            return GuardResult(False, "", "prompt_injection", name)
    return GuardResult(True, text)
