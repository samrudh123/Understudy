"""PII scrubbing applied before any question is written to disk.

The same function runs at write time in :class:`~understudy.querylog.logger.QueryLogger`
and again as a Spark UDF in the ``preprocess`` stage (defence in depth).
Detected spans are replaced with typed placeholders such as ``<EMAIL>``, so the
question keeps its shape (useful for clustering) without the personal detail.

Known limitation: a run of 10-13 digits with separators (for example
"1000 2000 3000") is treated as a phone number. That trade-off favours privacy.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_IPV4 = re.compile(
    r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"
)
# Institute-style roll numbers, e.g. DA24B025, CS21M104.
_ROLL_NO = re.compile(r"\b[A-Za-z]{2}\d{2}[A-Za-z]\d{3}\b")
# Runs of digits possibly separated by spaces, dashes or brackets (phones, cards, accounts).
_NUMBERISH = re.compile(r"(?<![\w])\+?\(?\d[\d \-()]{7,}\d(?![\w])")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _classify_number(span: str) -> str | None:
    digits = re.sub(r"\D", "", span)
    n = len(digits)
    if 13 <= n <= 19 and _luhn_ok(digits):
        return "CARD"
    if 10 <= n <= 13:
        return "PHONE"
    if n >= 9:
        return "ACCOUNT"
    return None


@dataclass
class ScrubResult:
    """Scrubbed text plus how many spans of each PII type were removed."""

    text: str
    counts: Counter = field(default_factory=Counter)

    @property
    def found(self) -> bool:
        """Return True if any PII was removed."""
        return bool(self.counts)


def scrub(text: str) -> ScrubResult:
    """Replace PII spans in ``text`` with ``<TYPE>`` placeholders."""
    counts: Counter = Counter()

    def _sub(pattern: re.Pattern[str], label: str, s: str) -> str:
        def repl(_m: re.Match[str]) -> str:
            counts[label] += 1
            return f"<{label}>"

        return pattern.sub(repl, s)

    out = _sub(_EMAIL, "EMAIL", text)
    out = _sub(_IPV4, "IP", out)
    out = _sub(_ROLL_NO, "ID", out)

    def number_repl(m: re.Match[str]) -> str:
        label = _classify_number(m.group(0))
        if label is None:
            return m.group(0)
        counts[label] += 1
        return f"<{label}>"

    out = _NUMBERISH.sub(number_repl, out)
    return ScrubResult(out, counts)


def scrub_text(text: str) -> str:
    """Return ``text`` with PII replaced (convenience wrapper around :func:`scrub`)."""
    return scrub(text).text
