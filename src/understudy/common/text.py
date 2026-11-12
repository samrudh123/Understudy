"""Text normalisation: the single, versioned definition of the student's input features.

Training (Spark preprocess), mining and online serving all call
:func:`normalize`, so the student never sees text that was cleaned differently
from what it was trained on (avoiding training/serving skew).
"""

from __future__ import annotations

import re
import unicodedata

NORMALIZER_VERSION = "v1"

_WS = re.compile(r"\s+")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', " ": " "})
_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")


def normalize(text: str) -> str:
    """Return a canonical form of a user question.

    NFKC unicode folding, straight quotes, control characters removed, lower-case,
    whitespace collapsed and trimmed.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    t = _CONTROL.sub(" ", t)
    t = _WS.sub(" ", t).strip().lower()
    return t


_PUNCT = re.compile(r"[^\w\s]")


def dedupe_key(text: str) -> str:
    """Key for near-duplicate detection: normalised, punctuation removed."""
    return " ".join(_PUNCT.sub(" ", normalize(text)).split())
