"""Sentence encoders behind one small interface.

* ``sentence-transformers/...`` : the real encoder (MiniLM), used by mining and the student.
* ``hash:<dim>``                : a deterministic hashing bag-of-words encoder with
  no downloads. Unit tests use it, and it is a cheap fallback.

Every encoder returns L2-normalised ``float32`` vectors, so a dot product is
cosine similarity.
"""

from __future__ import annotations

import re
import zlib
from functools import lru_cache
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9']+")


class Encoder(Protocol):
    """Anything that maps a list of texts to an ``(n, dim)`` L2-normalised array."""

    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # noqa: D102
        ...


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashingEncoder:
    """Signed feature hashing of unigrams and bigrams. Deterministic across processes."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode ``texts`` into hashed bag-of-n-gram vectors."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = _TOKEN.findall(t.lower())
            grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
            for g in grams:
                h = zlib.crc32(g.encode("utf-8"))
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return _l2_normalize(out)


class SentenceTransformerEncoder:
    """Thin wrapper around a pinned ``sentence-transformers`` model (CPU)."""

    def __init__(self, name: str, revision: str | None = None, batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model = SentenceTransformer(name, revision=revision, device="cpu")
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode ``texts`` with the transformer, L2-normalised."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)


@lru_cache(maxsize=4)
def get_encoder(name: str, revision: str | None = None) -> Encoder:
    """Return a cached encoder for ``name`` (``hash:<dim>`` or a sentence-transformers id)."""
    if name.startswith("hash:"):
        return HashingEncoder(int(name.split(":", 1)[1]))
    return SentenceTransformerEncoder(name, revision=revision)
