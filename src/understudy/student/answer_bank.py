"""``AnswerBank``: approved FAQ answers, versioned by content hash and shipped inside the model."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FAQ:
    """One approved question/answer pair."""

    faq_id: str
    question: str
    answer: str
    reviewer: str = ""
    prompt_version: str = ""


class AnswerBank:
    """Lookup of approved answers by ``faq_id``."""

    def __init__(self, faqs: list[FAQ]) -> None:
        self.faqs = {f.faq_id: f for f in faqs}
        blob = json.dumps([asdict(f) for f in sorted(faqs, key=lambda f: f.faq_id)], sort_keys=True)
        self.version = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def __len__(self) -> int:
        return len(self.faqs)

    def __contains__(self, faq_id: object) -> bool:
        return faq_id in self.faqs

    def get(self, faq_id: str) -> FAQ | None:
        """Return the FAQ or ``None``."""
        return self.faqs.get(faq_id)

    def to_json(self) -> dict:
        """Serialise (including the content version)."""
        return {"version": self.version, "faqs": [asdict(f) for f in self.faqs.values()]}

    def save(self, path: str | Path) -> None:
        """Write ``bank.json``."""
        Path(path).write_text(
            json.dumps(self.to_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> AnswerBank:
        """Read ``bank.json``."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([FAQ(**f) for f in data["faqs"]])
