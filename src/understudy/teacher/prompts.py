"""Versioned teacher prompts (``configs/prompts/teacher_vN.yaml``) and message builders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from understudy.common.config import load_yaml

# Bounded lengths keep the grammar from running past max_tokens (truncated JSON).
DISTIL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "maxLength": 160},
        "answer": {"type": "string", "maxLength": 600},
        "paraphrases": {
            "type": "array",
            "items": {"type": "string", "maxLength": 140},
            "minItems": 3,
            "maxItems": 10,
        },
    },
    "required": ["question", "answer", "paraphrases"],
}

GRADE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"relevant": {"type": "boolean"}},
    "required": ["relevant"],
}

Messages = list[dict[str, str]]


@dataclass(frozen=True)
class PromptSet:
    """One version of every teacher prompt."""

    version: str
    tasks: dict[str, dict[str, Any]]

    def settings(self, task: str) -> dict[str, Any]:
        """Return generation settings (max_tokens, temperature) for ``task``."""
        t = self.tasks[task]
        return {"max_tokens": int(t["max_tokens"]), "temperature": float(t["temperature"])}

    def answer(self, question: str) -> Messages:
        """Messages for a live fallback answer."""
        t = self.tasks["answer"]
        return [
            {"role": "system", "content": t["system"].strip()},
            {"role": "user", "content": t["user_template"].format(question=question)},
        ]

    def distil(self, examples: list[str]) -> Messages:
        """Messages that turn a cluster of user questions into one FAQ entry."""
        t = self.tasks["distil"]
        bullet = "\n".join(f"- {e}" for e in examples)
        return [
            {
                "role": "system",
                "content": t["system"].format(n_paraphrases=t["n_paraphrases"]).strip(),
            },
            {"role": "user", "content": t["user_template"].format(examples=bullet).strip()},
        ]

    def grade(self, question: str, faq_question: str, faq_answer: str) -> Messages:
        """Messages asking whether a student answer addresses the question."""
        t = self.tasks["grade"]
        return [
            {"role": "system", "content": t["system"].strip()},
            {
                "role": "user",
                "content": t["user_template"]
                .format(question=question, faq_question=faq_question, faq_answer=faq_answer)
                .strip(),
            },
        ]


def load_prompts(path: str | Path) -> PromptSet:
    """Load a prompt file and check it defines every task."""
    data = load_yaml(path)
    missing = {"answer", "distil", "grade"} - data.keys()
    if missing or "version" not in data:
        raise ValueError(f"Prompt file {path} is missing {sorted(missing) or ['version']}")
    return PromptSet(
        version=str(data["version"]), tasks={k: data[k] for k in data if k != "version"}
    )
