"""The student model: encoder + head + answer bank + threshold, as one versioned artifact.

Directory layout (identical locally under ``models/student/`` and inside the MLflow model)::

    encoder/   encoder.onnx  tokenizer.json  encoder_meta.json
    head/      head.npz  head.json
    bank.json
    student_meta.json   (tau, versions, lineage)

``StudentPredictor`` is the plain-Python runtime the edge uses. ``StudentPyfunc``
wraps it as an MLflow pyfunc for the registry and for batch scoring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from understudy.common.text import normalize
from understudy.student.answer_bank import AnswerBank
from understudy.student.encoder import OnnxEncoder
from understudy.student.head import FALLBACK, Head

META_FILE = "student_meta.json"


@dataclass(frozen=True)
class Prediction:
    """The student's best guess for one question."""

    faq_id: str | None  # best FAQ (None if the bank is empty)
    confidence: float  # probability of that FAQ
    fallback_vote: bool  # the head's own fallback class won
    top: list[tuple[str, float]]  # up to 3 (faq_id, prob), for "related questions"


class StudentPredictor:
    """Load a student directory and predict FAQs for questions."""

    def __init__(self, model_dir: str | Path, threads: int = 1) -> None:
        d = Path(model_dir)
        self.dir = d
        self.meta: dict[str, Any] = json.loads((d / META_FILE).read_text(encoding="utf-8"))
        self.encoder = OnnxEncoder(d / "encoder", threads=threads)
        self.head = Head.load(d / "head")
        self.bank = AnswerBank.load(d / "bank.json")
        self.tau = float(self.meta["tau"])
        self._faq_cols = np.array([i for i, lbl in enumerate(self.head.labels) if lbl != FALLBACK])
        self._fb_col = self.head.labels.index(FALLBACK) if FALLBACK in self.head.labels else None

    @property
    def version(self) -> str:
        """Human-readable version (registry version if known, else the local build id)."""
        return str(self.meta.get("registry_version") or self.meta.get("build_id", "local"))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Raw head probabilities for normalised ``texts``."""
        return self.head.predict_proba(self.encoder.encode([normalize(t) for t in texts]))

    def predict(self, texts: list[str]) -> list[Prediction]:
        """Predict the best FAQ for each question."""
        probs = self.predict_proba(texts)
        out = []
        for row in probs:
            faq_probs = row[self._faq_cols]
            order = np.argsort(-faq_probs)[:3]
            top = [(self.head.labels[self._faq_cols[i]], float(faq_probs[i])) for i in order]
            vote = self._fb_col is not None and int(np.argmax(row)) == self._fb_col
            out.append(Prediction(top[0][0] if top else None, top[0][1] if top else 0.0, vote, top))
        return out

    def accepts(self, p: Prediction, tau: float | None = None) -> bool:
        """Answer instantly only when the head does not vote fallback and confidence >= tau."""
        return (
            p.faq_id is not None
            and not p.fallback_vote
            and p.confidence >= (self.tau if tau is None else tau)
        )


def save_student(
    out_dir: str | Path, encoder_dir: str | Path, head: Head, bank: AnswerBank, meta: dict[str, Any]
) -> Path:
    """Assemble a student directory."""
    import shutil

    d = Path(out_dir)
    if d.exists():
        shutil.rmtree(d)
    (d / "encoder").mkdir(parents=True)
    for f in Path(encoder_dir).iterdir():
        shutil.copy(f, d / "encoder" / f.name)
    head.save(d / "head")
    bank.save(d / "bank.json")
    (d / META_FILE).write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    return d


try:  # MLflow is optional at the edge (it only needs the artifacts)
    import mlflow.pyfunc

    class StudentPyfunc(mlflow.pyfunc.PythonModel):
        """MLflow pyfunc wrapper: DataFrame ``text`` -> faq_id, confidence, accept."""

        def load_context(self, context) -> None:  # noqa: D102
            self.predictor = StudentPredictor(context.artifacts["student"])

        def predict(self, context, model_input, params=None):  # noqa: D102
            import pandas as pd

            texts = model_input["text"].astype(str).tolist()
            preds = self.predictor.predict(texts)
            return pd.DataFrame(
                {
                    "faq_id": [p.faq_id for p in preds],
                    "confidence": [p.confidence for p in preds],
                    "accept": [self.predictor.accepts(p) for p in preds],
                }
            )
except ImportError:  # pragma: no cover
    StudentPyfunc = None  # type: ignore[assignment,misc]
