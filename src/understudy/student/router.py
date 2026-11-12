"""``Router``: decide whether the student answers instantly or the question goes to the teacher."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from understudy.student.model import StudentPredictor

RELATED_MIN_CONFIDENCE = 0.25  # below this, a FAQ is not worth suggesting while the user waits


@dataclass(frozen=True)
class RouteDecision:
    """The routing outcome for one question."""

    use_student: bool
    faq_id: str | None
    confidence: float
    matched_question: str | None
    answer: str | None
    related: list[dict] = field(default_factory=list)  # other likely FAQs, for the waiting card
    inference_ms: float = 0.0


class Router:
    """Wrap a :class:`StudentPredictor` with the confidence threshold ``tau``."""

    def __init__(self, predictor: StudentPredictor, tau_override: float | None = None) -> None:
        self.predictor = predictor
        self.tau = predictor.tau if tau_override is None else float(tau_override)

    def route(self, question: str) -> RouteDecision:
        """Predict and apply the threshold."""
        return self.route_many([question])[0]

    def route_many(self, questions: list[str]) -> list[RouteDecision]:
        """Route a batch (one encoder call); used by the edge's micro-batcher."""
        t0 = time.perf_counter()
        preds = self.predictor.predict(questions)
        ms = (time.perf_counter() - t0) * 1000
        return [self._decide(p, ms) for p in preds]

    def _decide(self, pred, ms: float) -> RouteDecision:
        ok = self.predictor.accepts(pred, self.tau)
        faq = self.predictor.bank.get(pred.faq_id) if pred.faq_id else None
        related = [
            {
                "faq_id": fid,
                "question": self.predictor.bank.get(fid).question,
                "confidence": round(p, 3),
            }
            for fid, p in pred.top
            if p >= RELATED_MIN_CONFIDENCE and self.predictor.bank.get(fid) is not None
        ]
        return RouteDecision(
            use_student=ok and faq is not None,
            faq_id=pred.faq_id,
            confidence=pred.confidence,
            matched_question=faq.question if faq else None,
            answer=faq.answer if (ok and faq) else None,
            related=[] if ok else related,
            inference_ms=ms,
        )
