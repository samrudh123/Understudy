"""Routing metrics shared by ``train`` (threshold tuning) and ``evaluate_gate``.

The student either *accepts* (answers instantly with a FAQ) or *defers* to the teacher.

* **instant_precision**: of the questions answered instantly, the fraction answered with
  the right FAQ. This is the headline quality target (>= 0.95).
* **coverage**: of the questions the bank *can* answer, the fraction answered instantly
  and correctly.
* **oos_false_accept**: out-of-scope questions answered instantly (target <= 0.05).
* **uncovered_false_accept**: in-scope questions with no FAQ yet, answered instantly anyway.
* **traffic_precision / traffic_coverage**: the same, with each test question weighted by its
  intent's share of recent traffic. The CLINC test set is uniform over 150 intents, so
  uncovered intents outnumber covered ones about 6:1. That uniform view is a stress test.
  The traffic view estimates what users of the live system actually see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Outcomes:
    """Per-question outcomes, ready for any threshold."""

    confidence: np.ndarray  # top FAQ probability
    vote_ok: np.ndarray  # head did not vote fallback
    correct: np.ndarray  # top FAQ is a right answer
    answerable: np.ndarray  # a right FAQ exists in the bank
    is_oos: np.ndarray  # out of scope
    weight: np.ndarray | None = None  # traffic weight per question (test only)


def metrics_at(o: Outcomes, tau: float) -> dict[str, float]:
    """Routing metrics at threshold ``tau``."""
    acc = o.vote_ok & (o.confidence >= tau)
    n_acc = int(acc.sum())
    unc = ~o.answerable & ~o.is_oos

    def frac(num: np.ndarray, den: np.ndarray) -> float:
        d = int(den.sum())
        return float(num.sum() / d) if d else float("nan")

    out = {
        "tau": float(tau),
        "instant_precision": float((acc & o.correct).sum() / n_acc) if n_acc else 1.0,
        "coverage": frac(acc & o.correct & o.answerable, o.answerable),
        "oos_false_accept": frac(acc & o.is_oos, o.is_oos),
        "uncovered_false_accept": frac(acc & unc, unc),
        "false_accept": frac(acc & ~o.answerable, ~o.answerable),
        "accept_rate": float(n_acc / len(acc)) if len(acc) else 0.0,
    }
    if o.weight is not None:
        w = o.weight
        wa = float((w * acc).sum())
        out["traffic_precision"] = float((w * (acc & o.correct)).sum() / wa) if wa > 0 else 1.0
        wd = float((w * o.answerable).sum())
        out["traffic_coverage"] = (
            float((w * (acc & o.correct & o.answerable)).sum() / wd) if wd > 0 else float("nan")
        )
        out["traffic_accept_rate"] = wa / float(w.sum()) if w.sum() > 0 else 0.0
    return out


def choose_tau(
    o: Outcomes, min_precision: float, max_false_accept: float, grid: np.ndarray | None = None
) -> tuple[dict[str, float], bool]:
    """Return the threshold meeting both targets with max coverage, and whether one exists.

    If no threshold is feasible, returns the one with the best precision so the
    trial still gets a (penalised) score.
    """
    grid = np.round(np.arange(0.05, 1.0, 0.01), 2) if grid is None else grid
    best, best_any = None, None
    for t in grid:
        m = metrics_at(o, float(t))
        fa = max(m["false_accept"] if not np.isnan(m["false_accept"]) else 0.0, 0.0)
        feasible = (
            m["instant_precision"] >= min_precision
            and fa <= max_false_accept
            and m["coverage"] > 0  # "never answer" trivially meets both targets
        )
        if feasible and (best is None or m["coverage"] > best["coverage"]):
            best = m
        if best_any is None or (m["instant_precision"], m["coverage"]) > (
            best_any["instant_precision"],
            best_any["coverage"],
        ):
            best_any = m
    return (best, True) if best is not None else (best_any, False)
