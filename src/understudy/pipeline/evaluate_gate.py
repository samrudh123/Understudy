"""``evaluate_gate`` stage: decide whether the newly trained student replaces the champion.

The challenger (``models/student``) and, when one exists, the registered **champion** are
both evaluated on the same current test set. The gate passes only if the challenger:

* meets the absolute targets: traffic precision, out-of-scope false-accept, p95 latency,
* does not regress against the champion by more than the allowed precision/coverage drop,
* passes the behavioural checks: canonical questions route to their own FAQ, appended
  (scrubbed) PII does not change the answer, and empty or gibberish input defers.

The result is ``reports/gate.json``. A failed gate is a normal outcome, not a crash: the
``register`` stage reads it and simply does not promote.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from understudy.common.config import load_params, resolve
from understudy.common.logs import get_logger
from understudy.common.metrics import stage_run
from understudy.common.pii import scrub
from understudy.pipeline.approve import load_approvals
from understudy.pipeline.train import latency_ms, outcomes_from_intents
from understudy.student.evaluation import metrics_at
from understudy.student.model import StudentPredictor

log = get_logger(__name__)

GATE = "reports/gate.json"
_GIBBERISH = ["", "   ", "asdkj qwpeo zmxn", "????", "lorem ipsum dolor sit amet"]


def evaluate(pred: StudentPredictor, test: pd.DataFrame, faq_intent: dict[str, str]) -> dict:
    """Routing metrics of ``pred`` on ``test`` at its own tau."""
    probs = pred.head.predict_proba(pred.encoder.encode(test["text"].tolist()))
    o = outcomes_from_intents(pred.head, probs, test, faq_intent)
    return {k: round(v, 4) for k, v in metrics_at(o, pred.tau).items()}


def behavioural(pred: StudentPredictor) -> dict[str, float]:
    """Run simple invariance and sanity checks on the challenger."""
    faqs = list(pred.bank.faqs.values())
    own = pred.predict([f.question for f in faqs])
    canonical = sum(
        p.faq_id == f.faq_id and pred.accepts(p) for p, f in zip(own, faqs, strict=True)
    )
    noisy = pred.predict(
        [scrub(f"{f.question} my email is test.user@example.com").text for f in faqs]
    )
    invariant = sum(a.faq_id == b.faq_id for a, b in zip(own, noisy, strict=True))
    junk = pred.predict(_GIBBERISH)
    return {
        "canonical_recall": round(canonical / len(faqs), 4) if faqs else 0.0,
        "pii_invariance": round(invariant / len(faqs), 4) if faqs else 0.0,
        "gibberish_deferred": round(sum(not pred.accepts(p) for p in junk) / len(junk), 4),
    }


def champion_predictor(name: str) -> tuple[StudentPredictor | None, str | None, bool]:
    """Load the registered champion. Returns (predictor, version, registry_reachable)."""
    from understudy.common.tracking import TrackingUnavailableError, setup_mlflow
    from understudy.student.registry import CHAMPION, alias_version, download_student

    try:
        setup_mlflow()
    except TrackingUnavailableError as exc:
        log.warning(
            "registry unreachable: cannot compare with the champion", extra={"error": str(exc)}
        )
        return None, None, False
    mv = alias_version(name, CHAMPION)
    if mv is None:
        return None, None, True
    student = download_student(f"models:/{name}@{CHAMPION}")
    return StudentPredictor(student, threads=8), str(mv.version), True


def run(params: dict) -> dict[str, Any]:
    """Evaluate the challenger and write the gate decision."""
    g = params["gate"]
    with stage_run("evaluate_gate") as rep:
        train_info = json.loads(resolve("reports/train_run.json").read_text())
        if train_info.get("status") != "trained":
            result = {
                "passed": False,
                "reasons": [f"no trained model ({train_info.get('reason')})"],
            }
            resolve(GATE).write_text(json.dumps(result, indent=2) + "\n")
            return result

        test = pd.read_parquet(resolve("data/dataset/test.parquet"))
        faq_intent = {
            r.faq_id: r.source_intent for r in load_approvals().values() if r.source_intent
        }
        challenger = StudentPredictor(resolve("models/student"), threads=8)
        ch = evaluate(challenger, test, faq_intent)
        ch.update(behavioural(challenger))
        single = StudentPredictor(resolve("models/student"), threads=1)
        ch.update(
            {
                k: round(v, 2)
                for k, v in latency_ms(
                    single,
                    test["text"]
                    .sample(
                        min(int(params["train"]["latency_probe_queries"]), len(test)),
                        random_state=int(params["seed"]),
                    )
                    .tolist(),
                ).items()
            }
        )

        champ, champ_version, reachable = champion_predictor(params["register"]["model_name"])
        cp = evaluate(champ, test, faq_intent) if champ else None

        checks = {
            "traffic_precision": (
                ch["traffic_precision"] >= g["min_traffic_precision"],
                f">= {g['min_traffic_precision']}",
            ),
            "traffic_coverage": (
                ch["traffic_coverage"] >= g["min_traffic_coverage"],
                f">= {g['min_traffic_coverage']}",
            ),
            "oos_false_accept": (
                ch["oos_false_accept"] <= g["max_oos_false_accept"],
                f"<= {g['max_oos_false_accept']}",
            ),
            "latency_p95_ms": (
                ch["latency_p95_ms"] <= g["max_latency_p95_ms"],
                f"<= {g['max_latency_p95_ms']}",
            ),
            "canonical_recall": (
                ch["canonical_recall"] >= g["min_canonical_recall"],
                f">= {g['min_canonical_recall']}",
            ),
            "pii_invariance": (
                ch["pii_invariance"] >= g["min_pii_invariance"],
                f">= {g['min_pii_invariance']}",
            ),
            "gibberish_deferred": (
                ch["gibberish_deferred"] >= g["min_gibberish_deferred"],
                f">= {g['min_gibberish_deferred']}",
            ),
        }
        # Never pass silently without the regression check: no registry, no promotion.
        checks["registry_reachable"] = (reachable, "MLflow reachable (champion comparison)")
        if cp:
            checks["precision_vs_champion"] = (
                ch["traffic_precision"] >= cp["traffic_precision"] - g["max_precision_drop"],
                f">= champion {cp['traffic_precision']} - {g['max_precision_drop']}",
            )
            checks["coverage_vs_champion"] = (
                ch["traffic_coverage"] >= cp["traffic_coverage"] - g["max_coverage_drop"],
                f">= champion {cp['traffic_coverage']} - {g['max_coverage_drop']}",
            )
        passed = all(ok for ok, _ in checks.values())
        result = {
            "passed": passed,
            "reasons": [f"{k} failed ({rule})" for k, (ok, rule) in checks.items() if not ok],
            "checks": {k: {"passed": ok, "rule": rule} for k, (ok, rule) in checks.items()},
            "challenger": ch,
            "champion": cp,
            "champion_version": champ_version,
            "challenger_run_id": train_info["run_id"],
        }
        rep.rows_in = len(test)
        rep.extra.update(
            {
                "passed": passed,
                "traffic_precision": ch["traffic_precision"],
                "coverage": ch["traffic_coverage"],
            }
        )
    resolve(GATE).write_text(json.dumps(result, indent=2) + "\n")
    log.info("gate decided", extra={"passed": passed, "reasons": result["reasons"]})
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(run(load_params()), indent=2))


if __name__ == "__main__":
    main()
