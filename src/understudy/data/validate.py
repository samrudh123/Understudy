"""``validate`` stage: schema and value checks on the raw dataset and the log snapshot.

Writes ``reports/validation.json``. Exits non-zero (stopping ``dvc repro``) if any
hard check fails. Soft findings are recorded as warnings.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from understudy.common.config import load_params, resolve
from understudy.common.logs import get_logger
from understudy.common.metrics import stage_run
from understudy.common.schemas import FeedbackRecord, QueryLogRecord
from understudy.data.clinc import IN_SCOPE_SPLITS, OOS_LABEL, OOS_SPLITS, load_clinc
from understudy.data.ingest import FEEDBACK_SNAPSHOT, QUERYLOG_SNAPSHOT

log = get_logger(__name__)

REPORT_PATH = "reports/validation.json"


class Checks:
    """Collects named pass/fail checks and warnings."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []
        self.warnings: list[str] = []

    def check(self, name: str, ok: bool, detail: Any = None) -> bool:
        """Record a hard check."""
        self.results.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok:
            log.error("validation check failed", extra={"check": name, "detail": detail})
        return ok

    def warn(self, msg: str) -> None:
        """Record a soft finding that does not fail the pipeline."""
        self.warnings.append(msg)
        log.warning(msg)

    @property
    def passed(self) -> bool:
        """Return True if all hard checks passed."""
        return all(r["passed"] for r in self.results)


def validate_dataset(raw_dir: Path, vparams: dict, checks: Checks) -> dict[str, Any]:
    """Validate the CLINC150 files and return summary statistics."""
    try:
        clinc = load_clinc(raw_dir)
    except (OSError, json.JSONDecodeError, IndexError, TypeError) as exc:
        checks.check("dataset_loadable", False, str(exc))
        return {}
    checks.check("dataset_loadable", True)

    missing = [s for s in (*IN_SCOPE_SPLITS, *OOS_SPLITS) if not clinc.splits.get(s)]
    checks.check("all_splits_present", not missing, {"missing": missing})

    i2d = clinc.intent_to_domain
    checks.check(
        "domain_count",
        len(clinc.domains) == vparams["expected_domains"],
        {"found": len(clinc.domains), "expected": vparams["expected_domains"]},
    )
    checks.check(
        "intent_count",
        len(i2d) == vparams["expected_intents"],
        {"found": len(i2d), "expected": vparams["expected_intents"]},
    )
    dup_intents = [
        i for i, c in Counter(i for v in clinc.domains.values() for i in v).items() if c > 1
    ]
    checks.check("intent_in_exactly_one_domain", not dup_intents, {"duplicates": dup_intents})

    lo, hi = vparams["min_chars"], vparams["max_chars"]
    bad_rows: list[tuple[str, int]] = []
    unknown_labels: set[str] = set()
    per_split: dict[str, int] = {}
    for split, rows in clinc.splits.items():
        per_split[split] = len(rows)
        expect_oos = split in OOS_SPLITS
        for idx, (text, label) in enumerate(rows):
            if not isinstance(text, str) or not (lo <= len(text.strip()) <= hi):
                bad_rows.append((split, idx))
            if expect_oos and label != OOS_LABEL:
                unknown_labels.add(f"{split}:{label}")
            if not expect_oos and label not in i2d:
                unknown_labels.add(f"{split}:{label}")
    checks.check(
        "text_length_in_range", not bad_rows, {"bad_rows": bad_rows[:20], "n": len(bad_rows)}
    )
    checks.check("labels_known", not unknown_labels, {"unknown": sorted(unknown_labels)[:20]})

    by_intent = {s: {lbl for _, lbl in clinc.splits.get(s, [])} for s in IN_SCOPE_SPLITS}
    missing_cover = {
        s: sorted(set(i2d) - labels)[:10] for s, labels in by_intent.items() if set(i2d) - labels
    }
    checks.check("every_intent_in_every_split", not missing_cover, missing_cover)

    train_texts = {t.strip().lower() for s in ("train", "val") for t, _ in clinc.splits.get(s, [])}
    leaks = sum(1 for t, _ in clinc.splits.get("test", []) if t.strip().lower() in train_texts)
    if leaks:
        checks.warn(f"{leaks} test utterances also appear in train/val; preprocess drops them")
    return {
        "rows_per_split": per_split,
        "domains": len(clinc.domains),
        "intents": len(i2d),
        "test_train_overlap": leaks,
    }


def _validate_jsonl(
    path: Path, model: type, max_bad_frac: float, name: str, checks: Checks
) -> dict[str, Any]:
    total, bad, examples = 0, 0, []
    routes: Counter = Counter()
    weeks: Counter = Counter()
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                total += 1
                try:
                    rec = model.model_validate_json(line)
                except ValidationError as exc:
                    bad += 1
                    if len(examples) < 5:
                        examples.append({"line": lineno, "error": exc.errors()[0]["msg"]})
                    continue
                if isinstance(rec, QueryLogRecord):
                    routes[rec.route.value] += 1
                    if rec.sim_week is not None:
                        weeks[rec.sim_week] += 1
    frac = bad / total if total else 0.0
    checks.check(
        f"{name}_invalid_fraction",
        frac <= max_bad_frac,
        {"invalid": bad, "total": total, "fraction": round(frac, 4), "examples": examples},
    )
    out: dict[str, Any] = {"rows": total, "invalid_rows": bad}
    if routes:
        out["routes"] = dict(routes)
    if weeks:
        out["rows_per_sim_week"] = {str(k): v for k, v in sorted(weeks.items())}
    return out


def run(
    params: dict,
    raw_dir: str | Path,
    snapshot_dir: str | Path,
    report_path: str | Path = REPORT_PATH,
) -> bool:
    """Run all checks, write the report and return True if every hard check passed."""
    vparams = params["validate"]
    checks = Checks()
    with stage_run("validate") as rep:
        dataset = validate_dataset(resolve(raw_dir), vparams, checks)
        snap = resolve(snapshot_dir)
        logs = _validate_jsonl(
            snap / QUERYLOG_SNAPSHOT,
            QueryLogRecord,
            vparams["max_invalid_log_fraction"],
            "querylog",
            checks,
        )
        feedback = _validate_jsonl(
            snap / FEEDBACK_SNAPSHOT,
            FeedbackRecord,
            vparams["max_invalid_log_fraction"],
            "feedback",
            checks,
        )
        report = {
            "passed": checks.passed,
            "n_checks": len(checks.results),
            "n_failed": sum(not r["passed"] for r in checks.results),
            "dataset": dataset,
            "querylog": logs,
            "feedback": feedback,
            "checks": checks.results,
            "warnings": checks.warnings,
        }
        out = resolve(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        rep.rows_in = sum(dataset.get("rows_per_split", {}).values()) + logs["rows"]
        rep.rows_out = rep.rows_in
        rep.extra.update(passed=checks.passed, n_failed=report["n_failed"])
        if not checks.passed:
            raise SystemExit(1)  # recorded as a failed stage, stops dvc repro
    return checks.passed


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/clinc150")
    parser.add_argument("--snapshot-dir", default="data/snapshots")
    args = parser.parse_args(argv)
    try:
        run(load_params(), args.raw_dir, args.snapshot_dir)
    except SystemExit as exc:
        log.error("validation failed; see reports/validation.json")
        sys.exit(exc.code)


if __name__ == "__main__":
    main()
