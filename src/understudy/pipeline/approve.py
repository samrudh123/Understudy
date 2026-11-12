"""Human approval of teacher drafts: the gate between ``distil`` and ``build_dataset``.

Every decision is one JSON file under ``data/approvals/`` (git-tracked, auditable):

* **approve**: the draft, possibly edited by the reviewer, becomes FAQ ``faq-<id>``.
* **reject**: recorded so the cluster is never re-drafted.
* **merge**: the draft is a duplicate of an existing FAQ. Its user questions and
  paraphrases are added to that FAQ as extra training phrasings.

The pipeline console calls these functions. The CLI is for terminals and scripts::

    python -m understudy.pipeline.approve list
    python -m understudy.pipeline.approve approve cand-1234 --reviewer alice
    python -m understudy.pipeline.approve reject  cand-1234 --reviewer alice
    python -m understudy.pipeline.approve merge   cand-1234 --into faq-abcd --reviewer alice
    python -m understudy.pipeline.approve auto --reviewer sim-reviewer   # simulation only

``auto`` is a **simulated reviewer** for unattended demos and tests. It reads the
simulation ground truth (``dominant_sim_label``, ``purity``) that a real operator would
judge by reading the draft. It rejects impure or out-of-scope clusters and merges
duplicates of an intent that is already covered.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from understudy.common.config import resolve
from understudy.common.logs import get_logger
from understudy.common.schemas import ApprovalRecord, Draft
from understudy.common.text import dedupe_key

log = get_logger(__name__)

APPROVALS_DIR = "data/approvals"
DRAFTS_PATH = "data/drafts/drafts.json"


class ApprovalError(ValueError):
    """Invalid approval request (unknown draft, already decided, bad merge target...)."""


def faq_id_for(candidate_id: str) -> str:
    """Stable FAQ id derived from the candidate id."""
    return "faq-" + candidate_id.removeprefix("cand-")


def load_drafts(path: str | Path = DRAFTS_PATH) -> list[Draft]:
    """Load the current drafts (empty if ``distil`` has not run)."""
    p = resolve(path)
    if not p.is_file():
        return []
    return [Draft.model_validate(d) for d in json.loads(p.read_text(encoding="utf-8"))["drafts"]]


def load_approvals(approvals_dir: str | Path = APPROVALS_DIR) -> dict[str, ApprovalRecord]:
    """Return every decision keyed by ``faq_id``."""
    d = resolve(approvals_dir)
    out: dict[str, ApprovalRecord] = {}
    if d.is_dir():
        for f in sorted(d.glob("*.json")):
            rec = ApprovalRecord.model_validate_json(f.read_text(encoding="utf-8"))
            out[rec.faq_id] = rec
    return out


def pending(drafts: list[Draft], approvals: dict[str, ApprovalRecord]) -> list[Draft]:
    """Drafts nobody has decided on yet, most frequent first."""
    decided = {r.candidate_id for r in approvals.values()}
    for r in approvals.values():
        decided.update(r.merged_candidates)
    return sorted((d for d in drafts if d.candidate_id not in decided), key=lambda d: -d.support)


def _write(rec: ApprovalRecord, approvals_dir: str | Path) -> Path:
    d = resolve(approvals_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{rec.faq_id}.json"
    path.write_text(rec.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _dedupe(items: list[str], exclude: set[str]) -> list[str]:
    seen, out = set(exclude), []
    for t in items:
        k = dedupe_key(t)
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    return out


def decide(
    candidate_id: str,
    decision: str,
    reviewer: str,
    *,
    question: str | None = None,
    answer: str | None = None,
    paraphrases: list[str] | None = None,
    merge_into: str | None = None,
    drafts: list[Draft] | None = None,
    approvals_dir: str | Path = APPROVALS_DIR,
) -> ApprovalRecord:
    """Record a reviewer decision (``approve``, ``reject`` or ``merge``) for a pending draft."""
    if not reviewer.strip():
        raise ApprovalError("reviewer is required")
    drafts = load_drafts() if drafts is None else drafts
    approvals = load_approvals(approvals_dir)
    draft = next((d for d in pending(drafts, approvals) if d.candidate_id == candidate_id), None)
    if draft is None:
        raise ApprovalError(f"{candidate_id} is not a pending draft")
    now = datetime.now(UTC)

    if decision == "merge":
        target = approvals.get(merge_into or "")
        if target is None or target.decision != "approved":
            raise ApprovalError(f"merge target {merge_into!r} is not an approved FAQ")
        existing = {
            dedupe_key(t) for t in [target.question, *target.paraphrases, *target.log_examples]
        }
        target.paraphrases += _dedupe(paraphrases or draft.paraphrases, existing)
        existing |= {dedupe_key(t) for t in target.paraphrases}
        target.log_examples += _dedupe(draft.examples, existing)
        target.merged_candidates.append(candidate_id)
        target.support = (target.support or 0) + draft.support
        _write(target, approvals_dir)
        log.info("draft merged", extra={"candidate_id": candidate_id, "into": target.faq_id})
        return target

    if decision not in {"approve", "reject"}:
        raise ApprovalError(f"unknown decision {decision!r}")
    q = (question or draft.question).strip()
    a = (answer or draft.answer).strip()
    if decision == "approve" and (not q or not a):
        raise ApprovalError("an approved FAQ needs a question and an answer")
    rec = ApprovalRecord(
        faq_id=faq_id_for(candidate_id),
        candidate_id=candidate_id,
        decision="approved" if decision == "approve" else "rejected",
        question=q,
        answer=a,
        paraphrases=_dedupe(
            paraphrases if paraphrases is not None else draft.paraphrases, {dedupe_key(q)}
        ),
        log_examples=_dedupe(draft.examples, {dedupe_key(q)}),
        reviewer=reviewer.strip(),
        decided_at=now,
        teacher_model=draft.teacher_model,
        prompt_version=draft.prompt_version,
        support=draft.support,
        source_intent=draft.dominant_sim_label,
    )
    _write(rec, approvals_dir)
    log.info(
        "draft decided",
        extra={"candidate_id": candidate_id, "decision": rec.decision, "reviewer": rec.reviewer},
    )
    return rec


def auto_review(
    reviewer: str = "sim-reviewer",
    min_purity: float = 0.6,
    drafts: list[Draft] | None = None,
    approvals_dir: str | Path = APPROVALS_DIR,
) -> dict[str, int]:
    """Decide every pending draft as a simulated reviewer (simulation ground truth; demos only)."""
    drafts = load_drafts() if drafts is None else drafts
    counts = {"approved": 0, "rejected": 0, "merged": 0}
    for d in pending(drafts, load_approvals(approvals_dir)):
        covered = {
            r.source_intent: r.faq_id
            for r in load_approvals(approvals_dir).values()
            if r.decision == "approved" and r.source_intent
        }
        label = d.dominant_sim_label
        if label in (None, "oos") or (d.purity or 0.0) < min_purity:
            decide(d.candidate_id, "reject", reviewer, drafts=drafts, approvals_dir=approvals_dir)
            counts["rejected"] += 1
        elif label in covered:
            decide(
                d.candidate_id,
                "merge",
                reviewer,
                merge_into=covered[label],
                drafts=drafts,
                approvals_dir=approvals_dir,
            )
            counts["merged"] += 1
        else:
            decide(d.candidate_id, "approve", reviewer, drafts=drafts, approvals_dir=approvals_dir)
            counts["approved"] += 1
    return counts


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Review teacher drafts.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("approve", "reject", "merge"):
        p = sub.add_parser(name)
        p.add_argument("candidate_id")
        p.add_argument("--reviewer", required=True)
        if name == "merge":
            p.add_argument("--into", required=True)
    a = sub.add_parser("auto")
    a.add_argument("--reviewer", default="sim-reviewer")
    a.add_argument("--min-purity", type=float, default=0.6)
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for d in pending(load_drafts(), load_approvals()):
            print(
                f"{d.candidate_id}  support={d.support:<4} {d.question!r}\n    -> {d.answer[:100]}"
            )
    elif args.cmd == "auto":
        print(json.dumps(auto_review(args.reviewer, args.min_purity)))
    else:
        rec = decide(
            args.candidate_id, args.cmd, args.reviewer, merge_into=getattr(args, "into", None)
        )
        print(f"{rec.faq_id}: {rec.decision}")


if __name__ == "__main__":
    main()
