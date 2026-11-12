"""``distil`` stage (Ray): the teacher drafts one FAQ entry per newly-frequent question cluster.

For each mined candidate with status ``new`` that nobody has decided on yet, a pool of
Ray actors runs the teacher, one llama.cpp model per actor and in parallel. Each draft
has a clean canonical question, a reusable answer and N paraphrases, produced as
grammar-constrained JSON. Drafts go to ``data/drafts/drafts.json`` and wait for human
approval (console or ``python -m understudy.pipeline.approve``).

* Paraphrases are cleaned: de-duplicated, PII-checked, guard-checked and length-limited.
* Drafts are cached by (model, prompt version, examples), so re-running the pipeline
  after an approval does not regenerate anything.
* ``known_but_missed`` candidates are not drafted. They show the student's threshold is
  too strict, and they are reported instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from understudy.common.config import load_params, resolve
from understudy.common.logs import get_logger
from understudy.common.metrics import stage_run
from understudy.common.pii import scrub
from understudy.common.schemas import ApprovalRecord, Candidate, Draft
from understudy.common.text import dedupe_key
from understudy.gateway.guard import check_question

log = get_logger(__name__)

CANDIDATES_PATH = "data/candidates/candidates.json"
DRAFTS_PATH = "data/drafts/drafts.json"
REPORT_PATH = "reports/distil_metrics.json"


def decided_candidates(approvals_dir: Path) -> set[str]:
    """Return candidate ids that already have a human decision (approved, rejected or merged)."""
    done: set[str] = set()
    if approvals_dir.is_dir():
        for f in approvals_dir.glob("*.json"):
            rec = ApprovalRecord.model_validate_json(f.read_text(encoding="utf-8"))
            done.add(rec.candidate_id)
            done.update(rec.merged_candidates)
    return done


def clean_paraphrases(
    raw: list[Any], examples: list[str], question: str, max_chars: int
) -> list[str]:
    """Keep usable, distinct paraphrases that are not copies of the examples."""
    seen = {dedupe_key(e) for e in examples} | {dedupe_key(question)}
    out: list[str] = []
    for p in raw:
        if not isinstance(p, str):
            continue
        p = " ".join(p.split())
        key = dedupe_key(p)
        if not key or key in seen or not (3 <= len(p) <= max_chars):
            continue
        if scrub(p).counts or not check_question(p).ok:
            continue
        seen.add(key)
        out.append(p)
    return out


def cache_key(model: str, prompt_version: str, examples: list[str]) -> str:
    """Key a draft by everything that determines it."""
    blob = json.dumps([model, prompt_version, examples], ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def make_worker_class():
    """Define the Ray actor lazily so importing this module does not import Ray."""
    import ray

    @ray.remote
    class DistilWorker:
        def __init__(self, model_path: str, n_ctx: int, threads: int, prompt_file: str) -> None:
            from understudy.teacher.engine import make_engine
            from understudy.teacher.prompts import load_prompts

            self.engine = make_engine(model_path, n_ctx=n_ctx, n_threads=threads)
            self.prompts = load_prompts(prompt_file)

        def draft(self, candidate_id: str, examples: list[str]) -> dict[str, Any]:
            from understudy.teacher.prompts import DISTIL_SCHEMA

            t0 = time.perf_counter()
            settings = self.prompts.settings("distil")
            out, error = None, None
            for attempt in range(2):  # one retry, greedier and with fewer examples
                try:
                    out = self.engine.complete_json(
                        self.prompts.distil(examples if attempt == 0 else examples[:6]),
                        DISTIL_SCHEMA,
                        max_tokens=settings["max_tokens"],
                        temperature=settings["temperature"] if attempt == 0 else 0.0,
                    )
                    break
                except (ValueError, KeyError) as exc:  # malformed / truncated JSON
                    error = f"{type(exc).__name__}: {exc}"
            return {
                "candidate_id": candidate_id,
                "raw": out,
                "error": None if out is not None else error,
                "latency_s": round(time.perf_counter() - t0, 2),
                "model": self.engine.model_name,
            }

    return DistilWorker


def run(params: dict) -> dict[str, Any]:
    """Draft every undecided ``new`` candidate."""
    from understudy.teacher.download import model_path
    from understudy.teacher.prompts import load_prompts

    dp, tp = params["distil"], params["teacher"]
    prompts = load_prompts(tp["prompt_file"])
    fake = os.environ.get("UNDERSTUDY_TEACHER_FAKE", "").lower() in {"1", "true", "yes"}
    model_name = "fake-teacher" if fake else tp["model_file"]
    cache_dir = resolve(dp["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    with stage_run("distil") as rep:
        cands_file = resolve(CANDIDATES_PATH)
        candidates = [
            Candidate.model_validate(c)
            for c in json.loads(cands_file.read_text(encoding="utf-8"))["candidates"]
        ]
        decided = decided_candidates(resolve("data/approvals"))
        todo = [c for c in candidates if c.status == "new" and c.candidate_id not in decided]
        known = [c for c in candidates if c.status == "known_but_missed"]
        rep.rows_in = len(candidates)

        jobs, results, failed = [], {}, []
        for c in todo:
            ex = c.examples[: int(dp["max_examples"])]
            key = cache_key(model_name, prompts.version, ex)
            hit = cache_dir / f"{key}.json"
            if hit.is_file():
                results[c.candidate_id] = json.loads(hit.read_text(encoding="utf-8")) | {
                    "cached": True
                }
            else:
                jobs.append((c, ex, hit))

        t0 = time.perf_counter()
        if jobs:
            import ray
            from ray.util import ActorPool

            n_workers = max(1, min(int(dp["workers"]), len(jobs)))
            threads = int(dp["threads_per_worker"])
            env = {k: v for k, v in os.environ.items() if k.startswith("UNDERSTUDY_")}
            ray.init(
                num_cpus=n_workers * threads,
                include_dashboard=False,
                ignore_reinit_error=True,
                runtime_env={"env_vars": env},
                log_to_driver=False,
            )
            try:
                worker = make_worker_class().options(num_cpus=threads)
                path = str(model_path(tp))
                pool = ActorPool(
                    [
                        worker.remote(
                            path, int(tp["n_ctx"]), threads, str(resolve(tp["prompt_file"]))
                        )
                        for _ in range(n_workers)
                    ]
                )
                by_id = {c.candidate_id: (c, hit) for c, _, hit in jobs}
                for res in pool.map_unordered(
                    lambda a, job: a.draft.remote(job[0].candidate_id, job[1]), jobs
                ):
                    _, hit = by_id[res["candidate_id"]]
                    if res["raw"] is None:
                        failed.append(res["candidate_id"])
                        log.warning(
                            "draft failed",
                            extra={"candidate_id": res["candidate_id"], "error": res["error"]},
                        )
                        continue
                    hit.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
                    results[res["candidate_id"]] = res | {"cached": False}
                    log.info(
                        "draft ready",
                        extra={"candidate_id": res["candidate_id"], "latency_s": res["latency_s"]},
                    )
            finally:
                ray.shutdown()
        gen_s = time.perf_counter() - t0

        drafts: list[Draft] = []
        for c in todo:
            if c.candidate_id not in results:
                continue
            res = results[c.candidate_id]
            raw = res["raw"]
            question = " ".join(str(raw.get("question", "")).split()) or c.representative_question
            drafts.append(
                Draft(
                    candidate_id=c.candidate_id,
                    question=scrub(question).text,
                    answer=scrub(" ".join(str(raw.get("answer", "")).split())).text,
                    paraphrases=clean_paraphrases(
                        raw.get("paraphrases", []),
                        c.examples,
                        question,
                        int(dp["max_paraphrase_chars"]),
                    ),
                    examples=c.examples,
                    support=c.support,
                    share_of_fallbacks=c.share_of_fallbacks,
                    cohesion=c.cohesion,
                    teacher_model=res["model"],
                    prompt_version=prompts.version,
                    generated_at=datetime.now(UTC),
                    latency_s=res["latency_s"],
                    cached=res["cached"],
                    dominant_sim_label=c.dominant_sim_label,
                    purity=c.purity,
                )
            )
        out = resolve(DRAFTS_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"drafts": [d.model_dump(mode="json") for d in drafts]},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        report = {
            "candidates": len(candidates),
            "already_decided": sum(c.candidate_id in decided for c in candidates),
            "known_but_missed": len(known),
            "drafts": len(drafts),
            "generated": len(jobs) - len(failed),
            "failed": len(failed),
            "cache_hits": len(todo) - len(jobs),
            "generation_seconds": round(gen_s, 1),
            "drafts_per_minute": round(len(jobs) / gen_s * 60, 2) if jobs and gen_s else None,
            "workers": int(dp["workers"]),
            "mean_paraphrases": round(sum(len(d.paraphrases) for d in drafts) / len(drafts), 2)
            if drafts
            else 0,
            "teacher_model": model_name,
            "prompt_version": prompts.version,
        }
        rep.rows_out = len(drafts)
        rep.extra.update(report)
    resolve(REPORT_PATH).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(run(load_params()), indent=2))


if __name__ == "__main__":
    main()
