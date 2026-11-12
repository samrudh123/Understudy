"""``build_dataset`` stage: approved FAQs -> answer bank + train/val/test sets for the student.

Labels are an approved ``faq_id`` or ``__fallback__`` (the student must defer).

* **Positives** (per FAQ): the approved question (always train), plus the teacher's
  paraphrases and the real user phrasings from the mined cluster, split
  ``val_fraction`` to validation.
* **Negatives**:
  - CLINC's public out-of-scope set: ``oos_train`` to train, ``oos_val`` to val.
  - *Log negatives*: real fallback questions from the query logs that are far
    (cosine < ``neg_log_max_sim``) from every approved phrasing. They teach the
    student what it does *not* know. No ground truth is used to pick them. Their
    label noise is measured with the simulation labels and reported.
* **Test** (evaluation only, ground truth): the CLINC ``test`` split and ``oos_test``,
  never replayed as traffic. Each row keeps its ``true_intent``. ``evaluate`` maps FAQs
  to intents through ``source_intent`` in the approvals. ``traffic_weight`` re-weights
  each row by its intent's share of the recent query logs (simulation labels), so
  metrics can also be reported for the live traffic mix.

Outputs ``data/dataset/{train,val,test}.parquet``, ``data/dataset/bank.json`` and
``reports/dataset_stats.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from understudy.common.config import load_params, resolve
from understudy.common.logs import get_logger
from understudy.common.metrics import stage_run
from understudy.common.schemas import Route
from understudy.common.text import normalize
from understudy.pipeline.approve import load_approvals
from understudy.student.answer_bank import FAQ, AnswerBank

log = get_logger(__name__)

FALLBACK = "__fallback__"
OUT_DIR = "data/dataset"
STATS = "reports/dataset_stats.json"


def positives(approved: list, val_fraction: float, rng: np.random.Generator) -> pd.DataFrame:
    """Train/val rows for every approved FAQ."""
    rows = []
    for rec in approved:
        rows.append(
            {"text": rec.question, "label": rec.faq_id, "split": "train", "source": "question"}
        )
        seen = {normalize(rec.question)}
        items = []
        for src, texts in (("paraphrase", rec.paraphrases), ("log_example", rec.log_examples)):
            for t in texts:
                k = normalize(t)
                if k and k not in seen:
                    seen.add(k)
                    items.append((t, src))
        order = rng.permutation(len(items))
        n_val = int(round(len(items) * val_fraction)) if len(items) >= 3 else 0
        for rank, i in enumerate(order):
            t, src = items[i]
            rows.append(
                {
                    "text": t,
                    "label": rec.faq_id,
                    "split": "val" if rank < n_val else "train",
                    "source": src,
                }
            )
    return pd.DataFrame(rows, columns=["text", "label", "split", "source"])


def log_negatives(
    logs: pd.DataFrame | None,
    pos_texts: list[str],
    encoder,
    max_sim: float,
    cap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Fallback log questions far from every approved phrasing."""
    cols = ["text", "label", "source", "sim_label"]
    if logs is None or logs.empty or not pos_texts:
        return pd.DataFrame(columns=cols)
    fb = logs[logs["route"] != Route.STUDENT.value]
    fb = fb.drop_duplicates("text_norm")
    fb = fb[~fb["text_norm"].isin({normalize(t) for t in pos_texts})]
    if fb.empty:
        return pd.DataFrame(columns=cols)
    if len(fb) > cap * 3:
        fb = fb.iloc[rng.permutation(len(fb))[: cap * 3]]
    sims = (
        encoder.encode(fb["text_norm"].tolist())
        @ encoder.encode([normalize(t) for t in pos_texts]).T
    )
    keep = fb[sims.max(axis=1) < max_sim]
    if len(keep) > cap:
        keep = keep.iloc[rng.permutation(len(keep))[:cap]]
    return pd.DataFrame(
        {
            "text": keep["text_norm"].values,
            "label": FALLBACK,
            "source": "log_negative",
            "sim_label": keep["sim_label"].values,
        }
    )


def traffic_weights(test: pd.DataFrame, logs: pd.DataFrame | None, window_weeks: int) -> np.ndarray:
    """Weight test rows so each intent's total weight equals its share of recent traffic."""
    if logs is None or logs.empty or logs["sim_label"].isna().all():
        return np.full(len(test), 1.0 / max(len(test), 1))
    weeks = logs["week"].astype(int)
    sim = logs["source"] == "simulated" if "source" in logs else pd.Series(True, index=logs.index)
    latest = weeks[sim].max() if sim.any() else weeks.max()
    recent = logs[weeks > latest - window_weeks]
    share = recent["sim_label"].value_counts(normalize=True)
    per_intent = test["true_intent"].map(test["true_intent"].value_counts())
    return (test["true_intent"].map(share).fillna(0.0) / per_intent).to_numpy(dtype=float)


def _read_parquet(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.is_dir() and any(path.rglob("*.parquet")) else None


def run(params: dict) -> dict:
    """Build the bank and datasets."""
    from understudy.student.encoder import OnnxEncoder

    dp = params["dataset"]
    rng = np.random.default_rng(int(params["seed"]))
    out = resolve(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    with stage_run("build_dataset") as rep:
        approvals = load_approvals()
        approved = sorted(
            (r for r in approvals.values() if r.decision == "approved"), key=lambda r: r.faq_id
        )
        bank = AnswerBank(
            [FAQ(r.faq_id, r.question, r.answer, r.reviewer, r.prompt_version) for r in approved]
        )
        bank.save(out / "bank.json")

        clinc = pd.read_parquet(resolve("data/processed/clinc"))
        pos = positives(approved, float(dp["val_fraction"]), rng)

        logs = _read_parquet(resolve("data/processed/querylog"))
        encoder = OnnxEncoder(resolve("models/encoder"), threads=8)
        lneg = log_negatives(
            logs,
            pos["text"].tolist(),
            encoder,
            float(dp["neg_log_max_sim"]),
            int(dp["neg_log_max"]),
            rng,
        )
        n_lval = int(round(len(lneg) * float(dp["val_fraction"])))
        lneg["split"] = ["val" if i < n_lval else "train" for i in range(len(lneg))]

        oos = clinc[clinc["split"].isin(["oos_train", "oos_val"])]
        oos = pd.DataFrame(
            {
                "text": oos["text"].values,
                "label": FALLBACK,
                "source": "oos",
                "split": np.where(oos["split"] == "oos_train", "train", "val"),
            }
        )
        trainval = pd.concat([pos, oos, lneg.drop(columns="sim_label")], ignore_index=True)
        trainval["text"] = trainval["text"].map(normalize)
        trainval = trainval[trainval["text"].str.len() > 0].drop_duplicates(["text", "label"])

        covered = {r.source_intent for r in approved if r.source_intent}
        test = clinc[clinc["split"].isin(["test", "oos_test"])]
        test = pd.DataFrame(
            {
                "text": test["text_norm"].values,
                "true_intent": test["intent"].values,
                "kind": np.where(
                    test["intent"] == "oos",
                    "oos",
                    np.where(test["intent"].isin(covered), "covered", "uncovered"),
                ),
            }
        )

        test["traffic_weight"] = traffic_weights(test, logs, int(params["mine"]["window_weeks"]))

        for split in ("train", "val"):
            part = trainval[trainval["split"] == split].drop(columns="split")
            part.reset_index(drop=True).to_parquet(out / f"{split}.parquet", index=False)
        test.to_parquet(out / "test.parquet", index=False)

        noise = float(lneg["sim_label"].isin(covered).mean()) if len(lneg) and covered else None
        stats = {
            "faqs": len(bank),
            "bank_version": bank.version,
            "rejected": sum(r.decision == "rejected" for r in approvals.values()),
            "train_rows": int((trainval["split"] == "train").sum()),
            "val_rows": int((trainval["split"] == "val").sum()),
            "test_rows": len(test),
            "rows_by_source": trainval.groupby(["split", "source"])
            .size()
            .unstack(0)
            .fillna(0)
            .astype(int)
            .to_dict(),
            "log_negatives": len(lneg),
            "log_negative_label_noise": None if noise is None else round(noise, 4),
            "test_kinds": test["kind"].value_counts().to_dict(),
            "covered_intents": len(covered),
        }
        rep.rows_in = len(pos) + (0 if logs is None else len(logs))
        rep.rows_out = len(trainval) + len(test)
        rep.extra.update({k: stats[k] for k in ("faqs", "train_rows", "val_rows", "log_negatives")})
    resolve(STATS).write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    log.info("dataset built", extra={k: v for k, v in stats.items() if k != "rows_by_source"})
    return stats


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(run(load_params()), indent=2))


if __name__ == "__main__":
    main()
