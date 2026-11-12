"""``mine_frequent`` stage (Apache Spark): surface fallback questions that have *become* frequent.

Steps
    1. Keep the fallbacks (``route != student``) from the most recent ``window_weeks``.
    2. Count each distinct normalised question. Repetition is the frequency signal.
    3. Embed the distinct questions with the student's sentence encoder, inside Spark
       via ``mapInPandas``.
    4. Cluster with Spark ML KMeans (cosine distance, weighted by count). ``k`` is
       chosen from a grid by silhouette score.
    5. Score each cluster by support (how many questions) and cohesion (mean cosine
       similarity to the centroid). Pick a representative question. Compare the
       centroid against already-approved FAQs to separate genuinely ``new`` topics
       from ``known_but_missed`` ones, which point to a threshold problem rather than
       a knowledge gap.
    6. Keep clusters with enough support and cohesion, ranked by support.

Outputs ``data/candidates/candidates.json`` and ``reports/mine_metrics.json``.
Simulation labels are used *only* to report cluster purity, never to choose clusters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from understudy.common.config import load_params, resolve
from understudy.common.embeddings import get_encoder
from understudy.common.logs import get_logger
from understudy.common.metrics import stage_run
from understudy.common.schemas import ApprovalRecord, Candidate, Route
from understudy.common.spark import get_spark
from understudy.common.text import normalize

log = get_logger(__name__)

CANDIDATES_PATH = "data/candidates/candidates.json"
METRICS_PATH = "reports/mine_metrics.json"


def candidate_id(text_norm: str) -> str:
    """Stable id for a candidate, derived from its representative question."""
    return "cand-" + hashlib.sha1(text_norm.encode("utf-8")).hexdigest()[:10]


def load_approved_texts(approvals_dir: Path) -> list[str]:
    """Return normalised questions and paraphrases of every approved FAQ."""
    texts: list[str] = []
    if approvals_dir.is_dir():
        for f in sorted(approvals_dir.glob("*.json")):
            rec = ApprovalRecord.model_validate_json(f.read_text(encoding="utf-8"))
            if rec.decision == "approved":
                texts.extend(normalize(t) for t in [rec.question, *rec.paraphrases])
    return texts


def read_logs(spark: SparkSession, path: Path) -> DataFrame | None:
    """Read processed logs, or return None if the directory holds no Parquet data yet."""
    if not path.is_dir() or not any(path.rglob("*.parquet")):
        return None
    return spark.read.parquet(str(path))


def fallback_window(logs: DataFrame, window_weeks: int) -> tuple[DataFrame, dict[str, Any]]:
    """Restrict to the recent window, return distinct fallback questions with counts."""
    # The simulated clock defines "now" when simulated traffic exists. Live questions carry
    # real timestamps (a later week number) and so always count as recent.
    simulated = logs.filter(F.col("source") == "simulated")
    clock = simulated if simulated.limit(1).count() else logs
    latest = clock.agg(F.max("week")).first()[0]
    recent = logs.filter(F.col("week") > latest - window_weeks)
    per_week = (
        recent.groupBy("week")
        .agg(
            F.count("*").alias("total"),
            F.sum((F.col("route") != Route.STUDENT.value).cast("int")).alias("fallback"),
        )
        .orderBy("week")
        .collect()
    )
    fallbacks = recent.filter(F.col("route") != Route.STUDENT.value)
    distinct = fallbacks.groupBy("text_norm").agg(
        F.count("*").alias("count"),
        F.first("text").alias("text"),
        F.collect_list("sim_label").alias("sim_labels"),
    )
    info = {
        "latest_week": int(latest),
        "weeks": [int(r["week"]) for r in per_week],
        "fallback_ratio_per_week": {
            str(r["week"]): round(r["fallback"] / r["total"], 4) for r in per_week
        },
    }
    return distinct, info


def embed(spark_df: DataFrame, model: str, revision: str | None) -> DataFrame:
    """Add an ``embedding`` array column using the configured encoder (runs in Spark workers)."""
    out_schema = T.StructType(
        [*spark_df.schema.fields, T.StructField("embedding", T.ArrayType(T.FloatType()))]
    )

    def _embed(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        enc = get_encoder(model, revision)  # cached per Python worker
        for pdf in batches:
            vecs = enc.encode(pdf["text_norm"].tolist()) if len(pdf) else np.zeros((0, 1))
            pdf = pdf.copy()
            pdf["embedding"] = list(vecs)
            yield pdf

    return spark_df.mapInPandas(_embed, schema=out_schema)


def choose_k_and_cluster(
    df: DataFrame, n_rows: int, size_grid: list[int], seed: int
) -> tuple[DataFrame, int, float | None, dict[str, float]]:
    """Fit KMeans for each k in the grid; keep the best silhouette (cosine)."""
    ks = sorted({max(2, min(n_rows - 1, round(n_rows / s))) for s in size_grid})
    evaluator = ClusteringEvaluator(
        featuresCol="features", distanceMeasure="cosine", metricName="silhouette"
    )
    best: tuple[float, int, DataFrame] | None = None
    scores: dict[str, float] = {}
    for k in ks:
        km = KMeans(
            k=k,
            seed=seed,
            featuresCol="features",
            weightCol="weight",
            distanceMeasure="cosine",
            maxIter=50,
        )
        pred = km.fit(df).transform(df)
        sil = float(evaluator.evaluate(pred))
        scores[str(k)] = round(sil, 4)
        log.info("kmeans fitted", extra={"k": k, "silhouette": sil})
        if best is None or sil > best[0]:
            best = (sil, k, pred)
    assert best is not None
    return best[2], best[1], best[0], scores


def _centroid(g: pd.DataFrame) -> np.ndarray:
    emb = np.vstack(g["embedding"].to_numpy()).astype(np.float32)
    c = (emb * g["count"].to_numpy(dtype=np.float32)[:, None]).sum(0)
    return c / max(float(np.linalg.norm(c)), 1e-9)


def merge_similar_clusters(pdf: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Merge clusters whose centroids are closer than ``threshold`` (cosine), transitively.

    KMeans with a silhouette-chosen k tends to split one popular question into several
    clusters. Merging keeps one candidate (and later one FAQ entry) per topic.
    """
    labels = sorted(pdf["prediction"].unique())
    if len(labels) < 2:
        return pdf.assign(group=pdf["prediction"])
    cents = np.vstack([_centroid(pdf[pdf.prediction == lbl]) for lbl in labels])
    parent = list(range(len(labels)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sims = cents @ cents.T
    for i, j in zip(*np.where(np.triu(sims, k=1) >= threshold), strict=True):
        parent[find(int(i))] = find(int(j))
    mapping = {lbl: find(idx) for idx, lbl in enumerate(labels)}
    return pdf.assign(group=pdf["prediction"].map(mapping))


def summarise_clusters(
    pdf: pd.DataFrame, approved: np.ndarray, mp: dict, n_fallback: int
) -> list[Candidate]:
    """Turn clustered rows into ranked :class:`Candidate` objects (small data, pandas)."""
    cands: list[Candidate] = []
    pdf = merge_similar_clusters(pdf, mp["merge_sim_threshold"])
    for _, g in pdf.groupby("group"):
        support = int(g["count"].sum())
        if support < mp["min_support"]:
            continue
        emb = np.vstack(g["embedding"].to_numpy()).astype(np.float32)
        w = g["count"].to_numpy(dtype=np.float32)
        centroid = (emb * w[:, None]).sum(0)
        centroid /= max(np.linalg.norm(centroid), 1e-9)
        sims = emb @ centroid
        cohesion = float(np.average(sims, weights=w))
        if cohesion < mp["min_cohesion"]:
            continue
        order = np.lexsort((-w, -sims))  # closest to centroid, then most frequent
        rep = g.iloc[order[0]]
        known = (
            bool(approved.size) and float((approved @ centroid).max()) >= mp["known_sim_threshold"]
        )
        labels = Counter(
            lbl for lst in g["sim_labels"] for lbl in (lst if lst is not None else []) if lbl
        )
        dominant, purity = (None, None)
        if labels:
            dominant, top = labels.most_common(1)[0]
            purity = round(top / sum(labels.values()), 4)
        examples = g.sort_values("count", ascending=False)["text"].head(
            mp["examples_per_candidate"]
        )
        cands.append(
            Candidate(
                candidate_id=candidate_id(rep["text_norm"]),
                status="known_but_missed" if known else "new",
                representative_question=str(rep["text"]),
                support=support,
                n_unique=len(g),
                share_of_fallbacks=round(support / n_fallback, 4) if n_fallback else 0.0,
                cohesion=round(cohesion, 4),
                examples=[str(t) for t in examples],
                dominant_sim_label=dominant,
                purity=purity,
            )
        )
    cands.sort(key=lambda c: c.support, reverse=True)
    return cands[: mp["max_candidates"]]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def run(
    params: dict,
    logs_dir: str | Path,
    approvals_dir: str | Path,
    out_path: str | Path = CANDIDATES_PATH,
    metrics_path: str | Path = METRICS_PATH,
) -> dict:
    """Execute the stage and return the candidates payload."""
    mp = params["mine"]
    with stage_run("mine_frequent") as rep:
        spark = get_spark("understudy-mine", params["spark"])
        try:
            logs = read_logs(spark, resolve(logs_dir))
            meta: dict[str, Any] = {"window_weeks": mp["window_weeks"]}
            candidates: list[Candidate] = []
            n_fallback = n_distinct = 0
            if logs is not None and logs.limit(1).count():
                distinct, info = fallback_window(logs, mp["window_weeks"])
                meta.update(info)
                distinct = distinct.cache()
                n_distinct = distinct.count()
                n_fallback = int(distinct.agg(F.sum("count")).first()[0] or 0)
                if n_fallback >= mp["min_support"] and n_distinct >= 3:
                    emb = embed(distinct, mp["embedding_model"], mp.get("embedding_revision"))
                    feats = (
                        emb.withColumn("features", array_to_vector("embedding")).withColumn(
                            "weight", F.col("count").cast("double")
                        )
                    ).cache()
                    pred, k, sil, scores = choose_k_and_cluster(
                        feats, n_distinct, mp["avg_cluster_size_grid"], int(params["seed"])
                    )
                    meta.update(k=k, silhouette=round(sil, 4), silhouette_by_k=scores)
                    pdf = (
                        pred.withColumn("embedding", vector_to_array("features"))
                        .select(
                            "text", "text_norm", "count", "sim_labels", "embedding", "prediction"
                        )
                        .toPandas()
                    )
                    approved_texts = load_approved_texts(resolve(approvals_dir))
                    enc = get_encoder(mp["embedding_model"], mp.get("embedding_revision"))
                    approved = enc.encode(approved_texts) if approved_texts else np.zeros((0, 0))
                    candidates = summarise_clusters(pdf, approved, mp, n_fallback)
        finally:
            spark.stop()

        meta.update(n_fallback=n_fallback, n_distinct_fallback=n_distinct)
        payload = {"meta": meta, "candidates": [c.model_dump() for c in candidates]}
        _write(resolve(out_path), payload)

        purities = [c.purity for c in candidates if c.purity is not None]
        metrics = {
            "n_candidates": len(candidates),
            "n_new": sum(c.status == "new" for c in candidates),
            "n_known_but_missed": sum(c.status == "known_but_missed" for c in candidates),
            "n_fallback": n_fallback,
            "coverage_of_fallbacks": round(sum(c.support for c in candidates) / n_fallback, 4)
            if n_fallback
            else 0.0,
            "mean_purity": round(float(np.mean(purities)), 4) if purities else None,
            "distinct_true_intents": len(
                {c.dominant_sim_label for c in candidates if c.dominant_sim_label}
            ),
            "k": meta.get("k"),
            "silhouette": meta.get("silhouette"),
        }
        _write(resolve(metrics_path), metrics)
        rep.rows_in, rep.rows_out = n_fallback, len(candidates)
        rep.extra.update({k: v for k, v in metrics.items() if k != "n_fallback"})
    return payload


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", default="data/processed/querylog")
    parser.add_argument("--approvals-dir", default="data/approvals")
    args = parser.parse_args(argv)
    run(load_params(), args.logs_dir, args.approvals_dir)


if __name__ == "__main__":
    main()
