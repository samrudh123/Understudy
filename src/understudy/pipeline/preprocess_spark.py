"""``preprocess`` stage (Apache Spark): clean, scrub and de-duplicate.

Inputs
    * ``data/raw/clinc150``                 raw dataset
    * ``data/snapshots/querylog.jsonl``     validated log snapshot
Outputs
    * ``data/processed/clinc``      Parquet: split, text, text_norm, intent, domain
    * ``data/processed/querylog``   Parquet partitioned by ``week``

Logs are read with an explicit schema in PERMISSIVE mode. Malformed rows land in
``_corrupt_record`` and are dropped and counted, never silently coerced. Logs are
de-duplicated on ``request_id`` (retries and double writes), **not** on text,
because repeated text is the frequency signal ``mine_frequent`` needs.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from understudy.common.config import load_params, resolve
from understudy.common.logs import get_logger
from understudy.common.metrics import stage_run
from understudy.common.pii import scrub_text
from understudy.common.spark import get_spark
from understudy.common.text import NORMALIZER_VERSION, normalize
from understudy.data.clinc import load_clinc
from understudy.data.ingest import QUERYLOG_SNAPSHOT

log = get_logger(__name__)

QUERYLOG_SCHEMA = T.StructType(
    [
        T.StructField("schema_version", T.IntegerType()),
        T.StructField("request_id", T.StringType()),
        T.StructField("ts", T.StringType()),  # ISO-8601; cast below (robust to Z / microseconds)
        T.StructField("text", T.StringType()),
        T.StructField("route", T.StringType()),
        T.StructField("matched_faq_id", T.StringType()),
        T.StructField("confidence", T.DoubleType()),
        T.StructField("student_version", T.StringType()),
        T.StructField("latency_ms", T.DoubleType()),
        T.StructField("teacher_ttft_ms", T.DoubleType()),
        T.StructField("source", T.StringType()),
        T.StructField("sim_week", T.IntegerType()),
        T.StructField("sim_label", T.StringType()),
        T.StructField("sim_domain", T.StringType()),
        T.StructField("_corrupt_record", T.StringType()),
    ]
)


@F.pandas_udf(T.StringType())
def normalize_udf(s: pd.Series) -> pd.Series:
    """Vectorised wrapper around :func:`understudy.common.text.normalize`."""
    return s.fillna("").map(normalize)


@F.pandas_udf(T.StringType())
def scrub_udf(s: pd.Series) -> pd.Series:
    """Vectorised wrapper around :func:`understudy.common.pii.scrub_text`."""
    return s.fillna("").map(scrub_text)


def clinc_frame(spark: SparkSession, raw_dir: Path) -> DataFrame:
    """Flatten raw CLINC150 into one DataFrame of (split, text, intent, domain)."""
    clinc = load_clinc(raw_dir)
    i2d = clinc.intent_to_domain
    rows = [
        (split, text, label, i2d.get(label))
        for split, pairs in clinc.splits.items()
        for text, label in pairs
    ]
    schema = "split string, text string, intent string, domain string"
    return spark.createDataFrame(rows, schema=schema)


def clean_clinc(df: DataFrame, min_chars: int, max_chars: int) -> DataFrame:
    """Normalise text, filter by length, dedupe per split and drop test/train leakage."""
    df = df.withColumn("text_norm", normalize_udf("text"))
    df = df.filter(F.length("text_norm").between(min_chars, max_chars))
    df = df.dropDuplicates(["split", "text_norm", "intent"])
    seen = (
        df.filter(F.col("split").isin("train", "val", "oos_train", "oos_val"))
        .select("text_norm")
        .distinct()
        .withColumn("_seen", F.lit(True))
    )
    df = df.join(seen, "text_norm", "left")
    # Keep evaluation splits strictly unseen: drop test rows whose text appears in train/val.
    df = df.filter(~(F.col("split").isin("test", "oos_test") & F.col("_seen").isNotNull()))
    return df.drop("_seen").select("split", "text", "text_norm", "intent", "domain")


def read_querylog(spark: SparkSession, path: Path) -> DataFrame:
    """Read the JSONL snapshot with an explicit schema (malformed rows -> _corrupt_record)."""
    return (
        spark.read.schema(QUERYLOG_SCHEMA)
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .json(str(path))
    )


def clean_querylog(df: DataFrame, start_date: str, min_chars: int, max_chars: int) -> DataFrame:
    """Drop corrupt rows, dedupe by request_id, scrub PII again and derive the week."""
    df = df.withColumn("ts", F.col("ts").cast("timestamp"))
    df = df.filter(
        F.col("_corrupt_record").isNull()
        & F.col("request_id").isNotNull()
        & F.col("text").isNotNull()
        & F.col("ts").isNotNull()
    ).drop("_corrupt_record")
    # Load-test / drill traffic exercises the services but must never become training data.
    df = df.filter(F.col("source").isNull() | (F.col("source") != "test"))
    df = df.dropDuplicates(["request_id"])
    df = df.withColumn("text", scrub_udf("text"))
    df = df.withColumn("text_norm", normalize_udf("text"))
    df = df.filter(F.length("text_norm").between(min_chars, max_chars))
    start = date.fromisoformat(start_date).isoformat()
    derived_week = F.floor(F.datediff(F.to_date("ts"), F.lit(start)) / 7).cast("int")
    df = df.withColumn("week", F.coalesce(F.col("sim_week"), derived_week))
    return df.filter(F.col("week") >= 0)


def run(params: dict, raw_dir: str | Path, snapshot_dir: str | Path, out_dir: str | Path) -> dict:
    """Execute the stage and return row counts."""
    pp = params["preprocess"]
    out = resolve(out_dir)
    with stage_run("preprocess") as rep:
        spark = get_spark("understudy-preprocess", params["spark"])
        try:
            raw = clinc_frame(spark, resolve(raw_dir))
            n_raw = raw.count()
            clinc = clean_clinc(raw, pp["min_chars"], pp["max_chars"])
            clinc.repartition(1).write.mode("overwrite").parquet(str(out / "clinc"))
            n_clinc = spark.read.parquet(str(out / "clinc")).count()

            logs_raw = read_querylog(spark, resolve(snapshot_dir) / QUERYLOG_SNAPSHOT).cache()
            n_logs_raw = logs_raw.count()
            n_corrupt = logs_raw.filter(F.col("_corrupt_record").isNotNull()).count()
            logs = clean_querylog(
                logs_raw, params["simulation"]["start_date"], pp["min_chars"], pp["max_chars"]
            )
            logs.repartition("week").write.mode("overwrite").partitionBy("week").parquet(
                str(out / "querylog")
            )
            n_logs = logs.count()
        finally:
            spark.stop()
        counts = {
            "clinc_rows_in": n_raw,
            "clinc_rows_out": n_clinc,
            "querylog_rows_in": n_logs_raw,
            "querylog_rows_out": n_logs,
            "querylog_corrupt_rows": n_corrupt,
            "normalizer_version": NORMALIZER_VERSION,
        }
        rep.rows_in, rep.rows_out = n_raw + n_logs_raw, n_clinc + n_logs
        rep.extra.update(counts)
    return counts


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/clinc150")
    parser.add_argument("--snapshot-dir", default="data/snapshots")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args(argv)
    run(load_params(), args.raw_dir, args.snapshot_dir, args.out_dir)


if __name__ == "__main__":
    main()
