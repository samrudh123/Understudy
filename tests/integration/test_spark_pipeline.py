"""End-to-end run of the Phase 1 stages on synthetic data with a real local Spark."""

import json
from pathlib import Path

import pandas as pd
import pytest

from understudy.data import ingest, validate
from understudy.data.simulate_traffic import simulate_offline
from understudy.pipeline import mine_frequent_spark, preprocess_spark

pytestmark = pytest.mark.integration


@pytest.fixture
def processed(root: Path, params: dict, tiny_raw: Path) -> Path:
    simulate_offline(params, [0, 1, 2], tiny_raw, root / "data/querylogs", root / "data/approvals")
    # A duplicated line (same request_id) and a corrupt line must both be handled.
    log_file = root / "data/querylogs/sim-week-02.jsonl"
    first = log_file.read_text().splitlines()[0]
    load_test = json.loads(first) | {"request_id": "load-test-1", "source": "test"}
    load_test.pop("sim_week", None)
    with log_file.open("a") as fh:
        fh.write(first + "\n")
        fh.write("{not json\n")
        fh.write(json.dumps(load_test) + "\n")  # load-test traffic must never be mined
    ingest.ingest_logs(root / "data/querylogs", root / "data/snapshots")
    # 1 corrupt row out of 903 is under the 1% validation threshold.
    validate.run(params, tiny_raw, root / "data/snapshots")
    counts = preprocess_spark.run(
        params, tiny_raw, root / "data/snapshots", root / "data/processed"
    )
    assert counts["querylog_rows_in"] == 903
    assert counts["querylog_corrupt_rows"] == 1
    assert counts["querylog_rows_out"] == 900  # duplicate request_id and test traffic removed
    return root / "data/processed"


def test_preprocess_outputs(processed: Path) -> None:
    logs = pd.read_parquet(processed / "querylog")
    assert set(logs["week"].astype(int)) == {0, 1, 2}
    assert logs["request_id"].is_unique
    assert not logs["text"].str.contains("@example.com").any()
    assert (logs["text_norm"] == logs["text_norm"].str.lower()).all()

    clinc = pd.read_parquet(processed / "clinc")
    assert {"train", "val", "test", "oos_train", "oos_val", "oos_test"} == set(clinc["split"])
    train = set(clinc.loc[clinc.split.isin(["train", "val"]), "text_norm"])
    assert not set(clinc.loc[clinc.split == "test", "text_norm"]) & train


def test_mine_frequent_finds_the_new_domain(root: Path, params: dict, processed: Path) -> None:
    payload = mine_frequent_spark.run(params, processed / "querylog", root / "data/approvals")
    cands = payload["candidates"]
    assert cands, "expected at least one frequent fallback cluster"
    assert payload["meta"]["latest_week"] == 2 and payload["meta"]["weeks"] == [1, 2]
    supports = [c["support"] for c in cands]
    assert supports == sorted(supports, reverse=True)
    assert all(c["support"] >= params["mine"]["min_support"] for c in cands)
    # Week 2 introduces domain d3; its intents must surface as candidates.
    labels = {c["dominant_sim_label"] for c in cands}
    assert labels & {"recipe", "timer", "alarm"}
    metrics = json.loads((root / "reports/mine_metrics.json").read_text())
    assert metrics["n_candidates"] == len(cands)
    assert metrics["mean_purity"] > 0.7


def test_mine_frequent_cold_start_without_logs(root: Path, params: dict) -> None:
    payload = mine_frequent_spark.run(
        params, root / "data/processed/querylog", root / "data/approvals"
    )
    assert payload["candidates"] == []


def test_fallback_window_ignores_bootstrap_history_once_a_student_routes(params: dict) -> None:
    from understudy.common.spark import get_spark

    spark = get_spark("test-window", params["spark"])
    rows = [
        # (text_norm, route, source, student_version, week)
        ("old covered question", "student", "simulated", "oracle", 0),
        ("old fallback question", "teacher", "simulated", None, 1),
        ("library hours", "teacher", "live", "1", 15),
        ("library hours", "teacher", "live", "1", 15),
        ("pin change", "student", "live", "1", 15),
    ]
    logs = spark.createDataFrame(
        [(t, t, r, s, v, w, None) for t, r, s, v, w in rows],
        "text_norm string, text string, route string, source string, "
        "student_version string, week int, sim_label string",
    )
    distinct, info = mine_frequent_spark.fallback_window(logs, 2)
    got = {r["text_norm"]: r["count"] for r in distinct.collect()}
    assert got == {"library hours": 2}
    assert info["bootstrap_history_excluded"] is True and info["latest_week"] == 15

    # Cold start: only bootstrap history exists, so it is mined.
    distinct, info = mine_frequent_spark.fallback_window(logs.filter("week < 2"), 2)
    assert {r["text_norm"] for r in distinct.collect()} == {"old fallback question"}
    assert info["bootstrap_history_excluded"] is False
