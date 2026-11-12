"""Shared fixtures: a tiny, synthetic CLINC-shaped dataset and matching params.

Tests never touch the network or the real repo state. Every test that writes
files points ``UNDERSTUDY_ROOT`` at a temporary directory.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

DOMAINS = {
    "d1": ["reset_password", "card_lost", "balance"],
    "d2": ["book_flight", "hotel", "weather"],
    "d3": ["recipe", "timer", "alarm"],
}
CORE = {
    "reset_password": "reset my account password login",
    "card_lost": "lost stolen credit card block",
    "balance": "bank balance checking savings money",
    "book_flight": "book flight ticket airline plane",
    "hotel": "hotel room reservation stay night",
    "weather": "weather forecast rain sunny temperature",
    "recipe": "recipe cook dinner ingredients pasta",
    "timer": "kitchen timer countdown minutes oven",
    "alarm": "wake alarm morning clock set",
}
TRAIN_PREFIX = ["please", "can you", "i want to", "help me"]
TRAIN_SUFFIX = ["now", "today", "quickly"]


def _utts(intent: str, prefixes: list[str], suffixes: list[str]) -> list[list[str]]:
    return [[f"{p} {CORE[intent]} {s}", intent] for p in prefixes for s in suffixes]


def make_tiny_clinc(raw_dir: Path) -> Path:
    """Write data_full.json + domains.json with 3 domains x 3 intents."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, list] = {"train": [], "val": [], "test": []}
    for intents in DOMAINS.values():
        for i in intents:
            data["train"] += _utts(i, TRAIN_PREFIX, TRAIN_SUFFIX)
            data["val"] += _utts(i, ["would you"], ["soon", "later"])
            data["test"] += _utts(i, ["could you"], ["asap", "tonight"])
    data["oos_train"] = [
        [f"random unrelated question number {k} about stars", "oos"] for k in range(20)
    ]
    data["oos_val"] = [[f"another odd query {k} about poems", "oos"] for k in range(10)]
    data["oos_test"] = [[f"held out oos {k} about volcanoes", "oos"] for k in range(10)]
    (raw_dir / "data_full.json").write_text(json.dumps(data), encoding="utf-8")
    (raw_dir / "domains.json").write_text(json.dumps(DOMAINS), encoding="utf-8")
    return raw_dir


TINY_PARAMS = {
    "seed": 7,
    "simulation": {
        "start_date": "2026-08-03",
        "queries_per_week": 300,
        "initial_domains": ["d1", "d2"],
        "domain_schedule": {2: ["d3"]},
        "new_domain_share": 0.3,
        "zipf_s": 1.0,
        "oos_rate": 0.05,
        "pii_injection_rate": 0.05,
    },
    "validate": {
        "expected_intents": 9,
        "expected_domains": 3,
        "min_chars": 1,
        "max_chars": 500,
        "max_invalid_log_fraction": 0.01,
    },
    "spark": {"master": "local[2]", "driver_memory": "1g", "shuffle_partitions": 2},
    "preprocess": {"min_chars": 2, "max_chars": 300},
    "mine": {
        "window_weeks": 2,
        "embedding_model": "hash:256",
        "embedding_revision": None,
        "avg_cluster_size_grid": [4, 8],
        "min_support": 5,
        "min_cohesion": 0.3,
        "merge_sim_threshold": 0.8,
        "known_sim_threshold": 0.9,
        "max_candidates": 20,
        "examples_per_candidate": 5,
    },
    "teacher": {
        "model_repo": "none",
        "model_revision": "none",
        "model_file": "fake.gguf",
        "model_sha256": "none",
        "model_dir": "models/gguf",
        "n_ctx": 512,
        "prompt_file": "configs/prompts/teacher_v1.yaml",
    },
    "distil": {
        "workers": 2,
        "threads_per_worker": 1,
        "max_examples": 5,
        "max_paraphrase_chars": 160,
        "cache_dir": "data/teacher_cache",
    },
    "encoder": {"max_seq_len": 64},
    "dataset": {"val_fraction": 0.3, "neg_log_max_sim": 0.55, "neg_log_max": 200},
    "train": {
        "num_samples": 4,
        "max_concurrent": 2,
        "tau_min_precision": 0.9,
        "tau_max_false_accept": 0.1,
        "search_space": {
            "kind": ["logreg", "centroid"],
            "fallback_class": [True],
            "C": [1.0, 10.0],
            "hidden": [32],
            "alpha": [0.0001, 0.001],
            "temperature": [20.0, 40.0],
        },
        "latency_probe_queries": 20,
    },
    "gate": {
        "min_traffic_precision": 0.5,
        "min_traffic_coverage": 0.05,
        "max_oos_false_accept": 0.5,
        "max_latency_p95_ms": 1000,
        "max_precision_drop": 1.0,
        "max_coverage_drop": 1.0,
        "min_canonical_recall": 0.0,
        "min_pii_invariance": 0.0,
        "min_gibberish_deferred": 0.0,
    },
    "register": {"model_name": "understudy-student-test", "auto_promote": True},
}


@pytest.fixture
def params() -> dict:
    return copy.deepcopy(TINY_PARAMS)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throw-away repo root; all stage reports are written beneath it."""
    monkeypatch.setenv("UNDERSTUDY_ROOT", str(tmp_path))
    monkeypatch.delenv("UNDERSTUDY_PROMETHEUS_PUSHGATEWAY_URL", raising=False)
    return tmp_path


@pytest.fixture
def tiny_raw(root: Path) -> Path:
    return make_tiny_clinc(root / "data/raw/clinc150")
