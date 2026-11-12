import numpy as np
import pandas as pd
import pytest

from understudy.pipeline.mine_frequent_spark import (
    candidate_id,
    merge_similar_clusters,
    summarise_clusters,
)

pytestmark = pytest.mark.unit

MP = {
    "min_support": 3,
    "min_cohesion": 0.5,
    "merge_sim_threshold": 0.9,
    "known_sim_threshold": 0.95,
    "examples_per_candidate": 2,
    "max_candidates": 10,
}


def _row(text: str, vec: list[float], count: int, pred: int, label: str) -> dict:
    v = np.array(vec, dtype=np.float32)
    return {
        "text": text,
        "text_norm": text,
        "count": count,
        "embedding": v / np.linalg.norm(v),
        "prediction": pred,
        "sim_labels": np.array([label] * count),
    }


@pytest.fixture
def pdf() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("book a flight", [1, 0, 0], 5, 0, "book_flight"),
            _row("book flights please", [0.98, 0.05, 0], 4, 1, "book_flight"),  # split cluster
            _row("reset password", [0, 1, 0], 6, 2, "reset_password"),
            _row("forgot my password", [0, 0.97, 0.1], 2, 2, "reset_password"),
            _row("rare thing", [0, 0, 1], 1, 3, "oos"),  # below min_support
        ]
    )


def test_merge_joins_near_duplicate_clusters(pdf: pd.DataFrame) -> None:
    merged = merge_similar_clusters(pdf, 0.9)
    groups = merged.set_index("text")["group"]
    assert groups["book a flight"] == groups["book flights please"]
    assert groups["book a flight"] != groups["reset password"]


def test_summarise_ranks_filters_and_scores(pdf: pd.DataFrame) -> None:
    cands = summarise_clusters(pdf, np.zeros((0, 0)), MP, n_fallback=18)
    assert [c.dominant_sim_label for c in cands] == ["book_flight", "reset_password"]
    assert cands[0].support == 9 and cands[0].n_unique == 2
    assert cands[0].purity == 1.0 and cands[0].status == "new"
    assert cands[0].share_of_fallbacks == pytest.approx(0.5)
    assert len(cands[0].examples) == 2


def test_known_but_missed_when_close_to_approved_faq(pdf: pd.DataFrame) -> None:
    approved = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    cands = {c.dominant_sim_label: c for c in summarise_clusters(pdf, approved, MP, 18)}
    assert cands["reset_password"].status == "known_but_missed"
    assert cands["book_flight"].status == "new"


def test_candidate_id_is_stable() -> None:
    assert candidate_id("book a flight") == candidate_id("book a flight")
    assert candidate_id("book a flight").startswith("cand-")
