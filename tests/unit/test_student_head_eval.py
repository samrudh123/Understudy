import numpy as np
import pytest

from understudy.student.answer_bank import FAQ, AnswerBank
from understudy.student.evaluation import Outcomes, choose_tau, metrics_at
from understudy.student.head import FALLBACK, Head, train_head

pytestmark = pytest.mark.unit


def blobs(seed: int = 0):
    """Three well-separated classes plus scattered negatives, unit-normalised."""
    rng = np.random.default_rng(seed)
    centers = np.eye(8)[:3]
    x, y = [], []
    for i, c in enumerate(centers):
        pts = c + 0.05 * rng.normal(size=(20, 8))
        x.append(pts)
        y += [f"faq-{i}"] * 20
    neg = rng.normal(size=(30, 8))
    neg[:, :3] = 0
    x.append(neg)
    y += [FALLBACK] * 30
    x = np.vstack(x)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32), y


@pytest.mark.parametrize(
    "config",
    [
        {"kind": "logreg", "C": 5.0, "fallback_class": True},
        {"kind": "logreg", "C": 5.0, "fallback_class": False},
        {"kind": "mlp", "hidden": 16, "alpha": 1e-4, "fallback_class": True},
        {"kind": "centroid", "temperature": 30.0},
    ],
)
def test_heads_fit_and_round_trip(config, tmp_path) -> None:
    x, y = blobs()
    head = train_head(x, y, config)
    assert (FALLBACK in head.labels) == (
        config.get("fallback_class", False) and config["kind"] != "centroid"
    )
    probs = head.predict_proba(x)
    assert probs.shape == (len(x), len(head.labels))
    np.testing.assert_allclose(probs.sum(1), 1.0, rtol=1e-5)
    pred = np.array(head.labels)[probs.argmax(1)]
    assert (pred[:60] == np.array(y[:60])).mean() > 0.95
    head.save(tmp_path / "h")
    again = Head.load(tmp_path / "h")
    np.testing.assert_allclose(again.predict_proba(x), probs, rtol=1e-5)


def test_head_needs_two_classes() -> None:
    x, _ = blobs()
    with pytest.raises(ValueError):
        train_head(x[:20], ["faq-0"] * 20, {"kind": "logreg", "C": 1.0})


def test_metrics_and_tau_choice() -> None:
    conf = np.array([0.95, 0.9, 0.85, 0.6, 0.55, 0.3])
    o = Outcomes(
        confidence=conf,
        vote_ok=np.ones(6, bool),
        correct=np.array([1, 1, 1, 0, 1, 0], bool),
        answerable=np.array([1, 1, 1, 0, 1, 0], bool),
        is_oos=np.array([0, 0, 0, 1, 0, 0], bool),
        weight=np.array([0.3, 0.3, 0.1, 0.1, 0.1, 0.1]),
    )
    m = metrics_at(o, 0.5)
    assert m["instant_precision"] == pytest.approx(4 / 5)
    assert m["coverage"] == 1.0 and m["oos_false_accept"] == 1.0
    assert m["traffic_precision"] == pytest.approx(0.8 / 0.9)
    best, feasible = choose_tau(o, min_precision=1.0, max_false_accept=0.0)
    assert feasible and 0.6 < best["tau"] <= 0.85
    assert best["coverage"] == pytest.approx(3 / 4)


def test_tau_infeasible_returns_best_precision() -> None:
    o = Outcomes(
        np.array([0.9, 0.8]),
        np.ones(2, bool),
        np.array([0, 0], bool),
        np.array([1, 1], bool),
        np.array([0, 0], bool),
    )
    _, feasible = choose_tau(o, 0.99, 0.0)
    assert not feasible


def test_answer_bank_version_is_content_hash(tmp_path) -> None:
    a = AnswerBank([FAQ("faq-1", "q", "a"), FAQ("faq-2", "q2", "a2")])
    b = AnswerBank([FAQ("faq-2", "q2", "a2"), FAQ("faq-1", "q", "a")])
    assert a.version == b.version and len(a) == 2 and "faq-1" in a
    assert AnswerBank([FAQ("faq-1", "q", "changed")]).version != a.version
    a.save(tmp_path / "bank.json")
    assert AnswerBank.load(tmp_path / "bank.json").version == a.version
