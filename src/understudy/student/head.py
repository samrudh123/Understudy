"""Classification heads on top of the frozen sentence encoder.

Inference is plain numpy (no sklearn or torch at the edge). Training uses scikit-learn
and copies the weights out.

* ``logreg``   : multinomial logistic regression
* ``mlp``      : one hidden ReLU layer
* ``centroid`` : cosine similarity to each FAQ's mean embedding, softmax with a temperature

With ``fallback_class`` the ``__fallback__`` negatives become an extra class the head
can vote for. Otherwise negatives are only used to tune the threshold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FALLBACK = "__fallback__"


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Head:
    """A trained head: ``predict_proba(embeddings) -> (n, len(labels))``."""

    def __init__(
        self, kind: str, labels: list[str], weights: dict[str, np.ndarray], config: dict[str, Any]
    ) -> None:
        self.kind, self.labels, self.w, self.config = kind, labels, weights, config

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Class probabilities for L2-normalised embeddings ``x``."""
        w = self.w
        if self.kind == "logreg":
            return _softmax(x @ w["coef"].T + w["intercept"])
        if self.kind == "mlp":
            h = np.maximum(x @ w["w1"] + w["b1"], 0.0)
            return _softmax(h @ w["w2"] + w["b2"])
        if self.kind == "centroid":
            return _softmax((x @ w["centroids"].T) * float(self.config["temperature"]))
        raise ValueError(f"unknown head kind {self.kind!r}")

    def save(self, directory: str | Path) -> None:
        """Write ``head.npz`` + ``head.json``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "head.npz", **self.w)
        (d / "head.json").write_text(
            json.dumps({"kind": self.kind, "labels": self.labels, "config": self.config}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> Head:
        """Read a saved head."""
        d = Path(directory)
        meta = json.loads((d / "head.json").read_text(encoding="utf-8"))
        with np.load(d / "head.npz") as npz:
            w = {k: npz[k] for k in npz.files}
        return cls(meta["kind"], meta["labels"], w, meta["config"])


def train_head(x: np.ndarray, y: list[str], config: dict[str, Any], seed: int = 42) -> Head:
    """Fit a head described by ``config`` (``kind`` plus its hyper-parameters)."""
    kind = config["kind"]
    y = np.asarray(y)
    if not config.get("fallback_class", False) or kind == "centroid":
        keep = y != FALLBACK
        x, y = x[keep], y[keep]
    labels = sorted(set(y.tolist()))
    if len(labels) < 2:
        raise ValueError("need at least two classes to train a head")
    idx = {lbl: i for i, lbl in enumerate(labels)}
    yi = np.array([idx[v] for v in y])

    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(C=float(config["C"]), max_iter=3000, class_weight="balanced")
        m.fit(x, yi)
        w = {"coef": m.coef_.astype(np.float32), "intercept": m.intercept_.astype(np.float32)}
    elif kind == "mlp":
        from sklearn.neural_network import MLPClassifier

        m = MLPClassifier(
            hidden_layer_sizes=(int(config["hidden"]),),
            alpha=float(config["alpha"]),
            max_iter=600,
            random_state=seed,
        )
        m.fit(x, yi)
        w = {
            "w1": m.coefs_[0].astype(np.float32),
            "b1": m.intercepts_[0].astype(np.float32),
            "w2": m.coefs_[1].astype(np.float32),
            "b2": m.intercepts_[1].astype(np.float32),
        }
    elif kind == "centroid":
        cents = np.stack([x[yi == i].mean(axis=0) for i in range(len(labels))])
        cents /= np.linalg.norm(cents, axis=1, keepdims=True)
        w = {"centroids": cents.astype(np.float32)}
    else:
        raise ValueError(f"unknown head kind {kind!r}")
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    return Head(kind, labels, w, clean)
