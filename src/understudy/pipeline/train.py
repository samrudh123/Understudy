"""``train`` stage: Ray Tune searches student heads, MLflow records everything.

1. Embed train/val/test once with the INT8 ONNX encoder (the same graph the edge runs).
2. **Ray Tune** samples ``num_samples`` head configs (kind, regularisation, fallback
   class...). Each trial fits on train, then picks its own threshold ``tau`` on validation:
   the lowest ``tau`` with precision >= ``tau_min_precision`` and false-accept
   <= ``tau_max_false_accept``. The trial's score is the coverage at that ``tau``.
   Trials that cannot meet the targets get a negative score.
3. The best config is refit and evaluated on the held-out test set (ground truth), then
   assembled into a student directory (encoder + head + answer bank + tau).
4. **MLflow**: one parent run with params, lineage (git commit, data hash, prompt
   versions, teacher model), val/test metrics, latency, size, threshold curve,
   confusion matrix and trials table, plus one nested run per Tune trial. The student
   is logged as an MLflow **pyfunc** model.

Outputs ``models/student/``, ``reports/train_metrics.json``, ``reports/train_run.json``
and ``reports/plots/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from understudy.common.config import load_params, resolve
from understudy.common.lineage import git_commit, git_is_dirty
from understudy.common.logs import get_logger
from understudy.common.metrics import stage_run
from understudy.common.tracking import setup_mlflow
from understudy.pipeline.approve import load_approvals
from understudy.student.answer_bank import AnswerBank
from understudy.student.evaluation import Outcomes, choose_tau, metrics_at
from understudy.student.head import FALLBACK, Head, train_head
from understudy.student.model import StudentPredictor, save_student

log = get_logger(__name__)

EXPERIMENT = "understudy-student"
PKG_DIR = Path(__file__).resolve().parents[1]  # the installed ``understudy`` package
DATA_DIR = "data/dataset"
STUDENT_DIR = "models/student"
METRICS = "reports/train_metrics.json"
RUN_INFO = "reports/train_run.json"
PLOTS = "reports/plots"


def dir_hash(path: Path) -> str:
    """Content hash of a directory (stands in for the DVC data version)."""
    h = hashlib.md5()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(f.relative_to(path).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def outcomes_from_labels(
    head: Head, probs: np.ndarray, labels: np.ndarray, is_oos: np.ndarray
) -> Outcomes:
    """Outcomes when the right answer is a ``faq_id`` (validation)."""
    pred, conf, vote_ok = decode(head, probs)
    return Outcomes(conf, vote_ok, pred == labels, labels != FALLBACK, is_oos)


def outcomes_from_intents(
    head: Head, probs: np.ndarray, test: pd.DataFrame, faq_intent: dict[str, str]
) -> Outcomes:
    """Outcomes when correctness is judged by intent (test, simulation ground truth)."""
    pred, conf, vote_ok = decode(head, probs)
    pred_intent = np.array([faq_intent.get(p, "?") for p in pred])
    return Outcomes(
        conf,
        vote_ok,
        pred_intent == test["true_intent"].to_numpy(),
        (test["kind"] == "covered").to_numpy(),
        (test["kind"] == "oos").to_numpy(),
        test["traffic_weight"].to_numpy() if "traffic_weight" in test else None,
    )


def decode(head: Head, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best FAQ, its probability, and whether the head did *not* vote fallback."""
    labels = np.array(head.labels)
    faq_cols = np.where(labels != FALLBACK)[0]
    best = faq_cols[np.argmax(probs[:, faq_cols], axis=1)]
    conf = probs[np.arange(len(probs)), best]
    vote_ok = labels[np.argmax(probs, axis=1)] != FALLBACK
    return labels[best], conf, vote_ok


def score_trial(m: dict[str, float], feasible: bool) -> float:
    """Coverage if the targets are met, else a negative penalty."""
    if feasible:
        return m["coverage"]
    return -1.0 + 0.5 * m["instant_precision"] - m["false_accept"]


def trainable(config: dict[str, Any], data: dict[str, Any]) -> None:
    """One Ray Tune trial."""
    from ray import tune

    head = train_head(data["x_train"], data["y_train"], config, seed=data["seed"])
    probs = head.predict_proba(data["x_val"])
    o = outcomes_from_labels(head, probs, data["y_val"], data["oos_val"])
    m, feasible = choose_tau(o, data["min_precision"], data["max_fa"])
    tune.report(
        {
            **{f"val_{k}": v for k, v in m.items()},
            "feasible": feasible,
            "score": score_trial(m, feasible),
        }
    )


def search_space(ss: dict[str, Any]) -> dict[str, Any]:
    """Translate ``params.yaml`` into a Ray Tune search space."""
    from ray import tune

    return {
        "kind": tune.choice(ss["kind"]),
        "fallback_class": tune.choice(ss["fallback_class"]),
        "C": tune.loguniform(*ss["C"]),
        "hidden": tune.choice(ss["hidden"]),
        "alpha": tune.loguniform(*ss["alpha"]),
        "temperature": tune.uniform(*ss["temperature"]),
    }


def relevant(config: dict[str, Any]) -> dict[str, Any]:
    """Only the hyper-parameters the chosen head actually uses."""
    keys = {
        "logreg": ["C", "fallback_class"],
        "mlp": ["hidden", "alpha", "fallback_class"],
        "centroid": ["temperature"],
    }[config["kind"]]
    return {"kind": config["kind"], **{k: config[k] for k in keys}}


def latency_ms(predictor: StudentPredictor, texts: list[str]) -> dict[str, float]:
    """Single-question latency (encode + head), one thread, after warm-up."""
    for t in texts[:10]:
        predictor.predict([t])
    lat = []
    for t in texts:
        t0 = time.perf_counter()
        predictor.predict([t])
        lat.append((time.perf_counter() - t0) * 1000)
    return {
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
    }


def plot_threshold_curve(o: Outcomes, tau: float, path: Path) -> None:
    """Precision / coverage / false-accept against tau on validation."""
    from understudy.common.plotting import SERIES, TEXT, TEXT_2, plt, style_axes

    grid = np.round(np.arange(0.05, 1.0, 0.01), 2)
    rows = [metrics_at(o, float(t)) for t in grid]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (key, name) in enumerate(
        [
            ("instant_precision", "Instant-answer precision"),
            ("coverage", "Coverage"),
            ("false_accept", "False-accept rate"),
        ]
    ):
        ys = [r[key] for r in rows]
        ax.plot(grid, ys, color=SERIES[i], linewidth=2, label=name)
        ax.text(grid[-1] + 0.01, ys[-1], name, color=TEXT, fontsize=9, va="center")
    ax.axvline(tau, color=TEXT_2, linewidth=1, linestyle="--")
    ax.text(tau + 0.01, 0.5, f"chosen τ = {tau:.2f}", color=TEXT, fontsize=9)
    ax.set_xlim(0.05, 1.22)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=TEXT)
    style_axes(ax, "Validation routing metrics by confidence threshold", "threshold τ", "rate")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(true: list[str], pred: list[str], names: dict[str, str], path: Path) -> None:
    """Row-normalised confusion matrix on validation (accepted FAQ or fallback)."""
    from matplotlib.colors import LinearSegmentedColormap

    from understudy.common.plotting import SEQUENTIAL, SURFACE, TEXT, TEXT_2, plt

    labels = sorted(set(true) - {FALLBACK}) + [FALLBACK]
    idx = {lbl: i for i, lbl in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)))
    for t, p in zip(true, pred, strict=True):
        cm[idx[t], idx.get(p, idx[FALLBACK])] += 1
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    tick = [names.get(lbl, "fallback (defer)")[:38] for lbl in labels]
    size = max(6, 0.32 * len(labels) + 3)
    fig, ax = plt.subplots(figsize=(size + 2, size))
    fig.set_facecolor(SURFACE)
    im = ax.imshow(cm, cmap=LinearSegmentedColormap.from_list("seq", SEQUENTIAL), vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), tick, rotation=90, fontsize=7, color=TEXT_2)
    ax.set_yticks(range(len(labels)), tick, fontsize=7, color=TEXT_2)
    ax.set_xlabel("predicted", color=TEXT_2)
    ax.set_ylabel("true", color=TEXT_2)
    ax.set_title("Validation confusion (row-normalised)", loc="left", color=TEXT, fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.035)
    cb.ax.tick_params(labelsize=8, colors=TEXT_2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(params: dict) -> dict[str, Any]:
    """Tune, train, evaluate, log and save the student."""
    import mlflow
    import ray
    from ray import tune

    from understudy.student.encoder import OnnxEncoder

    tp, seed = params["train"], int(params["seed"])
    data_dir, enc_dir = resolve(DATA_DIR), resolve("models/encoder")
    plots = resolve(PLOTS)
    plots.mkdir(parents=True, exist_ok=True)

    with stage_run("train") as rep:
        train = pd.read_parquet(data_dir / "train.parquet")
        val = pd.read_parquet(data_dir / "val.parquet")
        test = pd.read_parquet(data_dir / "test.parquet")
        bank = AnswerBank.load(data_dir / "bank.json")
        rep.rows_in = len(train) + len(val)
        if len(bank) < 2:
            result = {"status": "skipped", "reason": f"need >= 2 approved FAQs, have {len(bank)}"}
            resolve(METRICS).write_text(json.dumps(result, indent=2) + "\n")
            resolve(RUN_INFO).write_text(json.dumps(result, indent=2) + "\n")
            rep.extra.update(result)
            return result

        approvals = load_approvals()
        faq_intent = {r.faq_id: r.source_intent for r in approvals.values() if r.source_intent}
        enc = OnnxEncoder(enc_dir, threads=16, batch_size=128)
        t0 = time.perf_counter()
        x_train, x_val, x_test = (enc.encode(df["text"].tolist()) for df in (train, val, test))
        embed_s = time.perf_counter() - t0

        data = {
            "x_train": x_train,
            "y_train": train["label"].tolist(),
            "x_val": x_val,
            "y_val": val["label"].to_numpy(),
            "oos_val": (val["source"] == "oos").to_numpy(),
            "min_precision": float(tp["tau_min_precision"]),
            "max_fa": float(tp["tau_max_false_accept"]),
            "seed": seed,
        }
        ray.init(
            num_cpus=int(tp["max_concurrent"]),
            include_dashboard=False,
            ignore_reinit_error=True,
            log_to_driver=False,
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tuner = tune.Tuner(
                    tune.with_parameters(trainable, data=data),
                    param_space=search_space(tp["search_space"]),
                    tune_config=tune.TuneConfig(
                        metric="score",
                        mode="max",
                        num_samples=int(tp["num_samples"]),
                        search_alg=tune.search.BasicVariantGenerator(random_state=seed),
                    ),
                    run_config=tune.RunConfig(storage_path=tmp, name="student", verbose=0),
                )
                t0 = time.perf_counter()
                grid = tuner.fit()
                tune_s = time.perf_counter() - t0
                trials = grid.get_dataframe()
                best_cfg = relevant(grid.get_best_result().config)
        finally:
            ray.shutdown()

        # Refit the winner, re-pick tau on validation, evaluate on the held-out test set.
        head = train_head(x_train, data["y_train"], best_cfg, seed=seed)
        o_val = outcomes_from_labels(
            head, head.predict_proba(x_val), data["y_val"], data["oos_val"]
        )
        m_val, feasible = choose_tau(o_val, data["min_precision"], data["max_fa"])
        tau = m_val["tau"]
        o_test = outcomes_from_intents(head, head.predict_proba(x_test), test, faq_intent)
        m_test = metrics_at(o_test, tau)

        data_version = dir_hash(data_dir)
        prompt_versions = sorted({r.prompt_version for r in approvals.values()})
        teacher_models = sorted({r.teacher_model for r in approvals.values()})
        cfg_hash = hashlib.md5(json.dumps(best_cfg, sort_keys=True).encode()).hexdigest()[:6]
        build_id = f"{data_version}-{cfg_hash}"
        meta = {
            "tau": tau,
            "head": best_cfg,
            "build_id": build_id,
            "bank_version": bank.version,
            "n_faqs": len(bank),
            "data_version": data_version,
            "git_commit": git_commit(),
            "encoder": json.loads((enc_dir / "encoder_meta.json").read_text()),
            "prompt_versions": prompt_versions,
            "teacher_models": teacher_models,
        }
        student_dir = save_student(resolve(STUDENT_DIR), enc_dir, head, bank, meta)
        predictor = StudentPredictor(student_dir, threads=1)
        lat = latency_ms(
            predictor,
            test["text"].sample(int(tp["latency_probe_queries"]), random_state=seed).tolist(),
        )
        size_mb = sum(f.stat().st_size for f in student_dir.rglob("*") if f.is_file()) / 2**20

        val_pred = [
            lbl if (ok and c >= tau) else FALLBACK
            for lbl, c, ok in zip(*decode(head, head.predict_proba(x_val)), strict=True)
        ]
        plot_threshold_curve(o_val, tau, plots / "threshold_curve.png")
        plot_confusion(
            val["label"].tolist(),
            val_pred,
            {f.faq_id: f.question for f in bank.faqs.values()},
            plots / "confusion_val.png",
        )

        metrics = {
            **{f"val_{k}": round(v, 4) for k, v in m_val.items() if k != "tau"},
            **{f"test_{k}": round(v, 4) for k, v in m_test.items() if k != "tau"},
            "tau": tau,
            "tau_feasible": feasible,
            **{k: round(v, 2) for k, v in lat.items()},
            "model_size_mb": round(size_mb, 2),
            "n_faqs": len(bank),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "tune_trials": len(trials),
            "tune_feasible_trials": int(trials["feasible"].sum()),
            "tune_seconds": round(tune_s, 1),
            "embed_seconds": round(embed_s, 1),
        }

        setup_mlflow(EXPERIMENT)
        with mlflow.start_run(run_name=f"student-{build_id}") as run:
            mlflow.set_tags(
                {
                    "git_commit": meta["git_commit"],
                    "git_dirty": str(git_is_dirty()),
                    "data_version": data_version,
                    "bank_version": bank.version,
                    "prompt_versions": ",".join(prompt_versions),
                    "teacher_models": ",".join(teacher_models),
                    "encoder_revision": meta["encoder"]["source_revision"],
                    "stage": "train",
                }
            )
            mlflow.log_params({f"head_{k}": v for k, v in best_cfg.items()})
            mlflow.log_params(
                {
                    "tau": tau,
                    "n_faqs": len(bank),
                    "seed": seed,
                    "tune_num_samples": tp["num_samples"],
                    "tau_min_precision": tp["tau_min_precision"],
                    "tau_max_false_accept": tp["tau_max_false_accept"],
                }
            )
            mlflow.log_metrics(
                {
                    k: float(v)
                    for k, v in metrics.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
            )
            for f in ("threshold_curve.png", "confusion_val.png"):
                mlflow.log_artifact(str(plots / f), artifact_path="plots")
            with tempfile.TemporaryDirectory() as tmp:
                tpath = Path(tmp) / "tune_trials.csv"
                trials.to_csv(tpath, index=False)
                mlflow.log_artifact(str(tpath))
            mlflow.log_artifact(str(resolve("reports/dataset_stats.json")))
            mlflow.pyfunc.log_model(
                name="student",
                python_model=str(PKG_DIR / "student" / "pyfunc_entry.py"),
                artifacts={"student": str(student_dir)},
                pip_requirements=["onnxruntime", "tokenizers", "numpy", "pandas"],
                input_example=pd.DataFrame({"text": ["how do i reset my password"]}),
                code_paths=[str(PKG_DIR)],
            )
            for _, t in trials.iterrows():
                cfg = relevant(
                    {k.removeprefix("config/"): t[k] for k in t.index if k.startswith("config/")}
                )
                with mlflow.start_run(run_name=f"trial-{t['trial_id']}", nested=True):
                    mlflow.log_params(cfg)
                    mlflow.log_metrics(
                        {k: float(t[k]) for k in t.index if k.startswith("val_") or k == "score"}
                    )
            run_id = run.info.run_id

        info = {
            "status": "trained",
            "run_id": run_id,
            "model_uri": f"runs:/{run_id}/student",
            "build_id": build_id,
            "experiment": EXPERIMENT,
        }
        resolve(METRICS).write_text(json.dumps(metrics, indent=2) + "\n")
        resolve(RUN_INFO).write_text(json.dumps(info, indent=2) + "\n")
        rep.rows_out = len(bank)
        rep.extra.update(
            {
                "run_id": run_id,
                "tau": tau,
                "test_instant_precision": metrics["test_instant_precision"],
                "test_coverage": metrics["test_coverage"],
            }
        )
    log.info("student trained", extra={**metrics, "run_id": run_id, "head": best_cfg})
    return {**info, "metrics": metrics, "head": best_cfg}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(run(load_params()), indent=2, default=str))


if __name__ == "__main__":
    main()
