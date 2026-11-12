"""``eda`` stage: exploratory data analysis of the dataset and the (simulated) query logs.

Writes ``reports/eda/summary.json`` (DVC metrics) plus three charts:

* ``intent_popularity.png``  questions per intent, ranked. A few questions dominate,
  which is why an instant-answer tier pays off.
* ``domain_mix_by_week.png`` traffic share per week: initial domains vs late arrivals vs
  out-of-scope. This is the drift the system must detect.
* ``question_length.png``    words per question in CLINC train. Bounds input validation.

Documented data biases: crowd-sourced English questions only; teacher-generated
paraphrases (Phase 3) will inherit the teacher's phrasing biases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402

from understudy.common.config import load_params, resolve  # noqa: E402
from understudy.common.metrics import stage_run  # noqa: E402
from understudy.common.plotting import SERIES, SURFACE, TEXT, TEXT_2, style_axes  # noqa: E402

OUT_DIR = "reports/eda"

_style = style_axes


def plot_intent_popularity(logs: pd.DataFrame, path: Path) -> dict:
    """Bar chart of questions per intent, ranked (in-scope traffic only)."""
    counts = logs.loc[
        logs.sim_label.notna() & (logs.sim_label != "oos"), "sim_label"
    ].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4))
    ranks = range(1, len(counts) + 1)
    ax.bar(ranks, counts.values, width=0.8, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    _style(ax, "Questions per intent, ranked (all simulated weeks)", "Intent rank", "Questions")
    top10 = float(counts.head(10).sum() / counts.sum()) if len(counts) else 0.0
    if len(counts):
        ax.annotate(
            f"top 10 intents = {top10:.0%} of in-scope traffic",
            xy=(10, counts.iloc[min(9, len(counts) - 1)]),
            xytext=(len(counts) * 0.35, counts.iloc[0] * 0.75),
            color=TEXT,
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": TEXT_2, "linewidth": 0.8},
        )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"intents_seen": int(len(counts)), "top10_share": round(top10, 4)}


def plot_domain_mix(logs: pd.DataFrame, initial: list[str], path: Path) -> dict:
    """Stacked bars of weekly traffic share: initial domains, each late domain, OOS."""

    def group(row: pd.Series) -> str:
        if row.sim_label == "oos" or row.sim_domain is None or pd.isna(row.sim_domain):
            return "out of scope"
        return "initial domains" if row.sim_domain in initial else row.sim_domain

    df = logs.assign(group=logs.apply(group, axis=1))
    share = pd.crosstab(df.week, df.group, normalize="index")
    late = [c for c in share.columns if c not in ("initial domains", "out of scope")]
    order = ["initial domains", *sorted(late), "out of scope"]
    share = share.reindex(columns=order, fill_value=0.0)
    colors = (SERIES * 2)[: len(order)] if len(order) <= len(SERIES) else None
    fig, ax = plt.subplots(figsize=(9, 4))
    bottom = pd.Series(0.0, index=share.index)
    for i, col in enumerate(order):
        ax.bar(
            share.index,
            share[col],
            bottom=bottom,
            width=0.7,
            label=col,
            color=colors[i] if colors else None,
            edgecolor=SURFACE,
            linewidth=2,
        )
        bottom += share[col]
    _style(
        ax,
        "Traffic mix by simulated week: late-arriving domains are the drift",
        "Simulated week",
        "Share of questions",
    )
    ax.set_xticks(share.index)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(
        frameon=False, fontsize=9, labelcolor=TEXT, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {
        "share_by_week": {
            str(w): {c: round(float(v), 4) for c, v in r.items()} for w, r in share.iterrows()
        }
    }


def plot_length(clinc: pd.DataFrame, path: Path) -> dict:
    """Histogram of words per question in the CLINC training split."""
    words = clinc.loc[clinc.split == "train", "text_norm"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(
        words, bins=range(1, int(words.max()) + 2), color=SERIES[0], edgecolor=SURFACE, linewidth=1
    )
    _style(ax, "Words per question (CLINC150 train)", "Words", "Questions")
    p95 = float(words.quantile(0.95))
    ax.axvline(p95, color=TEXT_2, linewidth=1, linestyle="--")
    ax.text(p95 + 0.3, ax.get_ylim()[1] * 0.9, f"p95 = {p95:.0f} words", color=TEXT, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {
        "words_mean": round(float(words.mean()), 2),
        "words_p95": p95,
        "words_max": int(words.max()),
    }


def _placeholder(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    fig.set_facecolor(SURFACE)
    ax.text(
        0.5,
        0.5,
        "No query logs yet - run the traffic simulator",
        ha="center",
        va="center",
        color=TEXT_2,
        fontsize=11,
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(params: dict, processed_dir: str | Path, out_dir: str | Path = OUT_DIR) -> dict:
    """Compute summary statistics and render the charts."""
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    processed = resolve(processed_dir)
    with stage_run("eda") as rep:
        clinc = pd.read_parquet(processed / "clinc")
        qdir = processed / "querylog"
        logs = (
            pd.read_parquet(qdir)
            if any(qdir.rglob("*.parquet"))
            else pd.DataFrame(columns=["week", "sim_label", "sim_domain", "route"])
        )
        logs["week"] = logs["week"].astype(int)
        summary: dict = {
            "clinc_rows_per_split": clinc.split.value_counts().sort_index().to_dict(),
            "clinc_intents": int(clinc.intent.nunique()),
            "log_rows": int(len(logs)),
            "log_weeks": sorted(int(w) for w in logs.week.unique()),
        }
        summary |= plot_length(clinc, out / "question_length.png")
        if len(logs):
            summary |= plot_intent_popularity(logs, out / "intent_popularity.png")
            summary |= plot_domain_mix(
                logs, params["simulation"]["initial_domains"], out / "domain_mix_by_week.png"
            )
            summary["pii_placeholders_in_logs"] = int(logs.text.str.contains(r"<[A-Z]+>").sum())
        else:  # cold start: DVC still expects every declared output
            for name in ("intent_popularity.png", "domain_mix_by_week.png"):
                _placeholder(out / name)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        rep.rows_in = len(clinc) + len(logs)
        rep.rows_out = len(summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args(argv)
    run(load_params(), args.processed_dir)


if __name__ == "__main__":
    main()
