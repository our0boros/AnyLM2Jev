#!/usr/bin/env python
"""Comparison figures (scikit-learn CalibrationDisplay + matplotlib).

Reads the JSON produced by ``45_debias_eval.py`` and writes the PNGs embedded in
the READMEs:

    docs/figures/calibration.png     top-label reliability, before/after debiasing
    docs/figures/risk_coverage.png   selective-prediction curves
    docs/figures/metrics_bars.png    accuracy / balanced accuracy / coverage@5%

For K > 2 there is no native multiclass ``CalibrationDisplay``, so calibration is
plotted with the standard **top-label** reduction: the binary target is "was the
top-1 prediction correct" and the score is the top-1 confidence.  That is exactly
the reliability diagram ECE is computed from, and it is well defined for any K.

    PYTHONPATH=src python scripts/70_figures.py
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# a small, consistent palette
C = {
    "p_raw": "#9aa0a6",
    "p_raw+batch": "#5f6368",
    "q*": "#1a73e8",
    "q*+batch": "#0b57d0",
    "p_head": "#e8710a",
    "p_head_avg": "#b31412",
}

# (method, prior, combine, label)
KEYS_FULL = [
    ("p_raw", "none", "mean", "p_raw"),
    ("p_raw", "batch", "mean", "p_raw+batch"),
    ("q*", "none", "logmean", "q*"),
    ("q*", "batch", "logmean", "q*+batch"),
    ("p_head", "none", "mean", "p_head"),
    ("p_head_avg", "none", "mean", "p_head_avg"),
]
# 3-way tasks have very coarse confidence, so fewer series keeps the panel readable
KEYS_3WAY = [
    ("p_raw", "none", "mean", "p_raw"),
    ("p_raw", "batch", "mean", "p_raw+batch"),
    ("p_head", "none", "mean", "p_head"),
    ("p_head_avg", "none", "mean", "p_head_avg"),
]


def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def rows_by_key(doc: dict) -> dict:
    return {f"{r['method']}|{r['prior']}|{r['combine']}": r for r in doc.get("rows", [])}


def series_from_curves(doc: dict, keys):
    curves = doc.get("curves", {})
    for method, prior, combine, label in keys:
        k = f"{method}|{prior}|{combine}"
        if k in curves:
            yield label, C.get(label), curves[k]


# ------------------------------------------------------------------ figures


def plot_calibration(docs, out: str) -> None:
    fig, axes = plt.subplots(1, len(docs), figsize=(6.2 * len(docs), 5.0))
    if len(docs) == 1:
        axes = [axes]
    for ax, (title, (doc, keys)) in zip(axes, docs.items()):
        for label, color, curve in series_from_curves(doc, keys):
            conf = np.array([b["avg_confidence"] for b in curve["reliability"] if b["count"]], dtype=float)
            acc = np.array([b["accuracy"] for b in curve["reliability"] if b["count"]], dtype=float)
            ax.plot(conf, acc, marker="o", ms=3.5, lw=1.6, color=color, label=label)
        ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#444", label="perfect")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("top-1 confidence")
        ax.set_ylabel("fraction correct")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Top-label reliability (closer to the diagonal = better calibrated)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def plot_risk_coverage(docs, out: str) -> None:
    fig, axes = plt.subplots(1, len(docs), figsize=(6.2 * len(docs), 5.0))
    if len(docs) == 1:
        axes = [axes]
    for ax, (title, (doc, keys)) in zip(axes, docs.items()):
        for label, color, curve in series_from_curves(doc, keys):
            rc = curve["risk_coverage"]["curve"]
            cov = np.array([p["coverage"] for p in rc], dtype=float)
            risk = np.array([p["risk"] for p in rc], dtype=float)
            ax.plot(cov, risk, marker="o", ms=3.5, lw=1.6, color=color, label=label)
        ax.axhline(0.05, ls=":", lw=1.2, color="#b31412")
        ax.text(0.03, 0.062, "5% risk budget", fontsize=8, color="#b31412")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("coverage (fraction answered)")
        ax.set_ylabel("selective risk (error rate)")
        ax.set_xlim(0.05, 1)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Risk-coverage: how much can be automated at a given error rate", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def plot_metric_bars(docs, out: str) -> None:
    metrics = [
        ("accuracy", "accuracy"),
        ("balanced_accuracy", "balanced acc"),
        ("coverage_at_5pct", "coverage@5%"),
    ]
    fig, axes = plt.subplots(1, len(docs), figsize=(7.4 * len(docs), 4.6), sharey=True)
    if len(docs) == 1:
        axes = [axes]
    for ax, (title, (doc, keys)) in zip(axes, docs.items()):
        labels = [k[3] for k in keys]
        rb = rows_by_key(doc)
        x = np.arange(len(labels))
        w = 0.26
        for mi, (field, mlabel) in enumerate(metrics):
            vals = []
            for method, prior, combine, _ in keys:
                r = rb.get(f"{method}|{prior}|{combine}")
                vals.append(float(r[field]) if r else np.nan)
            bars = ax.bar(x + (mi - 1) * w, vals, w, label=mlabel)
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("score")
    fig.suptitle("Accuracy, balanced accuracy and selective coverage", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="runs/eval")
    ap.add_argument("--out-dir", default="docs/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    docs = {}
    for title, fname, keys in (
        ("Banking77, 26-way, n=500", "fig_banking77.json", KEYS_FULL),
        ("WANLI, 3-way, n=500", "fig_wanli.json", KEYS_3WAY),
    ):
        doc = load_json(os.path.join(args.eval_dir, fname))
        if doc:
            docs[title] = (doc, keys)

    if not docs:
        raise SystemExit(f"no fig_*.json found in {args.eval_dir}; run scripts/45_debias_eval.py first")

    plot_calibration(docs, os.path.join(args.out_dir, "calibration.png"))
    plot_risk_coverage(docs, os.path.join(args.out_dir, "risk_coverage.png"))
    plot_metric_bars(docs, os.path.join(args.out_dir, "metrics_bars.png"))


if __name__ == "__main__":
    main()
