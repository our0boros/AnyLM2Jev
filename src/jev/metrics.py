"""Metrics for probabilistic decisions.

Accuracy alone is not evidence that a Jev-like readout works; the claims worth
testing are about *confidence*.  So everything here comes in pairs: a sharpness
metric (accuracy) and a calibration metric (ECE / Brier / NLL), plus an
order-consistency metric in :mod:`jev.induce`.
"""

from __future__ import annotations

import numpy as np


def _as_2d(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"expected [N, K] probabilities, got {probs.shape}")
    return probs


def accuracy(probs: np.ndarray, gold: np.ndarray) -> float:
    probs, gold = _as_2d(probs), np.asarray(gold)
    valid = gold >= 0
    if not valid.any():
        return float("nan")
    pred = probs[valid].argmax(axis=1)
    return float((pred == gold[valid]).mean())


def balanced_accuracy(probs: np.ndarray, gold: np.ndarray) -> float:
    """Macro-average recall over the classes present in `gold`."""
    probs, gold = _as_2d(probs), np.asarray(gold)
    valid = gold >= 0
    p, g = probs[valid], gold[valid]
    if len(g) == 0:
        return float("nan")
    pred = p.argmax(axis=1)
    recalls = [float((pred[g == c] == c).mean()) for c in np.unique(g)]
    return float(np.mean(recalls))


def nll(probs: np.ndarray, gold: np.ndarray) -> float:
    probs, gold = _as_2d(probs), np.asarray(gold)
    valid = gold >= 0
    p = np.clip(probs[valid, gold[valid]], 1e-12, 1.0)
    return float(-np.log(p).mean())


def brier(probs: np.ndarray, gold: np.ndarray) -> float:
    probs, gold = _as_2d(probs), np.asarray(gold)
    valid = gold >= 0
    p = probs[valid]
    y = np.zeros_like(p)
    y[np.arange(len(gold[valid])), gold[valid]] = 1.0
    return float(((p - y) ** 2).sum(axis=1).mean())


def top1_confidence(probs: np.ndarray) -> np.ndarray:
    return _as_2d(probs).max(axis=1)


def ece(probs: np.ndarray, gold: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error on the top-1 prediction."""
    probs, gold = _as_2d(probs), np.asarray(gold)
    valid = gold >= 0
    p, g = probs[valid], gold[valid]
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == g).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    err = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.sum() == 0:
            continue
        err += (in_bin.sum() / total) * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(err)


def reliability_curve(probs: np.ndarray, gold: np.ndarray, n_bins: int = 15) -> list[dict]:
    probs, gold = _as_2d(probs), np.asarray(gold)
    valid = gold >= 0
    p, g = probs[valid], gold[valid]
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == g).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        out.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "count": int(in_bin.sum()),
                "avg_confidence": float(conf[in_bin].mean()) if in_bin.sum() else None,
                "accuracy": float(correct[in_bin].mean()) if in_bin.sum() else None,
            }
        )
    return out


def risk_coverage(probs: np.ndarray, gold: np.ndarray) -> dict:
    """Selective-prediction curve: error rate among the most confident fraction."""
    probs, gold = _as_2d(probs), np.asarray(gold)
    valid = gold >= 0
    p, g = probs[valid], gold[valid]
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == g).astype(np.float64)
    order = np.argsort(-conf)
    correct = correct[order]
    n = len(correct)
    coverages = np.linspace(0.1, 1.0, 10)
    curve = []
    for c in coverages:
        m = max(1, int(round(c * n)))
        curve.append({"coverage": float(c), "risk": float(1.0 - correct[:m].mean())})
    risks = np.array([c["risk"] for c in curve])
    aurc = float(risks.mean())
    return {"curve": curve, "aurc": aurc}


def coverage_at_risk(probs: np.ndarray, gold: np.ndarray, target_risk: float = 0.05) -> float:
    """Largest fraction of items answerable at a selective error rate <= target.

    Sort by top-1 confidence, walk the prefix, and take the largest coverage whose
    cumulative error is still within ``target_risk``.  This is the number that
    decides whether a decision pipeline can actually be automated.
    """
    probs, gold = _as_2d(probs), np.asarray(gold)
    valid = gold >= 0
    p, g = probs[valid], gold[valid]
    n = len(g)
    if n == 0:
        return float("nan")
    conf = p.max(axis=1)
    wrong = (p.argmax(axis=1) != g).astype(np.float64)
    order = np.argsort(-conf, kind="stable")
    cum_wrong = np.cumsum(wrong[order])
    counts = np.arange(1, n + 1, dtype=np.float64)
    ok = (cum_wrong / counts) <= target_risk
    if not ok.any():
        return 0.0
    return float(counts[np.nonzero(ok)[0].max()] / n)


def order_consistency(view_probs: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Agreement of the top-1 option across permuted views of the same item.

    ``view_probs`` is [N, V, K] with distributions in *canonical* option order.
    """
    vp = np.asarray(view_probs, dtype=np.float64)
    n, v, k = vp.shape
    if mask is not None:
        m = np.asarray(mask)
    else:
        m = np.ones((n, k), dtype=bool)

    # total-variation distance between every pair of views
    tv_vals: list[float] = []
    argmax_agree: list[float] = []
    for a in range(v):
        for b in range(a + 1, v):
            valid_rows = m[:, 0]
            tva = 0.5 * np.abs(vp[:, a, :] - vp[:, b, :]).sum(axis=1)
            tv_vals.extend(tva[valid_rows].tolist())
            am = (vp[:, a, :].argmax(axis=1) == vp[:, b, :].argmax(axis=1)).astype(float)
            argmax_agree.extend(am[valid_rows].tolist())
    return {
        "mean_tv": float(np.mean(tv_vals)) if tv_vals else float("nan"),
        "argmax_agreement": float(np.mean(argmax_agree)) if argmax_agree else float("nan"),
    }


def summarize(probs: np.ndarray, gold: np.ndarray, n_bins: int = 15) -> dict:
    return {
        "accuracy": accuracy(probs, gold),
        "balanced_accuracy": balanced_accuracy(probs, gold),
        "nll": nll(probs, gold),
        "brier": brier(probs, gold),
        "ece": ece(probs, gold, n_bins=n_bins),
        "mean_confidence": float(top1_confidence(probs).mean()),
        "coverage_at_5pct": coverage_at_risk(probs, gold, target_risk=0.05),
    }
