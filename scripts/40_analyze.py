#!/usr/bin/env python
"""Stage 4: paired significance analysis for p_raw vs q* vs p_head.

Accuracy differences on a few hundred items are not self-evidently real, so this
recomputes per-item predictions (no LM needed: it reuses the stored hidden states)
and reports, for each comparison:

  * the accuracy and balanced-accuracy difference,
  * a paired bootstrap 95% CI for the accuracy difference,
  * a McNemar exact test on the discordant pairs.

    PYTHONPATH=src python scripts/40_analyze.py \
        --induce runs/induce/wanli_train_qwen35 \
        --test-induce runs/induce/wanli_test_qwen35 \
        --head runs/head/wanli_qwen35.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jev import induce as induce_mod  # noqa: E402
from jev.head import OptionScoreHead  # noqa: E402
from jev.metrics import balanced_accuracy  # noqa: E402


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def head_probs(head, hidden: torch.Tensor, device) -> np.ndarray:
    h = torch.from_numpy(hidden.astype(np.float32)).to(device)
    m = torch.ones((h.shape[0], h.shape[1]), dtype=torch.bool, device=device)
    head.eval()
    with torch.no_grad():
        return torch.softmax(head(h, m), dim=-1).cpu().numpy()


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial test on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def bootstrap_ci(correct_a: np.ndarray, correct_b: np.ndarray, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    diff = correct_b.astype(np.float64) - correct_a.astype(np.float64)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        samples[i] = diff[idx].mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--induce", required=True)
    ap.add_argument("--test-induce", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.head, map_location="cpu", weights_only=False)
    head = OptionScoreHead(ckpt["hidden_size"], n_layers=ckpt["n_layers"], dropout=ckpt.get("dropout", 0.0))
    head.load_state_dict(ckpt["state_dict"])
    head.to(device)

    test, _ = induce_mod.load(args.test_induce)
    gold = test["item_gold"].astype(np.int64)
    mask = gold >= 0
    praw = test["item_praw"][mask]
    qstar = test["item_qstar"][mask]
    hprobs = head_probs(head, test["view_option_hidden"][:, 0], device)[mask]
    g = gold[mask]

    preds = {"p_raw": praw, "q*": qstar, "p_head": hprobs}
    correct = {k: (v.argmax(1) == g) for k, v in preds.items()}
    n = len(g)

    print(f"n = {n}")
    for k, v in preds.items():
        print(f"  {k:8s} acc={correct[k].mean():.4f}  bal_acc={balanced_accuracy(v, g):.4f}")

    print("\npaired comparisons (vs p_raw):")
    rows = []
    for name in ("q*", "p_head"):
        a, b = correct["p_raw"], correct[name]
        b_disc = int((~a & b).sum())
        c_disc = int((a & ~b).sum())
        lo, hi = bootstrap_ci(a, b)
        p = mcnemar_exact(b_disc, c_disc)
        row = {
            "method": name,
            "acc_diff": float(b.mean() - a.mean()),
            "acc_diff_ci95": [lo, hi],
            "bal_acc_diff": float(balanced_accuracy(preds[name], g) - balanced_accuracy(praw, g)),
            "mcnemar_b_praw_wrong_head_right": b_disc,
            "mcnemar_c_praw_right_head_wrong": c_disc,
            "mcnemar_exact_p": p,
        }
        rows.append(row)
        print(
            f"  {name:8s} dAcc={row['acc_diff']:+.4f} CI95=[{lo:+.4f},{hi:+.4f}] "
            f"dBalAcc={row['bal_acc_diff']:+.4f}  discordant(b,c)=({b_disc},{c_disc}) p={p:.4f}"
        )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"n": n, "comparisons": rows}, fh, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
