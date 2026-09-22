#!/usr/bin/env python
"""Stage 3: compare p_raw / q* / p_head with calibration and order metrics (GPU)."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jev import induce as induce_mod  # noqa: E402
from jev.calibrate import apply_temperature, fit_temperature, to_logits  # noqa: E402
from jev.head import OptionScoreHead  # noqa: E402
from jev.metrics import order_consistency, reliability_curve, risk_coverage, summarize  # noqa: E402


def head_scores(head, hidden: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    head.eval()
    with torch.no_grad():
        return head(hidden, mask).cpu().numpy()


def hidden_view(arrays: dict, view: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    h = torch.from_numpy(arrays["view_option_hidden"][:, view].astype(np.float32)).to(device)
    m = torch.ones((h.shape[0], h.shape[1]), dtype=torch.bool, device=device)
    return h, m


def calibrated_ece(fit_logits, fit_gold, eval_logits, eval_gold):
    temp = fit_temperature(fit_logits, fit_gold)
    probs = apply_temperature(eval_logits, temp)
    return float(summarize(probs, eval_gold)["ece"]), float(temp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--induce", required=True, help="training induce prefix (val split + calibration)")
    ap.add_argument("--head", required=True, help="head checkpoint from 20_train_head.py")
    ap.add_argument("--test-induce", default="", help="optional separate test induce prefix")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt = torch.load(args.head, map_location="cpu", weights_only=False)
    head = OptionScoreHead(
        ckpt["hidden_size"], n_layers=ckpt["n_layers"], dropout=ckpt.get("dropout", 0.0)
    )
    head.load_state_dict(ckpt["state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    head.to(device)

    train_arrays, _ = induce_mod.load(args.induce)
    n = train_arrays["item_qstar"].shape[0]
    seed = ckpt["args"].get("seed", 0)
    val_frac = ckpt["args"].get("val_frac", 0.2)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx = perm[:n_val]
    val_gold = train_arrays["item_gold"][val_idx].astype(np.int64)

    # calibration references, all fitted on the training-time validation split
    fit_praw = train_arrays["view_label_logits"][val_idx, 0, :].astype(np.float64)
    fit_qstar = to_logits(train_arrays["item_qstar"][val_idx])
    vh, vm = hidden_view(train_arrays, 0, device)
    fit_head = head_scores(head, vh, vm)[val_idx]

    arrays = train_arrays
    if args.test_induce:
        arrays, _ = induce_mod.load(args.test_induce)
        sel = np.arange(arrays["item_qstar"].shape[0])
        eval_name = "test"
    else:
        sel = val_idx
        eval_name = "val (no --test-induce given)"

    K = arrays["item_qstar"].shape[1]
    gold = arrays["item_gold"][sel].astype(np.int64)
    mask = np.ones((len(sel), K), dtype=bool)
    v = arrays["view_option_hidden"].shape[1]

    praw = arrays["item_praw"][sel]
    qstar = arrays["item_qstar"][sel]
    view_probs = arrays["view_probs"][sel]
    eval_praw_logits = arrays["view_label_logits"][sel, 0, :].astype(np.float64)
    eval_qstar_logits = to_logits(qstar)

    hp_list, hp_logits_list = [], []
    for j in range(v):
        hj, mj = hidden_view(arrays, j, device)
        sc = head_scores(head, hj, mj)[sel]
        hp_logits_list.append(sc)
        hp_list.append(torch.softmax(torch.from_numpy(sc), dim=-1).numpy())
    hp_probs = hp_list[0]
    hp_avg = np.mean(hp_list, axis=0)

    rows = []

    def add(name, probs, logits=None, fit=None):
        row = {"method": name}
        row.update(summarize(probs, gold))
        if logits is not None and fit is not None:
            ece_cal, temp = calibrated_ece(fit[0], fit[1], logits, gold)
            row["ece_calibrated"] = ece_cal
            row["temperature"] = temp
        rows.append(row)

    add("p_raw (1 pass, no train)", praw, eval_praw_logits, (fit_praw, val_gold))
    add("q* (V passes, no train)", qstar, eval_qstar_logits, (fit_qstar, val_gold))
    add("p_head (1 pass, trained)", hp_probs, hp_logits_list[0], (fit_head, val_gold))
    add("p_head_avg (V passes)", hp_avg)

    # trivial references for context
    train_gold = train_arrays["item_gold"]
    counts = np.bincount(train_gold[train_gold >= 0], minlength=K)
    prior = counts / counts.sum()
    add("majority-class prior (train)", np.tile(prior, (len(gold), 1)))

    order_rows = [
        {
            "method": "order_consistency(p_raw views)",
            **order_consistency(view_probs, mask),
        },
        {
            "method": "order_consistency(p_head views)",
            **order_consistency(np.stack(hp_list, axis=1), mask),
        },
    ]

    print(f"eval on: {eval_name}   n={len(sel)}   views/item={v}   options={K}")
    header = (
        f"{'method':34s} {'acc':>7s} {'bal_acc':>8s} {'nll':>7s} {'brier':>7s} "
        f"{'ece':>7s} {'ece_cal':>8s} {'conf':>7s} {'cov@5':>7s}"
    )
    print(header)
    for r in rows:
        print(
            f"{r['method']:34s} {r.get('accuracy', float('nan')):7.3f} "
            f"{r.get('balanced_accuracy', float('nan')):8.3f} "
            f"{r.get('nll', float('nan')):7.3f} {r.get('brier', float('nan')):7.3f} "
            f"{r.get('ece', float('nan')):7.3f} {r.get('ece_calibrated', float('nan')):8.3f} "
            f"{r.get('mean_confidence', float('nan')):7.3f} "
            f"{r.get('coverage_at_5pct', float('nan')):7.3f}"
        )
    for r in order_rows:
        print(
            f"{r['method']:34s} mean_tv={r['mean_tv']:.3f}  "
            f"argmax_agreement={r['argmax_agreement']:.3f}"
        )

    payload = {
        "eval_on": eval_name,
        "n_eval": int(len(sel)),
        "n_views": int(v),
        "rows": rows,
        "order_consistency": order_rows,
        "reliability": {
            "p_raw": reliability_curve(praw, gold),
            "p_head": reliability_curve(hp_probs, gold),
        },
        "risk_coverage": {
            "p_raw": risk_coverage(praw, gold),
            "p_head": risk_coverage(hp_probs, gold),
        },
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
