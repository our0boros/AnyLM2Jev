#!/usr/bin/env python
"""Stage 4b: post-hoc debiasing eval (CPU, no LM).

Answers three questions on artefacts that already exist on disk:

1. does a **label-free prior correction** improve `p_raw` / `q*` / `p_head`, and by
   how much?
2. does combining views in **log space** (geometric mean, which cancels an additive
   position bias exactly) beat the arithmetic mean?
3. what does it do to the metric that decides automability, **coverage at 5% risk**?

Nothing here needs a GPU: the prior is reconstructed from ``view_order`` and the
stored per-view distributions.  Temperature is fitted on the training-time
validation split, exactly as in ``30_eval.py``, and the prior is estimated on the
training split only -- the test split is never touched.

    PYTHONPATH=src python scripts/45_debias_eval.py \
        --induce runs/induce/banking77_train_qwen35 \
        --test-induce runs/induce/banking77_test_qwen35 \
        --head runs/head/banking77_qwen35.pt \
        --out runs/eval/banking77_debias.json
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
from jev.calibrate import apply_temperature, fit_temperature, to_logits  # noqa: E402
from jev.debias import (  # noqa: E402
    apply_prior,
    batch_prior,
    content_free_prior,
    flip_rate,
    position_to_view,
    view_to_position,
)
from jev.debias import debias as run_debias  # noqa: E402
from jev.head import OptionScoreHead  # noqa: E402
from jev.metrics import reliability_curve, risk_coverage, summarize  # noqa: E402


def head_probs_all_views(head, arrays: dict, device) -> np.ndarray:
    """Head distribution for every (item, view), in canonical order [N, V, K].

    Done one view at a time: the full hidden-state block is ~1 GB for a 26-way
    train split, and only the per-view probabilities are needed downstream.
    """
    hidden = arrays["view_option_hidden"]
    n, v, k, _ = hidden.shape
    out = np.zeros((n, v, k), dtype=np.float64)
    head.eval()
    for j in range(v):
        h = torch.from_numpy(hidden[:, j].astype(np.float32)).to(device)
        mask = torch.ones((n, k), dtype=torch.bool, device=device)
        with torch.no_grad():
            out[:, j] = torch.softmax(head(h, mask), dim=-1).cpu().numpy()
    return out


def flip_rate_by_paraphrase(vp: np.ndarray, order: np.ndarray, para: np.ndarray) -> float:
    """Permutation-only flip rate: compare rotations within one paraphrase.

    Our view matrix mixes paraphrase and permutation, so the all-views ``flip``
    conflates the two.  A permutation-only flip compares rotations of one prompt,
    which is what this isolates.
    """
    pids = np.unique(para[0])
    vals = []
    for pid in pids:
        idx = np.nonzero(para[0] == pid)[0]
        if len(idx) > 1:
            vals.append(flip_rate(vp[:, idx], order[:, idx]))
    return float(np.mean(vals)) if vals else float("nan")


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def bootstrap_ci(correct_a: np.ndarray, correct_b: np.ndarray, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    diff = correct_b.astype(np.float64) - correct_a.astype(np.float64)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        samples[i] = diff[rng.integers(0, n, n)].mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--induce", required=True, help="training induce prefix")
    ap.add_argument("--test-induce", required=True)
    ap.add_argument("--head", default="", help="head checkpoint; omit for a q*-only (--no-hidden) induce run")
    ap.add_argument("--content-free-train", default="", help="npz from 12_induce_prior_cf.py")
    ap.add_argument("--content-free-test", default="", help="npz from 12_induce_prior_cf.py")
    ap.add_argument("--combines", default="mean,logmean")
    ap.add_argument("--priors", default="none,batch")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = None
    head = None
    if args.head:
        ckpt = torch.load(args.head, map_location="cpu", weights_only=False)
        head = OptionScoreHead(
            ckpt["hidden_size"], n_layers=ckpt["n_layers"], dropout=ckpt.get("dropout", 0.0)
        )
        head.load_state_dict(ckpt["state_dict"])
        head.to(device)

    train, _ = induce_mod.load(args.induce)
    test, _ = induce_mod.load(args.test_induce)
    if train["item_qstar"].shape[1] != test["item_qstar"].shape[1]:
        raise ValueError("train/test option counts differ")
    if train["view_probs"].shape[1] != test["view_probs"].shape[1]:
        raise ValueError("train/test view counts differ (views must be built with the same config)")
    k = test["item_qstar"].shape[1]
    n_tr = train["item_qstar"].shape[0]
    has_hidden = "view_option_hidden" in train and "view_option_hidden" in test
    if has_hidden and head is None:
        raise SystemExit("this induce pair has hidden states, so --head is required")
    seed = ckpt["args"].get("seed", 0) if ckpt else 0
    val_frac = ckpt["args"].get("val_frac", 0.2) if ckpt else 0.2
    rng = np.random.default_rng(seed)

    # ---------------- priors (estimated on the training split only) -------------
    priors: dict[str, dict[str, np.ndarray]] = {}
    if "content_free" in args.priors:
        if not (args.content_free_train and args.content_free_test):
            raise SystemExit("--priors content_free needs --content-free-train and --content-free-test")

        def _load_cf(path: str) -> np.ndarray:
            with np.load(path) as f:
                return content_free_prior(np.asarray(f["cf_position"], dtype=np.float64))

        priors["content_free"] = {"train": _load_cf(args.content_free_train), "test": _load_cf(args.content_free_test)}

    # A per-item prior may cover only a prefix of the training split (the
    # content-free pass is the expensive one and may have been limited).  Restrict
    # the training arrays to the covered prefix so the prior still lines up.
    n_cover = n_tr
    for entry in priors.values():
        if entry["train"].ndim == 3:
            n_cover = min(n_cover, int(entry["train"].shape[0]))
    if n_cover < n_tr:
        print(f"[prior] training side restricted to its first {n_cover} items to align with the prior")
        train = {
            kk: (vv[:n_cover] if getattr(vv, "ndim", 0) >= 1 and vv.shape[0] == n_tr else vv)
            for kk, vv in train.items()
        }
        n_tr = n_cover

    if "batch" in args.priors:
        # label-free, from real inputs; a shared [V, K] prior over display slots
        bp = batch_prior(view_to_position(train["view_probs"], train["view_order"]))
        priors["batch"] = {"train": bp, "test": bp}

    # validation split, reproduced from the head checkpoint's training args
    val_idx = rng.permutation(n_tr)[: max(1, int(round(val_frac * n_tr)))]
    val_gold = train["item_gold"][val_idx].astype(np.int64)

    hp_train = head_probs_all_views(head, train, device) if has_hidden else None
    hp_test = head_probs_all_views(head, test, device) if has_hidden else None

    def slice_prior(p: np.ndarray | None, nv: int) -> np.ndarray | None:
        if p is None:
            return None
        return p[:nv] if p.ndim == 2 else p[:, :nv]

    gold = test["item_gold"].astype(np.int64)
    valid = gold >= 0

    # ------------------------------- methods -----------------------------------
    # (name, test per-view canonical probs, train per-view canonical probs)
    methods = [
        ("p_raw", test["view_probs"][:, :1], train["view_probs"][:, :1]),
        ("q*", test["view_probs"], train["view_probs"]),
    ]
    if has_hidden:
        methods += [
            ("p_head", hp_test[:, :1], hp_train[:, :1]),
            ("p_head_avg", hp_test, hp_train),
        ]
    vo_test, vo_train = test["view_order"], train["view_order"]

    rows, preds = [], {}
    curves: dict[str, dict] = {}
    for name, vp_te, vp_tr in methods:
        nv = vp_te.shape[1]
        vo_te_j, vo_tr_j = vo_test[:, :nv], vo_train[:, :nv]
        for prior_name in args.priors.split(","):
            entry = priors.get(prior_name)
            prior_tr = slice_prior(entry["train"], nv) if entry else None
            prior_te = slice_prior(entry["test"], nv) if entry else None
            for combine in args.combines.split(","):
                probs_te = run_debias(vp_te, vo_te_j, prior_te, combine=combine)
                probs_tr = run_debias(vp_tr, vo_tr_j, prior_tr, combine=combine)

                row = {"method": name, "prior": prior_name, "combine": combine}
                row.update(summarize(probs_te[valid], gold[valid]))
                temp = fit_temperature(to_logits(probs_tr[val_idx]), val_gold)
                row["temperature"] = float(temp)
                row["ece_calibrated"] = float(
                    summarize(apply_temperature(to_logits(probs_te[valid]), temp), gold[valid])["ece"]
                )
                rows.append(row)
                preds[(name, prior_name, combine)] = (probs_te[valid].argmax(1) == gold[valid])
                curves[f"{name}|{prior_name}|{combine}"] = {
                    "reliability": reliability_curve(probs_te[valid], gold[valid]),
                    "risk_coverage": risk_coverage(probs_te[valid], gold[valid]),
                }

    # ---------------------------- paired stats ---------------------------------
    # paired stats against the free single-pass baseline, whatever combine it ran with
    base_key = next(
        (kk for kk in preds if kk[0] == "p_raw" and kk[1] == "none"), next(iter(preds))
    )
    base_correct = preds[base_key]
    for r in rows:
        key = (r["method"], r["prior"], r["combine"])
        c = preds[key]
        r["acc_diff_vs_praw"] = float(c.mean() - base_correct.mean())
        lo, hi = bootstrap_ci(base_correct, c, n_boot=args.n_boot)
        r["acc_diff_ci95"] = [lo, hi]
        b_disc = int((~base_correct & c).sum())
        c_disc = int((base_correct & ~c).sum())
        r["mcnemar_p"] = mcnemar_exact(b_disc, c_disc)

    # ------------------------------- report ------------------------------------
    tag = os.path.basename(args.test_induce)
    print(f"debias eval on {tag}   n={int(valid.sum())}   options={k}   views={test['view_probs'].shape[1]}")
    print(
        f"{'method':11s} {'prior':13s} {'comb':8s} {'acc':>7s} {'bal_acc':>8s} {'nll':>7s} "
        f"{'ece':>7s} {'ece_cal':>8s} {'cov@5':>7s} {'dAcc':>7s} {'p':>8s}"
    )
    for r in rows:
        print(
            f"{r['method']:11s} {r['prior']:13s} {r['combine']:8s} {r['accuracy']:7.3f} "
            f"{r['balanced_accuracy']:8.3f} {r['nll']:7.3f} {r['ece']:7.3f} "
            f"{r['ece_calibrated']:8.3f} {r['coverage_at_5pct']:7.3f} "
            f"{r['acc_diff_vs_praw']:+7.4f} {r['mcnemar_p']:8.4f}"
        )

    # --------------------------- order stability ------------------------------
    para_te = test.get("view_paraphrase")
    print("\norder stability (raw readout per view, before marginalization):")
    stability = []
    sources = [("p_raw", test["view_probs"])]
    if has_hidden:
        sources.append(("p_head", hp_test))
    for src_name, vp in sources:
        for prior_name in args.priors.split(","):
            entry = priors.get(prior_name)
            p_te = slice_prior(entry["test"], vp.shape[1]) if entry else None
            if p_te is not None:
                corr = position_to_view(apply_prior(view_to_position(vp, vo_test), p_te), vo_test)
            else:
                corr = vp
            entry = {
                "source": src_name,
                "prior": prior_name,
                "flip_all_views": flip_rate(corr, vo_test),
                "flip_permutation_only": (
                    flip_rate_by_paraphrase(corr, vo_test, para_te) if para_te is not None else None
                ),
            }
            stability.append(entry)
            extra = (
                f"  flip(perm-only)={entry['flip_permutation_only']:.3f}"
                if entry["flip_permutation_only"] is not None
                else ""
            )
            print(f"  {src_name:8s} prior={prior_name:13s} flip(all views)={entry['flip_all_views']:.3f}{extra}")

    print("\nlabel marginals (gold):")
    print("  train", np.bincount(train["item_gold"][train["item_gold"] >= 0], minlength=k) / max(1, n_tr))
    print("  test ", np.bincount(gold[valid], minlength=k) / max(1, int(valid.sum())))
    for pname, entry in priors.items():
        p = np.asarray(entry["test"], dtype=np.float64)
        per_position = p.reshape(-1, p.shape[-1]).mean(axis=0)
        top = np.argsort(-per_position)[:5]
        print(f"  prior[{pname}] top positions: " + ", ".join(f"{i}:{per_position[i]:.3f}" for i in top))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        payload = {
            "test_induce": args.test_induce,
            "n": int(valid.sum()),
            "options": int(k),
            "rows": rows,
            "order_stability": stability,
            "curves": curves,
            "priors": {k2: {kk: vv.tolist() for kk, vv in v.items()} for k2, v in priors.items()},
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
