#!/usr/bin/env python
"""Stage 2: distil q* into the readout head (GPU or CPU, head-only)."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jev import induce as induce_mod  # noqa: E402
from jev.calibrate import apply_temperature, to_logits  # noqa: E402
from jev.debias import batch_prior, view_to_position  # noqa: E402
from jev.debias import debias as run_debias  # noqa: E402
from jev.distill import soft_kl  # noqa: E402
from jev.head import OptionScoreHead  # noqa: E402
from jev.metrics import summarize  # noqa: E402


def evaluate_head(head, hidden, target, gold, mask, temperature=1.0):
    head.eval()
    with torch.no_grad():
        scores = head(hidden, mask)
        probs = torch.softmax(scores / temperature, dim=-1).cpu().numpy()
    metrics = summarize(probs, gold)
    metrics["loss_vs_qstar"] = float(soft_kl(scores, target, mask))
    return probs, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--induce", required=True, help="prefix produced by 10_induce.py")
    ap.add_argument("--out", required=True, help="checkpoint path")
    ap.add_argument("--views", choices=["canonical", "all"], default="all")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--head-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--debias",
        choices=["none", "batch"],
        default="none",
        help="label-free position-prior correction folded into the q* target",
    )
    ap.add_argument("--combine", choices=["mean", "logmean"], default="mean")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    arrays, items = induce_mod.load(args.induce)
    hidden_all = torch.from_numpy(arrays["view_option_hidden"].astype(np.float32))  # [N,V,K,H]
    # The distillation target is always *debiased first, distilled second*: if the
    # prior correction were applied to the head's output afterwards it would double
    # correct, because the head has already absorbed whatever bias is in q*.
    prior = None
    if args.debias == "batch":
        prior = batch_prior(view_to_position(arrays["view_probs"], arrays["view_order"]))
    target_np = run_debias(
        arrays["view_probs"], arrays["view_order"], prior, combine=args.combine
    )
    target = torch.from_numpy(target_np.astype(np.float32))  # [N,K]
    print(f"target: debias={args.debias} combine={args.combine} n_views={arrays['view_probs'].shape[1]}")
    gold = arrays["item_gold"].astype(np.int64)
    n, v, k, h = hidden_all.shape
    mask = torch.ones((n, k), dtype=torch.bool)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(args.val_frac * n)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    if args.views == "all":
        # flatten (item, view) pairs for training
        tr_hidden = hidden_all[train_idx].reshape(-1, k, h)
        tr_target = target[train_idx].repeat_interleave(v, dim=0)
        tr_mask = mask[train_idx].repeat_interleave(v, dim=0)
        va_hidden = hidden_all[val_idx][:, 0, :, :].contiguous()  # canonical view
    else:
        tr_hidden = hidden_all[train_idx, 0]
        tr_target = target[train_idx]
        tr_mask = mask[train_idx]
        va_hidden = hidden_all[val_idx, 0].contiguous()

    va_target = target[val_idx]
    va_gold = gold[val_idx]
    va_mask = mask[val_idx]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    head = OptionScoreHead(h, n_layers=args.head_layers, dropout=args.dropout).to(device)
    tr_hidden = tr_hidden.to(device)
    tr_target = tr_target.to(device)
    tr_mask = tr_mask.to(device)
    va_hidden = va_hidden.to(device)
    va_target = va_target.to(device)
    va_mask = va_mask.to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_train = tr_hidden.shape[0]
    best = None
    history = []
    for epoch in range(args.epochs):
        order = torch.randperm(n_train, device=device)
        running = 0.0
        for s in range(0, n_train, args.batch_size):
            idx = order[s : s + args.batch_size]
            opt.zero_grad(set_to_none=True)
            scores = head(tr_hidden[idx], tr_mask[idx])
            loss = soft_kl(scores, tr_target[idx], tr_mask[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            running += float(loss.detach()) * len(idx)
        train_loss = running / max(1, n_train)

        # fit temperature on the validation split, then report
        with torch.no_grad():
            va_scores = head(va_hidden, va_mask).cpu().numpy()
        temp = _fit_temp(va_scores, va_target.cpu().numpy())
        _, va_metrics = evaluate_head(head, va_hidden, va_target, va_gold, va_mask, temperature=temp)
        record = {"epoch": epoch, "train_loss": train_loss, "val": va_metrics, "temperature": temp}
        history.append(record)
        if best is None or va_metrics["nll"] < best[0]:
            best = (va_metrics["nll"], {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, temp)
        print(json.dumps(record))

    head.load_state_dict(best[1])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "hidden_size": h,
            "n_layers": args.head_layers,
            "dropout": args.dropout,
            "temperature": best[2],
            "args": vars(args),
            "val_metrics": best[0],
        },
        args.out,
    )
    print("saved", args.out, "best val nll", best[0])


def _fit_temp(scores: np.ndarray, target: np.ndarray) -> float:
    """Temperature fitted against the soft target q* (not hard labels)."""
    logits = scores.astype(np.float64)

    def objective(log_t: float) -> float:
        p = apply_temperature(logits, float(np.exp(log_t)))
        return float(-(target * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1).mean())

    grid = np.linspace(-2.0, 3.0, 97)
    losses = [objective(lt) for lt in grid]
    best = int(np.argmin(losses))
    lo = grid[max(0, best - 1)]
    hi = grid[min(len(grid) - 1, best + 1)]
    fine = np.linspace(lo, hi, 97)
    fine_losses = [objective(lt) for lt in fine]
    return float(np.exp(fine[int(np.argmin(fine_losses))]))


if __name__ == "__main__":
    main()
