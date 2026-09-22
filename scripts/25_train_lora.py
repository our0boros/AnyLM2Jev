#!/usr/bin/env python
"""Stage 2b: LoRA on the frozen backbone + head, trained end-to-end against q*.

Stage 2 (`20_train_head.py`) only re-parameterises the readout over frozen
features.  That caps the method at whatever the frozen representation encodes --
fine for 3-way NLI, not obviously fine for a 26-way intent decision.  This stage
lets the backbone move (low-rank adapters) so the representation itself can
specialise, while the head and the soft-KL objective stay the same.

Targets are the `q*` produced by stage 1; no LM forward is stored here, the model
is run on the fly so gradients flow into the adapters.

    PYTHONPATH=src python scripts/25_train_lora.py \
        --induce runs/induce/banking77_train_qwen35 \
        --model Qwen/Qwen3.5-2B-Base \
        --out runs/lora/banking77_qwen35
"""

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
from jev.distill import soft_kl  # noqa: E402
from jev.head import OptionScoreHead  # noqa: E402
from jev.metrics import summarize  # noqa: E402
from jev.modeling import OptionReader, load_lm  # noqa: E402
from jev.prompts import build_prompt, default_orders, make_labels, make_views  # noqa: E402


def build_item_prompts(item, view, labels):
    texts = [item.options[c] for c in view.order]
    prompt, _labels, _loff, eoff = build_prompt(
        item.state, item.question, texts, view.paraphrase_id, labels=labels
    )
    return prompt, eoff, list(view.order)


def canonical_index(order, probs_display):
    """Map display-order probabilities back to canonical option order."""
    out = np.zeros(len(order), dtype=np.float64)
    out[list(order)] = probs_display
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--induce", required=True, help="stage-1 prefix (gives q* and items)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--out", required=True, help="output directory for adapter + head")
    ap.add_argument("--paraphrases", type=int, default=2)
    ap.add_argument("--max-orders", type=int, default=3)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="all-linear")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--batch-items", type=int, default=4)
    ap.add_argument("--head-layers", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    arrays, items = induce_mod.load(args.induce)
    if args.limit:
        items = items[: args.limit]
        for key in ("item_qstar", "item_gold", "item_praw", "item_n_options"):
            if key in arrays:
                arrays[key] = arrays[key][: args.limit]
    n = len(items)
    k = items[0].n_options
    labels = make_labels(k)
    qstar = torch.from_numpy(arrays["item_qstar"].astype(np.float32))
    gold = arrays["item_gold"].astype(np.int64)

    orders = default_orders(k, args.max_orders)
    all_views = make_views(k, args.paraphrases, orders=orders)
    print(f"items={n} options={k} views/item={len(all_views)} "
          f"(paraphrases={args.paraphrases}, orders={len(orders)})")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(args.val_frac * n)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, tok = load_lm(args.model, dtype=args.dtype, device=args.device)
    reader = OptionReader(model, tok, device=args.device, labels=labels)

    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    # peft creates adapters in fp32; the Qwen3.5 hybrid tower is bf16, and some
    # linear layers are not matched by peft's autocast, so align explicitly.
    _dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    for p in model.parameters():
        if p.dtype.is_floating_point and p.dtype != _dtype:
            p.data = p.data.to(_dtype)
    model.print_trainable_parameters()

    head = OptionScoreHead(reader.hidden_size or int(model.config.hidden_size),
                           n_layers=args.head_layers).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
            {"params": head.parameters(), "lr": args.head_lr},
        ]
    )

    def score_items(indices, view, batch_items):
        """Return head probabilities (canonical order) and gold for given items."""
        model.eval()
        head.eval()
        probs = np.zeros((len(indices), k), dtype=np.float64)
        with torch.no_grad():
            for s in range(0, len(indices), batch_items):
                chunk = indices[s : s + batch_items]
                prompts, eoffs, orders = [], [], []
                for i in chunk:
                    p, eo, o = build_item_prompts(items[int(i)], view, labels)
                    prompts.append(p)
                    eoffs.append(eo)
                    orders.append(o)
                hidden = reader.read_grad(prompts, eoffs, max_length=args.max_length)
                scores = head(hidden.float())
                p_disp = torch.softmax(scores, dim=-1).cpu().numpy()
                for row, o in enumerate(orders):
                    probs[s + row] = canonical_index(o, p_disp[row])
        return probs

    def evaluate(tag):
        va = [int(i) for i in val_idx]
        probs = score_items(va, all_views[0], args.batch_items)  # canonical view only
        g = gold[val_idx]
        m = summarize(probs, g)
        return {"tag": tag, **{key: float(val) for key, val in m.items()}}

    history = []
    best = None
    for epoch in range(args.epochs):
        model.train()
        head.train()
        order = rng.permutation(train_idx)
        running, seen = 0.0, 0
        for s in range(0, len(order), args.batch_items):
            chunk = order[s : s + args.batch_items]
            prompts, eoffs, targets = [], [], []
            for i in chunk:
                view = all_views[int(rng.integers(0, len(all_views)))]
                p, eo, o = build_item_prompts(items[int(i)], view, labels)
                prompts.append(p)
                eoffs.append(eo)
                # scores are produced in DISPLAY order, so the canonical q* must
                # be permuted into display order too (identity view hides this bug)
                targets.append(qstar[int(i)][list(o)])
            hidden = reader.read_grad(prompts, eoffs, max_length=args.max_length)
            scores = head(hidden.float())
            target = torch.stack(targets).to(device)
            mask = torch.ones_like(target, dtype=torch.bool)
            loss = soft_kl(scores, target, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            running += float(loss.detach()) * len(chunk)
            seen += len(chunk)
        metrics = evaluate(f"epoch{epoch}")
        record = {"epoch": epoch, "train_loss": running / max(1, seen), "val": metrics}
        history.append(record)
        print(json.dumps(record))
        if best is None or metrics["nll"] < best[0]:
            best = (metrics["nll"], metrics)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(os.path.join(args.out, "adapter"))
    torch.save(
        {
            "state_dict": head.state_dict(),
            "hidden_size": reader.hidden_size,
            "n_layers": args.head_layers,
            "labels": labels,
            "args": vars(args),
            "best_val": best[1] if best else None,
        },
        os.path.join(args.out, "head.pt"),
    )
    with open(os.path.join(args.out, "history.json"), "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print("saved", args.out, "best val nll", best[0] if best else None)


if __name__ == "__main__":
    main()
