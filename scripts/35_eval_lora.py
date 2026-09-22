#!/usr/bin/env python
"""Stage 3b: evaluate a LoRA-adapted backbone + head against p_raw and q*.

`30_eval.py` is LM-free because it reads stage-1 hidden states.  A LoRA model has
different hidden states, so this stage runs the adapted model on the fly and
compares it with the *same* frozen baselines stored by stage 1.

    PYTHONPATH=src python scripts/35_eval_lora.py \
        --induce runs/induce/banking77_train_qwen35 \
        --test-induce runs/induce/banking77_test_qwen35 \
        --lora-dir runs/lora/banking77_qwen35 \
        --head runs/lora/banking77_qwen35/head.pt \
        --out runs/eval/banking77_lora.json
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
from jev.head import OptionScoreHead  # noqa: E402
from jev.metrics import order_consistency, summarize  # noqa: E402
from jev.modeling import OptionReader, _find_token, load_lm  # noqa: E402
from jev.prompts import build_prompt, default_orders, make_labels, make_views  # noqa: E402


def canonical(order, display: np.ndarray) -> np.ndarray:
    out = np.zeros(len(order), dtype=np.float64)
    out[list(order)] = display
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--induce", required=True)
    ap.add_argument("--test-induce", required=True)
    ap.add_argument("--lora-dir", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--paraphrases", type=int, default=2)
    ap.add_argument("--max-orders", type=int, default=3)
    ap.add_argument("--batch-items", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    train_arrays, _ = induce_mod.load(args.induce)
    test_arrays, test_items = induce_mod.load(args.test_induce)
    n_train = train_arrays["item_qstar"].shape[0]
    k = test_arrays["item_qstar"].shape[1]
    labels = [str(x) for x in test_arrays["labels"]] if "labels" in test_arrays else make_labels(k)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_train)
    n_val = max(1, int(round(args.val_frac * n_train)))
    val_idx = perm[:n_val]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base, tok = load_lm(args.model, dtype=args.dtype, device=args.device)
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, os.path.join(args.lora_dir, "adapter"))
    _dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    for p in model.parameters():
        if p.dtype.is_floating_point and p.dtype != _dtype:
            p.data = p.data.to(_dtype)
    model.eval()
    reader = OptionReader(model, tok, device=args.device, labels=labels)

    ckpt = torch.load(args.head, map_location="cpu", weights_only=False)
    head = OptionScoreHead(ckpt["hidden_size"], n_layers=ckpt["n_layers"], dropout=0.0)
    head.load_state_dict(ckpt["state_dict"])
    head.to(device)
    head.eval()

    orders = default_orders(k, args.max_orders)
    views = make_views(k, args.paraphrases, orders=orders)
    label_ids = torch.tensor(reader.label_token_ids, device=device)

    def score(items, indices, view):
        head_p = np.zeros((len(indices), k))
        head_z = np.zeros((len(indices), k))
        raw_p = np.zeros((len(indices), k))
        raw_z = np.zeros((len(indices), k))
        with torch.no_grad():
            for s in range(0, len(indices), args.batch_items):
                chunk = indices[s : s + args.batch_items]
                prompts, eoffs, order = [], [], []
                for i in chunk:
                    it = items[int(i)]
                    texts = [it.options[c] for c in view.order]
                    p, _l, _lo, eo = build_prompt(it.state, it.question, texts,
                                                  view.paraphrase_id, labels=labels)
                    prompts.append(p)
                    eoffs.append(eo)
                    order.append(list(view.order))
                last_logits, hidden, offsets = reader._forward(prompts, args.max_length)
                rows = []
                for bi, offs in enumerate(eoffs):
                    rows.append(torch.stack([hidden[bi, _find_token(offsets[bi], o)]
                                             for o in offs], dim=0))
                opt_hidden = torch.stack(rows, dim=0)
                hz = head(opt_hidden.float())
                hp = torch.softmax(hz, dim=-1).cpu().numpy()
                hz = hz.cpu().numpy()
                rz = last_logits[:, label_ids].float()
                rp = torch.softmax(rz, dim=-1).cpu().numpy()
                rz = rz.cpu().numpy()
                for r, o in enumerate(order):
                    head_p[s + r] = canonical(o, hp[r])
                    head_z[s + r] = canonical(o, hz[r])
                    raw_p[s + r] = canonical(o, rp[r])
                    raw_z[s + r] = canonical(o, rz[r])
        return head_p, head_z, raw_p, raw_z

    _, val_items = induce_mod.load(args.induce)
    vi = [int(i) for i in val_idx]
    val_head_p, val_head_z, val_raw_p, val_raw_z = score(val_items, vi, views[0])
    val_gold = train_arrays["item_gold"][val_idx].astype(np.int64)
    fit_praw = train_arrays["view_label_logits"][val_idx, 0, :].astype(np.float64)
    fit_qstar = to_logits(train_arrays["item_qstar"][val_idx])

    ti = list(range(len(test_items)))
    test_gold = test_arrays["item_gold"].astype(np.int64)
    head_p, head_z, raw_p, raw_z = score(test_items, ti, views[0])
    head_view_probs = []
    for j in range(len(views)):
        hp, _hz, _rp, _rz = score(test_items, ti, views[j])
        head_view_probs.append(hp)
    head_avg = np.mean(head_view_probs, axis=0)

    def row(name, probs, logits=None, fit=None, fit_gold=None):
        r = {"method": name}
        r.update(summarize(probs, test_gold))
        if logits is not None and fit is not None:
            temp = fit_temperature(fit, fit_gold)
            r["ece_calibrated"] = float(summarize(apply_temperature(logits, temp), test_gold)["ece"])
            r["temperature"] = float(temp)
        return r

    rows = [
        row("p_raw (frozen LM, 1 pass)", test_arrays["item_praw"],
            logits=test_arrays["view_label_logits"][:, 0, :].astype(np.float64),
            fit=fit_praw, fit_gold=val_gold),
        row("q* (frozen LM, V passes)", test_arrays["item_qstar"],
            logits=to_logits(test_arrays["item_qstar"]), fit=fit_qstar, fit_gold=val_gold),
        row("p_lora_head (1 pass)", head_p, logits=head_z, fit=val_head_z, fit_gold=val_gold),
        row("p_lora_head_avg (V passes)", head_avg),
        row("p_lora_raw (LoRA logits)", raw_p, logits=raw_z, fit=val_raw_z, fit_gold=val_gold),
    ]

    print(f"eval on: test   n={len(test_gold)}   views/item={len(views)}   options={k}")
    print(f"{'method':30s} {'acc':>7s} {'bal_acc':>8s} {'nll':>7s} {'brier':>7s} {'ece':>7s} {'ece_cal':>8s} {'conf':>7s}")
    for r in rows:
        print(f"{r['method']:30s} {r.get('accuracy', float('nan')):7.3f} "
              f"{r.get('balanced_accuracy', float('nan')):8.3f} {r.get('nll', float('nan')):7.3f} "
              f"{r.get('brier', float('nan')):7.3f} {r.get('ece', float('nan')):7.3f} "
              f"{r.get('ece_calibrated', float('nan')):8.3f} {r.get('mean_confidence', float('nan')):7.3f}")

    oc_raw = order_consistency(test_arrays["view_probs"], np.ones((len(test_gold), k), bool))
    oc_head = order_consistency(np.stack(head_view_probs, axis=1), np.ones((len(test_gold), k), bool))
    print(f"order_consistency(frozen p_raw views)  mean_tv={oc_raw['mean_tv']:.3f} "
          f"argmax_agreement={oc_raw['argmax_agreement']:.3f}")
    print(f"order_consistency(lora head views)     mean_tv={oc_head['mean_tv']:.3f} "
          f"argmax_agreement={oc_head['argmax_agreement']:.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows, "order_frozen": oc_raw, "order_lora_head": oc_head}, fh, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
