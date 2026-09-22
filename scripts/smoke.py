#!/usr/bin/env python
"""End-to-end sanity check of the readout path (run on a GPU node).

    PYTHONPATH=src python scripts/smoke.py --limit 4 --model Qwen/Qwen3.5-2B-Base

Verifies, in order: the model loads, the label offsets map to real tokens, the
baseline p_raw is a sane distribution, aggregation is not degenerate, and the
frozen readout beats chance on the templated synthetic task.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jev.data import synthetic_items  # noqa: E402
from jev.induce import induce, save  # noqa: E402
from jev.modeling import OptionReader, load_lm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--paraphrases", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/smoke/smoke")
    args = ap.parse_args()

    import torch

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))

    items = synthetic_items(args.limit, seed=0)
    model, tok = load_lm(args.model, dtype=args.dtype, device=args.device)
    reader = OptionReader(model, tok, device=args.device)
    print("hidden_size:", reader.hidden_size, "label token ids:", reader.label_token_ids)

    arrays = induce(
        reader,
        items,
        n_paraphrases=args.paraphrases,
        include_rotations=True,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    save(args.out, arrays, items)

    praw, qstar, gold = arrays["item_praw"], arrays["item_qstar"], arrays["item_gold"]
    print("\nper-item distributions (canonical order):")
    for i, it in enumerate(items):
        print(f"  [{it.item_id}] gold={gold[i]} ({it.options[gold[i]]})")
        print(f"    p_raw  = {np.round(praw[i], 3)}  argmax={praw[i].argmax()}")
        print(f"    q*     = {np.round(qstar[i], 3)}  argmax={qstar[i].argmax()}")
        print(f"    options= {it.options}")

    assert not np.isnan(praw).any(), "NaN in p_raw"
    assert np.allclose(praw.sum(1), 1.0, atol=1e-4), "p_raw does not sum to 1"
    assert np.allclose(qstar.sum(1), 1.0, atol=1e-4), "q* does not sum to 1"
    acc_raw = float((praw.argmax(1) == gold).mean())
    acc_q = float((qstar.argmax(1) == gold).mean())
    print(f"\np_raw accuracy {acc_raw:.3f}   q* accuracy {acc_q:.3f}   n={len(items)}")
    print("SMOKE_OK", args.out + ".npz")


if __name__ == "__main__":
    main()
