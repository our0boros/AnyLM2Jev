#!/usr/bin/env python
"""Stage 1 (noul variant): option-conditioned yes/no readout (GPU).

Supports arbitrary cardinality, unlike the choice format's 26-option label ceiling.

    PYTHONPATH=src python scripts/11_induce_noul.py \
        --data data/banking77full_train.jsonl --out runs/induce/banking77full_noul
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jev import induce as induce_mod  # noqa: E402
from jev.modeling import OptionReader, load_lm  # noqa: E402
from jev.noul import induce_noul  # noqa: E402
from jev.schema import read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--paraphrases", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    items = read_jsonl(args.data)
    if args.limit:
        items = items[: args.limit]
    print(f"items: {len(items)}  options/item: {items[0].n_options}  paraphrases: {args.paraphrases}")

    model, tok = load_lm(args.model, dtype=args.dtype, device=args.device)
    reader = OptionReader(model, tok, device=args.device)

    arrays = induce_noul(
        reader,
        items,
        n_paraphrases=args.paraphrases,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    induce_mod.save(args.out, arrays, items)
    print("saved", args.out + ".npz")
    print("q* shape", arrays["item_qstar"].shape, "p_raw shape", arrays["item_praw"].shape)


if __name__ == "__main__":
    main()
