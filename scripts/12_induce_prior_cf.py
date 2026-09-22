#!/usr/bin/env python
"""Stage 1b: content-free label prior (contextual calibration, Zhao et al. 2021).

The batch prior in :mod:`jev.debias` removes the model's marginal preference among
labels using real inputs.  This stage computes the *other* label-free estimator:
render each prompt with the **state replaced by a content-free probe**
(``N/A``, ``""``, ``[MASK]``) and read the label distribution.  Dividing that out
removes the part of the preference that has nothing to do with the input.

The option *texts* are part of our prompt, so the probe distribution is per item;
the output is ``cf_position`` [N, V, P, K] in display order, ready for
:func:`jev.debias.content_free_prior`.

    python scripts/12_induce_prior_cf.py --model Qwen/Qwen3.5-2B-Base \\
      --data data/banking77_test.jsonl --out runs/prior/banking77_cf_test \\
      --paraphrases 2 --max-orders 3
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jev import induce as induce_mod  # noqa: E402
from jev.modeling import OptionReader, load_lm  # noqa: E402
from jev.prompts import build_prompt, make_labels  # noqa: E402
from jev.schema import read_jsonl  # noqa: E402

PROBES = ("N/A", "", "[MASK]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True, help="output npz path")
    ap.add_argument("--paraphrases", type=int, default=4)
    ap.add_argument("--no-rotations", action="store_true")
    ap.add_argument("--max-orders", type=int, default=0, help="must match the induce run")
    ap.add_argument("--seed", type=int, default=0, help="must match the induce run")
    ap.add_argument("--probes", default=",".join(PROBES))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    items = read_jsonl(args.data)
    if args.limit:
        items = items[: args.limit]
    probes = [p for p in args.probes.split(",")]

    # the same view grid as the induce run, so the priors line up view-for-view
    all_views = induce_mod.build_views(
        items,
        args.paraphrases,
        include_rotations=not args.no_rotations,
        max_orders=(args.max_orders or None),
        seed=args.seed,
    )
    n = len(items)
    v = len(all_views[0])
    k = items[0].n_options
    if any(it.n_options != k for it in items):
        raise ValueError("content-free prior expects a constant option count")

    labels = make_labels(k)
    prompts, label_offsets, index = [], [], []
    for i, it in enumerate(items):
        for j, view in enumerate(all_views[i]):
            texts = [it.options[c] for c in view.order]
            for pi, probe in enumerate(probes):
                prompt, _lab, char_offs, _ends = build_prompt(
                    probe, it.question, texts, view.paraphrase_id, labels=labels
                )
                prompts.append(prompt)
                label_offsets.append(char_offs)
                index.append((i, j, pi))

    print(f"items={n} views/item={v} probes={len(probes)} prompts={len(prompts)}")

    model, tok = load_lm(args.model, dtype=args.dtype, device=args.device)
    reader = OptionReader(model, tok, device=args.device, labels=labels)
    read = reader.read(
        prompts, label_offsets, batch_size=args.batch_size, max_length=args.max_length, keep_hidden=False
    )
    logits = read["last_label_logits"].numpy()  # [N*V*P, K]

    cf = np.zeros((n, v, len(probes), k), dtype=np.float32)
    for (i, j, pi), z in zip(index, logits):
        z = np.asarray(z, dtype=np.float64)
        z = z - z.max()
        p = np.exp(z)
        cf[i, j, pi] = (p / p.sum()).astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, cf_position=cf)
    print("saved", args.out, "cf_position", cf.shape)


if __name__ == "__main__":
    main()
