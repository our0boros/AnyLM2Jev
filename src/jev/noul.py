"""The noul (option-conditioned yes/no) formulation.

Jev's own primitive is a Bernoulli per option, not a softmax over an option list.
Rendering one prompt per option has two structural advantages over the choice
format:

* **arbitrary cardinality** -- no single-token label ceiling, so the full 77-way
  Banking77 task becomes reachable; and
* **order invariance by construction** -- the option order never appears in the
  context, so there is no permutation sensitivity to average away.

The output arrays deliberately match :mod:`jev.induce`, so the existing head
training / evaluation / analysis stages work unchanged.  The per-option `p_yes`
values are turned into an option distribution via ``softmax(logit(p_yes))``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .modeling import OptionReader
from .prompts import build_noul_prompt
from .schema import DecisionItem


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def induce_noul(
    reader: OptionReader,
    items: Sequence[DecisionItem],
    n_paraphrases: int = 4,
    batch_size: int = 16,
    max_length: int = 512,
) -> dict:
    n = len(items)
    k = items[0].n_options
    if any(it.n_options != k for it in items):
        raise ValueError("induce_noul expects a constant option count per run")
    v = n_paraphrases

    prompts: list[str] = []
    for it in items:
        for pid in range(v):
            for opt in it.options:
                prompts.append(build_noul_prompt(it.state, it.question, opt, pid))

    read = reader.read_noul(
        prompts, batch_size=batch_size, max_length=max_length, keep_hidden=True
    )
    p_yes = read["p_yes"].numpy().reshape(n, v, k).astype(np.float64)
    hidden = read["option_hidden"].numpy().reshape(n, v, k, -1)

    eps = 1e-6
    p_clip = np.clip(p_yes, eps, 1.0 - eps)
    logits = np.log(p_clip / (1.0 - p_clip)).astype(np.float32)  # [N, V, K]
    probs = _softmax(logits.astype(np.float64)).astype(np.float32)

    identity = np.tile(np.arange(k, dtype=np.int16), (n, v, 1))
    return {
        "view_order": identity,
        "view_paraphrase": np.tile(np.arange(v, dtype=np.int16), (n, 1)),
        "view_label_logits": logits,
        "view_probs": probs,
        "view_option_hidden": hidden.astype(np.float16),
        "view_p_yes": p_yes.astype(np.float32),
        "item_praw": probs[:, 0, :].copy(),
        "item_qstar": probs.mean(axis=1).astype(np.float32),
        "item_gold": np.asarray(
            [(-1 if it.gold is None else it.gold) for it in items], dtype=np.int16
        ),
        "item_n_options": np.asarray([it.n_options for it in items], dtype=np.int16),
    }
