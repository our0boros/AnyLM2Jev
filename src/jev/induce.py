"""Self-consistency targets from a frozen LM.

For each item we render every ``(paraphrase, option-order)`` view, read the LM,
and keep three things per view: the canonical-order label-logit distribution, the
canonical-order per-option hidden states, and the permutation itself.

From those we build

* ``p_raw``  -- the *first* (canonical) view's distribution, i.e. the free
  single-forward-pass SemIf-style baseline;
* ``q*``     -- the mean distribution across views: the self-consistency target
  the head is distilled towards, and also the multi-pass reference.

The gap between ``p_raw`` and ``q*`` is the gain available *without any
training*; the head's job is to compress it into one forward pass.
"""

from __future__ import annotations

import json
from typing import Sequence

import numpy as np

from .modeling import OptionReader
from .prompts import View, build_prompt, default_orders, make_labels, make_views
from .schema import DecisionItem

try:  # progress is nice-to-have, never required
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def build_views(
    items: Sequence[DecisionItem],
    n_paraphrases: int,
    include_rotations: bool = True,
    max_orders: int | None = None,
    seed: int = 0,
) -> list[list[View]]:
    n_options = items[0].n_options
    orders = default_orders(
        n_options,
        max_orders if max_orders is not None else (None if include_rotations else 1),
        seed=seed,
    )
    return [make_views(it.n_options, n_paraphrases, orders=orders) for it in items]


def induce(
    reader: OptionReader,
    items: Sequence[DecisionItem],
    n_paraphrases: int = 4,
    include_rotations: bool = True,
    max_orders: int | None = None,
    seed: int = 0,
    batch_size: int = 8,
    max_length: int = 512,
    keep_hidden: bool = True,
) -> dict:
    """Run the frozen LM over all views and return the aggregation arrays.

    ``keep_hidden=False`` skips storing per-option hidden states, which is all the
    ``q*`` path needs and is ~10x smaller on disk.  The returned dict then has no
    ``view_option_hidden`` key, so head training/eval cannot run on it.
    """
    n = len(items)
    k_set = {it.n_options for it in items}
    if len(k_set) != 1:
        raise ValueError(
            f"induce expects a constant option count per run, got {sorted(k_set)}; "
            "split the data or pad options first"
        )
    k = k_set.pop()

    labels = make_labels(k)
    if getattr(reader, "labels", None) is None or len(reader.labels) != k:
        # keep the reader's label scheme in sync with the option count
        reader.labels = labels
        reader.label_token_ids = [
            reader.tok.encode(label, add_special_tokens=False)[0] for label in labels
        ]

    all_views = build_views(items, n_paraphrases, include_rotations, max_orders, seed)
    v = len(all_views[0])
    if any(len(views) != v for views in all_views):
        raise ValueError("inconsistent view count across items")

    prompts: list[str] = []
    offsets: list[list[int]] = []
    hidden_offsets: list[list[int]] = []
    flat: list[tuple[int, int]] = []  # (item_index, view_index)
    for i, it in enumerate(items):
        for j, view in enumerate(all_views[i]):
            texts = [it.options[c] for c in view.order]
            prompt, _letters, char_offsets, text_end_offsets = build_prompt(
                it.state, it.question, texts, view.paraphrase_id, labels=labels
            )
            prompts.append(prompt)
            offsets.append(char_offsets)
            hidden_offsets.append(text_end_offsets)
            flat.append((i, j))

    read = reader.read(
        prompts,
        offsets,
        hidden_offsets=hidden_offsets if keep_hidden else None,
        batch_size=batch_size,
        max_length=max_length,
        keep_hidden=keep_hidden,
    )
    label_logits = read["last_label_logits"].numpy()  # [N*V, K]

    h = 0
    option_hidden = None
    if keep_hidden:
        option_hidden = read["option_hidden"].numpy()  # [N*V, K, H]
        h = option_hidden.shape[-1]

    view_order = np.zeros((n, v, k), dtype=np.int16)
    view_para = np.zeros((n, v), dtype=np.int16)
    view_logits = np.zeros((n, v, k), dtype=np.float32)
    view_probs = np.zeros((n, v, k), dtype=np.float32)
    view_hidden = np.zeros((n, v, k, h), dtype=np.float16) if keep_hidden else None

    for row, (i, j) in enumerate(flat):
        logits_disp = label_logits[row]
        hidden_disp = option_hidden[row] if option_hidden is not None else None
        view = all_views[i][j]
        order = np.asarray(view.order)
        z = np.asarray(logits_disp, dtype=np.float64)
        z = z - z.max()
        p_disp = np.exp(z)
        p_disp = p_disp / p_disp.sum()

        canon_probs = np.zeros(k, dtype=np.float32)
        canon_logits = np.zeros(k, dtype=np.float32)
        canon_probs[order] = p_disp.astype(np.float32)
        canon_logits[order] = np.asarray(logits_disp, dtype=np.float32)

        view_order[i, j] = order
        view_para[i, j] = view.paraphrase_id
        view_logits[i, j] = canon_logits
        view_probs[i, j] = canon_probs
        if view_hidden is not None and hidden_disp is not None:
            canon_hidden = np.zeros((k, h), dtype=np.float16)
            canon_hidden[order] = hidden_disp.astype(np.float16)
            view_hidden[i, j] = canon_hidden

    gold = np.asarray([(-1 if it.gold is None else it.gold) for it in items], dtype=np.int16)
    out = {
        "labels": np.asarray(labels),
        "view_order": view_order,
        "view_paraphrase": view_para,
        "view_label_logits": view_logits,
        "view_probs": view_probs,
        "item_praw": view_probs[:, 0, :].copy(),
        "item_qstar": view_probs.mean(axis=1).astype(np.float32),
        "item_gold": gold,
        "item_n_options": np.asarray([it.n_options for it in items], dtype=np.int16),
    }
    if view_hidden is not None:
        out["view_option_hidden"] = view_hidden
    return out


def save(path: str, arrays: dict, items: Sequence[DecisionItem]) -> None:
    np.savez_compressed(path + ".npz", **arrays)
    with open(path + ".items.jsonl", "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it.to_json(), ensure_ascii=False) + "\n")


def load(path: str) -> tuple[dict, list[DecisionItem]]:
    with np.load(path + ".npz", allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    items: list[DecisionItem] = []
    with open(path + ".items.jsonl", "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(DecisionItem.from_json(json.loads(line)))
    return arrays, items
