"""Prompt rendering for choice-form decision queries.

Design choice: every option is rendered **in one shared context**, and the model
is read at (a) the final token for the logit baseline and (b) the end of each
option's text for the trained head.  Reading the head input at the option's
*label* letter would be a bug: a causal LM at that position has not read the
option yet.

A *view* is a `(paraphrase_id, order)` pair: a wording of the task plus a
permutation of the options.  Aggregating `p_raw` across views is the
self-consistency signal we distil into the head.

Labels must be single tokens for the logit baseline to be well defined, so the
choice format supports up to 26 options (``A``..``Z``).  Higher cardinalities
need the option-conditioned yes/no ("noul") formulation instead.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Wording variants.  All of them place the option block verbatim so that we can
# locate each label character by substring search.
PARAPHRASES: tuple[str, ...] = (
    "{state}\n\nQuestion: {question}\n\nOptions:\n{options}\n\nAnswer with the letter of the correct option.\nAnswer:",
    "{state}\n\n{question}\n\n{options}\n\nThe correct option is:",
    "Task: {question}\n\nContext:\n{state}\n\nCandidates:\n{options}\n\nBest candidate:",
    "{state}\n\nQ: {question}\n\n{options}\n\nChoose one letter:",
)


@dataclass(frozen=True)
class View:
    """A wording plus an option permutation."""

    paraphrase_id: int
    order: tuple[int, ...]  # canonical option indices, in display order


def make_labels(n_options: int) -> list[str]:
    """Single-token labels for ``n_options`` options (``A``..``Z``)."""
    if n_options > len(LABELS):
        raise ValueError(
            f"{n_options} options exceed the {len(LABELS)} single-token labels; "
            "use the option-conditioned noul formulation for higher cardinality"
        )
    return list(LABELS[:n_options])


def default_orders(n_options: int, max_orders: int | None = None, seed: int = 0) -> list[tuple[int, ...]]:
    """Option permutations to use.

    ``max_orders=None`` -> identity plus all cyclic rotations (cheap and complete
    for small option sets).  Otherwise identity plus ``max_orders-1`` seeded
    random permutations, which is what keeps a 26-way task affordable.
    """
    identity = tuple(range(n_options))
    if max_orders is None or max_orders >= n_options:
        return [identity] + [
            tuple((i + shift) % n_options for i in range(n_options))
            for shift in range(1, n_options)
        ]

    orders = [identity]
    seen = {identity}
    rng = random.Random(seed)
    guard = 0
    while len(orders) < max_orders and guard < 10_000:
        guard += 1
        perm = list(range(n_options))
        rng.shuffle(perm)
        perm_t = tuple(perm)
        if perm_t not in seen:
            seen.add(perm_t)
            orders.append(perm_t)
    return orders


def make_views(
    n_options: int,
    n_paraphrases: int,
    orders: Sequence[tuple[int, ...]] | None = None,
    include_rotations: bool = True,
    max_orders: int | None = None,
    seed: int = 0,
) -> list[View]:
    if orders is None:
        orders = default_orders(
            n_options,
            max_orders if max_orders is not None else (None if include_rotations else 1),
            seed=seed,
        )
    views: list[View] = []
    for pid in range(n_paraphrases):
        for order in orders:
            views.append(View(paraphrase_id=pid, order=tuple(order)))
    return views


def render_options(
    option_texts: Sequence[str], labels: Sequence[str] | None = None
) -> tuple[str, list[int], list[int]]:
    """Render the option block.

    Returns ``(block, label_offsets, text_end_offsets)`` where the offsets are
    character positions *inside* the block: the first character of each label and
    the last character of each option's text.  The label offset is what the logit
    baseline reads; the text-end offset is where a causal model has finished
    reading the option, which is the only useful place to take a hidden state.
    """
    labels = list(labels) if labels is not None else make_labels(len(option_texts))
    lines: list[str] = []
    label_off: list[int] = []
    end_off: list[int] = []
    cursor = 0
    for i, text in enumerate(option_texts):
        line = f"{labels[i]}. {text}"
        label_off.append(cursor)
        end_off.append(cursor + len(line) - 1)
        lines.append(line)
        cursor += len(line) + 1  # +1 for the newline
    return "\n".join(lines), label_off, end_off


def build_prompt(
    state: str,
    question: str,
    option_texts_in_order: Sequence[str],
    paraphrase_id: int,
    labels: Sequence[str] | None = None,
) -> tuple[str, list[str], list[int], list[int]]:
    """Return (prompt, labels, label offsets, option-text-end offsets)."""
    labels = list(labels) if labels is not None else make_labels(len(option_texts_in_order))
    block, inner_label, inner_end = render_options(option_texts_in_order, labels)
    template = PARAPHRASES[paraphrase_id % len(PARAPHRASES)]
    prompt = template.format(state=state, question=question, options=block)
    base = prompt.index(block)
    return prompt, labels, [base + off for off in inner_label], [base + off for off in inner_end]


# ------------------------------------------------------------------ noul (yes/no)
#
# The option-conditioned yes/no formulation.  Each option gets its own prompt, so
# there is no option ordering in the context at all: this interface is
# order-invariant by construction and supports arbitrary cardinality (Jev's own
# primitive).  The readout is the final-token "Yes" vs "No" contrast.

NOUL_PARAPHRASES: tuple[str, ...] = (
    "{state}\n\n{question}\n\nOption: {option}\nIs this option correct? Answer Yes or No:\nAnswer:",
    "Context:\n{state}\n\nQuestion: {question}\nCandidate: {option}\nDoes the candidate apply? Yes or No:",
    "{state}\n\n{question}\n\nConsider this option: {option}\nIs it a correct answer? Yes/No:",
    "Task: {question}\n\n{state}\n\nIs the following correct: {option}\nAnswer Yes or No:",
)


def build_noul_prompt(state: str, question: str, option_text: str, paraphrase_id: int) -> str:
    template = NOUL_PARAPHRASES[paraphrase_id % len(NOUL_PARAPHRASES)]
    return template.format(state=state, question=question, option=option_text)
