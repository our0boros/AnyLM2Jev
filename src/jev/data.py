"""Decision datasets.

The primary source is WANLI (zero-shot NLI), the same external check the SemIf
reproduction used, because it has verifiable gold labels and a known baseline
(Qwen3.5-4B direct option logits scored 0.637 balanced accuracy on WANLI).  A
small templated generator is provided as an offline fallback and for smoke runs.
"""

from __future__ import annotations

import random
from typing import Iterable, Optional

from .schema import DecisionItem

NLI_OPTIONS = (
    "entailment: the premise entails the hypothesis",
    "neutral: the premise neither entails nor contradicts the hypothesis",
    "contradiction: the premise contradicts the hypothesis",
)
NLI_QUESTION = "What is the relationship between the premise and the hypothesis?"


def nli_item(idx: int, premise: str, hypothesis: str, label: int, source: str = "wanli") -> DecisionItem:
    return DecisionItem(
        item_id=f"{source}-{idx}",
        state=f"Premise: {premise.strip()}\nHypothesis: {hypothesis.strip()}",
        question=NLI_QUESTION,
        options=list(NLI_OPTIONS),
        gold=int(label),
        task="choice",
        meta={"source": source},
    )


def from_wanli(records: Iterable[dict], limit: Optional[int] = None) -> list[DecisionItem]:
    items: list[DecisionItem] = []
    for i, r in enumerate(records):
        if limit is not None and len(items) >= limit:
            break
        items.append(
            nli_item(i, r["premise"], r["hypothesis"], int(r["label"]), source="wanli")
        )
    return items


_CITIES = ["Lisbon", "Osaka", "Quito", "Riga", "Perth", "Cairo", "Bern", "Lima"]
_NAMES = ["Ada", "Bram", "Cleo", "Dario", "Elin", "Faisal", "Greta", "Hugo"]
_COLORS = ["teal", "amber", "indigo", "olive"]


def synthetic_items(n: int, seed: int = 0, n_options: int = 3) -> list[DecisionItem]:
    """A trivially verifiable extraction task, for smoke tests without network."""
    rng = random.Random(seed)
    items: list[DecisionItem] = []
    for i in range(n):
        name = rng.choice(_NAMES)
        city = rng.choice(_CITIES)
        colour = rng.choice(_COLORS)
        distractors = rng.sample([c for c in _CITIES if c != city], n_options - 1)
        options = [city] + distractors
        rng.shuffle(options)
        state = (
            f"{name} moved to {city} last spring. {name} likes the {colour} trams there "
            f"and writes about them every week."
        )
        items.append(
            DecisionItem(
                item_id=f"synth-{i}",
                state=state,
                question=f"Which city does {name} live in?",
                options=options,
                gold=options.index(city),
                task="choice",
                meta={"source": "synthetic"},
            )
        )
    return items
