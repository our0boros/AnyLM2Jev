"""Core data types for the Jev-style decision interface.

A Jev request is a `state` (free text or JSON) plus one or more questions, each
of which is a Choice, a Score or a Noul (a Bernoulli / yes-no probability).  A
reproduction only needs the *choice* case to run a first experiment, so this
module keeps the record format general but the pipeline uses `task="choice"`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class DecisionItem:
    """One decision query with a finite, ordered option set.

    Attributes
    ----------
    item_id : stable identifier, unique within a split
    state   : the unstructured context handed to the model
    question: what is being asked about the state
    options : option *texts* in canonical order
    gold    : index into `options`, or None when there is no reference answer
    task    : "choice" | "noul" | "score"
    meta    : free-form provenance (dataset, original label, difficulty, ...)
    """

    item_id: str
    state: str
    question: str
    options: list[str]
    gold: Optional[int] = None
    task: str = "choice"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError(f"{self.item_id}: empty option list")
        if self.gold is not None and not (0 <= self.gold < len(self.options)):
            raise ValueError(f"{self.item_id}: gold {self.gold} out of range")

    @property
    def n_options(self) -> int:
        return len(self.options)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "DecisionItem":
        return cls(**d)


def read_jsonl(path: str) -> list[DecisionItem]:
    import json

    out: list[DecisionItem] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(DecisionItem.from_json(json.loads(line)))
    return out


def write_jsonl(items: list[DecisionItem], path: str) -> None:
    import json

    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it.to_json(), ensure_ascii=False) + "\n")
