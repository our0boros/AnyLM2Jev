"""Forward-KL distillation of the aggregated target into the readout head.

The loss is deliberately *not* cross-entropy against hard labels.  Hard labels
discard exactly the confidence information that distinguishes a Jev-like readout
from a plain classifier, and they are the usual reason teacher distillation into
a weaker student hurts calibration.
"""

from __future__ import annotations

import torch


def soft_kl(scores: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Forward KL(target || softmax(scores)), averaged over valid options.

    ``scores``/``target`` are [B, K]; ``mask`` is [B, K] bool.
    """
    logp = torch.log_softmax(scores, dim=-1)
    per_item = -(target * logp).sum(dim=-1)  # [B]
    return per_item.mean()


def entropy(scores: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(scores, dim=-1)
    return -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1).mean()


def train_step(head, optimizer, hidden, target, mask, entropy_weight: float = 0.0) -> float:
    head.train()
    optimizer.zero_grad(set_to_none=True)
    scores = head(hidden, mask)
    loss = soft_kl(scores, target, mask)
    if entropy_weight:
        loss = loss - entropy_weight * entropy(scores)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())
