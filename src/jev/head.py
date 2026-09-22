"""Option-score head.

The head consumes the hidden state at each option's label token and emits one
scalar per option; a softmax over those scalars is the predicted distribution.
This is deliberately small: the frozen backbone does the work, the head only
re-parameterises the readout.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OptionScoreHead(nn.Module):
    def __init__(self, hidden_size: int, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        dim = hidden_size
        for _ in range(max(1, n_layers - 1)):
            layers += [nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, option_hidden: torch.Tensor, mask: torch.Tensor | None = None):
        """``option_hidden`` [B, K, H] -> scores [B, K] (masked entries -inf)."""
        scores = self.mlp(option_hidden).squeeze(-1) * self.logit_scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        return scores

    def probabilities(self, option_hidden: torch.Tensor, mask: torch.Tensor | None = None):
        return torch.softmax(self.forward(option_hidden, mask), dim=-1)


class PooledLinearHead(nn.Module):
    """Ablation: mean-pool the option hidden states, then one linear map.

    Kept as a baseline so we can tell whether the per-option structure matters.
    """

    def __init__(self, hidden_size: int, n_options: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, n_options)

    def forward(self, option_hidden: torch.Tensor, mask: torch.Tensor | None = None):
        h = option_hidden
        if mask is not None:
            h = h * mask.unsqueeze(-1)
            h = h.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            h = h.mean(1)
        return self.out(torch.tanh(self.proj(h)))
