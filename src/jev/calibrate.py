"""Post-hoc temperature scaling.

A trained head inherits whatever miscalibration the aggregated target carries, so
every method here gets the same post-hoc treatment: fit one temperature on a
validation split by minimising NLL, then report the test ECE before and after.

Implemented with numpy only (no scipy) so the pipeline can run against a minimal
environment.
"""

from __future__ import annotations

import numpy as np


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, gold: np.ndarray, temperature: float) -> float:
    p = apply_temperature(logits, temperature)
    return float(-np.log(np.clip(p[np.arange(len(gold)), gold], 1e-12, 1.0)).mean())


def fit_temperature(logits: np.ndarray, gold: np.ndarray, n_grid: int = 97) -> float:
    """One scalar temperature minimising NLL on (logits, gold).

    Coarse log-grid followed by a local refinement; monotone in 1/T and smooth,
    so this is enough for the single-parameter problem.
    """
    logits = np.asarray(logits, dtype=np.float64)
    gold = np.asarray(gold)
    valid = gold >= 0
    logits, gold = logits[valid], gold[valid]
    if len(gold) == 0 or logits.shape[1] < 2:
        return 1.0

    grid = np.linspace(-2.0, 3.0, n_grid)  # log-temperature
    losses = [_nll(logits, gold, float(np.exp(lt))) for lt in grid]
    best = int(np.argmin(losses))

    lo = grid[max(0, best - 1)]
    hi = grid[min(len(grid) - 1, best + 1)]
    fine = np.linspace(lo, hi, n_grid)
    fine_losses = [_nll(logits, gold, float(np.exp(lt))) for lt in fine]
    return float(np.exp(fine[int(np.argmin(fine_losses))]))


def to_logits(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    return np.log(p)
