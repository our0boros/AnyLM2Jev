"""Label-free debiasing of the readout.

Two biases survive even a frozen-LM choice readout, and both belong to the readout
rather than to the model:

* **position bias** -- which slot in the option list the model likes.  Averaging
  over cyclic shifts (every option lands in every slot once) removes it when the
  bias is additive in logit space, because a geometric mean cancels an additive
  per-slot offset exactly: ``logit(i at slot j) = c_i + b_j`` averages to
  ``c_i + mean(b) - mean(log Z)``, i.e. ``softmax(c)``.
* **label / prior bias** -- which *label* the model likes regardless of the input
  (``Yes`` over ``No``, common classes over rare ones).  Estimate the prior without
  labels and divide it out.

Both corrections are training-free and need no labels, so they can be folded into
the self-consistency target ``q*`` *before* the head is distilled towards it.  The
batch estimator follows Zhou et al. (ICLR 2024) and the content-free estimator
Zhao et al. (ICML 2021); permutation marginalization follows Zheng et al.
(ICLR 2024).  Because our arrays store canonical order plus the permutation,
position space is reconstructed rather than stored.

Everything here is pure numpy: no torch, no transformers.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8

PRIOR_MODES = ("none", "batch", "content_free")
COMBINE_MODES = ("mean", "logmean")


def _renorm(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, None)
    return p / p.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------- position space


def view_to_position(view_probs: np.ndarray, view_order: np.ndarray) -> np.ndarray:
    """Canonical-order distributions -> display-position distributions.

    ``view_probs`` [N, V, K] is indexed by canonical option index; ``view_order``
    [N, V, K] holds the canonical index shown at each position.  Returns [N, V, K]
    where index ``j`` now means "the option displayed in slot ``j``".
    """
    vp = np.asarray(view_probs, dtype=np.float64)
    order = np.asarray(view_order, dtype=np.int64)
    if vp.shape != order.shape:
        raise ValueError(f"shape mismatch: view_probs {vp.shape} vs view_order {order.shape}")
    return np.take_along_axis(vp, order, axis=-1)


def position_to_view(p_pos: np.ndarray, view_order: np.ndarray) -> np.ndarray:
    """Inverse of :func:`view_to_position`."""
    p_pos = np.asarray(p_pos, dtype=np.float64)
    order = np.asarray(view_order, dtype=np.int64)
    n, v, k = order.shape
    inv = np.empty_like(order)
    np.put_along_axis(
        inv, order, np.broadcast_to(np.arange(k, dtype=np.int64), (n, v, k)), axis=-1
    )
    return np.take_along_axis(p_pos, inv, axis=-1)


# ----------------------------------------------------------------------- priors


def batch_prior(p_pos: np.ndarray) -> np.ndarray:
    """Label-free prior from real inputs: the mean position distribution.

    ``p_pos`` [N, V, K] -> prior [V, K].  Low variance and the more reliable of the
    two estimators here; it assumes the label marginal of the batch is not extreme.
    """
    return _renorm(np.asarray(p_pos, dtype=np.float64).mean(axis=0))


def content_free_prior(cf_probs: np.ndarray) -> np.ndarray:
    """Prior from content-free probes.

    ``cf_probs`` is ``[V, P, K]`` (a prior shared across items, when the option set
    is the same everywhere) or ``[N, V, P, K]`` (per item, which is what real option
    texts require).  The probe axis is averaged out, so the result is ``[V, K]`` or
    ``[N, V, K]`` respectively.
    """
    cf = np.asarray(cf_probs, dtype=np.float64)
    if cf.ndim not in (3, 4):
        raise ValueError(f"expected [V, P, K] or [N, V, P, K] content-free probabilities, got {cf.shape}")
    return _renorm(cf.mean(axis=-2))


def apply_prior(p: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """``normalize(p / prior)`` with broadcasting over leading axes."""
    p = np.asarray(p, dtype=np.float64)
    if p.shape[-1] != prior.shape[-1]:
        raise ValueError(f"prior {prior.shape} does not match {p.shape}")
    return _renorm(p / np.clip(prior, EPS, None))


# ------------------------------------------------------------------- aggregation


def marginalize(p_pos: np.ndarray, view_order: np.ndarray, combine: str = "mean") -> np.ndarray:
    """[N, V, K] position space -> [N, K] canonical, combining across views."""
    if combine not in COMBINE_MODES:
        raise ValueError(f"combine must be one of {COMBINE_MODES}")
    canon = position_to_view(p_pos, view_order)  # [N, V, K]
    if combine == "mean":
        return _renorm(canon.mean(axis=1))
    logp = np.log(np.clip(canon, EPS, None)).mean(axis=1)
    logp = logp - logp.max(axis=-1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=-1, keepdims=True)


def debias(
    view_probs: np.ndarray,
    view_order: np.ndarray,
    prior: np.ndarray | None = None,
    combine: str = "mean",
) -> np.ndarray:
    """Full L0 pipeline for one method: prior correction, then marginalization.

    ``view_probs`` [N, V, K] canonical (V may be 1 for a single-pass readout),
    ``view_order`` [N, V, K], ``prior`` [V, K] or None for no prior correction.
    """
    p_pos = view_to_position(view_probs, view_order)
    if prior is not None:
        p_pos = apply_prior(p_pos, np.asarray(prior))
    return marginalize(p_pos, view_order, combine=combine)


def flip_rate(view_probs: np.ndarray, view_order: np.ndarray | None = None) -> float:
    """Fraction of views whose top option differs from the first view's top option.

    This is the standard order-flip rate, per item and averaged: the fraction of
    views whose top option differs from the first view's.  ``view_probs`` is
    canonical-order [N, V, K], so the winner is already an option index and
    ``view_order`` is only accepted for call-site symmetry with :func:`debias`.
    """
    vp = np.asarray(view_probs, dtype=np.float64)
    winners = vp.argmax(axis=-1)  # [N, V], already in canonical option space
    if vp.shape[1] <= 1:
        return 0.0
    return float(np.mean(winners[:, 1:] != winners[:, :1]))
