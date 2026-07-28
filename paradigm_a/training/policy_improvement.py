"""KL/entropy-regularised policy improvement.

The improved policy is

    pi_new(a|s)  proportional to  exp( (Q(s,a) + beta * log pi_ref(a|s)) / (alpha + beta) )

restricted to legal actions and renormalised over them.

* ``alpha`` (entropy) pulls the result toward uniform-over-legal;
* ``beta`` (reverse KL) pulls it toward the reference policy, which bounds how
  far one improvement step can move;
* as ``alpha, beta -> 0`` the operator degenerates to greedy ``argmax Q``.

Both a NumPy version (used while acting in self-play) and a Torch version (used
to build training targets) are provided; they compute the same quantity.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

NEG_INF = -1e9
_LOG_EPS = 1e-8


def improved_policy_np(
    q_values: np.ndarray,
    reference_policy: np.ndarray,
    legal_mask: np.ndarray,
    alpha: float,
    beta: float,
    temperature: float = 1.0,
    q_scale: float = 1.0,
) -> np.ndarray:
    """Improved distribution for one state; illegal actions are exactly zero.

    ``q_scale`` divides Q before exponentiation so that ``alpha`` and ``beta``
    are dimensionless and portable across reward modes.
    """
    q_values = np.asarray(q_values, dtype=np.float64)
    legal = np.asarray(legal_mask, dtype=np.float64) > 0
    if not legal.any():
        raise ValueError("no legal actions")

    reference = np.asarray(reference_policy, dtype=np.float64) * legal
    total = reference.sum()
    reference = reference / total if total > 0 else legal / legal.sum()
    log_reference = np.log(np.clip(reference, _LOG_EPS, None))

    denominator = max(alpha + beta, _LOG_EPS)
    scores = (q_values / max(q_scale, _LOG_EPS) + beta * log_reference) / denominator

    if temperature <= 0:
        out = np.zeros_like(scores)
        out[int(np.argmax(np.where(legal, scores, -np.inf)))] = 1.0
        return out.astype(np.float32)

    scores = np.where(legal, scores / temperature, NEG_INF)
    scores = scores - scores.max()
    weights = np.exp(scores) * legal
    total = weights.sum()
    if total <= 0:
        return (legal / legal.sum()).astype(np.float32)
    return (weights / total).astype(np.float32)


def improved_policy_torch(
    q_values: torch.Tensor,
    log_reference_policy: torch.Tensor,
    legal_mask: torch.Tensor,
    alpha: float,
    beta: float,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """Batched improved policy target, shape ``[batch, action_count]``.

    ``log_reference_policy`` must already be masked (illegal entries very
    negative).  The result is detached: it is a target, not a differentiable
    quantity.
    """
    denominator = max(alpha + beta, _LOG_EPS)
    log_reference = torch.clamp(log_reference_policy, min=-30.0)
    scores = (q_values / max(q_scale, _LOG_EPS) + beta * log_reference) / denominator
    scores = scores.masked_fill(legal_mask <= 0, NEG_INF)
    probs = torch.softmax(scores, dim=-1)
    probs = probs * (legal_mask > 0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(_LOG_EPS)
    return probs.detach()


def masked_entropy(probs: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Per-row entropy over legal actions."""
    safe = torch.clamp(probs, min=_LOG_EPS)
    terms = -probs * torch.log(safe) * (legal_mask > 0)
    return terms.sum(dim=-1)


def masked_kl(p: torch.Tensor, q: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """KL(p || q) restricted to legal actions (diagnostic)."""
    p_safe = torch.clamp(p, min=_LOG_EPS)
    q_safe = torch.clamp(q, min=_LOG_EPS)
    terms = p * (torch.log(p_safe) - torch.log(q_safe)) * (legal_mask > 0)
    return terms.sum(dim=-1)


def expected_value(
    probs: np.ndarray, q_values: np.ndarray, legal_mask: Optional[np.ndarray] = None
) -> float:
    """V(s) = sum_a pi(a|s) Q(s,a) over legal actions -- the bootstrap value."""
    probs = np.asarray(probs, dtype=np.float64)
    q_values = np.asarray(q_values, dtype=np.float64)
    if legal_mask is not None:
        legal = np.asarray(legal_mask, dtype=np.float64) > 0
        probs = probs * legal
        q_values = np.where(legal, q_values, 0.0)
    total = probs.sum()
    if total <= 0:
        return 0.0
    return float((probs * q_values).sum() / total)
