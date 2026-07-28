"""Action-space helpers shared by the environment, the network and the API.

The action *ids* themselves live in :mod:`environment.state` because they are a
property of the rules; this module re-exports them and adds encoding and
masking utilities.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from paradigm_a.environment.state import (  # noqa: F401  (re-exported)
    ACTION_ID_FROM_NAME,
    ACTION_NAMES,
    AGGRESSIVE_ACTIONS,
    ALL_IN,
    BET_ACTIONS,
    BET_LARGE,
    BET_MEDIUM,
    BET_SMALL,
    CALL,
    CHECK,
    FOLD,
    NUM_ACTIONS,
    RAISE_ACTIONS,
    RAISE_LARGE,
    RAISE_MEDIUM,
    RAISE_SMALL,
)

NEG_INF = -1e9


def action_one_hot(action_id: int, num_actions: int = NUM_ACTIONS) -> np.ndarray:
    vec = np.zeros(num_actions, dtype=np.float32)
    vec[action_id] = 1.0
    return vec


def mask_from_ids(ids: Sequence[int], num_actions: int = NUM_ACTIONS) -> np.ndarray:
    mask = np.zeros(num_actions, dtype=np.float32)
    for i in ids:
        mask[i] = 1.0
    return mask


def masked_softmax(logits: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax restricted to legal actions; illegal entries are exactly zero."""
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if mask.sum() <= 0:
        raise ValueError("no legal actions in mask")

    if temperature <= 0:
        # Deterministic: all probability on the best legal action.
        scores = np.where(mask > 0, logits, -np.inf)
        out = np.zeros_like(logits)
        out[int(np.argmax(scores))] = 1.0
        return out.astype(np.float32)

    scaled = np.where(mask > 0, logits / temperature, NEG_INF)
    scaled = scaled - scaled.max()
    exp = np.exp(scaled) * mask
    total = exp.sum()
    if total <= 0:  # numerical fallback: uniform over legal actions
        return (mask / mask.sum()).astype(np.float32)
    return (exp / total).astype(np.float32)


def apply_temperature(probs: np.ndarray, mask: np.ndarray, temperature: float) -> np.ndarray:
    """Re-sharpen an existing distribution over legal actions."""
    probs = np.asarray(probs, dtype=np.float64) * (mask > 0)
    if temperature <= 0:
        out = np.zeros_like(probs)
        out[int(np.argmax(np.where(mask > 0, probs, -np.inf)))] = 1.0
        return out.astype(np.float32)
    with np.errstate(divide="ignore"):
        logits = np.log(np.clip(probs, 1e-12, None))
    return masked_softmax(logits, mask, temperature)


def sample_action(probs: np.ndarray, rng) -> int:
    """Sample an action id from a distribution that is already masked."""
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if total <= 0:
        raise ValueError("cannot sample from an all-zero distribution")
    probs = probs / total
    return int(rng.choices(range(len(probs)), weights=probs.tolist(), k=1)[0])


def probs_to_dict(
    probs: Sequence[float],
    mask: Optional[Sequence[float]] = None,
    names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Public-API view: every action named, illegal ones exactly 0.0."""
    names = names if names is not None else ACTION_NAMES
    out: Dict[str, float] = {}
    for i, name in enumerate(names):
        if mask is not None and not mask[i]:
            out[name] = 0.0
        else:
            out[name] = float(probs[i])
    return out


def q_values_to_dict(
    q_values: Sequence[float],
    mask: Sequence[float],
    names: Optional[Sequence[str]] = None,
) -> Dict[str, Optional[float]]:
    """Public-API view: illegal actions report ``None`` rather than a number."""
    names = names if names is not None else ACTION_NAMES
    return {
        name: (float(q_values[i]) if mask[i] else None) for i, name in enumerate(names)
    }


def legal_action_names(
    mask: Sequence[float], names: Optional[Sequence[str]] = None
) -> List[str]:
    names = names if names is not None else ACTION_NAMES
    return [name for i, name in enumerate(names) if mask[i]]
