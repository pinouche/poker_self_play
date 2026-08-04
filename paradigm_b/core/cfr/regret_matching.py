"""Regret matching — the local rule every CFR variant is built from.

Given the cumulative regret for each action at an information set, play each
action in proportion to how much regret it has accumulated, and play uniformly
when nothing is positive.  Hart & Mas-Colell showed the resulting per-infoset
regret grows sublinearly; CFR's folk theorem then bounds the exploitability of
the *average* strategy by the average regret.
"""

from __future__ import annotations

import numpy as np


def regret_matching(regrets: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """Strategy proportional to positive regret (uniform if there is none)."""
    positive = np.maximum(regrets, 0.0)
    total = positive.sum()
    if out is None:
        out = np.empty_like(positive)
    if total > 0.0:
        np.divide(positive, total, out=out)
    else:
        out.fill(1.0 / len(regrets))
    return out


def regret_matching_batch(regrets: np.ndarray) -> np.ndarray:
    """Row-wise :func:`regret_matching` for a ``(n, num_actions)`` block."""
    positive = np.maximum(regrets, 0.0)
    totals = positive.sum(axis=-1, keepdims=True)
    uniform = np.full_like(positive, 1.0 / positive.shape[-1])
    return np.where(totals > 0.0, positive / np.where(totals > 0.0, totals, 1.0), uniform)
