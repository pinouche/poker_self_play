"""Action translation: mapping a real bet onto the abstraction.

The abstraction knows a handful of bet sizes.  A real opponent does not, and
will bet 63% of the pot when the tree contains only 50% and 100%.  Without a
rule for that the agent simply cannot be handed the action, which is the
difference between a solver and something that can sit at a table.

The rule here is the **pseudo-harmonic mapping** of Ganzfried & Sandholm (2013).
Given a bet ``x`` between neighbouring abstraction sizes ``A`` and ``B``, it maps
to ``A`` with probability

    f(A, B, x) = ((B - x) * (1 + A)) / ((B - A) * (1 + x))

with all three expressed as fractions of the pot.  Two properties matter.  It is
**randomised**, so an opponent cannot learn which side of the boundary a given
size lands on and exploit the seam.  And it is *not* linear: the weighting leans
toward the smaller size, which is what stops the classic attack where an
opponent bets just over a boundary all game and has every such bet treated as
the much larger one.

Translation is lossy by construction — the agent answers a question slightly
different from the one it was asked — and that loss is a floor on exploitability
no amount of search removes.  The only cure is a finer abstraction.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from paradigm_b.holdem.engine.betting import CALL, FIRST_BET, Betting


def pseudo_harmonic_weight(smaller: float, larger: float, actual: float) -> float:
    """Probability of mapping ``actual`` down to ``smaller`` rather than up."""
    if actual <= smaller:
        return 1.0
    if actual >= larger:
        return 0.0
    numerator = (larger - actual) * (1.0 + smaller)
    denominator = (larger - smaller) * (1.0 + actual)
    if denominator <= 0.0:
        return 1.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def bet_sizes_in_pots(betting: Betting) -> List[Tuple[int, float]]:
    """Each legal aggressive action with the pot fraction it actually commits."""
    player = betting.to_move()
    owed = betting.to_call(player)
    pot_after_call = betting.pot + owed
    sizes = []
    for action in betting.legal_actions():
        if not betting.is_aggressive(action):
            continue
        total = betting.raise_total(player, action)
        extra = total - betting.contributions[player] - owed
        sizes.append((action, extra / max(pot_after_call, 1)))
    return sorted(sizes, key=lambda pair: pair[1])


def translate_bet(betting: Betting, amount: int) -> Dict[int, float]:
    """Distribution over abstraction actions for a real raise to ``amount``.

    ``amount`` is the opponent's total contribution after the raise, in chips —
    the same convention the tree uses internally.
    """
    player = betting.to_move()
    owed = betting.to_call(player)
    pot_after_call = betting.pot + owed
    extra = amount - betting.contributions[player] - owed
    actual = extra / max(pot_after_call, 1)

    sizes = bet_sizes_in_pots(betting)
    if not sizes:
        return {CALL: 1.0}
    if actual <= sizes[0][1]:
        return {sizes[0][0]: 1.0}
    if actual >= sizes[-1][1]:
        return {sizes[-1][0]: 1.0}

    for (low_action, low), (high_action, high) in zip(sizes, sizes[1:]):
        if low <= actual <= high:
            weight = pseudo_harmonic_weight(low, high, actual)
            return {low_action: weight, high_action: 1.0 - weight}
    return {sizes[-1][0]: 1.0}


def translate_action(
    betting: Betting, amount: int, rng: np.random.Generator
) -> int:
    """Sample one abstraction action for a real bet — randomised on purpose."""
    distribution = translate_bet(betting, amount)
    actions = list(distribution)
    weights = np.array([distribution[a] for a in actions], dtype=np.float64)
    weights /= weights.sum()
    return int(rng.choice(actions, p=weights))
