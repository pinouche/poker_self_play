"""Showdown counterfactual values in linear time.

The quantity wanted at a showdown is, for each of my 1,326 possible hands, the
opponent's range mass I beat minus the mass I lose to — with every hand that
shares a card with mine excluded, because the opponent cannot hold it.

Written directly that is a 1,326 x 1,326 comparison per showdown node, which is
1.7M operations repeated at every node of every CFR iteration.  Sorting the
hands by strength once per board turns it into prefix sums: the mass weaker than
me is a lookup, and the blocked part of it is a second lookup in a per-card
prefix table.  The board-dependent part is cached, so a node costs O(cards) work
on top of two 1,326-element gathers.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import numpy as np

from holdem.combos import CARD_IN_COMBO, COMBO_CARDS, NUM_CARDS, NUM_COMBOS
from holdem.strength import ILLEGAL, hand_ranks, sorted_by_strength


@dataclass(frozen=True)
class ShowdownIndex:
    """Board-dependent scaffolding, computed once and reused every iteration.

    The blocking correction needs, for each hand, the opponent mass sitting on
    each of its two cards below and above its own strength.  Done densely that
    is a ``(52, n)`` prefix table — but every hand belongs to exactly two of
    those rows, so all but ``2n`` of its 52n entries are structurally zero.
    Storing the rows end to end instead makes it one ``2n`` cumulative sum, and
    since the group boundaries depend only on the board, the positions to read
    off can be precomputed too.  On a river board that is 2,162 elements per
    call instead of 56,212.
    """

    order: np.ndarray  # legal combos, weakest first
    group_start: np.ndarray  # per combo: position where its tie group begins
    group_end: np.ndarray  # per combo: position just past its tie group
    legal: np.ndarray  # boolean mask over all 1326 combos
    flat_hands: np.ndarray  # (2n,) hands grouped by card, each group by strength
    card_offset: np.ndarray  # (53,) where each card's group starts
    lo_a: np.ndarray  # per combo: offsets into the flat cumulative sum
    lo_b: np.ndarray
    hi_a: np.ndarray
    hi_b: np.ndarray


@lru_cache(maxsize=4096)
def showdown_index(board: Tuple[int, ...]) -> ShowdownIndex:
    ranks = hand_ranks(board)
    order = sorted_by_strength(board)
    sorted_ranks = ranks[order]
    # Where each run of equal strength starts and ends, mapped back to combos.
    starts = np.searchsorted(sorted_ranks, sorted_ranks, side="left")
    ends = np.searchsorted(sorted_ranks, sorted_ranks, side="right")
    group_start = np.zeros(NUM_COMBOS, dtype=np.int64)
    group_end = np.zeros(NUM_COMBOS, dtype=np.int64)
    group_start[order] = starts
    group_end[order] = ends

    # Hands grouped by the cards they use, each group ordered by strength.
    position = np.zeros(NUM_COMBOS, dtype=np.int64)
    position[order] = np.arange(len(order))
    groups, positions = [], []
    card_offset = np.zeros(NUM_CARDS + 1, dtype=np.int64)
    for card in range(NUM_CARDS):
        holders = np.flatnonzero(CARD_IN_COMBO[card] & (ranks != ILLEGAL))
        holders = holders[np.argsort(position[holders], kind="stable")]
        groups.append(holders)
        positions.append(position[holders])
        card_offset[card + 1] = card_offset[card] + len(holders)
    flat_hands = (
        np.concatenate(groups) if groups else np.zeros(0, dtype=np.int64)
    )

    # Where each hand's tie-group boundaries fall inside its cards' groups.
    card_a, card_b = COMBO_CARDS[:, 0], COMBO_CARDS[:, 1]
    lo_a, lo_b = np.zeros(NUM_COMBOS, np.int64), np.zeros(NUM_COMBOS, np.int64)
    hi_a, hi_b = np.zeros(NUM_COMBOS, np.int64), np.zeros(NUM_COMBOS, np.int64)
    for combo in np.flatnonzero(ranks != ILLEGAL):
        a, b = card_a[combo], card_b[combo]
        first, last = group_start[combo], group_end[combo]
        lo_a[combo] = card_offset[a] + np.searchsorted(positions[a], first)
        hi_a[combo] = card_offset[a] + np.searchsorted(positions[a], last)
        lo_b[combo] = card_offset[b] + np.searchsorted(positions[b], first)
        hi_b[combo] = card_offset[b] + np.searchsorted(positions[b], last)

    return ShowdownIndex(
        order=order,
        group_start=group_start,
        group_end=group_end,
        legal=ranks != ILLEGAL,
        flat_hands=flat_hands,
        card_offset=card_offset,
        lo_a=lo_a,
        lo_b=lo_b,
        hi_a=hi_a,
        hi_b=hi_b,
    )


def showdown_values(board: Tuple[int, ...], reach: np.ndarray, stake: float) -> np.ndarray:
    """``(1326,)`` counterfactual values against an opponent holding ``reach``.

    Positive where the hand wins more opponent mass than it loses to.  Ties
    contribute nothing, which is also how blocked hands contribute nothing: both
    sit inside the tie group that the prefix sums skip over.
    """
    index = showdown_index(board)

    # Mass strictly weaker than each position, and strictly stronger.
    prefix = np.concatenate(([0.0], np.cumsum(reach[index.order])))
    total = prefix[-1]

    # The same restricted to each card, over the compacted per-card groups.
    flat = np.concatenate(([0.0], np.cumsum(reach[index.flat_hands])))
    base = flat[index.card_offset[:-1]]
    card_a, card_b = COMBO_CARDS[:, 0], COMBO_CARDS[:, 1]
    base_a, base_b = base[card_a], base[card_b]
    card_total = flat[index.card_offset[1:]] - base

    first, last = index.group_start, index.group_end
    weaker = (
        prefix[first] - (flat[index.lo_a] - base_a) - (flat[index.lo_b] - base_b)
    )
    stronger = (total - prefix[last]) - (
        (card_total[card_a] - (flat[index.hi_a] - base_a))
        + (card_total[card_b] - (flat[index.hi_b] - base_b))
    )
    return stake * (weaker - stronger) * index.legal


def showdown_values_brute_force(
    board: Tuple[int, ...], reach: np.ndarray, stake: float
) -> np.ndarray:
    """The definition, written out.  Used only to test the fast version."""
    ranks = hand_ranks(board)
    values = np.zeros(NUM_COMBOS)
    for mine in range(NUM_COMBOS):
        if ranks[mine] == ILLEGAL:
            continue
        my_cards = set(COMBO_CARDS[mine])
        total = 0.0
        for theirs in range(NUM_COMBOS):
            if ranks[theirs] == ILLEGAL or reach[theirs] == 0.0:
                continue
            if my_cards & set(COMBO_CARDS[theirs]):
                continue
            if ranks[mine] > ranks[theirs]:
                total += reach[theirs]
            elif ranks[mine] < ranks[theirs]:
                total -= reach[theirs]
        values[mine] = stake * total
    return values
