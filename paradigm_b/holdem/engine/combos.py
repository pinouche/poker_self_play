"""Two-card hand combinations and the blocking arithmetic they force.

Leduc had six possible hands and card removal was "not the board card, not the
opponent's card".  Hold'em has 1,326 two-card combinations and every one of them
blocks 51 others: any hand sharing either card with mine is one the opponent
cannot hold.  That turns card removal from a footnote into the dominant cost of
every value computation, and it is where a range-based solver is easiest to get
subtly wrong.

Everything here is precomputed once and shared: the combo table, the card
membership masks, and the per-card index lists the showdown routine sums over.
Ranges are always full 1,326-vectors with impossible combos masked to zero,
rather than re-indexed per board — the 17% waste buys a fixed layout that the
network, the solver and the tests can all share.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple

import numpy as np

NUM_CARDS = 52
NUM_COMBOS = NUM_CARDS * (NUM_CARDS - 1) // 2  # 1326

# combo index -> its two card ids, and the reverse lookup.
COMBO_CARDS = np.array(
    [(a, b) for a in range(NUM_CARDS) for b in range(a + 1, NUM_CARDS)], dtype=np.int64
)
INDEX_OF_PAIR = np.full((NUM_CARDS, NUM_CARDS), -1, dtype=np.int64)
for _i, (_a, _b) in enumerate(COMBO_CARDS):
    INDEX_OF_PAIR[_a, _b] = _i
    INDEX_OF_PAIR[_b, _a] = _i

# CARD_IN_COMBO[c, i] is True when combo i uses card c.
CARD_IN_COMBO = np.zeros((NUM_CARDS, NUM_COMBOS), dtype=bool)
CARD_IN_COMBO[COMBO_CARDS[:, 0], np.arange(NUM_COMBOS)] = True
CARD_IN_COMBO[COMBO_CARDS[:, 1], np.arange(NUM_COMBOS)] = True


def combo_index(card_a: int, card_b: int) -> int:
    return int(INDEX_OF_PAIR[card_a, card_b])


@lru_cache(maxsize=None)
def board_mask(board: Tuple[int, ...]) -> np.ndarray:
    """1.0 for combos that do not use a board card, 0.0 for the rest."""
    mask = np.ones(NUM_COMBOS)
    for card in board:
        mask[CARD_IN_COMBO[card]] = 0.0
    return mask


@lru_cache(maxsize=None)
def num_legal_combos(num_board_cards: int) -> int:
    free = NUM_CARDS - num_board_cards
    return free * (free - 1) // 2


@lru_cache(maxsize=None)
def pair_correction(num_board_cards: int) -> float:
    """Rescaling that turns a product of ranges into the true joint deal.

    The two players hold disjoint hands, so the legal pairs are fewer than the
    product of the marginals suggests.  Every hand blocks exactly the same
    number of opponent hands, which makes the correction a single constant:
    (hands available) / (hands available to an opponent who is not me).
    """
    free = NUM_CARDS - num_board_cards
    compatible = (free - 2) * (free - 3) // 2
    return num_legal_combos(num_board_cards) / compatible


def card_masses(reach: np.ndarray) -> np.ndarray:
    """``(52,)`` total range mass sitting on each individual card.

    Two scatter-adds rather than a ``(52, 1326)`` matrix product: a hand touches
    exactly two cards, so the dense form does twenty-six times the arithmetic to
    add up the same numbers.
    """
    return np.bincount(
        COMBO_CARDS[:, 0], weights=reach, minlength=NUM_CARDS
    ) + np.bincount(COMBO_CARDS[:, 1], weights=reach, minlength=NUM_CARDS)


def compatible_mass(reach: np.ndarray) -> np.ndarray:
    """For each combo, the opponent mass that does *not* block it.

    Inclusion-exclusion: everything, minus the hands using my first card, minus
    those using my second, plus my own hand back (it was subtracted twice).
    """
    masses = card_masses(reach)
    total = reach.sum()
    return total - masses[COMBO_CARDS[:, 0]] - masses[COMBO_CARDS[:, 1]] + reach


def compatible_mass_batch(reach: np.ndarray) -> np.ndarray:
    """:func:`compatible_mass` for a ``(K, 1326)`` stack of ranges."""
    masses = reach @ CARD_IN_COMBO.T  # (K, 52)
    total = reach.sum(axis=1, keepdims=True)
    return total - masses[:, COMBO_CARDS[:, 0]] - masses[:, COMBO_CARDS[:, 1]] + reach


def cards_to_str(cards: Sequence[int]) -> str:
    ranks, suits = "23456789TJQKA", "cdhs"
    return " ".join(f"{ranks[c // 4]}{suits[c % 4]}" for c in cards)
