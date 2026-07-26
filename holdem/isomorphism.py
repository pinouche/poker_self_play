"""Suit isomorphism: boards that are the same board with the suits renamed.

A flop of A♠K♠7♥ plays identically to A♥K♥7♦ — the suits are labels, and only
the pattern of which cards share a suit matters.  Treating them as distinct
means solving, caching and learning the same position several times over.

Collapsing them is not an optimisation you can skip at scale.  There are 22,100
flops but only **1,755** genuinely distinct ones, so recognising the symmetry
removes more than nine tenths of the work — and the same argument applies to the
network, which otherwise has to learn separately that each of twelve relabelings
of one flop is worth the same.

A board is canonicalised by relabelling its suits in a fixed order: suits that
appear more often on the board come first, ties broken by the ranks they hold.
The permutation used is returned as well, because a range over combinations has
to be permuted the same way to stay aligned with the board.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Sequence, Tuple

import numpy as np

from holdem.combos import COMBO_CARDS, INDEX_OF_PAIR, NUM_CARDS, NUM_COMBOS

NUM_SUITS = 4


def suit_of(card: int) -> int:
    return card % NUM_SUITS


def rank_of(card: int) -> int:
    return card // NUM_SUITS


@lru_cache(maxsize=None)
def canonical_board(board: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """``(canonical board, suit relabelling)`` for ``board``.

    ``relabelling[s]`` is the canonical name of original suit ``s``.
    """
    by_suit: Dict[int, list] = {s: [] for s in range(NUM_SUITS)}
    for card in board:
        by_suit[suit_of(card)].append(rank_of(card))

    order = sorted(
        range(NUM_SUITS),
        key=lambda s: (-len(by_suit[s]), [-r for r in sorted(by_suit[s], reverse=True)]),
    )
    relabelling = [0] * NUM_SUITS
    for new_label, original in enumerate(order):
        relabelling[original] = new_label

    canonical = tuple(
        sorted(rank_of(c) * NUM_SUITS + relabelling[suit_of(c)] for c in board)
    )
    return canonical, tuple(relabelling)


@lru_cache(maxsize=None)
def card_permutation(relabelling: Tuple[int, ...]) -> np.ndarray:
    """``(52,)`` map from original card id to relabelled card id."""
    out = np.empty(NUM_CARDS, dtype=np.int64)
    for card in range(NUM_CARDS):
        out[card] = rank_of(card) * NUM_SUITS + relabelling[suit_of(card)]
    return out


@lru_cache(maxsize=None)
def combo_permutation(relabelling: Tuple[int, ...]) -> np.ndarray:
    """``(1326,)`` map from original combination index to relabelled index."""
    cards = card_permutation(relabelling)
    out = np.empty(NUM_COMBOS, dtype=np.int64)
    for combo in range(NUM_COMBOS):
        a, b = COMBO_CARDS[combo]
        out[combo] = INDEX_OF_PAIR[cards[a], cards[b]]
    return out


def canonicalise_range(weights: np.ndarray, relabelling: Tuple[int, ...]) -> np.ndarray:
    """Re-index a range so it lines up with the canonical board."""
    permutation = combo_permutation(relabelling)
    out = np.zeros_like(weights)
    out[permutation] = weights
    return out


def count_canonical(num_cards: int, limit: int | None = None) -> int:
    """How many distinct boards of this size remain after collapsing suits."""
    from itertools import combinations

    seen = set()
    for board in combinations(range(NUM_CARDS), num_cards):
        seen.add(canonical_board(board)[0])
        if limit is not None and len(seen) >= limit:
            break
    return len(seen)
