"""Hand strength for every combination on a given board.

Ranks are dense integers: equal strength means an equal integer, and a higher
integer beats a lower one.  Collapsing the evaluator's tuples to dense ranks is
what lets the showdown routine be a prefix sum instead of 1,326^2 comparisons.

Evaluating all 1,326 combinations on a board costs about 4ms, and a turn board
has 44 rivers, so the results are cached per board.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

from environment.cards import Card
from environment.hand_evaluator import best_hand_of
from holdem.combos import COMBO_CARDS, NUM_COMBOS, board_mask

ILLEGAL = -1


@lru_cache(maxsize=4096)
def hand_ranks(board: Tuple[int, ...]) -> np.ndarray:
    """``(1326,)`` dense strength ranks; ``ILLEGAL`` where the board blocks."""
    board_cards = [Card.from_id(c) for c in board]
    mask = board_mask(board)
    scores = {}
    for combo in range(NUM_COMBOS):
        if mask[combo] == 0.0:
            continue
        hole = [Card.from_id(int(c)) for c in COMBO_CARDS[combo]]
        scores[combo] = best_hand_of(hole, board_cards)

    ranks = np.full(NUM_COMBOS, ILLEGAL, dtype=np.int64)
    order = {score: dense for dense, score in enumerate(sorted(set(scores.values())))}
    for combo, score in scores.items():
        ranks[combo] = order[score]
    return ranks


@lru_cache(maxsize=4096)
def sorted_by_strength(board: Tuple[int, ...]) -> np.ndarray:
    """Legal combos, weakest first."""
    ranks = hand_ranks(board)
    legal = np.flatnonzero(ranks != ILLEGAL)
    return legal[np.argsort(ranks[legal], kind="stable")]
