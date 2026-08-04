"""Hand strength for every combination on a given board.

Ranks are dense integers: equal strength means an equal integer, and a higher
integer beats a lower one.  Collapsing the evaluator's tuples to dense ranks is
what lets the showdown routine be a prefix sum instead of 1,326^2 comparisons.

**All 1,326 combos are evaluated at once, in array form.**  The obvious
implementation calls :func:`common.hand_evaluator.best_hand_of` in a Python
loop, and that is what this module used to do — 6.1ms per board, which sounds
negligible until you count the boards.  A depth-limit-1 *turn* tree contains
~96 five-card terminal nodes (the all-in lines, where the river is dealt
straight to showdown), every one of them a board this has never seen, so a
single turn label pays for ~96 of these before any CFR arithmetic happens at
all.  Measured over the default street mixture that was 27% of generation time
at the paper-sized network, and it was being counted as "CFR tree work".

So the evaluator here is vectorised rather than looped: one ``(n, 52)`` card
presence matrix, viewed as ``(n, 13, 4)`` because a card id is
``rank_index * 4 + suit``, and every category test becomes an array reduction
over it.  The result is a packed integer per combo whose ordering is identical
to the tuple ordering ``evaluate_hand`` produces — which is all the dense
ranking downstream actually needs.  ``hand_ranks_reference`` keeps the looped
version, and the tests check the two agree combo-for-combo.

The second thing that matters is that there are only thirteen ranks.  Written
with ``np.sort``, the kicker and multiplicity lookups cost ten sorts of an
``(n, 13)`` array and the vectorised evaluator came out only 4x ahead of the
loop.  A set of ranks is a 13-bit integer, so every one of those questions —
the five best ranks, the best excluding a given one, the straight it makes —
is a lookup in a table of all 8,192 possible sets, built once at import.  That
turns the sorts into gathers and the loop into a bitmask, and is worth another
order of magnitude.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

from common.cards import Card
from common.hand_evaluator import best_hand_of
from paradigm_b.holdem.engine.combos import COMBO_CARDS, NUM_CARDS, NUM_COMBOS, board_mask

ILLEGAL = -1

NUM_RANKS = 13
NUM_SUITS = 4
MIN_RANK = 2  # rank index 0 is a deuce, index 12 an ace

# Categories, in the order ``common.hand_evaluator`` numbers them.
HIGH_CARD, PAIR, TWO_PAIR, THREE_OF_A_KIND = 0, 1, 2, 3
STRAIGHT, FLUSH, FULL_HOUSE, FOUR_OF_A_KIND, STRAIGHT_FLUSH = 4, 5, 6, 7, 8


# --- tables over all 8,192 sets of ranks --------------------------------
# A set of ranks is a 13-bit integer, bit r meaning "rank index r is present".
_ALL_SETS = np.arange(1 << NUM_RANKS)
_SET_BITS = ((_ALL_SETS[:, None] >> np.arange(NUM_RANKS)) & 1).astype(bool)
RANK_BIT = (1 << np.arange(NUM_RANKS)).astype(np.int64)


def _build_top_table() -> np.ndarray:
    """``(8192, 5)`` the five highest *ranks* in each set, zero-padded."""
    indices = np.where(_SET_BITS, np.arange(NUM_RANKS), -1)
    ordered = np.sort(indices, axis=1)[:, ::-1][:, :5]
    return np.where(ordered >= 0, ordered + MIN_RANK, 0).astype(np.int64)


def _build_straight_table() -> np.ndarray:
    """``(8192,)`` best straight's high rank in each set, 0 where there is none."""
    high = np.zeros(len(_ALL_SETS), dtype=np.int64)
    bits = _SET_BITS
    # A-2-3-4-5: the ace plays low, so it is not a contiguous window.
    wheel = bits[:, 0] & bits[:, 1] & bits[:, 2] & bits[:, 3] & bits[:, 12]
    high = np.where(wheel, 5, high)
    for top in range(6, 15):  # ascending, so the highest straight overwrites
        index = top - MIN_RANK
        high = np.where(bits[:, index - 4 : index + 1].all(axis=1), top, high)
    return high


TOP_RANKS = _build_top_table()
STRAIGHT_HIGH = _build_straight_table()


def _bits_of(present: np.ndarray) -> np.ndarray:
    """``(n,)`` rank-set integers from an ``(n, 13)`` presence array."""
    return present.astype(np.int64) @ RANK_BIT


def _without(bits: np.ndarray, *ranks: np.ndarray) -> np.ndarray:
    """``bits`` with each given rank cleared.

    Rows where a rank is ``-1`` (no such rank) take a branch whose score is
    discarded, so clamping the shift rather than masking the row is enough.
    """
    for rank in ranks:
        bits = bits & ~(np.int64(1) << np.maximum(np.asarray(rank) - MIN_RANK, 0))
    return bits


def _nth_set_rank(bits: np.ndarray, nth: int = 0) -> np.ndarray:
    """The ``nth``-highest rank in each set, ``-1`` when the set is smaller."""
    chosen = TOP_RANKS[bits, nth]
    return np.where(chosen > 0, chosen, -1)


def _pack(category, *tiebreakers) -> np.ndarray:
    """Category and up to five tiebreakers into one order-preserving integer.

    Four bits each, most significant first, exactly as tuple comparison reads
    them.  Ranks run 2..14 so they fit, and shorter tuples pad with zeros —
    safe because tuples of differing length never share a category, and the
    category is compared first.
    """
    score = np.asarray(category, dtype=np.int64) << 20
    for shift, value in zip((16, 12, 8, 4, 0), tiebreakers):
        score = score | (np.asarray(value, dtype=np.int64) << shift)
    return score


def _scores(board: Tuple[int, ...], legal: np.ndarray) -> np.ndarray:
    """Packed comparable scores for the combos indexed by ``legal``."""
    count = len(legal)
    rows = np.arange(count)

    presence = np.zeros((count, NUM_CARDS), dtype=bool)
    presence[:, list(board)] = True
    presence[rows, COMBO_CARDS[legal, 0]] = True
    presence[rows, COMBO_CARDS[legal, 1]] = True
    # A card id is rank_index * 4 + suit, so this view is (row, rank, suit).
    grid = presence.reshape(count, NUM_RANKS, NUM_SUITS)

    rank_counts = grid.sum(axis=2)
    suit_counts = grid.sum(axis=1)

    # --- flush and straight flush ---------------------------------------
    flush_suit = suit_counts.argmax(axis=1)
    has_flush = suit_counts.max(axis=1) >= 5
    flush_bits = _bits_of(grid[rows, :, flush_suit]) * has_flush
    flush_ranks = TOP_RANKS[flush_bits]
    straight_flush_high = STRAIGHT_HIGH[flush_bits]
    has_straight_flush = straight_flush_high > 0

    # --- multiplicity groups --------------------------------------------
    quad = _nth_set_rank(_bits_of(rank_counts == 4))
    trips_bits = _bits_of(rank_counts == 3)
    trips, second_trips = _nth_set_rank(trips_bits, 0), _nth_set_rank(trips_bits, 1)
    pair_bits = _bits_of(rank_counts == 2)
    pair, second_pair = _nth_set_rank(pair_bits, 0), _nth_set_rank(pair_bits, 1)
    # A second set of trips can play as the full house's pair.
    house_pair = np.maximum(second_trips, pair)

    present_bits = _bits_of(rank_counts > 0)
    straight = STRAIGHT_HIGH[present_bits]

    # --- kickers ----------------------------------------------------------
    plain = TOP_RANKS[present_bits]
    without_quad = TOP_RANKS[_without(present_bits, quad)]
    without_trips = TOP_RANKS[_without(present_bits, trips)]
    without_pair = TOP_RANKS[_without(present_bits, pair)]
    without_both_pairs = TOP_RANKS[_without(present_bits, pair, second_pair)]

    # --- assemble, weakest first so stronger categories overwrite ---------
    score = _pack(HIGH_CARD, plain[:, 0], plain[:, 1], plain[:, 2], plain[:, 3], plain[:, 4])
    score = np.where(
        pair >= 0,
        _pack(PAIR, pair, without_pair[:, 0], without_pair[:, 1], without_pair[:, 2]),
        score,
    )
    score = np.where(
        second_pair >= 0,
        _pack(TWO_PAIR, pair, second_pair, without_both_pairs[:, 0]),
        score,
    )
    score = np.where(
        trips >= 0,
        _pack(THREE_OF_A_KIND, trips, without_trips[:, 0], without_trips[:, 1]),
        score,
    )
    score = np.where(straight > 0, _pack(STRAIGHT, straight), score)
    score = np.where(
        has_flush,
        _pack(FLUSH, *(flush_ranks[:, i] for i in range(5))),
        score,
    )
    score = np.where(
        (trips >= 0) & (house_pair >= 0), _pack(FULL_HOUSE, trips, house_pair), score
    )
    score = np.where(quad >= 0, _pack(FOUR_OF_A_KIND, quad, without_quad[:, 0]), score)
    return np.where(has_straight_flush, _pack(STRAIGHT_FLUSH, straight_flush_high), score)


@lru_cache(maxsize=4096)
def hand_ranks(board: Tuple[int, ...]) -> np.ndarray:
    """``(1326,)`` dense strength ranks; ``ILLEGAL`` where the board blocks."""
    legal = np.flatnonzero(board_mask(board))
    ranks = np.full(NUM_COMBOS, ILLEGAL, dtype=np.int64)
    if len(legal):
        scores = _scores(board, legal)
        # Dense ranking: identical scores must collapse to the same integer,
        # because that is what makes a tie a tie in the showdown prefix sums.
        ranks[legal] = np.unique(scores, return_inverse=True)[1]
    return ranks


def hand_ranks_reference(board: Tuple[int, ...]) -> np.ndarray:
    """The looped evaluator, kept as the thing :func:`hand_ranks` is tested against."""
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
