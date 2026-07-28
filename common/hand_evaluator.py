"""Best-five-of-seven hand evaluation.

``evaluate_hand`` returns a plain tuple whose natural Python ordering *is* the
poker ordering: bigger tuple beats smaller tuple, equal tuples split the pot.
The first element is the hand category, the rest are tiebreakers in descending
significance.

Correctness is prioritised over speed here; a straightforward counting
evaluator handles a few hundred thousand hands per second, which is far more
than self-play needs.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Sequence, Tuple

from .cards import Card

HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8

NUM_CATEGORIES = 9

HAND_CATEGORY_NAMES = [
    "HIGH_CARD",
    "PAIR",
    "TWO_PAIR",
    "THREE_OF_A_KIND",
    "STRAIGHT",
    "FLUSH",
    "FULL_HOUSE",
    "FOUR_OF_A_KIND",
    "STRAIGHT_FLUSH",
]

HandRank = Tuple[int, ...]


def _straight_high(ranks_desc_unique: Sequence[int]) -> int:
    """Highest card of the best straight, or 0 if there is none.

    ``ranks_desc_unique`` must be unique ranks in descending order.  The wheel
    (A-2-3-4-5) is handled by treating an ace as an additional low card.
    """
    ranks = list(ranks_desc_unique)
    if ranks and ranks[0] == 14:
        ranks.append(1)  # ace plays low
    run = 1
    for i in range(1, len(ranks)):
        if ranks[i] == ranks[i - 1] - 1:
            run += 1
            if run >= 5:
                # Descending scan, so the first hit is the highest straight.
                return ranks[i] + 4
        else:
            run = 1
    return 0


def evaluate_hand(cards: Sequence[Card]) -> HandRank:
    """Evaluate 5, 6 or 7 cards and return a comparable rank tuple."""
    if len(cards) < 5:
        raise ValueError(f"need at least 5 cards to evaluate, got {len(cards)}")

    ranks = [c.rank for c in cards]
    suits = [c.suit for c in cards]

    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    unique_desc = sorted(rank_counts, reverse=True)

    flush_suit = next((s for s, n in suit_counts.items() if n >= 5), None)

    # Straight flush -------------------------------------------------------
    if flush_suit is not None:
        flush_ranks = sorted({c.rank for c in cards if c.suit == flush_suit}, reverse=True)
        sf_high = _straight_high(flush_ranks)
        if sf_high:
            return (STRAIGHT_FLUSH, sf_high)

    # Group ranks by multiplicity, each group ordered high to low.
    by_count: dict = {}
    for rank, count in rank_counts.items():
        by_count.setdefault(count, []).append(rank)
    for group in by_count.values():
        group.sort(reverse=True)

    quads = by_count.get(4, [])
    trips = by_count.get(3, [])
    pairs = by_count.get(2, [])

    # Four of a kind -------------------------------------------------------
    if quads:
        quad = quads[0]
        kicker = max(r for r in unique_desc if r != quad)
        return (FOUR_OF_A_KIND, quad, kicker)

    # Full house -----------------------------------------------------------
    if trips:
        top_trips = trips[0]
        # A second set of trips can serve as the pair.
        pair_candidates = [r for r in trips[1:]] + pairs
        if pair_candidates:
            return (FULL_HOUSE, top_trips, max(pair_candidates))

    # Flush ----------------------------------------------------------------
    if flush_suit is not None:
        flush_ranks = sorted((c.rank for c in cards if c.suit == flush_suit), reverse=True)
        return (FLUSH, *flush_ranks[:5])

    # Straight -------------------------------------------------------------
    high = _straight_high(unique_desc)
    if high:
        return (STRAIGHT, high)

    # Three of a kind ------------------------------------------------------
    if trips:
        trip = trips[0]
        kickers = [r for r in unique_desc if r != trip][:2]
        return (THREE_OF_A_KIND, trip, *kickers)

    # Two pair -------------------------------------------------------------
    if len(pairs) >= 2:
        hi, lo = pairs[0], pairs[1]
        kicker = max(r for r in unique_desc if r not in (hi, lo))
        return (TWO_PAIR, hi, lo, kicker)

    # One pair -------------------------------------------------------------
    if len(pairs) == 1:
        pair = pairs[0]
        kickers = [r for r in unique_desc if r != pair][:3]
        return (PAIR, pair, *kickers)

    # High card ------------------------------------------------------------
    return (HIGH_CARD, *unique_desc[:5])


def hand_category(rank: HandRank) -> int:
    return rank[0]


def describe_hand(rank: HandRank) -> str:
    return HAND_CATEGORY_NAMES[rank[0]]


def best_hand_of(hole: Sequence[Card], board: Sequence[Card]) -> HandRank:
    """Evaluate a player's holding against the board."""
    return evaluate_hand(list(hole) + list(board))


def compare_hands(a: HandRank, b: HandRank) -> int:
    """-1 if ``a`` loses, 0 on a tie, +1 if ``a`` wins."""
    if a > b:
        return 1
    if a < b:
        return -1
    return 0


def rank_all(hands: Sequence[Sequence[Card]]) -> List[HandRank]:
    return [evaluate_hand(h) for h in hands]
