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

**The gathers are compiled.**  This is the hottest leaf in the solver — roughly
25,000 calls per training label — and in numpy it is ten fancy-index gathers and
two cumulative sums over 1,326 elements, each allocating a temporary.  The
arithmetic is trivial; the cost is almost entirely numpy's per-operation
overhead and the memory traffic of those temporaries.  Written as one explicit
loop and compiled with numba it is **3.0x faster** (24.4us -> 8.1us), because
the loop fuses every gather into a single pass with no intermediates.

Numba is optional.  If it is not installed the numpy implementation is used
instead and everything still works, just slower; ``showdown_values_numpy`` is
kept as both the fallback and the thing the compiled kernel is tested against.
Note the trade: the first call in each process pays JIT compilation, so
``cache=True`` is essential here — eight actor processes must load the compiled
kernel from disk rather than each rebuilding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Tuple

import numpy as np

from paradigm_b.holdem.engine.combos import CARD_IN_COMBO, COMBO_CARDS, NUM_CARDS, NUM_COMBOS
from paradigm_b.holdem.engine.strength import ILLEGAL, hand_ranks, sorted_by_strength

# Contiguous copies: ``COMBO_CARDS[:, 0]`` is a strided view, and handing a
# strided array to a compiled kernel costs more than the copy ever will.
CARD_A = np.ascontiguousarray(COMBO_CARDS[:, 0])
CARD_B = np.ascontiguousarray(COMBO_CARDS[:, 1])

try:  # pragma: no cover - depends on the environment, both paths are tested
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


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
    # Scratch for the compiled kernel's two prefix sums, sized by this board and
    # reused across its calls so a 25,000-call label does not allocate 25,000
    # pairs of temporaries.  Safe to share because the index is cached per board
    # and the solver is single-threaded within a process -- the parallelism here
    # is actor *processes* (see arm2_iterative/actors.py), which do not share
    # these objects.  Threading the solver would require making these per-call.
    prefix: np.ndarray = field(default=None, repr=False, compare=False)
    flat: np.ndarray = field(default=None, repr=False, compare=False)


@lru_cache(maxsize=4096)
def showdown_index(board: Tuple[int, ...]) -> ShowdownIndex:
    """Build the scaffolding for one board.

    Written with the two obvious Python loops — 52 cards, then one pass over
    the ~1,081 legal combos doing four :func:`numpy.searchsorted` calls each —
    this cost 4.6ms per board, and a turn label needs ~96 of them.  Both loops
    are gone.

    The per-card loop becomes one ``argsort`` of the ``(card, hand)`` pairs.
    The per-combo loop becomes four whole-array searches: the group boundaries
    are only ever looked up *within* a card's block of ``flat_hands``, and the
    blocks are laid out in card order with positions ascending inside each, so
    ``card * (NUM_COMBOS + 1) + position`` is monotone across the entire array.
    A search for that key therefore lands inside the right card's block and
    nowhere else, which makes 4,324 group-local searches one global one.
    """
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
    legal = ranks != ILLEGAL
    # Every legal hand contributes exactly two (card, hand) entries, so the
    # table can be written down directly.  Reading them off
    # ``nonzero(CARD_IN_COMBO[:, legal])`` instead materialised a (52, ~1081)
    # boolean and scanned all 56,000 cells to find the ~2,160 that are set —
    # twenty-six times the work, on the second-hottest board-table path.
    #
    # The pairing is what has to survive: ``repeat`` gives each hand twice,
    # ``COMBO_CARDS[...].reshape(-1)`` gives its two cards in the same order.
    # The subsequent sort makes the order this arrives in irrelevant anyway —
    # ``card * scale + position`` is unique per entry, because a position
    # identifies one hand — so the result is the same array either way.
    legal_hands = np.flatnonzero(legal)
    hands = np.repeat(legal_hands, 2)
    card_of = COMBO_CARDS[legal_hands].reshape(-1)
    scale = np.int64(NUM_COMBOS + 1)
    keys = card_of.astype(np.int64) * scale + position[hands]
    ordering = np.argsort(keys, kind="stable")
    flat_hands = hands[ordering]
    flat_keys = keys[ordering]
    card_offset = np.zeros(NUM_CARDS + 1, dtype=np.int64)
    np.cumsum(np.bincount(card_of, minlength=NUM_CARDS), out=card_offset[1:])

    # Where each hand's tie-group boundaries fall inside its cards' groups.
    card_a, card_b = COMBO_CARDS[:, 0], COMBO_CARDS[:, 1]
    lo_a = np.searchsorted(flat_keys, card_a * scale + group_start)
    hi_a = np.searchsorted(flat_keys, card_a * scale + group_end)
    lo_b = np.searchsorted(flat_keys, card_b * scale + group_start)
    hi_b = np.searchsorted(flat_keys, card_b * scale + group_end)
    # Illegal combos are never read, but the looped version left them at zero
    # and the tests compare the two field for field.
    for offsets in (lo_a, hi_a, lo_b, hi_b):
        offsets[~legal] = 0

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
        prefix=np.zeros(len(order) + 1),
        flat=np.zeros(len(flat_hands) + 1),
    )


def _showdown_loop(
    reach, stake, order, flat_hands, card_offset, group_start, group_end,
    lo_a, hi_a, lo_b, hi_b, legal, card_a, card_b, prefix, flat, out,
):
    """One pass over the combos; the compiled body of :func:`showdown_values`.

    Deliberately written as scalar loops rather than array expressions: that is
    what lets the two prefix sums and the ten per-combo lookups happen without
    materialising a single temporary.  Kept importable unjitted so the tests can
    exercise it directly, but it is only ever *called* through the jitted
    version — in pure Python this loop is far slower than the numpy form.
    """
    count = order.shape[0]
    prefix[0] = 0.0
    for i in range(count):
        prefix[i + 1] = prefix[i] + reach[order[i]]
    total = prefix[count]

    flat_count = flat_hands.shape[0]
    flat[0] = 0.0
    for i in range(flat_count):
        flat[i + 1] = flat[i] + reach[flat_hands[i]]

    for combo in range(out.shape[0]):
        if not legal[combo]:
            out[combo] = 0.0
            continue
        a = card_a[combo]
        b = card_b[combo]
        base_a = flat[card_offset[a]]
        base_b = flat[card_offset[b]]
        total_a = flat[card_offset[a + 1]] - base_a
        total_b = flat[card_offset[b + 1]] - base_b
        weaker = (
            prefix[group_start[combo]]
            - (flat[lo_a[combo]] - base_a)
            - (flat[lo_b[combo]] - base_b)
        )
        stronger = (total - prefix[group_end[combo]]) - (
            (total_a - (flat[hi_a[combo]] - base_a))
            + (total_b - (flat[hi_b[combo]] - base_b))
        )
        out[combo] = stake * (weaker - stronger)


_showdown_compiled = (
    njit(cache=True, fastmath=True)(_showdown_loop) if HAVE_NUMBA else None
)


def showdown_values(
    board: Tuple[int, ...],
    reach: np.ndarray,
    stake: float,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """``(1326,)`` counterfactual values against an opponent holding ``reach``.

    Positive where the hand wins more opponent mass than it loses to.  Ties
    contribute nothing, which is also how blocked hands contribute nothing: both
    sit inside the tie group that the prefix sums skip over.

    Dispatches to the compiled kernel when it is available and ``reach`` is
    already the contiguous float64 the solver produces.  Anything else — no
    numba, a float32 range from the batched generator, a strided view — takes
    the numpy path rather than paying a copy to enter the fast one.

    ``out`` writes the result into a caller-owned row instead of a fresh array.
    The kernel always wrote into a buffer it had just allocated, and a terminal
    node has somewhere to put the answer already — its own ``(2, 1326)`` value
    array — so at ~40,000 terminal evaluations per label that is 40,000
    allocations spent on nothing.  Must be contiguous float64; a row of a
    C-contiguous ``(2, 1326)`` array is.
    """
    if _showdown_compiled is None or reach.dtype != np.float64 or not reach.flags.c_contiguous:
        values = showdown_values_numpy(board, reach, stake)
        if out is None:
            return values
        out[:] = values
        return out
    index = showdown_index(board)
    if out is None:
        out = np.empty(NUM_COMBOS)
    _showdown_compiled(
        reach, float(stake), index.order, index.flat_hands, index.card_offset,
        index.group_start, index.group_end, index.lo_a, index.hi_a, index.lo_b,
        index.hi_b, index.legal, CARD_A, CARD_B, index.prefix, index.flat, out,
    )
    return out


def showdown_values_numpy(
    board: Tuple[int, ...], reach: np.ndarray, stake: float
) -> np.ndarray:
    """The array implementation: fallback when numba is absent, and the reference.

    Ten fancy-index gathers and two cumulative sums, each allocating.  Retained
    because it is what the compiled kernel is checked against, and because a
    working install without numba is a supported configuration.
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


def showdown_values_batch(
    board: Tuple[int, ...], reach: np.ndarray, stake: float
) -> np.ndarray:
    """:func:`showdown_values` for a batch of ranges sharing one board.

    ``reach`` is ``(K, 1326)``.  The sorting and the per-card grouping depend
    only on the board, so they are computed once and every situation in the
    batch rides on the same index — the whole point of batching by board.
    """
    index = showdown_index(board)
    weights = reach[:, index.order]
    prefix = np.concatenate(
        (np.zeros((len(reach), 1)), np.cumsum(weights, axis=1)), axis=1
    )
    total = prefix[:, -1:]

    flat = np.concatenate(
        (
            np.zeros((len(reach), 1)),
            np.cumsum(reach[:, index.flat_hands], axis=1),
        ),
        axis=1,
    )
    base = flat[:, index.card_offset[:-1]]
    card_a, card_b = COMBO_CARDS[:, 0], COMBO_CARDS[:, 1]
    base_a, base_b = base[:, card_a], base[:, card_b]
    card_total = flat[:, index.card_offset[1:]] - base

    first, last = index.group_start, index.group_end
    weaker = (
        prefix[:, first] - (flat[:, index.lo_a] - base_a) - (flat[:, index.lo_b] - base_b)
    )
    stronger = (total - prefix[:, last]) - (
        (card_total[:, card_a] - (flat[:, index.hi_a] - base_a))
        + (card_total[:, card_b] - (flat[:, index.hi_b] - base_b))
    )
    return stake * (weaker - stronger) * index.legal
