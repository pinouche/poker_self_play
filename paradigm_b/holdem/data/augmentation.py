"""Free labels from suit and chip symmetries the game already has.

A solved belief state stays solved when the suits are renamed.  A♠K♠7♥ with a
range over 1,326 combinations is the same position as A♥K♥7♦ with that range
relabelled the same way, and its counterfactual values are the same numbers in
permuted order.  So one CFR solve can be turned into up to **24** distinct
training examples — every permutation of the four suits — for the price of two
gathers, no search and no estimation error.

Chip scaling is the second exact symmetry.  Multiplying absolute pot, stack,
and counterfactual values by one factor leaves ranges, cards, stack-to-pot
ratios, and normalised betting history unchanged.  Applying it to encoded PBS
data also supports the paper's continuous training stacks without rebuilding
an integer-chip game tree.

This is what makes a value network's job easier.  Without it
the network has to learn separately that each relabelling of a board is worth
the same thing; with it, that fact is handed to it as data.

**Both isomorphisms are applied when a row is read, not when it is written.**
:func:`transform_batch` is the live path, and the batched form of
:func:`relabel_example` and :func:`scale_example`, which stay as the per-example
statement of each symmetry.

The paper stores ``K`` transformed copies of each solve and periodically
rewrites a slice of the buffer to renew them, which costs ``K`` times the bytes
per solve and a buffer-sized write on every renewal pass.  Drawing the
transformation at sample time instead gives one canonical row per solve, an
unbounded number of orbit points rather than ``K`` frozen ones, and no
write-back at all — which is what makes a replay buffer larger than memory
affordable.  See :func:`transform_batch` for the details that change once rows
are canonical.

Two things this deliberately does *not* do:

**It does not charge the label budget.**  ``label_budget`` exists to bound
*search*, and a relabelled row runs none.  Charging for it would make a run
with augmentation on look identical in spend to one without, while doing a
fraction of the solving — which is exactly the comparison the budgets exist to
make honest.  Transforming on read makes this automatic rather than a rule to
keep: nothing is added to anything, so there is nothing to charge.

**It does not touch the journal.**  The journal records what search concluded,
and :mod:`~paradigm_b.holdem.harness.accuracy` scores predictions against it.  A
transformed row is derived from those conclusions rather than additional
evidence about them, so counting one there would inflate the record and let a
run grade itself on its own permutations.  The journal sees the canonical
labels, which is what search actually produced.

**It draws from a generator of its own.**  The buffer holds the stream these
are drawn from rather than using the caller's.  In the synchronous loop one
generator feeds both generation and training, so a transform drawn from it
would advance the stream that decides which situation comes next — and the
``suit_augmentations=0`` ablation would silently stop being a comparison.

"""

from __future__ import annotations

from itertools import permutations
from typing import Tuple

import numpy as np

from paradigm_b.holdem.engine.combos import CARD_IN_COMBO, NUM_COMBOS
from paradigm_b.holdem.engine.isomorphism import (
    NUM_SUITS,
    card_permutation,
    combo_permutation,
)
from paradigm_b.holdem.net.features import (
    BOARD_DIM,
    BOARD_OFFSET,
    CARD_SET_DIM,
    NUM_CARD_SETS,
    POT_CHIPS_INDEX,
    RANGE_DIM,
    STACK_CHIPS_INDEX,
)

NUM_PLAYERS = 2

IDENTITY: Tuple[int, ...] = tuple(range(NUM_SUITS))
# Every relabelling of the four suits is an exact symmetry of hold'em — there is
# no suit ranking for one to break — so all 24 are usable.
#
# The identity is among them, and is drawn like any other.  It is *not* wasted:
# the paper stores K transformations of each PBS and does not additionally store
# the untransformed original, so the sampled orientation has no privileged
# status and reaches the buffer only when the identity is drawn — and even then
# it arrives under its own chip scale.  ``RELABELLINGS`` remains the
# identity-free tuple for callers that want a guaranteed *change*.
ALL_RELABELLINGS: Tuple[Tuple[int, ...], ...] = tuple(permutations(range(NUM_SUITS)))
RELABELLINGS: Tuple[Tuple[int, ...], ...] = tuple(
    p for p in ALL_RELABELLINGS if p != IDENTITY
)
MAX_AUGMENTATIONS = len(ALL_RELABELLINGS)  # 24
MIN_CHIP_SCALE = 0.75
MAX_CHIP_SCALE = 1.25


def relabel_features(features: np.ndarray, relabelling: Tuple[int, ...]) -> np.ndarray:
    """``features`` with both ranges and the board sets moved to new suits.

    Everything after the board — the pot and stack scalars, whose street it is,
    the betting history — is untouched, because renaming a suit cannot change
    any of it.  That is the whole reason this is free.
    """
    combos = combo_permutation(relabelling)
    cards = card_permutation(relabelling)

    out = np.array(features, copy=True)

    ranges = np.asarray(features[:RANGE_DIM]).reshape(NUM_PLAYERS, NUM_COMBOS)
    moved = np.zeros_like(ranges)
    moved[:, combos] = ranges
    out[:RANGE_DIM] = moved.reshape(-1)

    board = np.asarray(
        features[BOARD_OFFSET : BOARD_OFFSET + BOARD_DIM]
    ).reshape(NUM_CARD_SETS, CARD_SET_DIM)
    moved_board = np.zeros_like(board)
    moved_board[:, cards] = board
    out[BOARD_OFFSET : BOARD_OFFSET + BOARD_DIM] = moved_board.reshape(-1)

    return out


def relabel_example(example, relabelling: Tuple[int, ...]):
    """One :class:`~paradigm_b.holdem.selfplay.Example` under a suit relabelling.

    The D_pi targets are dropped rather than permuted.  A policy example carries
    a ``(1326, MAX_ACTIONS)`` target whose rows would move the same way, but the
    policy network is off by default and warm-starting it from augmented copies
    is a change to theta_pi's data distribution that has not been measured here.
    Value data is where the symmetry pays.
    """
    from paradigm_b.holdem.selfplay import Example  # circular at module import time

    combos = combo_permutation(relabelling)
    cards = card_permutation(relabelling)

    mask = np.zeros_like(example.mask)
    mask[combos] = example.mask

    values = np.zeros_like(example.values)
    values[:, combos] = example.values

    return Example(
        features=relabel_features(example.features, relabelling),
        mask=mask,
        values=values,
        policies=(),
        board=tuple(sorted(int(cards[c]) for c in example.board)),
    )


def scale_example(example, scale: float):
    """One example under the continuous chip isomorphism.

    No accumulated factor is carried any more.  Stored rows are canonical, so
    every scale is applied to an unscaled original and is exactly one draw from
    ``[0.75, 1.25]``.  The in-place renewal this replaces had to divide the
    previous factor back out first, because composing two draws from that
    interval is a product, not a draw — and a product with ``E[log s] < 0``,
    which walked the pot down over repeated renewals if it was got wrong.
    """
    from paradigm_b.holdem.selfplay import Example  # circular at module import time

    if scale <= 0.0:
        raise ValueError("chip scale must be positive")
    features = np.array(example.features, copy=True)
    features[[POT_CHIPS_INDEX, STACK_CHIPS_INDEX]] *= scale
    return Example(
        features=features,
        mask=np.array(example.mask, copy=True),
        values=np.asarray(example.values) * scale,
        policies=(),
        board=example.board,
    )


def board_masks(boards: np.ndarray) -> np.ndarray:
    """``(n, 1326)`` masks for a batch of boards, without touching a cache.

    ``combos.board_mask`` is ``lru_cache(maxsize=None)``, which is right for
    generation — a trajectory asks about the same handful of boards over and
    over — and wrong here.  Sampling asks about 1,024 mostly-distinct boards
    per batch, forty times an iteration, for the length of a run; an unbounded
    cache of 1,326-float arrays keyed on every board a multi-million-row buffer
    has ever held is a slow leak with no hit rate to show for it.

    So the mask is rebuilt from ``CARD_IN_COMBO`` instead: five gathers of an
    ``(n, 1326)`` boolean, which is cheaper than the dictionary lookup it
    replaces once the boards stop repeating.  ``-1`` marks a slot the board has
    not dealt yet, which is what makes one fixed-width array cover the flop,
    the turn and the river alike.
    """
    boards = np.asarray(boards)
    blocked = np.zeros((len(boards), NUM_COMBOS), dtype=bool)
    for slot in range(boards.shape[1]):
        cards = boards[:, slot]
        rows = np.nonzero(cards >= 0)[0]
        if rows.size:
            blocked[rows] |= CARD_IN_COMBO[cards[rows]]
    return (~blocked).astype(np.float32)


def transform_batch(
    features: np.ndarray,
    targets: np.ndarray,
    boards: np.ndarray,
    rng: np.random.Generator,
    *,
    relabel: bool = True,
    rescale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialise stored canonical rows as one float32 training batch.

    This is where both isomorphisms are applied now: not once at write time and
    then frozen into the buffer, but freshly on every draw.  It subsumes the
    paper's two augmentation clauses at once, and is why neither ``K`` copies
    nor a renewal pass exists any more.

    * ``K`` transformations per stored PBS was a *storage* multiplier — K rows
      for one solve, each pinned to one point of the orbit for the rest of the
      run.  Drawing the transformation here instead gives one row per solve and
      an unbounded number of orbit points, so the same bytes hold K times the
      distinct belief states.
    * *"periodically reapply transformations to samples already in the replay
      buffer"* was an in-place rewrite of a slice of the buffer.  A row that is
      re-transformed every time it is read is renewed continuously and for
      free, and — the reason this matters on disk — nothing is written back.

    Stored rows are **canonical**: the untransformed encoding at chip scale
    1.0.  That is what removes the ratio bookkeeping the in-place version
    needed.  Composing two draws from ``[0.75, 1.25]`` is not a draw from that
    interval but a product, so renewal had to track each row's accumulated
    factor and divide it back out; a canonical row has nothing to divide out
    and each draw is exactly one uniform sample.

    ``masks`` are not stored either — they are a pure function of the board,
    which is 5 bytes against the 5,304 the mask itself would cost.  The mask is
    rebuilt canonically and then moved by the same combo permutation as
    everything else, which is equivalent to (and cheaper than) permuting the
    board cards and rebuilding it.

    Returns the ``(features, masks, targets)`` triple the gradient loop wants,
    in float32, freshly allocated and safe for the caller to keep.
    """
    features = np.asarray(features, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    boards = np.asarray(boards)
    rows = len(features)
    masks = board_masks(boards)
    if rows == 0:
        return features, masks, targets

    if relabel:
        # One permutation per row, applied in at most 24 vectorised groups
        # rather than row by row.  This runs inside every gradient step now, so
        # the per-row form is not affordable the way it was during an idle
        # renewal pass.
        chosen = rng.integers(0, len(ALL_RELABELLINGS), size=rows)
        ranges = features[:, :RANGE_DIM].reshape(rows, NUM_PLAYERS, NUM_COMBOS)
        board_sets = features[:, BOARD_OFFSET : BOARD_OFFSET + BOARD_DIM].reshape(
            rows, NUM_CARD_SETS, CARD_SET_DIM
        )
        moved_ranges = np.zeros_like(ranges)
        moved_board_sets = np.zeros_like(board_sets)
        moved_masks = np.zeros_like(masks)
        moved_targets = np.zeros_like(targets)
        for value in np.unique(chosen):
            group = np.nonzero(chosen == value)[0]
            combos = combo_permutation(ALL_RELABELLINGS[int(value)])
            cards = card_permutation(ALL_RELABELLINGS[int(value)])
            # Scatter into a zeroed copy of the group's slice, then write the
            # slice back: fancy indexing on the left of a plain slice is the
            # one form numpy will not fuse, and the two-step is what keeps this
            # a handful of contiguous copies.
            block = np.zeros_like(ranges[group])
            block[:, :, combos] = ranges[group]
            moved_ranges[group] = block

            block = np.zeros_like(board_sets[group])
            block[:, :, cards] = board_sets[group]
            moved_board_sets[group] = block

            block = np.zeros_like(masks[group])
            block[:, combos] = masks[group]
            moved_masks[group] = block

            block = np.zeros_like(targets[group])
            block[:, :, combos] = targets[group]
            moved_targets[group] = block
        features[:, :RANGE_DIM] = moved_ranges.reshape(rows, RANGE_DIM)
        features[:, BOARD_OFFSET : BOARD_OFFSET + BOARD_DIM] = moved_board_sets.reshape(
            rows, BOARD_DIM
        )
        masks = moved_masks
        targets = moved_targets

    if rescale:
        scales = rng.uniform(MIN_CHIP_SCALE, MAX_CHIP_SCALE, size=rows).astype(np.float32)
        features[:, [POT_CHIPS_INDEX, STACK_CHIPS_INDEX]] *= scales[:, None]
        targets *= scales[:, None, None]

    return features, masks, targets
