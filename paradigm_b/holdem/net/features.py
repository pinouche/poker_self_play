"""Encoding a hold'em public belief state for the value network.

The core modelling decision of stage 4, and the one that does not carry over
from Leduc.  There the input was six numbers per player; here it is 1,326, and
they are not exchangeable — combos have structure (rank, suit, whether they pair
the board) that a flat vector hides.

The ranges stay flat and unabstracted, while public cards follow Deep CFR's
structured representation.  Flop, turn, and river are separate permutation-
invariant sets; the network sums rank, suit, and card-identity embeddings inside
each set.  Betting history uses fixed sequential slots containing a presence
bit and the chips added by that action as a fraction of the stack.

Layout: two 1,326-entry ranges, three 52-card set indicators, eight public
scalars, then four rounds of six two-value betting-history slots.

Preflop needs no special case here: its board sets are simply all zero, which
is what distinguishes an undealt board from a dealt one, and its betting lands
in the first history slot rather than being right-aligned away.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Sequence

import numpy as np

from paradigm_b.holdem.engine.combos import NUM_CARDS, NUM_COMBOS

RANGE_DIM = 2 * NUM_COMBOS
BOARD_OFFSET = RANGE_DIM
NUM_CARD_SETS = 3
CARD_SET_DIM = NUM_CARDS
BOARD_DIM = NUM_CARD_SETS * CARD_SET_DIM
SCALAR_OFFSET = BOARD_OFFSET + BOARD_DIM
NUM_SCALARS = 8
# preflop, flop, turn, river.  A shorter game right-aligns into these slots (see
# ``round_offset`` below), so a turn endgame's two rounds land in the last two
# and a whole hand fills all four.
NUM_BETTING_ROUNDS = 4
MAX_ACTIONS_PER_ROUND = 6
BETTING_SLOT_DIM = 2
BETTING_HISTORY_DIM = NUM_BETTING_ROUNDS * MAX_ACTIONS_PER_ROUND * BETTING_SLOT_DIM
HISTORY_OFFSET = SCALAR_OFFSET + NUM_SCALARS
# Chips that correspond to a scaled value of 1.0.  Fixed and global so that the
# network's output scaling is exact whatever the pot and stack happen to be —
# with randomised situations they vary by an order of magnitude.
GLOBAL_CHIP_SCALE = 1000.0
INPUT_DIM = HISTORY_OFFSET + BETTING_HISTORY_DIM
PUBLIC_DIM = BOARD_DIM + NUM_SCALARS + BETTING_HISTORY_DIM


POT_CHIPS_INDEX = SCALAR_OFFSET + 5
# The two absolute-chip scalars, and the only inputs the chip isomorphism in
# ``data/augmentation.py`` touches: every other scalar is already a ratio and is
# therefore invariant under a uniform rescaling of the money.
STACK_CHIPS_INDEX = SCALAR_OFFSET + 6


@lru_cache(maxsize=8192)
def encode_public(public) -> np.ndarray:
    """Structured board sets, public scalars, and ordered betting history.

    Cached, and the cache is worth more than it looks.  A depth-limited solve
    re-encodes its entire leaf frontier on *every* CFR iteration — the ranges
    move, but the public states do not — so without this the betting history of
    each of ~360 leaves is replayed forty times per solve, through a chain of
    frozen-dataclass ``replace`` calls that cost more than everything else in
    the encoder put together.  It was 16% of generation time.

    ``PublicState`` is a frozen dataclass and the public tree already keys a
    dict on it, so it is hashable by construction.  The returned array is
    marked read-only because callers copy it into a larger buffer and a cached
    array must not be writable by them.  8,192 entries is ~13MB and comfortably
    covers a whole solve, which is where all the reuse is; sizing it to span
    many situations would only multiply memory across actor processes.
    """
    betting = public.betting
    out = np.zeros(PUBLIC_DIM)
    for set_index, cards in enumerate(
        (public.board[:3], public.board[3:4], public.board[4:5])
    ):
        offset = set_index * CARD_SET_DIM
        for card in cards:
            out[offset + card] = 1.0
    stack = float(betting.stack) or 1.0
    to_move = betting.to_move() if not betting.is_terminal else 0
    scalars = out[BOARD_DIM : BOARD_DIM + NUM_SCALARS]
    scalars[0] = betting.pot / (2.0 * stack + betting.starting_pot)
    scalars[1] = betting.contributions[0] / stack
    scalars[2] = betting.contributions[1] / stack
    scalars[3] = float(to_move)
    scalars[4] = float(betting.betting_round)
    # Absolute chips, on one global scale: this is what converts the network's
    # pot-sized output back into money, so it must not depend on the stack.
    scalars[5] = betting.pot / GLOBAL_CHIP_SCALE
    scalars[6] = stack / GLOBAL_CHIP_SCALE
    scalars[7] = float(
        any(betting.is_aggressive(action) for action in betting.history[betting.betting_round])
    )
    out[BOARD_DIM + NUM_SCALARS :] = _encode_betting_history(betting)
    out.setflags(write=False)
    return out


def _encode_betting_history(betting) -> np.ndarray:
    """Replay the public line to recover each action's actual chip increment."""
    round_offset = NUM_BETTING_ROUNDS - betting.num_rounds
    if round_offset < 0:
        raise ValueError(
            f"betting state has {betting.num_rounds} rounds; "
            f"maximum is {NUM_BETTING_ROUNDS}"
        )
    encoded = np.zeros(
        (NUM_BETTING_ROUNDS, MAX_ACTIONS_PER_ROUND, BETTING_SLOT_DIM)
    )
    replay = replace(
        betting,
        betting_round=0,
        history=tuple(() for _ in betting.history),
        # The blinds, not zero.  They are forced chips rather than actions, so
        # they never appear in ``history``; starting the replay from (0, 0)
        # would charge the small blind's call the big blind's whole two chips
        # instead of the one it actually adds, and every preflop increment
        # downstream of it would be wrong by the same amount.
        contributions=betting.blinds,
        folder=-1,
        showdown=False,
        awaiting_board=False,
    )
    stack = float(betting.stack) or 1.0
    for round_index, actions in enumerate(betting.history[:NUM_BETTING_ROUNDS]):
        encoded_round = round_offset + round_index
        if encoded_round >= NUM_BETTING_ROUNDS:
            break
        if round_index and replay.awaiting_board:
            replay = replay.deal_board()
        if len(actions) > MAX_ACTIONS_PER_ROUND:
            raise ValueError(
                f"betting round has {len(actions)} actions; "
                f"maximum is {MAX_ACTIONS_PER_ROUND}"
            )
        for position, action in enumerate(actions):
            player = replay.to_move()
            contribution_before = replay.contributions[player]
            replay = replay.apply(action)
            encoded[encoded_round, position, 0] = 1.0
            encoded[encoded_round, position, 1] = (
                replay.contributions[player] - contribution_before
            ) / stack
    return encoded.reshape(-1)


def encode_pbs(pbs) -> np.ndarray:
    out = np.empty(INPUT_DIM)
    out[:RANGE_DIM] = pbs.ranges.reshape(-1)
    out[BOARD_OFFSET:] = encode_public(pbs.public)
    return out


def encode_batch(states: Sequence) -> np.ndarray:
    out = np.empty((len(states), INPUT_DIM))
    for i, pbs in enumerate(states):
        out[i] = encode_pbs(pbs)
    return out


def possible_from_features(features: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Pass-through helper kept for symmetry with the Leduc encoder."""
    return masks
