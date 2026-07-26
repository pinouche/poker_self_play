"""Encoding a public belief state as network input.

The input is deliberately close to the raw PBS: the two range vectors as they
are, plus the public facts that determine what the pot is worth.  No card
abstraction and no bucketing — Leduc has six hands, and the point of the ReBeL
formulation is that the range *is* the input.  (At hold'em scale this is the one
design decision that has to change: 1,326 combos per player is where bucketing
or an embedding becomes unavoidable.)

Layout, 28 floats:

===========  ====  ==================================================
offset       size  meaning
===========  ====  ==================================================
0            6     player 0's range
6            6     player 1's range
12           6     board card, one-hot
18           1     is the board out at all
19           2     betting round, one-hot
21           2     player to act, one-hot
23           1     raises so far this round / 2
24           1     facing a bet
25           2     each player's contribution / 13
27           1     pot / 26
===========  ====  ==================================================
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from belief.public_tree import PublicState
from belief.ranges import NUM_HANDS, NUM_PLAYERS, PBS
from game.actions import RAISE
from game.leduc import MAX_RAISES_PER_ROUND, NUM_CARDS, NUM_ROUNDS

RANGE_SLICE = slice(0, NUM_PLAYERS * NUM_HANDS)
BOARD_SLICE = slice(12, 12 + NUM_CARDS)
PUBLIC_SLICE = slice(12, 28)
POT_INDEX = 27
INPUT_DIM = 28
RANGE_DIM = NUM_PLAYERS * NUM_HANDS
PUBLIC_DIM = INPUT_DIM - RANGE_DIM

# The largest a single contribution and the pot can get in Leduc: 1 ante + 2 + 2
# in round one, + 4 + 4 in round two.
MAX_CONTRIBUTION = 13.0
MAX_POT = 2.0 * MAX_CONTRIBUTION


def encode_public(public: PublicState) -> np.ndarray:
    """The public half of the input vector."""
    out = np.zeros(PUBLIC_DIM)
    betting = public.betting
    if public.board >= 0:
        out[public.board] = 1.0
        out[NUM_CARDS] = 1.0
    offset = NUM_CARDS + 1
    out[offset + betting.betting_round] = 1.0
    offset += NUM_ROUNDS
    to_move = betting.to_move() if not betting.is_terminal else 0
    out[offset + to_move] = 1.0
    offset += NUM_PLAYERS
    raises = betting.history[betting.betting_round].count(RAISE)
    out[offset] = raises / MAX_RAISES_PER_ROUND
    out[offset + 1] = float(
        betting.contributions[to_move] < betting.contributions[1 - to_move]
    )
    out[offset + 2] = betting.contributions[0] / MAX_CONTRIBUTION
    out[offset + 3] = betting.contributions[1] / MAX_CONTRIBUTION
    out[offset + 4] = betting.pot / MAX_POT
    return out


def encode_pbs(pbs: PBS) -> np.ndarray:
    """One PBS as a flat feature vector."""
    out = np.empty(INPUT_DIM)
    out[RANGE_SLICE] = pbs.ranges.reshape(-1)
    out[PUBLIC_SLICE] = encode_public(pbs.public)
    return out


def encode_batch(states: Sequence[PBS]) -> np.ndarray:
    """``(N, INPUT_DIM)`` features for a batch of belief states."""
    out = np.empty((len(states), INPUT_DIM))
    for i, pbs in enumerate(states):
        out[i] = encode_pbs(pbs)
    return out


def ranges_from_features(features: np.ndarray) -> np.ndarray:
    """Recover the ``(..., 2, NUM_HANDS)`` ranges from encoded features."""
    return features[..., RANGE_SLICE].reshape(*features.shape[:-1], NUM_PLAYERS, NUM_HANDS)


def hand_mask_from_features(features: np.ndarray) -> np.ndarray:
    """Which hands are still possible: everything except the board card."""
    return 1.0 - features[..., BOARD_SLICE]
