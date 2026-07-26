"""Encoding a hold'em public belief state for the value network.

The core modelling decision of stage 4, and the one that does not carry over
from Leduc.  There the input was six numbers per player; here it is 1,326, and
they are not exchangeable — combos have structure (rank, suit, whether they pair
the board) that a flat vector hides.

This encoder stays deliberately flat and unabstracted: both range vectors as
they are, plus the board as a 52-way indicator and the pot and stacks in
fractions.  No bucketing, no clustering, which is what ReBeL argues for — the
network is given the raw belief and left to find its own structure.  The cost is
a wide input layer and a wide output layer; the benefit is that no information
is thrown away before the network sees it, and there is no abstraction to blame
when values come out wrong.

Layout: 1326 + 1326 range entries, then 52 board indicators, then five scalars.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from holdem.combos import NUM_CARDS, NUM_COMBOS

RANGE_DIM = 2 * NUM_COMBOS
BOARD_OFFSET = RANGE_DIM
SCALAR_OFFSET = BOARD_OFFSET + NUM_CARDS
NUM_SCALARS = 7
# Chips that correspond to a scaled value of 1.0.  Fixed and global so that the
# network's output scaling is exact whatever the pot and stack happen to be —
# with randomised situations they vary by an order of magnitude.
GLOBAL_CHIP_SCALE = 1000.0
INPUT_DIM = SCALAR_OFFSET + NUM_SCALARS
PUBLIC_DIM = NUM_CARDS + NUM_SCALARS


POT_CHIPS_INDEX = BOARD_OFFSET + NUM_CARDS + 5


def encode_public(public) -> np.ndarray:
    """Board indicators plus the scalars that set the scale of the pot."""
    betting = public.betting
    out = np.zeros(PUBLIC_DIM)
    for card in public.board:
        out[card] = 1.0
    stack = float(betting.stack) or 1.0
    to_move = betting.to_move() if not betting.is_terminal else 0
    scalars = out[NUM_CARDS:]
    scalars[0] = betting.pot / (2.0 * stack + betting.starting_pot)
    scalars[1] = betting.contributions[0] / stack
    scalars[2] = betting.contributions[1] / stack
    scalars[3] = float(to_move)
    scalars[4] = float(betting.betting_round)
    # Absolute chips, on one global scale: this is what converts the network's
    # pot-sized output back into money, so it must not depend on the stack.
    scalars[5] = betting.pot / GLOBAL_CHIP_SCALE
    scalars[6] = stack / GLOBAL_CHIP_SCALE
    return out


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
