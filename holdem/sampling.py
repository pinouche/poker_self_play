"""Random training situations, in the style DeepStack and ReBeL generate them.

The first hold'em training run kept the board, the pot and the starting ranges
fixed, and every trajectory therefore began from a byte-identical belief state.
Half the training data was one input repeated thousands of times, and the
network was never asked the question that matters — *what is a board you have
not seen worth?*

DeepStack fixed this by generating each training situation at random: a random
board, a random pot, and random ranges.  That is what this module does.  The
ranges deliberately span shapes real betting produces — uniform when nobody has
shown anything, sharply strength-tilted after a raise, capped when a player has
only called, and sparse when a line is very narrow — because a value network is
only useful on the range shapes it has actually seen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from holdem.betting import Betting
from holdem.combos import NUM_CARDS, NUM_COMBOS, board_mask
from holdem.public_tree import PublicState
from holdem.space import TurnEndgameSpace
from holdem.strength import ILLEGAL, hand_ranks

RANGE_STYLES = ("uniform", "dirichlet", "tilted", "capped", "sparse")


@dataclass
class SituationConfig:
    """The distribution training situations are drawn from."""

    board_cards: int = 4  # 4 = turn endgame
    min_pot: int = 10
    max_pot: int = 120
    min_stack: int = 40
    max_stack: int = 200
    max_raises: int = 1
    # Boards to exclude — the held-out set, when measuring generalisation.
    excluded_boards: Tuple[Tuple[int, ...], ...] = ()


def sample_board(
    rng: np.random.Generator, num_cards: int = 4, excluded: Sequence = ()
) -> Tuple[int, ...]:
    """A random board, canonicalised by sorting so lookups share a cache."""
    excluded = {tuple(sorted(b)) for b in excluded}
    while True:
        board = tuple(sorted(rng.choice(NUM_CARDS, size=num_cards, replace=False)))
        if board not in excluded:
            return tuple(int(c) for c in board)


def sample_range(
    rng: np.random.Generator, board: Tuple[int, ...], style: Optional[str] = None
) -> np.ndarray:
    """One player's range on ``board``, drawn from a mixture of shapes."""
    mask = board_mask(board)
    legal = np.flatnonzero(mask)
    style = style or RANGE_STYLES[int(rng.integers(len(RANGE_STYLES)))]

    weights = np.zeros(NUM_COMBOS)
    if style == "uniform":
        weights[legal] = 1.0
    elif style == "dirichlet":
        concentration = float(np.exp(rng.uniform(np.log(0.2), np.log(8.0))))
        weights[legal] = rng.dirichlet(np.full(len(legal), concentration))
    else:
        ranks = hand_ranks(board)[legal].astype(np.float64)
        strength = (ranks - ranks.min()) / max(ranks.max() - ranks.min(), 1.0)
        if style == "tilted":
            # Positive slope: a range that has bet.  Negative: one that has
            # called down and been capped.
            slope = float(rng.normal(0.0, 4.0))
            weights[legal] = np.exp(slope * (strength - strength.mean()))
        elif style == "capped":
            keep = strength <= rng.uniform(0.3, 0.95)
            weights[legal] = np.where(keep, 1.0, 0.02)
        else:  # sparse
            keep = rng.random(len(legal)) < rng.uniform(0.1, 0.6)
            if not keep.any():
                keep[rng.integers(len(legal))] = True
            weights[legal] = keep * rng.random(len(legal))

    total = weights.sum()
    if total <= 0.0:
        weights = mask.copy()
        total = weights.sum()
    return weights / total


def sample_situation(
    rng: np.random.Generator, config: SituationConfig | None = None
) -> Tuple[TurnEndgameSpace, PublicState, np.ndarray]:
    """A random (board, pot, stack, ranges) endgame to train or test on."""
    config = config or SituationConfig()
    board = sample_board(rng, config.board_cards, config.excluded_boards)
    pot = int(rng.integers(config.min_pot, config.max_pot + 1))
    stack = int(rng.integers(config.min_stack, config.max_stack + 1))
    betting = Betting(starting_pot=pot, stack=stack, max_raises=config.max_raises)
    space = TurnEndgameSpace(board)
    reach = np.stack([sample_range(rng, board) for _ in range(2)])
    return space, PublicState(betting=betting, board=board), reach


def held_out_boards(
    rng: np.random.Generator, count: int, num_cards: int = 4
) -> Tuple[Tuple[int, ...], ...]:
    """Boards reserved for evaluation and never trained on."""
    boards = []
    seen = set()
    while len(boards) < count:
        board = sample_board(rng, num_cards)
        if board not in seen:
            seen.add(board)
            boards.append(board)
    return tuple(boards)
