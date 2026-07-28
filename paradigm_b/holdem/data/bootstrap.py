"""Bootstrapped training data for the streets above the river.

The river can be solved exactly and cheaply, so its examples are ground truth.
Nothing above it can: an exact turn target means solving the turn *and* every
river below it, and an exact flop target is worse again.

ReBeL's answer is to bootstrap.  A turn target is produced by a depth-limited
turn solve that asks the network what the river belief states are worth — so the
cost of a turn example is one small solve plus some network calls, not a search
to the end of the game.  The same trick one street higher gives flop examples
from turn values.  This is the mechanism that makes deep games affordable, and
it is the part of ReBeL that a purely sampled DeepStack-style pipeline gives up.

Written once, generically over the starting street: a three-card root produces
turn examples, a four-card root produces river-boundary ones.  What changes is
only how many cards are on the table.

**Ordering matters.**  A turn target bootstrapped from an inaccurate river
network is confident nonsense, and training on it teaches the error.  The river
layer has to be good before this is worth running, which is why the training
loop treats the streets as a curriculum rather than a mixture to be stirred at
once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from paradigm_b.core.cfr.tabular_cfr import CFRConfig
from paradigm_b.holdem.engine.betting import Betting
from paradigm_b.holdem.engine.combos import NUM_COMBOS, board_mask
from paradigm_b.holdem.net.features import INPUT_DIM, encode_pbs
from paradigm_b.holdem.data.generation import Examples, GenerationConfig, REFERENCE_POT
from paradigm_b.holdem.engine.public_tree import PublicState, build_endgame_tree
from paradigm_b.holdem.data.sampling import sample_board, sample_range
from paradigm_b.holdem.engine.space import EndgameSpace
from paradigm_b.core.search.subgame import SubgameSolver

NUM_PLAYERS = 2


@dataclass
class BootstrapConfig:
    """How bootstrapped examples are drawn and solved."""

    board_cards: int = 4  # 4 -> turn examples; 3 -> flop examples
    situations_per_board: int = 8
    cfr_iterations: int = 40
    min_spr: float = 0.5
    max_spr: float = 6.0
    min_pot: int = 10
    max_pot: int = 400
    max_raises: int = 1
    num_rounds: int = 2
    excluded_boards: Tuple[Tuple[int, ...], ...] = ()
    cfr: CFRConfig = field(default_factory=CFRConfig.dcfr)


def street_state(config: BootstrapConfig, pot: int, stack: int) -> Betting:
    """A betting state at the start of this street, with the past in the pot."""
    return Betting(
        starting_pot=pot,
        stack=stack,
        max_raises=config.max_raises,
        num_rounds=config.num_rounds,
        betting_round=0,
        contributions=(0, 0),
    )


def normalise_values(
    values: np.ndarray, reach: np.ndarray, correction: float
) -> Optional[np.ndarray]:
    """Counterfactual values -> the network's per-hand convention."""
    masses = reach.sum(axis=1)
    if masses.min() <= 0.0:
        return None
    out = np.empty_like(values)
    out[0] = values[0] / (correction * masses[1])
    out[1] = values[1] / (correction * masses[0])
    return out


def bootstrap_for_board(
    board: Tuple[int, ...],
    leaf_value_fn,
    rng: np.random.Generator,
    config: BootstrapConfig,
) -> Examples:
    """Depth-limited solves on one board, evaluated by ``leaf_value_fn``.

    One solve per situation rather than a batch: the tree has leaves, and the
    batched solver deliberately handles only terminal-only trees.  The leaves of
    a *single* solve are already evaluated in one batched network call, which is
    where most of the work is.
    """
    space = EndgameSpace(board)
    features: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    mask = board_mask(board)

    for _ in range(config.situations_per_board):
        spr = float(rng.uniform(config.min_spr, config.max_spr))
        pot = int(rng.integers(config.min_pot, config.max_pot + 1))
        stack = max(int(round(spr * pot)), 2)
        public = PublicState(betting=street_state(config, pot, stack), board=board)
        reach = np.stack([sample_range(rng, board) for _ in range(2)])

        tree = build_endgame_tree(public, depth_limit=1)
        solver = SubgameSolver(
            tree,
            leaf_value_fn=leaf_value_fn if tree.leaves() else None,
            config=config.cfr,
            space=space,
        )
        solver.solve(reach=reach, iterations=config.cfr_iterations)
        values = normalise_values(
            solver.root_values(), reach, space.pair_correction
        )
        if values is None:
            continue
        features.append(encode_pbs(space.pbs(public, reach)))
        targets.append(values * mask)

    if not features:
        return Examples.concatenate([])
    return Examples(
        features=np.stack(features).astype(np.float32),
        masks=np.repeat(mask[None].astype(np.float32), len(features), axis=0),
        targets=np.stack(targets).astype(np.float32),
    )


def generate_bootstrapped(
    leaf_value_fn,
    num_examples: int,
    config: BootstrapConfig | None = None,
    seed: int = 0,
) -> Examples:
    """Sampled situations one street above whatever ``leaf_value_fn`` values.

    Single-process by design: ``leaf_value_fn`` is normally a neural network,
    and forking one copy per worker costs more in memory and thread contention
    than it returns at this scale.  The parallel path is the river generator,
    which is pure numpy.
    """
    config = config or BootstrapConfig()
    rng = np.random.default_rng(seed)
    boards_needed = max(
        int(np.ceil(num_examples / config.situations_per_board)), 1
    )
    parts = []
    for _ in range(boards_needed):
        board = sample_board(rng, config.board_cards, config.excluded_boards)
        parts.append(bootstrap_for_board(board, leaf_value_fn, rng, config))
    return Examples.concatenate(parts)
