"""Fast generation of river training situations.

The value network in a turn-rooted agent is queried at exactly one kind of
belief state: the start of the river.  Nothing else.  So the fastest way to
train it is to make those directly — sample a board, a stack-to-pot ratio and a
pair of ranges, solve the river subgame, record the values — rather than to play
self-play trajectories whose expensive half produces turn examples the network
is never asked about.

Three things make this cheap:

* **The turn solve disappears.**  It was the costly half of a trajectory and it
  produced an off-distribution example.
* **Situations batch by board.**  The sort order and per-card index a showdown
  needs are properties of the board, so K range pairs share them and solve in
  one vectorised pass.
* **The pot is a scale factor.**  At a fixed stack-to-pot ratio every payoff
  scales linearly with the pot, so one solve at a reference pot yields labelled
  examples at any pot — verified exact except where integer chip rounding bites
  at very small pots.

Generation is embarrassingly parallel across boards, so the work is spread over
processes; that turns out to be a far larger lever than anything inside the
solver.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from paradigm_b.core.cfr.tabular_cfr import CFRConfig
from paradigm_b.holdem.data.batched import BatchedRiverSolver
from paradigm_b.holdem.engine.betting import Betting
from paradigm_b.holdem.engine.combos import NUM_COMBOS, board_mask
from paradigm_b.holdem.net.features import INPUT_DIM, encode_pbs
from paradigm_b.holdem.engine.public_tree import PublicState
from paradigm_b.holdem.data.sampling import sample_board, sample_range
from paradigm_b.holdem.engine.space import EndgameSpace

NUM_PLAYERS = 2
REFERENCE_POT = 100  # large enough that integer bet rounding is negligible


@dataclass
class GenerationConfig:
    """What the sampled river situations look like."""

    situations_per_board: int = 64
    cfr_iterations: int = 60
    min_spr: float = 0.5
    max_spr: float = 6.0
    min_pot: int = 10
    max_pot: int = 400
    max_raises: int = 1
    dtype: str = "float32"
    excluded_boards: Tuple[Tuple[int, ...], ...] = ()
    cfr: CFRConfig = field(default_factory=CFRConfig.dcfr)


@dataclass
class Examples:
    features: np.ndarray  # (n, INPUT_DIM)
    masks: np.ndarray  # (n, 1326)
    targets: np.ndarray  # (n, 2, 1326)

    def __len__(self) -> int:
        return len(self.features)

    @staticmethod
    def concatenate(parts: Sequence["Examples"]) -> "Examples":
        parts = [p for p in parts if len(p)]
        if not parts:
            return Examples(
                np.zeros((0, INPUT_DIM), np.float32),
                np.zeros((0, NUM_COMBOS), np.float32),
                np.zeros((0, NUM_PLAYERS, NUM_COMBOS), np.float32),
            )
        return Examples(
            np.concatenate([p.features for p in parts]),
            np.concatenate([p.masks for p in parts]),
            np.concatenate([p.targets for p in parts]),
        )


def river_state(pot: int, stack: int, max_raises: int) -> Betting:
    """A river start: everything won so far is dead money in the pot."""
    return Betting(
        starting_pot=pot,
        stack=stack,
        max_raises=max_raises,
        betting_round=1,
        contributions=(0, 0),
    )


def generate_for_board(
    board: Tuple[int, ...], rng: np.random.Generator, config: GenerationConfig
) -> Examples:
    """Solve one board's worth of river situations in a single batched pass."""
    count = config.situations_per_board
    spr = float(rng.uniform(config.min_spr, config.max_spr))
    reference_stack = max(int(round(spr * REFERENCE_POT)), 2)

    space = EndgameSpace(board[:4])
    root = PublicState(
        betting=river_state(REFERENCE_POT, reference_stack, config.max_raises),
        board=board,
    )
    reaches = np.stack(
        [np.stack([sample_range(rng, board) for _ in range(2)]) for _ in range(count)]
    )

    solver = BatchedRiverSolver(
        space,
        root,
        batch_size=count,
        config=config.cfr,
        dtype=np.dtype(config.dtype),
    )
    values = solver.solve(reaches, config.cfr_iterations).astype(np.float64)
    values /= space.pair_correction  # the network's convention: normalised ranges

    mask = board_mask(board)
    features = np.empty((count, INPUT_DIM), dtype=np.float32)
    targets = np.empty((count, NUM_PLAYERS, NUM_COMBOS), dtype=np.float32)
    for i in range(count):
        # Pot is a pure scale, so each example gets its own without re-solving.
        pot = int(rng.integers(config.min_pot, config.max_pot + 1))
        stack = max(int(round(spr * pot)), 2)
        scale = pot / REFERENCE_POT
        public = PublicState(
            betting=river_state(pot, stack, config.max_raises), board=board
        )
        features[i] = encode_pbs(space.pbs(public, reaches[i]))
        targets[i] = values[i] * scale * mask
    return Examples(
        features=features,
        masks=np.repeat(mask[None].astype(np.float32), count, axis=0),
        targets=targets,
    )


def _worker(args) -> Examples:
    seed, boards, config = args
    rng = np.random.default_rng(seed)
    return Examples.concatenate(
        [generate_for_board(board, rng, config) for board in boards]
    )


def generate(
    num_examples: int,
    config: GenerationConfig | None = None,
    seed: int = 0,
    workers: Optional[int] = None,
) -> Examples:
    """Generate roughly ``num_examples`` labelled river belief states."""
    config = config or GenerationConfig()
    workers = workers if workers is not None else max(os.cpu_count() // 2, 1)
    rng = np.random.default_rng(seed)

    boards_needed = max(int(np.ceil(num_examples / config.situations_per_board)), 1)
    boards = [
        sample_board(rng, 5, config.excluded_boards) for _ in range(boards_needed)
    ]

    if workers <= 1:
        return _worker((seed, boards, config))

    chunks = [boards[i::workers] for i in range(workers)]
    tasks = [
        (int(rng.integers(1 << 31)), chunk, config) for chunk in chunks if chunk
    ]
    import multiprocessing as mp

    with mp.get_context("fork").Pool(len(tasks)) as pool:
        return Examples.concatenate(pool.map(_worker, tasks))
