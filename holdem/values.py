"""Leaf evaluators for hold'em endgames."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from cfr.tabular_cfr import CFRConfig
from holdem.combos import NUM_COMBOS, board_mask
from holdem.features import encode_batch
from holdem.net import HoldemValueNet
from holdem.public_tree import build_turn_tree
from holdem.space import TurnEndgameSpace
from search.subgame import SubgameSolver

NUM_PLAYERS = 2


class RiverLeafValues:
    """Solve each river subgame exactly, keeping one warm solver per board.

    The hold'em analogue of ``NestedLeafValues``: a river subgame is a real
    game, so its values are not free, and re-solving 48 of them from cold on
    every trunk iteration is unaffordable.  Warm-starting makes it affordable
    and — as on Leduc — more accurate per unit of work, because the trunk's
    ranges move slowly.
    """

    def __init__(
        self,
        space: TurnEndgameSpace,
        iterations_per_call: int = 2,
        warmup: int = 60,
        config: CFRConfig | None = None,
    ) -> None:
        self.space = space
        self.iterations_per_call = iterations_per_call
        self.warmup = warmup
        self.config = config or CFRConfig.dcfr()
        self._solvers: Dict[object, SubgameSolver] = {}
        self.iteration_count = 0

    def __call__(self, states: Sequence) -> np.ndarray:
        out = np.empty((len(states), NUM_PLAYERS, NUM_COMBOS))
        for i, pbs in enumerate(states):
            solver = self._solvers.get(pbs.public)
            if solver is None:
                solver = SubgameSolver(
                    build_turn_tree(pbs.public, depth_limit=None),
                    config=self.config,
                    space=self.space,
                )
                self._solvers[pbs.public] = solver
                iterations = self.warmup
            else:
                iterations = self.iterations_per_call
            solver.reset_values()
            solver.solve(reach=pbs.ranges, iterations=iterations)
            self.iteration_count += iterations
            out[i] = solver.root_values() / self.space.pair_correction
        return out


class NetLeafValues:
    """Leaf evaluator backed by a :class:`HoldemValueNet`."""

    def __init__(
        self,
        net: HoldemValueNet,
        space: Optional[TurnEndgameSpace] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        # No board is baked in: the mask comes from each state's own board, so
        # one evaluator serves every situation in a randomised training run.
        self.net = net
        self.space = space
        self.device = torch.device(device)
        self.net.to(self.device)
        self.net.eval()
        self.state_count = 0

    @torch.no_grad()
    def __call__(self, states: Sequence) -> np.ndarray:
        features = torch.as_tensor(
            encode_batch(states), dtype=torch.float32, device=self.device
        )
        masks = torch.as_tensor(
            np.stack([board_mask(tuple(s.public.board)) for s in states]),
            dtype=torch.float32,
            device=self.device,
        )
        self.state_count += len(states)
        return self.net(features, masks).cpu().numpy().astype(np.float64)


class ZeroLeafValues:
    def __call__(self, states: Sequence) -> np.ndarray:
        return np.zeros((len(states), NUM_PLAYERS, NUM_COMBOS))
