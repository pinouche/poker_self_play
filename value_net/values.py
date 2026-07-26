"""Leaf evaluators: the exact one (ground truth) and the learned one.

Both satisfy the same contract — a batch of public belief states in, per-hand
values for both players out, in chips, for the *normalised* ranges the PBS
carries.  Search does not care which it is given, which is exactly what makes
the value network testable: solve a subgame with exact leaf values, solve it
again with the network, and compare exploitability.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from belief.public_tree import build_public_tree
from belief.ranges import NUM_HANDS, NUM_PLAYERS, PAIR_CORRECTION, PBS
from cfr.tabular_cfr import CFRConfig
from search.evaluate import range_values
from search.policy import StrategyMap
from search.subgame import SubgameSolver
from value_net.features import encode_batch
from value_net.net import PBSValueNet

EXACT_ITERATIONS = 200


def exact_pbs_values(
    pbs: PBS, iterations: int = EXACT_ITERATIONS, config: CFRConfig | None = None
) -> np.ndarray:
    """Solve the subgame below ``pbs`` and return its per-hand values.

    The ground truth the value network is trying to imitate: run range-form CFR
    from this belief state to the end of the game and read off what each hand is
    worth.  Values are for normalised ranges, so the reach mass is divided back
    out (the ranges sum to one, leaving only the card-removal correction).

    The iteration-averaged values are used rather than the values of the average
    strategy — see :meth:`SubgameSolver.root_values` for why that distinction is
    load-bearing rather than cosmetic.
    """
    tree = build_public_tree(pbs.public, depth_limit=None)
    solver = SubgameSolver(tree, config=config or CFRConfig.dcfr())
    solver.solve(reach=pbs.ranges, iterations=iterations)
    return solver.root_values() / PAIR_CORRECTION


class ExactLeafValues:
    """Leaf evaluator that re-solves each leaf subgame exactly.

    Used to separate two questions that are easy to confuse: is the
    *depth-limited search* sound, and is the *network* accurate?  With this
    evaluator only the first is being tested.
    """

    def __init__(
        self,
        iterations: int = EXACT_ITERATIONS,
        config: CFRConfig | None = None,
        cache: bool = True,
    ) -> None:
        self.iterations = iterations
        self.config = config or CFRConfig.dcfr()
        self._cache: Optional[Dict[Tuple, np.ndarray]] = {} if cache else None
        self.solve_count = 0

    def __call__(self, states: Sequence[PBS]) -> np.ndarray:
        out = np.empty((len(states), NUM_PLAYERS, NUM_HANDS))
        for i, pbs in enumerate(states):
            key = None
            if self._cache is not None:
                key = (pbs.public, np.round(pbs.ranges, 9).tobytes())
                hit = self._cache.get(key)
                if hit is not None:
                    out[i] = hit
                    continue
            values = exact_pbs_values(pbs, self.iterations, self.config)
            self.solve_count += 1
            if key is not None:
                self._cache[key] = values
            out[i] = values
        return out


class NestedLeafValues:
    """Leaf evaluator that keeps one solver per leaf and never restarts it.

    :class:`ExactLeafValues` solves each leaf subgame from scratch on every
    query, which is both slow and — at any affordable iteration count —
    inaccurate.  That inaccuracy does not wash out: CFR-D's trunk inherits the
    leaf-value error as a *constant* term, so no amount of trunk iteration
    removes it, and the depth-limited scheme sits on a floor it cannot get below.

    The fix is the one Burch et al. describe for CFR-D: interleave the trunk and
    subgame iterations instead of nesting them.  Each leaf keeps a persistent
    regret minimiser, and a query runs a handful more iterations on it.  Because
    the trunk's ranges move slowly between iterations, a warm-started solver is
    already near-converged for the ranges it is being asked about — so a few
    iterations per query beat hundreds from cold, at a fraction of the cost.

    Only the value *average* is reset per query; the regrets, which are what took
    the work to learn, carry over.
    """

    def __init__(
        self,
        iterations_per_call: int = 2,
        warmup: int = 200,
        config: CFRConfig | None = None,
    ) -> None:
        self.iterations_per_call = iterations_per_call
        self.warmup = warmup
        self.config = config or CFRConfig.dcfr()
        self._solvers: Dict[object, SubgameSolver] = {}
        self.iteration_count = 0

    def __call__(self, states: Sequence[PBS]) -> np.ndarray:
        out = np.empty((len(states), NUM_PLAYERS, NUM_HANDS))
        for i, pbs in enumerate(states):
            solver = self._solvers.get(pbs.public)
            if solver is None:
                solver = SubgameSolver(
                    build_public_tree(pbs.public, depth_limit=None), config=self.config
                )
                self._solvers[pbs.public] = solver
                iterations = self.warmup
            else:
                iterations = self.iterations_per_call
            solver.reset_values()
            solver.solve(reach=pbs.ranges, iterations=iterations)
            self.iteration_count += iterations
            out[i] = solver.root_values() / PAIR_CORRECTION
        return out


class BlueprintLeafValues:
    """Leaf evaluator that evaluates a *fixed continuation policy*.

    The consistent leaf value function: it answers "what is each hand worth if
    play continues with this policy", which is the question a depth-limited
    search actually needs answered.  Contrast :class:`ExactLeafValues`, which
    answers "what is each hand worth at an equilibrium of the leaf subgame" — a
    different and, on its own, unsound number, because the trunk then assumes a
    per-hand split of the subgame value that the continuation need not deliver.

    This is also the target the ReBeL value network converges to: the values of
    the policy its own search plays.
    """

    def __init__(self, strategies: StrategyMap) -> None:
        self.strategies = strategies

    def __call__(self, states: Sequence[PBS]) -> np.ndarray:
        out = np.empty((len(states), NUM_PLAYERS, NUM_HANDS))
        for i, pbs in enumerate(states):
            tree = build_public_tree(pbs.public, depth_limit=None)
            out[i] = range_values(tree, self.strategies, pbs.ranges) / PAIR_CORRECTION
        return out


class NetLeafValues:
    """Leaf evaluator backed by a :class:`PBSValueNet`."""

    def __init__(
        self, net: PBSValueNet, device: str | torch.device = "cpu", eval_mode: bool = True
    ) -> None:
        self.net = net
        self.device = torch.device(device)
        self.net.to(self.device)
        if eval_mode:
            self.net.eval()
        self.call_count = 0
        self.state_count = 0

    @torch.no_grad()
    def __call__(self, states: Sequence[PBS]) -> np.ndarray:
        features = torch.as_tensor(
            encode_batch(states), dtype=torch.float32, device=self.device
        )
        values = self.net(features)
        self.call_count += 1
        self.state_count += len(states)
        return values.detach().cpu().numpy().astype(np.float64)


class ZeroLeafValues:
    """Leaf evaluator that predicts nothing — the untrained baseline.

    Search with this is "assume every hand is worth zero at the depth limit",
    which is what a fresh network effectively does.  Its exploitability is the
    number a trained network has to beat.
    """

    def __call__(self, states: Sequence[PBS]) -> np.ndarray:
        return np.zeros((len(states), NUM_PLAYERS, NUM_HANDS))
