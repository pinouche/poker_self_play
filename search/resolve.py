"""Continual re-solving: play a whole game out of depth-limited searches.

DeepStack's central idea, and how a ReBeL agent acts.  Rather than carrying a
strategy for the whole game, the agent keeps only the current public belief
state.  At the start of every betting round it solves the subgame it can see —
this round's betting, with the value network standing in for everything beyond —
plays that solution, and when the round closes it propagates both ranges to the
next round's belief state and solves again.

Two things this exercises that a single solve does not.  The strategy played in
round two comes from a *different* solve than the one that produced the ranges
it inherits, so any inconsistency in the value network shows up as
exploitability.  And the ranges are the agent's own: it must reason about what
its own betting revealed.

Walking every reachable belief state (rather than one sampled trajectory) gives
a complete tabular policy, which the exact best-response routine can then score.
That is the only honest way to answer "is search actually making this less
exploitable".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from belief.public_tree import PublicNode, PublicState, PublicTree, build_public_tree
from belief.ranges import NUM_PLAYERS, PBS, initial_reach
from cfr.policy import TabularPolicy
from cfr.tabular_cfr import CFRConfig
from game.tree import GameTree
from search.evaluate import leaf_counterfactual_values, range_values, reaches_at
from search.policy import StrategyMap, strategy_map, tabular_policy_from_strategies
from search.space import LEDUC_SPACE, HandSpace
from search.subgame import Gadget, LeafValueFn, SubgameSolver


@dataclass
class ResolveConfig:
    """Knobs of the search itself — the ones that trade strength for time."""

    iterations: int = 100
    depth_limit: int = 1  # betting rounds expanded per solve
    cfr: CFRConfig = field(default_factory=CFRConfig.dcfr)
    # Hold each re-solve to the counterfactual values the previous solve
    # promised (see :class:`Gadget`).  Off, re-solving is measurably exploitable.
    safe_resolving: bool = True


@dataclass
class ResolveTrace:
    """One solve: where it was rooted and what it produced."""

    pbs: PBS
    root_values: np.ndarray  # (2, NUM_HANDS), iteration-averaged
    reach: np.ndarray


class ContinualResolver:
    """Produce behaviour for a whole game by re-solving round by round."""

    def __init__(
        self,
        leaf_value_fn: Optional[LeafValueFn],
        config: ResolveConfig | None = None,
        space: Optional[HandSpace] = None,
        tree_builder=None,
    ) -> None:
        self.leaf_value_fn = leaf_value_fn
        self.config = config or ResolveConfig()
        self.space = space or LEDUC_SPACE
        self.build_tree = tree_builder or build_public_tree

    def run(
        self, root: PublicState | None = None, reach: np.ndarray | None = None
    ) -> Tuple[StrategyMap, List[ResolveTrace]]:
        """Solve every reachable belief state, breadth first.

        Returns the strategy at every public decision node together with a trace
        of the solves — the traces are what the ReBeL loop trains on.
        """
        root = root if root is not None else PublicState()
        reach = (
            self.space.initial_reach()
            if reach is None
            else np.asarray(reach, dtype=np.float64)
        )
        strategies: StrategyMap = {}
        traces: List[ResolveTrace] = []
        # Each entry is a belief state to solve, plus the counterfactual values
        # the solve that produced it promised each player (None at the root).
        frontier: List[Tuple[PublicState, np.ndarray, Optional[np.ndarray]]] = [
            (root, reach, None)
        ]

        while frontier:
            public, node_reach, promised = frontier.pop()
            safe = promised is not None and self.config.safe_resolving
            if safe:
                local, root_values = self._solve_safely(public, node_reach, promised)
            else:
                solver = self._solve(public, node_reach)
                local = strategy_map(solver)
                root_values = solver.root_values()
            strategies.update(local)
            traces.append(
                ResolveTrace(
                    pbs=self.space.pbs(public, node_reach),
                    root_values=root_values,
                    reach=node_reach.copy(),
                )
            )
            tree = self.build_tree(public, depth_limit=self.config.depth_limit)
            for point, point_reach in self._resolve_points(tree, local, node_reach):
                if point_reach.sum(axis=1).min() <= 0.0:
                    continue  # a belief state neither player can reach
                promise = self._promise(
                    point, point_reach, local, public.betting_round
                )
                frontier.append((point.public, point_reach, promise))
        return strategies, traces

    def _resolve_points(
        self, tree: PublicTree, strategies: StrategyMap, reach: np.ndarray
    ) -> List[Tuple[PublicNode, np.ndarray]]:
        """Where to stop and solve again, with the ranges arriving there.

        Every start of a betting round below this solve's root.  When the depth
        limit is one round those are exactly the leaves, and this is ordinary
        depth-limited search.  When the lookahead reaches further, they are the
        points DeepStack calls again at instead of following the plan it already
        computed — which matters, because the plan was computed against ranges
        that the opponent's actual betting has since narrowed.
        """
        points = tree.round_starts()
        if not points:
            return []
        return reaches_at(
            tree, strategies, reach, {node.public for node in points}, space=self.space
        )

    def _promise(
        self,
        point: PublicNode,
        reach: np.ndarray,
        strategies: StrategyMap,
        root_round: int,
    ) -> Optional[np.ndarray]:
        """The counterfactual values the solve above just committed to here.

        At a depth-limit leaf that is whatever the value function said.  Deeper
        in, it is what the strategy just computed actually delivers, which is
        the stronger promise: it was produced by search rather than predicted.
        """
        if point.is_leaf:
            if self.leaf_value_fn is None:
                return None
            return leaf_counterfactual_values(
                point.public, reach, self.leaf_value_fn, space=self.space
            )
        remaining = self.config.depth_limit
        if remaining is not None:
            remaining -= point.public.betting_round - root_round
        subtree = self.build_tree(point.public, depth_limit=remaining)
        return range_values(
            subtree, strategies, reach, self.leaf_value_fn, space=self.space
        )

    def policy(
        self,
        world_tree: GameTree,
        root: PublicState | None = None,
        reach: np.ndarray | None = None,
    ) -> TabularPolicy:
        """The full-game policy this agent plays, as a tabular policy."""
        strategies, _ = self.run(root=root, reach=reach)
        return tabular_policy_from_strategies(strategies, world_tree)

    def _solve(
        self, public: PublicState, reach: np.ndarray, gadget: Optional[Gadget] = None
    ) -> SubgameSolver:
        tree = self.build_tree(public, depth_limit=self.config.depth_limit)
        solver = SubgameSolver(
            tree,
            leaf_value_fn=self.leaf_value_fn if tree.leaves() else None,
            config=self.config.cfr,
            space=self.space,
        )
        solver.solve(reach=reach, iterations=self.config.iterations, gadget=gadget)
        return solver

    def _solve_safely(
        self, public: PublicState, reach: np.ndarray, promised: np.ndarray
    ) -> Tuple[StrategyMap, np.ndarray]:
        """One gadget solve per player; each supplies only its own strategy.

        A gadget solve answers "what can *this* player do that still honours the
        promise made to the other".  The opponent's half of that solve is the
        worst case used to enforce the promise, not a policy, so it is discarded
        and the other player's own solve supplies it instead.
        """
        local: StrategyMap = {}
        root_values = np.zeros_like(promised)
        for player in range(NUM_PLAYERS):
            solver = self._solve(
                public, reach, gadget=Gadget(player=player, promised=promised[1 - player])
            )
            for node in solver.tree.decision_nodes():
                if node.player == player:
                    local[node.public] = solver.average_strategy(node)
            root_values[player] = solver.root_values()[player]
        return local, root_values
