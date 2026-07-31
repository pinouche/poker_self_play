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

from paradigm_b.core.belief.public_tree import (
    LEAF,
    PublicNode,
    PublicState,
    PublicTree,
    build_public_tree,
)
from paradigm_b.core.belief.ranges import NUM_PLAYERS, PBS, initial_reach
from paradigm_b.core.cfr.policy import TabularPolicy
from paradigm_b.core.cfr.tabular_cfr import CFRConfig
from paradigm_b.core.game.tree import GameTree
from paradigm_b.core.search.evaluate import (
    leaf_counterfactual_values,
    leaf_reaches,
    range_values,
    reaches_at,
)
from paradigm_b.core.search.policy import StrategyMap, strategy_map, tabular_policy_from_strategies
from paradigm_b.core.search.space import LEDUC_SPACE, HandSpace
from paradigm_b.core.search.subgame import Gadget, LeafValueFn, SubgameSolver


@dataclass
class ResolveConfig:
    """Knobs of the search itself — the ones that trade strength for time."""

    iterations: int = 100
    depth_limit: int = 1  # betting rounds expanded per solve
    cfr: CFRConfig = field(default_factory=CFRConfig.dcfr)
    # Hold each re-solve to the counterfactual values the previous solve
    # promised (see :class:`Gadget`).  Off, re-solving is measurably exploitable.
    safe_resolving: bool = True
    # How many betting rounds below the root to keep re-solving for.  ``None``
    # resolves every reachable belief state to the end of the game, which is
    # what you want on the turn (241 solves) and cannot afford on the flop,
    # where the turn and river fan out to roughly 11,000.  Setting it to *n*
    # stops the recursion at round ``root + n``, so those belief states become
    # leaves priced by ``leaf_value_fn`` instead of solved.  Match it to the
    # depth limit of the tree the strategy is scored on, or the agent will be
    # measured on decision nodes it was never asked to produce behaviour for.
    max_resolve_rounds: Optional[int] = None


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
        root_round = root.betting_round
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
                if self._beyond_resolve_limit(point.public, root_round):
                    continue  # a leaf of the bounded tree: valued, not solved
                promise = self._promise(
                    point, point_reach, local, public.betting_round
                )
                frontier.append((point.public, point_reach, promise))
        return strategies, traces

    def _beyond_resolve_limit(self, public: PublicState, root_round: int) -> bool:
        limit = self.config.max_resolve_rounds
        return limit is not None and public.betting_round - root_round >= limit

    def _resolve_points(
        self, tree: PublicTree, strategies: StrategyMap, reach: np.ndarray
    ) -> List[Tuple[PublicNode, np.ndarray]]:
        """Where to stop and solve again, with the ranges arriving there.

        Every start of a betting round below this solve's root.  When the
        lookahead reaches past a board deal these are nodes of this tree, and
        they are the points DeepStack calls again at instead of following the
        plan it already computed — which matters, because the plan was computed
        against ranges that the opponent's actual betting has since narrowed.

        With the depth limit at one round the tree stops *in front of* the deal,
        so there are no such nodes and the next belief states have to be dealt
        out by hand — see :meth:`_across_the_deal`.
        """
        points = tree.round_starts()
        if points:
            return reaches_at(
                tree, strategies, reach, {node.public for node in points}, space=self.space
            )
        return self._across_the_deal(tree, strategies, reach)

    def _across_the_deal(
        self, tree: PublicTree, strategies: StrategyMap, reach: np.ndarray
    ) -> List[Tuple[PublicNode, np.ndarray]]:
        """Step each end-of-round leaf over the chance node in front of it.

        A subgame that ends where the betting ends leaves the agent holding a
        belief state the board has not been dealt into yet, so the states to
        solve next are one outcome away: every card the deck can still turn
        over, with both ranges narrowed by it.  This is the enumeration that
        used to happen inside the tree, moved to the only place that can still
        afford to do it — outside the CFR loop, once per solve rather than once
        per iteration.

        The synthesised nodes are marked ``LEAF`` so that :meth:`_promise` holds
        the next solve to the value network's opinion of the post-deal belief
        state, which is exactly the promise a post-deal leaf carried before.

        A deal of more than one card is left alone.  Only the flop is like that,
        and its 19,600 outcomes are not something continual re-solving can walk;
        the pre-deal leaf keeps its network value and the recursion stops there,
        which is the honest behaviour for a bounded agent.
        """
        out: List[Tuple[PublicNode, np.ndarray]] = []
        for leaf, leaf_reach in leaf_reaches(tree, strategies, reach, space=self.space):
            public = leaf.public
            if not public.awaiting_board or public.cards_to_deal != 1:
                continue
            for card in public.undealt_cards():
                mask = self.space.deal_mask(card)
                child_reach = leaf_reach * mask
                if child_reach.sum() <= 0.0:
                    continue
                out.append(
                    (
                        PublicNode(
                            index=-1, kind=LEAF, public=public.with_board(card)
                        ),
                        child_reach,
                    )
                )
        return out

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
