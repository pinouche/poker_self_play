"""Best response in range form, over a public tree.

The cheap counterpart of the exact world-tree best response: because a
counterfactual value vector already gives the value of *every* hand, a best
response needs no extra machinery — at its own decision nodes the responder
simply takes, hand by hand, the action with the highest value.  It may play a
different action with each hand, which is exactly what knowing your own cards
buys you.

Two uses:

* **Subgame exploitability.**  Given the ranges arriving at a subgame, how much
  can a best responder beat a candidate strategy inside it?  That is how you
  tell "this re-solve found a different equilibrium" (same value) from "this
  re-solve is wrong" (worse value).
* **Local best response (LBR)**, later: the same traversal against a truncated
  tree is the standard cheap lower bound on exploitability in games too large
  for the exact version.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from belief.public_tree import PublicNode, PublicState, PublicTree
from belief.ranges import NUM_PLAYERS, PBS
from search.space import LEDUC_SPACE, HandSpace
from search.policy import StrategyMap
from search.subgame import LeafValueFn, terminal_values


class RangeBestResponse:
    """Exact best response to a fixed strategy on a (sub)public tree."""

    def __init__(
        self,
        tree: PublicTree,
        strategies: StrategyMap,
        responder: int,
        leaf_value_fn: Optional[LeafValueFn] = None,
        space: Optional[HandSpace] = None,
    ) -> None:
        self.tree = tree
        self.strategies = strategies
        self.responder = responder
        self.leaf_value_fn = leaf_value_fn
        self.space = space or LEDUC_SPACE
        self.choices: Dict[PublicState, np.ndarray] = {}

    def values(self, reach: np.ndarray) -> np.ndarray:
        """Per-hand values for the responder, given both players' reaches.

        ``reach[responder]`` is ignored for the values themselves — a
        counterfactual value never includes your own probabilities — but it is
        still needed to keep the opponent's belief updates right.
        """
        return self._walk(self.tree.root, reach.copy())

    def value(self, reach: np.ndarray) -> float:
        """The responder's expected payoff: their reach-weighted hand values."""
        per_hand = self.values(reach)
        return float((reach[self.responder] * per_hand).sum())

    def _walk(self, node: PublicNode, reach: np.ndarray) -> np.ndarray:
        if node.is_terminal:
            return self.space.terminal_values(node.public, reach)[self.responder]
        if node.is_leaf:
            if self.leaf_value_fn is None:
                raise ValueError("a depth-limited tree needs a leaf value function")
            predicted = self.leaf_value_fn([self.space.pbs(node.public, reach)])[0]
            masses = reach.sum(axis=1)
            return (
                self.space.pair_correction
                * masses[1 - self.responder]
                * predicted[self.responder]
            )
        if node.is_chance:
            values = np.zeros(self.space.num_hands)
            weight = self.space.chance_weight(node.public)
            for board, child in zip(node.boards, node.children):
                mask = self.space.deal_mask(board)
                child_reach = reach * mask
                if child_reach.sum() == 0.0:
                    continue
                values += weight * self._walk(child, child_reach) * mask
            return values

        if node.player == self.responder:
            # Free to pick a different action for every hand.
            action_values = np.empty((node.num_actions, self.space.num_hands))
            for j, child in enumerate(node.children):
                action_values[j] = self._walk(child, reach)
            best = action_values.max(axis=0)
            choice = np.zeros((self.space.num_hands, node.num_actions))
            choice[np.arange(self.space.num_hands), action_values.argmax(axis=0)] = 1.0
            self.choices[node.public] = choice
            return best

        strategy = self.strategies[node.public]
        values = np.zeros(self.space.num_hands)
        for j, child in enumerate(node.children):
            child_reach = reach.copy()
            child_reach[node.player] = reach[node.player] * strategy[:, j]
            values += self._walk(child, child_reach)
        return values


def subgame_exploitability(
    tree: PublicTree,
    strategies: StrategyMap,
    reach: np.ndarray,
    leaf_value_fn: Optional[LeafValueFn] = None,
    space: Optional[HandSpace] = None,
) -> Tuple[float, List[float]]:
    """How much both best responders beat ``strategies`` inside this subgame.

    Returns the total (the sum over players, which is zero when the strategy is
    an equilibrium of the subgame given these ranges) and the per-player values.
    """
    values = [
        RangeBestResponse(tree, strategies, player, leaf_value_fn, space).value(reach)
        for player in range(NUM_PLAYERS)
    ]
    return float(sum(values)), values
