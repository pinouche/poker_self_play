"""Exact best response and exploitability.

This is the ground truth of paradigm B.  Paradigm A could only ever ask "does
it beat that opponent?"; here we ask "how much does the *best possible*
opponent win?", which is a property of the strategy alone.  For a two-player
zero-sum game the exploitability of a profile is

    (BR_0(sigma_1) + BR_1(sigma_0)) / 2

in chips per hand — zero exactly at a Nash equilibrium, and it is the number
every stage of the roadmap is judged by.

The best response is computed in two passes over the tree.  First the
counterfactual reach of every history under the *opponents'* strategy (chance
included).  Then a memoised bottom-up pass: at each of the responder's
information sets the action is chosen once, maximising the reach-weighted sum
of the values of every history in that set — the responder cannot tell those
histories apart, so it must commit to a single action for all of them.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from cfr.policy import TabularPolicy
from game.tree import GameTree, Node


class BestResponse:
    """A best response to ``policy`` for one player."""

    def __init__(self, tree: GameTree, policy: TabularPolicy, player: int) -> None:
        self.tree = tree
        self.policy = policy
        self.player = player
        self._cf_reach = np.zeros(tree.num_nodes)
        self._value: Dict[int, float] = {}
        self._choice: Dict[int, int] = {}
        self._accumulate_reach(tree.root, 1.0)
        self.value = self._node_value(tree.root)

    # --- pass 1: how often do the others let us get here? ------------------
    def _accumulate_reach(self, node: Node, reach: float) -> None:
        self._cf_reach[node.index] = reach
        if node.is_terminal:
            return
        if node.is_chance:
            for child, prob in zip(node.children, node.chance_probs):
                self._accumulate_reach(child, reach * prob)
            return
        if node.player == self.player:
            # Our own probabilities are excluded from a counterfactual reach.
            for child in node.children:
                self._accumulate_reach(child, reach)
            return
        probs = self.policy.action_probs(node.infoset)
        for child, prob in zip(node.children, probs):
            self._accumulate_reach(child, reach * prob)

    # --- pass 2: bottom-up values, one decision per information set --------
    def _node_value(self, node: Node) -> float:
        cached = self._value.get(node.index)
        if cached is not None:
            return cached

        if node.is_terminal:
            value = float(node.payoffs[self.player])
        elif node.is_chance:
            value = sum(
                prob * self._node_value(child)
                for child, prob in zip(node.children, node.chance_probs)
            )
        elif node.player != self.player:
            probs = self.policy.action_probs(node.infoset)
            value = sum(
                float(prob) * self._node_value(child)
                for child, prob in zip(node.children, probs)
                if prob > 0.0
            )
        else:
            value = self._node_value(node.children[self._decide(node.infoset)])

        self._value[node.index] = value
        return value

    def _decide(self, infoset: int) -> int:
        """Pick the single action this information set commits to."""
        cached = self._choice.get(infoset)
        if cached is not None:
            return cached
        nodes = self.tree.infoset_nodes[infoset]
        num_actions = self.tree.num_actions(infoset)
        totals = np.zeros(num_actions)
        for node in nodes:
            reach = self._cf_reach[node.index]
            if reach == 0.0:
                continue
            for j in range(num_actions):
                totals[j] += reach * self._node_value(node.children[j])
        choice = int(np.argmax(totals))
        self._choice[infoset] = choice
        return choice

    def policy_table(self) -> TabularPolicy:
        """The best response as a (deterministic) policy over the whole tree."""
        probs = np.zeros((self.tree.num_infosets, self.tree.game.max_actions))
        for infoset in range(self.tree.num_infosets):
            n = self.tree.num_actions(infoset)
            if self.tree.infoset_player[infoset] == self.player:
                probs[infoset, self._decide(infoset)] = 1.0
            else:
                probs[infoset, :n] = self.policy.probs[infoset, :n]
        return TabularPolicy(tree=self.tree, probs=probs)


def best_response_value(tree: GameTree, policy: TabularPolicy, player: int) -> float:
    """Value to ``player`` of playing a best response to the rest of ``policy``."""
    return BestResponse(tree, policy, player).value


def best_response_values(tree: GameTree, policy: TabularPolicy) -> List[float]:
    return [
        best_response_value(tree, policy, p) for p in range(tree.game.num_players)
    ]


def exploitability(tree: GameTree, policy: TabularPolicy) -> float:
    """Distance to Nash in chips per hand (0 at equilibrium, two-player zero-sum)."""
    values = best_response_values(tree, policy)
    return float(sum(values) / len(values))


def expected_values(tree: GameTree, policy: TabularPolicy) -> np.ndarray:
    """Expected payoff per seat when everyone follows ``policy``."""
    return expected_values_at(tree.root, policy)


def expected_values_at(node: Node, policy: TabularPolicy) -> np.ndarray:
    """Expected payoff per seat from ``node`` onwards under ``policy``.

    Conditioning on a subtree is how per-hand values are checked: the value of
    the root's first chance child is the expected payoff given that deal.
    """
    return _expected_values(node, policy)


def _expected_values(node: Node, policy: TabularPolicy) -> np.ndarray:
    if node.is_terminal:
        return node.payoffs
    if node.is_chance:
        acc = None
        for child, prob in zip(node.children, node.chance_probs):
            v = _expected_values(child, policy) * prob
            acc = v if acc is None else acc + v
        return acc
    probs = policy.action_probs(node.infoset)
    acc = None
    for child, prob in zip(node.children, probs):
        if prob == 0.0:
            continue
        v = _expected_values(child, policy) * float(prob)
        acc = v if acc is None else acc + v
    if acc is None:  # every action has zero probability: should not happen
        raise ValueError("policy assigns no probability at an information set")
    return acc
