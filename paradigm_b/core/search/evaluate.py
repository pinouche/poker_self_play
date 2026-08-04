"""Evaluate a fixed strategy in range form.

No regret, no updates: given a strategy for every public decision node, what is
each hand worth?  This is the value function *of a policy*, as opposed to the
value function of an equilibrium, and the distinction turns out to matter a
great deal.

A depth-limited search is only sound if the values it uses at the depth limit
are the values of the continuation that will actually be played.  Handing it the
*equilibrium* value of each leaf subgame is not enough: the total value of a
subgame is pinned down by the arriving ranges, but the per-hand counterfactual
values are not — different equilibria of the same subgame split that total
differently across hands, and the trunk's decisions are made hand by hand.  Feed
it one split and then play another and the trunk's assumptions are silently
violated.

So this module provides the consistent alternative: pick a continuation policy,
evaluate *it*, and search against those numbers.  That is also exactly the
target ReBeL's value network converges to — the value of the policy its own
search produces.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

import numpy as np

from paradigm_b.core.belief.public_tree import PublicNode, PublicTree
from paradigm_b.core.belief.ranges import NUM_PLAYERS, PBS
from paradigm_b.core.search.space import LEDUC_SPACE, HandSpace
from paradigm_b.core.search.policy import StrategyMap
from paradigm_b.core.search.subgame import LeafValueFn, terminal_values


def range_values(
    tree: PublicTree,
    strategies: StrategyMap,
    reach: np.ndarray,
    leaf_value_fn: Optional[LeafValueFn] = None,
    space: Optional[HandSpace] = None,
) -> np.ndarray:
    """``(2, num_hands)`` counterfactual values of ``strategies`` at the root."""
    return _walk(
        tree.root,
        np.asarray(reach, dtype=np.float64).copy(),
        strategies,
        leaf_value_fn,
        space or LEDUC_SPACE,
    )


def leaf_reaches(
    tree: PublicTree,
    strategies: StrategyMap,
    reach: np.ndarray,
    space: Optional[HandSpace] = None,
) -> List[Tuple[PublicNode, np.ndarray]]:
    """Reach vectors at every leaf of ``tree`` under ``strategies``."""
    out: List[Tuple[PublicNode, np.ndarray]] = []
    _descend(
        tree.root,
        np.asarray(reach, dtype=np.float64).copy(),
        strategies,
        out,
        space=space or LEDUC_SPACE,
    )
    return out


def decision_reaches(
    tree: PublicTree,
    strategies: StrategyMap,
    reach: np.ndarray,
    space: Optional[HandSpace] = None,
) -> List[Tuple[PublicNode, np.ndarray]]:
    """Reach vectors at *every* decision node, descending through them.

    :func:`reaches_at` stops at the states it is asked for, which is what
    continual re-solving wants; this one keeps going, which is what ReBeL's
    ``for beta in G: add {beta, pi_bar(beta)} to D_pi`` wants — every public
    belief state the subgame contains, each with the ranges that arrive there
    under ``strategies``.
    """
    out: List[Tuple[PublicNode, np.ndarray]] = []
    _descend_decisions(
        tree.root,
        np.asarray(reach, dtype=np.float64).copy(),
        strategies,
        out,
        space or LEDUC_SPACE,
    )
    return out


def _descend_decisions(
    node: PublicNode,
    reach: np.ndarray,
    strategies: StrategyMap,
    out: List[Tuple[PublicNode, np.ndarray]],
    space: HandSpace,
) -> None:
    if node.is_terminal or node.is_leaf:
        return
    if node.is_chance:
        for board, child in zip(node.boards, node.children):
            child_reach = reach * space.deal_mask(board)
            if child_reach.sum() > 0.0:
                _descend_decisions(child, child_reach, strategies, out, space)
        return
    out.append((node, reach))
    strategy = strategies[node.public]
    for j, child in enumerate(node.children):
        child_reach = reach.copy()
        child_reach[node.player] = reach[node.player] * strategy[:, j]
        _descend_decisions(child, child_reach, strategies, out, space)


def reaches_at(
    tree: PublicTree,
    strategies: StrategyMap,
    reach: np.ndarray,
    wanted: Set,
    space: Optional[HandSpace] = None,
) -> List[Tuple[PublicNode, np.ndarray]]:
    """Reach vectors at a chosen set of public states, which are not descended past."""
    out: List[Tuple[PublicNode, np.ndarray]] = []
    _descend(
        tree.root,
        np.asarray(reach, dtype=np.float64).copy(),
        strategies,
        out,
        stop_at=wanted,
        space=space or LEDUC_SPACE,
    )
    return out


def _descend(
    node: PublicNode,
    reach: np.ndarray,
    strategies: StrategyMap,
    out: List[Tuple[PublicNode, np.ndarray]],
    stop_at: Optional[Set] = None,
    space: Optional[HandSpace] = None,
) -> None:
    space = space or LEDUC_SPACE
    if stop_at is not None and node.public in stop_at:
        out.append((node, reach))
        return
    if node.is_leaf:
        if stop_at is None:
            out.append((node, reach))
        return
    if node.is_terminal:
        return
    if node.is_chance:
        for board, child in zip(node.boards, node.children):
            child_reach = reach * space.deal_mask(board)
            if child_reach.sum() > 0.0:
                _descend(child, child_reach, strategies, out, stop_at, space)
        return
    strategy = strategies[node.public]
    for j, child in enumerate(node.children):
        child_reach = reach.copy()
        child_reach[node.player] = reach[node.player] * strategy[:, j]
        _descend(child, child_reach, strategies, out, stop_at, space)


def leaf_counterfactual_values(
    public,
    reach: np.ndarray,
    leaf_value_fn: LeafValueFn,
    space: Optional[HandSpace] = None,
) -> np.ndarray:
    """The counterfactual values a leaf was *promised*, as the solver saw them.

    Scaling a value function's per-hand output (which assumes normalised ranges)
    back up by the opponent's actual reach mass, plus the card-removal
    correction.  These are the numbers the re-solving gadget holds the next
    solve to.
    """
    space = space or LEDUC_SPACE
    predicted = np.asarray(leaf_value_fn([space.pbs(public, reach)]))[0]
    masses = reach.sum(axis=1)
    values = np.empty((NUM_PLAYERS, space.num_hands))
    values[0] = space.pair_correction * masses[1] * predicted[0]
    values[1] = space.pair_correction * masses[0] * predicted[1]
    return values


def _walk(
    node: PublicNode,
    reach: np.ndarray,
    strategies: StrategyMap,
    leaf_value_fn: Optional[LeafValueFn],
    space: HandSpace,
) -> np.ndarray:
    if node.is_terminal:
        return space.terminal_values(node.public, reach)

    if node.is_leaf:
        if leaf_value_fn is None:
            raise ValueError("a depth-limited tree needs a leaf value function")
        predicted = np.asarray(leaf_value_fn([space.pbs(node.public, reach)]))[0]
        masses = reach.sum(axis=1)
        values = np.empty((NUM_PLAYERS, space.num_hands))
        values[0] = space.pair_correction * masses[1] * predicted[0]
        values[1] = space.pair_correction * masses[0] * predicted[1]
        return values

    if node.is_chance:
        values = np.zeros((NUM_PLAYERS, space.num_hands))
        weight = space.chance_weight(node.public)
        for board, child in zip(node.boards, node.children):
            mask = space.deal_mask(board)
            child_reach = reach * mask
            if child_reach.sum() == 0.0:
                continue
            values += (
                weight * _walk(child, child_reach, strategies, leaf_value_fn, space) * mask
            )
        return values

    strategy = strategies[node.public]
    player = node.player
    opponent = 1 - player
    values = np.zeros((NUM_PLAYERS, space.num_hands))
    for j, child in enumerate(node.children):
        child_reach = reach.copy()
        child_reach[player] = reach[player] * strategy[:, j]
        child_values = _walk(child, child_reach, strategies, leaf_value_fn, space)
        values[player] += strategy[:, j] * child_values[player]
        values[opponent] += child_values[opponent]
    return values
