"""Convert range-form solutions into world-tree policies.

The range solver produces, at every public node, a strategy per hand.  The
stage-0 exploitability routine wants a strategy per *world* information set.
The two line up exactly: a Leduc information set is (my rank, the board rank,
the betting), and a public node plus a hand determines all three.  Suits are
strategically irrelevant, so the two cards of a rank get the same strategy and
averaging them is a no-op that doubles as a symmetry check.

This is what makes the whole roadmap measurable: anything that produces
behaviour on the public tree — a single range solve, or a chain of depth-limited
re-solves driven by a value network — becomes a tabular policy that the exact
best-response routine can score.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from belief.public_tree import PublicState
from cfr.policy import TabularPolicy
from game.actions import history_str
from game.leduc import NUM_CARDS, rank_of
from game.tree import GameTree
from search.subgame import SubgameSolver

# Strategy per public state: a ``(NUM_HANDS, num_actions)`` matrix, actions in
# the order given by ``public.legal_actions()``.
StrategyMap = Dict[PublicState, np.ndarray]


def world_infoset_key(public: PublicState, hand: int):
    """The world-tree information-set key seen by the player to act."""
    board = public.board
    return (
        rank_of(hand),
        rank_of(board) if board >= 0 else -1,
        history_str(public.betting.history[0]),
        history_str(public.betting.history[1]),
    )


def strategy_map(solver: SubgameSolver, average: bool = True) -> StrategyMap:
    """Read every decision node's strategy out of a solved public tree."""
    return {
        node.public: (
            solver.average_strategy(node) if average else solver.current_strategy(node)
        )
        for node in solver.tree.decision_nodes()
    }


def tabular_policy_from_strategies(
    strategies: StrategyMap, world_tree: GameTree
) -> TabularPolicy:
    """Project public-state strategies onto ``world_tree``.

    Public states the strategies do not cover keep the uniform default; that is
    only ever the case for parts of the tree the caller never solved.
    """
    policy = TabularPolicy.uniform(world_tree)
    totals = np.zeros((world_tree.num_infosets, world_tree.game.max_actions))
    counts = np.zeros(world_tree.num_infosets)

    for public, strategy in strategies.items():
        num_actions = strategy.shape[1]
        board = public.board
        for hand in range(NUM_CARDS):
            if hand == board:
                continue  # that card is on the table; nobody holds it
            infoset = world_tree.index_of_key.get(world_infoset_key(public, hand))
            if infoset is None:
                continue
            totals[infoset, :num_actions] += strategy[hand]
            counts[infoset] += 1.0

    covered = counts > 0
    policy.probs[covered] = totals[covered] / counts[covered, None]
    policy.check_valid()
    return policy


def extract_tabular_policy(
    solver: SubgameSolver, world_tree: GameTree, average: bool = True
) -> TabularPolicy:
    """Project a single solved public tree onto ``world_tree``."""
    return tabular_policy_from_strategies(strategy_map(solver, average), world_tree)


def suit_symmetry_error(solver: SubgameSolver) -> float:
    """Largest disagreement between the two suits of a rank.

    Zero up to floating point in any correct solve: the deal is suit-symmetric,
    so nothing can distinguish the jack of one suit from the jack of the other.
    """
    worst = 0.0
    for node in solver.tree.decision_nodes():
        strategy = solver.average_strategy(node)
        for rank_start in range(0, NUM_CARDS, 2):
            first, second = rank_start, rank_start + 1
            if node.public.board in (first, second):
                continue
            worst = max(worst, float(np.abs(strategy[first] - strategy[second]).max()))
    return worst
