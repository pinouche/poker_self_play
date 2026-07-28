"""Solve many river situations at once, on one shared tree.

Data generation is the bottleneck for a value network, and a single river solve
is small: eight decision nodes, thirteen terminals, ranges of 1,326 floats.  At
that size the per-node Python work — refreshing strategies, discounting regrets,
dispatching on node type — costs about as much as the arithmetic it wraps, and
it is paid once per node per iteration no matter how much data is being pushed
through.

So push more through.  Situations that share a board and a betting tree can be
solved simultaneously by giving every array a leading batch axis: the reaches
become ``(K, 2, 1326)``, the regrets ``(K, 1326, actions)``, and the traversal
runs once for all K.  The arithmetic scales with K; the Python does not.

Batching by *board* is what makes this legal — the sort order, the tie groups
and the per-card index the showdown depends on are all board-dependent and
therefore shared.  Pot is not a constraint: values scale linearly with the pot
at a fixed stack-to-pot ratio, so one solve at a reference pot serves every pot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from paradigm_b.core.cfr.tabular_cfr import CFRConfig
from paradigm_b.holdem.engine.combos import CARD_IN_COMBO, COMBO_CARDS, NUM_COMBOS, board_mask
from paradigm_b.holdem.engine.public_tree import PublicNode, PublicState, build_endgame_tree
from paradigm_b.holdem.engine.showdown import showdown_values_batch
from paradigm_b.holdem.engine.space import EndgameSpace

NUM_PLAYERS = 2


def compatible_mass_batch(reach: np.ndarray) -> np.ndarray:
    """``(K, 1326)`` opponent mass that does not block each hand."""
    masses = reach @ CARD_IN_COMBO.T  # (K, 52)
    total = reach.sum(axis=1, keepdims=True)
    return total - masses[:, COMBO_CARDS[:, 0]] - masses[:, COMBO_CARDS[:, 1]] + reach


class BatchedRiverSolver:
    """Range-form CFR over a terminal-only tree, for K situations at once.

    Deliberately narrower than :class:`~search.subgame.SubgameSolver`: no depth
    limit, no value function, no re-solving gadget.  A river subgame needs none
    of them, and leaving them out keeps the traversal to array operations.
    """

    def __init__(
        self,
        space: EndgameSpace,
        root: PublicState,
        batch_size: int,
        config: CFRConfig | None = None,
        dtype: np.dtype = np.float64,
    ) -> None:
        self.space = space
        self.tree = build_endgame_tree(root, depth_limit=None)
        if self.tree.leaves():
            raise ValueError("a batched river solve must run to real terminals")
        self.batch_size = batch_size
        self.config = config or CFRConfig.dcfr()
        self.board = tuple(root.board)
        # Single precision roughly halves the memory traffic, and this is
        # bandwidth-bound arithmetic on arrays far too large for cache.  The
        # targets it produces are regression labels, not a converged strategy,
        # so the precision it costs does not propagate anywhere.
        self.dtype = dtype
        self.mask = board_mask(self.board).astype(dtype)

        self.decisions = self.tree.decision_nodes()
        shape = lambda node: (batch_size, NUM_COMBOS, node.num_actions)  # noqa: E731
        self.regrets = {n.index: np.zeros(shape(n), dtype=dtype) for n in self.decisions}
        self.strategy = {
            n.index: np.full(shape(n), 1.0 / n.num_actions, dtype=dtype)
            for n in self.decisions
        }
        self.iteration = 0
        self._value_sum = np.zeros((batch_size, NUM_PLAYERS, NUM_COMBOS), dtype=dtype)
        self._value_weight = 0.0

    # --- driving -----------------------------------------------------------
    def solve(self, reach: np.ndarray, iterations: int) -> np.ndarray:
        """Run CFR for every situation; returns ``(K, 2, 1326)`` root values."""
        reach = np.asarray(reach, dtype=self.dtype)
        if reach.shape != (self.batch_size, NUM_PLAYERS, NUM_COMBOS):
            raise ValueError(f"reach must be {(self.batch_size, NUM_PLAYERS, NUM_COMBOS)}")
        for _ in range(iterations):
            self.iteration += 1
            self._refresh()
            self._update_player = (self.iteration - 1) % NUM_PLAYERS
            values = self._walk(self.tree.root, reach[:, 0].copy(), reach[:, 1].copy())
            self._value_sum += values
            self._value_weight += 1.0
            self._discount()
        return self.root_values()

    def root_values(self) -> np.ndarray:
        if self._value_weight == 0.0:
            return np.zeros((self.batch_size, NUM_PLAYERS, NUM_COMBOS))
        return self._value_sum / self._value_weight

    def _refresh(self) -> None:
        for node in self.decisions:
            regrets = self.regrets[node.index]
            positive = np.maximum(regrets, 0.0)
            totals = positive.sum(axis=2, keepdims=True)
            self.strategy[node.index] = np.where(
                totals > 0.0,
                positive / np.where(totals > 0.0, totals, 1.0),
                1.0 / node.num_actions,
            )

    def _discount(self) -> None:
        cfg = self.config
        t = float(self.iteration)
        for node in self.decisions:
            regrets = self.regrets[node.index]
            if cfg.plus:
                np.maximum(regrets, 0.0, out=regrets)
            positive = (
                t**cfg.regret_alpha / (t**cfg.regret_alpha + 1.0)
                if cfg.regret_alpha is not None
                else 1.0
            )
            negative = (
                t**cfg.regret_beta / (t**cfg.regret_beta + 1.0)
                if cfg.regret_beta is not None
                else 1.0
            )
            if positive != 1.0 or negative != 1.0:
                # One pass, no mask allocation and no fancy indexing.
                regrets *= np.where(regrets > 0.0, positive, negative)
        if cfg.strategy_gamma is not None:
            factor = (t / (t + 1.0)) ** cfg.strategy_gamma
            self._value_sum *= factor
            self._value_weight *= factor

    # --- the traversal -----------------------------------------------------
    def _walk(
        self, node: PublicNode, reach_zero: np.ndarray, reach_one: np.ndarray
    ) -> np.ndarray:
        if node.is_terminal:
            return self._terminal(node.public, reach_zero, reach_one)

        player = node.player
        opponent = 1 - player
        mine = reach_zero if player == 0 else reach_one
        theirs = reach_one if player == 0 else reach_zero
        strategy = self.strategy[node.index]
        values = np.zeros(
            (self.batch_size, NUM_PLAYERS, NUM_COMBOS), dtype=self.dtype
        )
        # (K, hands, actions) throughout, so the sums below are plain products
        # rather than transposes.
        action_values = np.empty(
            (self.batch_size, NUM_COMBOS, node.num_actions), dtype=self.dtype
        )

        for j, child in enumerate(node.children):
            child_mine = mine * strategy[:, :, j]
            child_values = (
                self._walk(child, child_mine, theirs)
                if player == 0
                else self._walk(child, theirs, child_mine)
            )
            action_values[:, :, j] = child_values[:, player]
            values[:, opponent] += child_values[:, opponent]

        values[:, player] = (strategy * action_values).sum(axis=2)
        if self._update_player == player:
            self.regrets[node.index] += action_values - values[:, player][:, :, None]
        return values

    def _terminal(
        self, public: PublicState, reach_zero: np.ndarray, reach_one: np.ndarray
    ) -> np.ndarray:
        betting = public.betting
        values = np.empty(
            (self.batch_size, NUM_PLAYERS, NUM_COMBOS), dtype=self.dtype
        )
        correction = self.space.pair_correction
        opposing = (reach_one, reach_zero)
        if betting.folder >= 0:
            winner = 1 - betting.folder
            stake = abs(betting.fold_returns()[winner])
            for player in range(NUM_PLAYERS):
                sign = 1.0 if player == winner else -1.0
                values[:, player] = (
                    sign
                    * stake
                    * correction
                    * compatible_mass_batch(opposing[player])
                    * self.mask
                )
            return values
        stake = betting.showdown_stake()
        values[:, 0] = correction * showdown_values_batch(self.board, reach_one, stake)
        values[:, 1] = correction * showdown_values_batch(self.board, reach_zero, stake)
        return values
