"""Counterfactual regret minimisation over an explicit game tree.

One traversal per iteration, full-width (no sampling).  At every information
set of the traversing player the algorithm compares the value of each action
against the value of the current strategy, weights the difference by the
*counterfactual* reach — the probability the opponents and chance let the
player arrive there — and accumulates it as regret.  Regret matching turns the
accumulated regret back into a strategy, and the **time-average** of those
strategies converges to a Nash equilibrium in two-player zero-sum games.

Four variants share the traversal, differing only in how regret and the
average are weighted over time:

``vanilla``   Zinkevich et al. 2007.  Nothing is discounted; converges at the
              theoretical O(1/sqrt(T)) and no faster.
``cfr_plus``  CFR+ (Tammelin 2014).  Cumulative regret is floored at zero, so
              an action recovers the instant it looks good again, and iteration
              *t* gets weight *t* in the average.
``linear``    Linear CFR (Brown & Sandholm 2019).  Regret and average both
              weighted by *t*.
``dcfr``      Discounted CFR, same paper: positive regret discounted by
              t^1.5/(t^1.5+1), negative regret halved every iteration, average
              weighted by t^2.  The fastest of the four on Leduc by ~10x, and
              the default here.

One subtlety that costs an order of magnitude if you get it wrong: the profile
must be frozen for the duration of a traversal.  An information set is reached
from many histories, and all of them have to be evaluated against the *same*
strategy or the accumulated quantity stops being a counterfactual regret.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from paradigm_b.core.cfr.policy import TabularPolicy
from paradigm_b.core.game.tree import GameTree, Node


@dataclass
class CFRConfig:
    """Weighting scheme.  Use the named constructors rather than the fields."""

    plus: bool = False  # floor cumulative regret at zero (regret matching+)
    alternating: bool = True  # update one player per iteration
    linear_averaging: bool = False  # weight iteration t by t in the average
    # Added to that weight, so iteration *t* counts ``t + average_offset``.
    # CFR+ conventionally uses weight *t*; ReBeL's Algorithm 2 writes its
    # averages as the recursion ``x_bar <- (t/(t+2))*x_bar + (2/(t+2))*x_t``,
    # which is the same thing with weight *t+1*.  Solving the recursion:
    # matching ``w_t / W_t = 2/(t+2)`` and ``W_{t-1}/W_t = t/(t+2)`` needs
    # ``W_t = (t+1)(t+2)/2``, hence ``w_t = W_t - W_{t-1} = t + 1``.
    average_offset: int = 0
    regret_alpha: Optional[float] = None  # discount exponent, positive regret
    regret_beta: Optional[float] = None  # discount exponent, negative regret
    strategy_gamma: Optional[float] = None  # discount exponent, average strategy

    @staticmethod
    def vanilla() -> "CFRConfig":
        return CFRConfig(plus=False, alternating=False)

    @staticmethod
    def cfr_plus() -> "CFRConfig":
        return CFRConfig(plus=True, alternating=True, linear_averaging=True)

    @staticmethod
    def linear() -> "CFRConfig":
        return CFRConfig(
            alternating=True, regret_alpha=1.0, regret_beta=1.0, strategy_gamma=1.0
        )

    @staticmethod
    def linear_cfr_d() -> "CFRConfig":
        """The weighting ReBeL's Algorithm 2 (Linear CFR-D) specifies.

        Regret is weighted by *t* — Linear CFR, which is Discounted CFR with
        alpha = beta = 1 and is implemented here as the equivalent per-iteration
        discount ``t/(t+1)``, so regret accrued at iteration *s* retains
        ``prod_{u=s..T} u/(u+1) = s/(T+1)``.

        Both averages — the average strategy and the average root value the
        value network trains on — are weighted explicitly by ``t + 1`` rather
        than by discounting, which is the pseudocode's ``(t/(t+2))`` recursion
        exactly.  ``strategy_gamma`` is therefore ``None``: discounting *and*
        weighting would count the schedule twice.

        This is not the same as :meth:`dcfr`, which is faster on Leduc but
        weights iteration *s* by ``s^2``.  Use dcfr when you want the strongest
        solver; use this when you want the paper's algorithm.
        """
        return CFRConfig(
            plus=False,
            alternating=True,
            linear_averaging=True,
            average_offset=1,
            regret_alpha=1.0,
            regret_beta=1.0,
            strategy_gamma=None,
        )

    @staticmethod
    def dcfr() -> "CFRConfig":
        return CFRConfig(
            alternating=True, regret_alpha=1.5, regret_beta=0.0, strategy_gamma=2.0
        )

    def average_weight(self, iteration: float) -> float:
        """Weight iteration ``t`` carries in the strategy and value averages."""
        if not self.linear_averaging:
            return 1.0
        return float(iteration) + float(self.average_offset)

    def cumulative_average_weight(self, iteration: float) -> float:
        """Total average weight ``sum_{t=1..iteration} average_weight(t)``.

        What a warm start has to seed so that resuming at ``t_warm`` continues
        the same running average rather than restarting it.
        """
        if not self.linear_averaging:
            return float(iteration)
        offset = float(self.average_offset)
        total = float(iteration)
        return total * (total + 1.0) / 2.0 + offset * total

    def initial_average_weight(self, warm_iterations: float = 0.0) -> float:
        """Weight carried by the value/strategy average *before* the loop runs.

        Algorithm 2 seeds ``v(beta_r) = COMPUTE_EV(G, pi_{t_warm})`` and
        ``pi_bar = pi_{t_warm}`` before iterating, and the ``(t/(t+2))``
        recursion then decays that seed.  For the recursion to hold, the seed
        must enter with weight ``W_{t_warm} = (t_warm+1)(t_warm+2)/2``, which is
        one more than the weight ``t_warm`` real iterations would have
        accumulated — with no warm start that is a weight of exactly 1.
        """
        if not self.linear_averaging:
            return 1.0
        return 1.0 + self.cumulative_average_weight(warm_iterations)


class CFRSolver:
    """Tabular CFR on a pre-expanded :class:`GameTree`."""

    def __init__(self, tree: GameTree, config: CFRConfig | None = None) -> None:
        self.tree = tree
        self.config = config or CFRConfig.dcfr()
        num_infosets = tree.num_infosets
        max_actions = tree.game.max_actions
        self.regrets = np.zeros((num_infosets, max_actions))
        self.strategy_sum = np.zeros((num_infosets, max_actions))
        self._strategy = np.zeros((num_infosets, max_actions))
        self._num_actions = np.array(
            [tree.num_actions(i) for i in range(num_infosets)], dtype=np.int64
        )
        self._legal = np.zeros((num_infosets, max_actions), dtype=bool)
        for i in range(num_infosets):
            self._legal[i, : self._num_actions[i]] = True
        self._uniform = self._legal / self._num_actions[:, None]
        self._strategy[:] = self._uniform
        self.iteration = 0

    # --- driving -----------------------------------------------------------
    def iterate(self, iterations: int = 1) -> None:
        for _ in range(iterations):
            self.iteration += 1
            self._refresh_strategy()
            weight = self.config.average_weight(self.iteration)
            players = (
                [(self.iteration - 1) % self.tree.game.num_players]
                if self.config.alternating
                else list(range(self.tree.game.num_players))
            )
            for player in players:
                self._walk(self.tree.root, 1.0, 1.0, player, weight)
            self._discount()

    def _refresh_strategy(self) -> None:
        positive = np.maximum(self.regrets, 0.0) * self._legal
        totals = positive.sum(axis=1, keepdims=True)
        np.divide(positive, np.where(totals > 0.0, totals, 1.0), out=self._strategy)
        empty = (totals <= 0.0).ravel()
        self._strategy[empty] = self._uniform[empty]

    def _discount(self) -> None:
        """Down-weight the past, as Linear/Discounted CFR prescribe."""
        cfg = self.config
        t = float(self.iteration)
        if cfg.regret_alpha is not None:
            factor = t**cfg.regret_alpha / (t**cfg.regret_alpha + 1.0)
            positive = self.regrets > 0.0
            self.regrets[positive] *= factor
        if cfg.regret_beta is not None:
            factor = t**cfg.regret_beta / (t**cfg.regret_beta + 1.0)
            negative = self.regrets < 0.0
            self.regrets[negative] *= factor
        if cfg.strategy_gamma is not None:
            self.strategy_sum *= (t / (t + 1.0)) ** cfg.strategy_gamma

    # --- the traversal -----------------------------------------------------
    def _walk(
        self, node: Node, my_reach: float, opp_reach: float, player: int, weight: float
    ) -> float:
        """Expected value of ``node`` for ``player`` under the current profile.

        ``my_reach`` is ``player``'s own contribution to the reach probability;
        ``opp_reach`` is everyone else's, chance included.  Regrets need the
        latter only — that is what makes them *counterfactual*.
        """
        if node.is_terminal:
            return float(node.payoffs[player])

        if node.is_chance:
            value = 0.0
            for child, prob in zip(node.children, node.chance_probs):
                value += prob * self._walk(child, my_reach, opp_reach * prob, player, weight)
            return value

        infoset = node.infoset
        num_actions = int(self._num_actions[infoset])
        strategy = self._strategy[infoset, :num_actions]

        if node.player != player:
            value = 0.0
            for j in range(num_actions):
                p = strategy[j]
                if p == 0.0:
                    continue
                value += p * self._walk(
                    node.children[j], my_reach, opp_reach * p, player, weight
                )
            return value

        # The traversing player's own decision: evaluate every action.
        action_values = np.empty(num_actions)
        for j in range(num_actions):
            action_values[j] = self._walk(
                node.children[j], my_reach * strategy[j], opp_reach, player, weight
            )
        value = float(strategy @ action_values)

        regrets = self.regrets[infoset, :num_actions]
        regrets += opp_reach * (action_values - value)
        if self.config.plus:
            np.maximum(regrets, 0.0, out=regrets)
        self.strategy_sum[infoset, :num_actions] += weight * my_reach * strategy
        return value

    # --- read-out ----------------------------------------------------------
    def current_policy(self) -> TabularPolicy:
        self._refresh_strategy()
        return TabularPolicy(tree=self.tree, probs=self._strategy.copy())

    def average_policy(self) -> TabularPolicy:
        """The time-average strategy — the one that converges to equilibrium."""
        totals = self.strategy_sum.sum(axis=1, keepdims=True)
        probs = np.where(
            totals > 0.0, self.strategy_sum / np.where(totals > 0.0, totals, 1.0), self._uniform
        )
        return TabularPolicy(tree=self.tree, probs=probs)
