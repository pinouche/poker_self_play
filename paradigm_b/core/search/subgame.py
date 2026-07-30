"""Range-form CFR over a public tree, optionally depth-limited.

This is the search step of DeepStack and ReBeL, and the same code serves three
jobs:

* ``depth_limit=None`` with no leaf-value function — solve a whole (sub)game
  exactly.  Used to validate the belief machinery against the stage-0 tabular
  solver, and to generate ground-truth values for training.
* ``depth_limit=1`` with a value network — the real thing: solve the current
  betting round, and ask the network what the resulting belief states are worth
  instead of searching to the end of the game.
* Rooted at any PBS, not just the game root — which is what "continual
  re-solving" needs.

The bookkeeping to keep straight is whose probabilities are in which vector.
``reach[i]`` is player ``i``'s *own* contribution to the probability of being
here with each hand.  A counterfactual value for player ``i`` deliberately
excludes it: it answers "what would this be worth had I arrived here holding
``h``", so it is linear in the *opponent's* reach vector — one matrix product
per terminal node, all six hands at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from paradigm_b.core.belief.public_tree import PublicNode, PublicState, PublicTree, build_public_tree
from paradigm_b.core.belief.ranges import NUM_PLAYERS, PBS
from paradigm_b.core.cfr.tabular_cfr import CFRConfig
from paradigm_b.core.search.space import LEDUC_SPACE, HandSpace

# A leaf evaluator maps a *batch* of public belief states to per-hand values for
# both players, in chips, assuming the normalised ranges carried by each PBS:
# ``(N, 2, NUM_HANDS)``.  Batched because the caller is a neural network and a
# depth-limited solve queries every leaf on every iteration — thirty separate
# forward passes per iteration would dominate the run time.
LeafValueFn = Callable[[Sequence[PBS]], np.ndarray]


@dataclass
class Gadget:
    """Safe re-solving: let the opponent refuse the subgame.

    Re-solving a subgame from ranges alone is *unsafe*.  The total value of a
    subgame is pinned down by the arriving ranges, but the split of that total
    across the opponent's individual hands is not — and the strategy that led
    here was chosen on the assumption of one particular split.  Deliver a
    different one and the opponent can profit by steering into this subgame with
    the hands you shortchanged.  Measured on Leduc, re-solving the second round
    from equilibrium ranges takes exploitability from 0.002 to 0.117 chips.

    The fix (Burch et al. 2014; DeepStack's re-solving gadget) is to make the
    promise explicit.  Before the subgame starts, the opponent chooses, for each
    hand separately, whether to play it or to *opt out* and collect the
    counterfactual value the previous solve promised that hand.  Any strategy we
    adopt must therefore be good enough that opting out is not attractive, which
    is exactly the guarantee that was missing.

    ``player`` is whose strategy this solve is for; only that player's strategy
    should be read out of it.  The opponent's side of a gadget solve is a
    worst-case construction, not a policy.
    """

    player: int
    promised: np.ndarray  # (NUM_HANDS,) opponent counterfactual values

    @property
    def opponent(self) -> int:
        return 1 - self.player


@dataclass
class SubgameSolution:
    """What a solve returns: values at the root and the policy that produced them."""

    root_values: np.ndarray  # (2, NUM_HANDS) counterfactual values
    root_pbs: PBS
    iterations: int


@dataclass
class WarmStart:
    """What ``INITIALIZE_POLICY(G, theta_pi)`` produced, ready to hand to a solve.

    ReBeL's Algorithm 2 does not always start CFR from the uniform policy: when
    a policy network is available it initialises the subgame's profile to that
    network's prediction and starts the iteration counter at ``t_warm``, so the
    warm profile carries the weight ``t_warm`` iterations of search would have
    earned rather than being immediately averaged away.

    ``strategies`` and ``reaches`` are indexed by ``PublicNode.index`` and hold
    ``None`` for every node that is not a decision node.  ``reaches`` is needed
    because CFR's average strategy is reach-weighted: seeding it correctly
    requires knowing how much of each player's range arrives at each node under
    pi_0, which the initialiser computed on its way down the tree.
    """

    strategies: List[Optional[np.ndarray]]
    reaches: List[Optional[np.ndarray]]
    iterations: int  # t_warm


class SubgameSolver:
    """CFR over ranges on a public tree.

    A single traversal produces both players' counterfactual values, so both
    players can be updated from one walk.  ``config.alternating`` instead walks
    twice and updates one player each time, which converges faster for the same
    reason it does in the tabular solver — the responding player is always
    reacting to a profile that has just been improved.
    """

    def __init__(
        self,
        tree: PublicTree,
        leaf_value_fn: Optional[LeafValueFn] = None,
        config: CFRConfig | None = None,
        space: HandSpace | None = None,
    ) -> None:
        if tree.leaves() and leaf_value_fn is None:
            raise ValueError("a depth-limited tree needs a leaf value function")
        self.tree = tree
        self.leaf_value_fn = leaf_value_fn
        self.config = config or CFRConfig.dcfr()
        self.space = space or LEDUC_SPACE
        NUM_HANDS = self.space.num_hands
        num_nodes = tree.num_nodes
        self.regrets: List[Optional[np.ndarray]] = [None] * num_nodes
        self.strategy_sum: List[Optional[np.ndarray]] = [None] * num_nodes
        self.strategy: List[Optional[np.ndarray]] = [None] * num_nodes
        for node in tree.decision_nodes():
            shape = (self.space.num_hands, node.num_actions)
            self.regrets[node.index] = np.zeros(shape)
            self.strategy_sum[node.index] = np.zeros(shape)
            self.strategy[node.index] = np.full(shape, 1.0 / node.num_actions)
        self.iteration = 0
        self._weight = 1.0
        self._update_player: Optional[int] = None
        self._active_strategy = self.strategy
        self._leaf_cache: Dict[int, np.ndarray] = {}
        self._store_leaves = False
        # Per-iteration leaf frontiers, when asked for: ReBeL descends into a
        # leaf sampled from a *random* iteration's policy, not the average one.
        self.iteration_leaves: List[List[Tuple[PublicNode, np.ndarray]]] = []
        # The profile as it stood at the top of ``capture_strategy_at``, i.e.
        # pi_{t-1} for the sampled iteration t.  Algorithm 2 draws t_sample
        # *before* the loop and calls SAMPLE_LEAF once, at that iteration, so a
        # single snapshot is all a descent needs — keeping every iteration's
        # frontier instead costs a leaf-reach descent and an array per leaf per
        # iteration, for T-1 frontiers that are thrown away.
        self.captured_strategy: Optional[List[Optional[np.ndarray]]] = None
        self.gadget_regrets = np.zeros((self.space.num_hands, 2))
        # Values at the root, averaged over iterations: the ReBeL training target.
        self._value_sum = np.zeros((NUM_PLAYERS, self.space.num_hands))
        self._value_weight = 0.0
        self.iteration_values: List[np.ndarray] = []

    # --- driving -----------------------------------------------------------
    def solve(
        self,
        reach: np.ndarray | None = None,
        iterations: int = 100,
        store_iteration_values: bool = False,
        store_iteration_leaves: bool = False,
        gadget: Optional[Gadget] = None,
        warm_start: Optional[WarmStart] = None,
        capture_strategy_at: Optional[int] = None,
        seed_root_value: bool = False,
    ) -> SubgameSolution:
        """Run CFR in this subgame.

        ``iterations`` is how many iterations to *run*, except when
        ``warm_start`` is given: then it is read as Algorithm 2's ``T``, the
        iteration to finish at, and the loop runs ``T - t_warm`` times so that
        ``t`` covers ``t_warm+1 ... T`` exactly as the pseudocode writes it.

        ``capture_strategy_at`` is Algorithm 2's ``t_sample``: at the top of
        that iteration the current profile — pi_{t-1}, since regret matching
        for iteration ``t`` has not run yet — is snapshotted into
        :attr:`captured_strategy` for the caller's ``SAMPLE_LEAF``.

        ``seed_root_value`` adds the pre-loop ``v(beta_r) = COMPUTE_EV(G,
        pi_{t_warm})`` term to the value average.  It costs one extra traversal
        and one extra leaf-evaluator call per solve, and with no warm start it
        enters the average with weight 1 against a total of ``(T+1)(T+2)/2`` —
        0.1% at T=40.  It is off by default so that solvers which are not
        running Algorithm 2 (exact labelling, continual re-solving, the leaf
        oracles) do not pay for it.
        """
        reach = self.space.initial_reach() if reach is None else np.asarray(reach, dtype=np.float64)
        root_public = self.tree.root.public
        self.captured_strategy = None
        if gadget is not None:
            self.gadget_regrets = np.zeros((self.space.num_hands, 2))
        if warm_start is not None:
            self._apply_warm_start(warm_start)
            iterations = max(iterations - warm_start.iterations, 0)
        if seed_root_value:
            self._seed_root_value(reach)
        self._store_leaves = store_iteration_leaves
        for _ in range(iterations):
            self.iteration += 1
            if capture_strategy_at is not None and self.iteration == capture_strategy_at:
                self.captured_strategy = [
                    None if s is None else s.copy() for s in self.strategy
                ]
            self._refresh_strategies()
            weight = self.config.average_weight(self.iteration)
            self._weight = weight
            self._update_player = (
                (self.iteration - 1) % NUM_PLAYERS if self.config.alternating else None
            )
            iteration_reach = reach
            follow = None
            if gadget is not None:
                follow = self._gadget_follow_probabilities()
                iteration_reach = reach.copy()
                iteration_reach[gadget.opponent] = reach[gadget.opponent] * follow
            self._evaluate_leaves(iteration_reach)
            values = self._walk(self.tree.root, iteration_reach.copy())
            if gadget is not None:
                self._update_gadget(gadget, follow, values)
            self._value_sum += weight * values
            self._value_weight += weight
            if store_iteration_values:
                self.iteration_values.append(values.copy())
            self._discount()
        return SubgameSolution(
            root_values=self.root_values(),
            root_pbs=self.space.pbs(root_public, reach),
            iterations=self.iteration,
        )

    # --- Algorithm 2's initialisation --------------------------------------
    def _apply_warm_start(self, warm: WarmStart) -> None:
        """Adopt pi_0 from a policy network and start counting at ``t_warm``.

        Three things have to line up for the warm profile to *stay* warm.

        The current profile is pi_0 outright.  The accumulated regret is set to
        ``t_warm * pi_0``: regret matching normalises, so any positive multiple
        of pi_0 reproduces pi_0 exactly, and the multiple is what decides how
        many real iterations of contrary evidence it takes to move away from it
        — scaling by ``t_warm`` makes that cost the same as if the warm profile
        had genuinely won ``t_warm`` iterations of regret.

        The average strategy is seeded with pi_0 at the weight ``t_warm``
        iterations would have earned, reach-weighted like every other
        contribution to it, so ``pi_bar`` starts at pi_0 rather than at zero.
        """
        warm_iterations = max(int(warm.iterations), 0)
        weight = self.config.initial_average_weight(warm_iterations)
        for node in self.tree.decision_nodes():
            strategy = warm.strategies[node.index]
            if strategy is None:
                continue
            strategy = np.asarray(strategy, dtype=np.float64)
            self.strategy[node.index] = strategy.copy()
            self.regrets[node.index] = strategy * float(warm_iterations)
            node_reach = warm.reaches[node.index]
            own = (
                np.ones((self.space.num_hands, 1))
                if node_reach is None
                else np.asarray(node_reach, dtype=np.float64)[node.player][:, None]
            )
            self.strategy_sum[node.index] = weight * own * strategy
        self.iteration = warm_iterations

    def _seed_root_value(self, reach: np.ndarray) -> None:
        """Algorithm 2's pre-loop ``v(beta_r) = COMPUTE_EV(G, pi_{t_warm})``.

        The value average starts from the expected value of the initial policy
        — uniform with no warm start, the policy network's profile with one —
        carrying the weight the ``(t/(t+2))`` recursion assigns it.
        """
        weight = self.config.initial_average_weight(self.iteration)
        self._value_sum += weight * self.evaluate(reach, average=False)
        self._value_weight += weight

    def evaluate(
        self, reach: np.ndarray | None = None, average: bool = True
    ) -> np.ndarray:
        """Counterfactual values of a *fixed* strategy, updating nothing.

        The values of the average strategy, which is not the same object as the
        average of the per-iteration values in :meth:`root_values`.  Used to
        check the range machinery against the world tree, and to produce exact
        value targets for a leaf.
        """
        reach = self.space.initial_reach() if reach is None else np.asarray(reach, dtype=np.float64)
        strategies: List[Optional[np.ndarray]] = [None] * self.tree.num_nodes
        for node in self.tree.decision_nodes():
            strategies[node.index] = (
                self.average_strategy(node) if average else self.current_strategy(node)
            )
        previous_strategy, previous_update = self._active_strategy, self._update_player
        self._active_strategy, self._update_player = strategies, -1
        try:
            self._evaluate_leaves(reach)
            return self._walk(self.tree.root, reach.copy())
        finally:
            self._active_strategy = previous_strategy
            self._update_player = previous_update

    def reset_values(self) -> None:
        """Forget the accumulated value average, keeping the regrets.

        For a solver that is re-used across calls with *different* ranges: the
        strategy it has learned is still worth keeping, but values computed for
        the ranges of a previous call describe a different question.
        """
        self._value_sum[:] = 0.0
        self._value_weight = 0.0

    def root_values(self) -> np.ndarray:
        """The iteration-averaged counterfactual values at the root.

        ReBeL trains the value network on exactly this quantity, and it is also
        what search should be fed at a depth limit — *not* the values of the
        average strategy, which :meth:`evaluate` returns.  The difference matters
        for the hands a player never actually holds here.

        A reach-weighted average strategy is undefined for a hand with zero
        reach, so it falls back to uniform — i.e. to nonsense.  Those are exactly
        the hands whose counterfactual values a parent node's regrets depend on:
        "how much would folding this hand have cost me" is a question about a
        hand you are currently never continuing with.  Per-iteration values come
        from regret-matched strategies, which are well defined for every hand, so
        averaging those keeps the whole vector meaningful.
        """
        if self._value_weight == 0.0:
            return np.zeros((NUM_PLAYERS, self.space.num_hands))
        return self._value_sum / self._value_weight

    # --- the re-solving gadget ---------------------------------------------
    def _gadget_follow_probabilities(self) -> np.ndarray:
        """Per hand, how much of the opponent's range enters the subgame."""
        positive = np.maximum(self.gadget_regrets, 0.0)
        totals = positive.sum(axis=1)
        follow = np.where(totals > 0.0, positive[:, 0] / np.where(totals > 0.0, totals, 1.0), 0.5)
        return follow

    def _update_gadget(
        self, gadget: Gadget, follow: np.ndarray, values: np.ndarray
    ) -> None:
        """Regret-match the opponent's opt-out decision against its promise."""
        entering = values[gadget.opponent]
        opting_out = gadget.promised
        realised = follow * entering + (1.0 - follow) * opting_out
        self.gadget_regrets[:, 0] += entering - realised
        self.gadget_regrets[:, 1] += opting_out - realised
        if self.config.plus:
            np.maximum(self.gadget_regrets, 0.0, out=self.gadget_regrets)

    def _refresh_strategies(self) -> None:
        for node in self.tree.decision_nodes():
            regrets = self.regrets[node.index]
            positive = np.maximum(regrets, 0.0)
            totals = positive.sum(axis=1, keepdims=True)
            strategy = np.where(
                totals > 0.0,
                positive / np.where(totals > 0.0, totals, 1.0),
                1.0 / node.num_actions,
            )
            self.strategy[node.index] = strategy

    def _discount(self) -> None:
        cfg = self.config
        t = float(self.iteration)
        for node in self.tree.decision_nodes():
            regrets = self.regrets[node.index]
            if cfg.plus:
                np.maximum(regrets, 0.0, out=regrets)
            if cfg.regret_alpha is not None:
                factor = t**cfg.regret_alpha / (t**cfg.regret_alpha + 1.0)
                regrets[regrets > 0.0] *= factor
            if cfg.regret_beta is not None:
                factor = t**cfg.regret_beta / (t**cfg.regret_beta + 1.0)
                regrets[regrets < 0.0] *= factor
            if cfg.strategy_gamma is not None:
                self.strategy_sum[node.index] *= (t / (t + 1.0)) ** cfg.strategy_gamma
        if cfg.strategy_gamma is not None:
            # The value average is discounted like the strategy average, so that
            # the reported root values describe the late (converged) iterates
            # rather than being dragged down by the first few.
            factor = (t / (t + 1.0)) ** cfg.strategy_gamma
            self._value_sum *= factor
            self._value_weight *= factor

    # --- leaves ------------------------------------------------------------
    def leaf_reaches(
        self, reach: np.ndarray, average: bool = False
    ) -> List[Tuple[PublicNode, np.ndarray]]:
        """Reach vectors at every leaf under the current (or average) strategy.

        The same descent the value traversal does, but carrying probabilities
        down instead of values up.  Used to build the leaf belief states, and by
        continual re-solving to hand a leaf's ranges to the next solve.
        """
        strategies = self._active_strategy
        if average:
            strategies = [None] * self.tree.num_nodes
            for node in self.tree.decision_nodes():
                strategies[node.index] = self.average_strategy(node)
        out: List[Tuple[PublicNode, np.ndarray]] = []
        self._descend_reach(self.tree.root, reach, strategies, out)
        return out

    def _descend_reach(
        self,
        node: PublicNode,
        reach: np.ndarray,
        strategies: List[Optional[np.ndarray]],
        out: List[Tuple[PublicNode, np.ndarray]],
    ) -> None:
        if node.is_leaf:
            out.append((node, reach))
            return
        if node.is_terminal:
            return
        if node.is_chance:
            for board, child in zip(node.boards, node.children):
                child_reach = reach * self.space.deal_mask(board)
                if child_reach.sum() > 0.0:
                    self._descend_reach(child, child_reach, strategies, out)
            return
        strategy = strategies[node.index]
        for j, child in enumerate(node.children):
            child_reach = reach.copy()
            child_reach[node.player] = reach[node.player] * strategy[:, j]
            self._descend_reach(child, child_reach, strategies, out)

    def _evaluate_leaves(self, reach: np.ndarray) -> None:
        """Query the leaf evaluator once for the whole frontier."""
        self._leaf_cache: Dict[int, np.ndarray] = {}
        if not self.tree.leaves():
            return
        frontier = self.leaf_reaches(reach)
        if self._store_leaves:
            self.iteration_leaves.append(frontier)
        if not frontier:
            return
        states = [self.space.pbs(node.public, node_reach) for node, node_reach in frontier]
        predicted = np.asarray(self.leaf_value_fn(states), dtype=np.float64)
        expected = (len(states), NUM_PLAYERS, self.space.num_hands)
        if predicted.shape != expected:
            raise ValueError(
                f"leaf value function returned {predicted.shape}, expected {expected}"
            )
        for (node, node_reach), values in zip(frontier, predicted):
            masses = node_reach.sum(axis=1)
            correction = self.space.pair_correction
            scaled = np.empty((NUM_PLAYERS, self.space.num_hands))
            scaled[0] = correction * masses[1] * values[0]
            scaled[1] = correction * masses[0] * values[1]
            self._leaf_cache[node.index] = scaled

    # --- the traversal -----------------------------------------------------
    def _walk(self, node: PublicNode, reach: np.ndarray) -> np.ndarray:
        if node.is_terminal:
            return self.space.terminal_values(node.public, reach)
        if node.is_leaf:
            return self._leaf_cache[node.index]
        if node.is_chance:
            return self._chance_values(node, reach)
        return self._decision_values(node, reach)

    def _chance_values(self, node: PublicNode, reach: np.ndarray) -> np.ndarray:
        """Deal the board.

        Given two distinct private cards, each of the four remaining cards is
        the board with probability 1/4.  Zeroing the dealt card in both reach
        vectors and masking it out of the returned values makes that exact: the
        pairs where a player holds the board card simply contribute nothing.
        """
        values = np.zeros((NUM_PLAYERS, self.space.num_hands))
        weight = self.space.chance_weight(node.public)
        for board, child in zip(node.boards, node.children):
            mask = self.space.deal_mask(board)
            child_reach = reach * mask
            if child_reach.sum() == 0.0:
                continue
            values += weight * self._walk(child, child_reach) * mask
        return values

    def _decision_values(self, node: PublicNode, reach: np.ndarray) -> np.ndarray:
        player = node.player
        opponent = 1 - player
        strategy = self._active_strategy[node.index]
        values = np.zeros((NUM_PLAYERS, self.space.num_hands))
        action_values = np.empty((node.num_actions, self.space.num_hands))

        for j, child in enumerate(node.children):
            child_reach = reach.copy()
            child_reach[player] = reach[player] * strategy[:, j]
            child_values = self._walk(child, child_reach)
            action_values[j] = child_values[player]
            # The opponent's values already carry our action probabilities,
            # because they went down inside ``child_reach``.
            values[opponent] += child_values[opponent]

        values[player] = np.einsum("ha,ah->h", strategy, action_values)
        if self._update_player in (None, player):
            regrets = self.regrets[node.index]
            regrets += action_values.T - values[player][:, None]
            self.strategy_sum[node.index] += (
                self._weight * reach[player][:, None] * strategy
            )
        return values

    # --- read-out ----------------------------------------------------------
    def average_strategy(self, node: PublicNode) -> np.ndarray:
        """The time-average strategy at ``node``, per hand."""
        totals = self.strategy_sum[node.index].sum(axis=1, keepdims=True)
        return np.where(
            totals > 0.0,
            self.strategy_sum[node.index] / np.where(totals > 0.0, totals, 1.0),
            1.0 / node.num_actions,
        )

    def current_strategy(self, node: PublicNode) -> np.ndarray:
        return self.strategy[node.index]


def terminal_values(
    public: PublicState, reach: np.ndarray, space: HandSpace | None = None
) -> np.ndarray:
    """Exact counterfactual values at a real terminal node."""
    return (space or LEDUC_SPACE).terminal_values(public, reach)


def solve_public_game(
    iterations: int = 300,
    config: CFRConfig | None = None,
    root: PublicState | None = None,
    reach: np.ndarray | None = None,
) -> SubgameSolver:
    """Solve a whole game (no depth limit, exact terminals) in range form."""
    tree = build_public_tree(root, depth_limit=None)
    solver = SubgameSolver(tree, leaf_value_fn=None, config=config)
    solver.solve(reach=reach, iterations=iterations)
    return solver
