"""ReBeL self play on a postflop hold'em endgame.

Algorithm 2 of Brown et al. 2020 — ReBeL with Linear CFR-D data generation —
line by line, with the differences hold'em forces:

* a trajectory starts on any postflop street and follows the sampled CFR
  iterate's reach distribution through every later street; the river solve runs
  to real showdowns, so its values are exact;
* every example carries its own hand mask, because which of the 1,326 combos
  are possible depends on all five board cards.

The correspondence to the pseudocode, since the whole point of this module is
that it *is* the pseudocode:

===============================  =======================================
``REBEL-LINEAR-CFR-D``           :func:`collect_trajectory`
``CONSTRUCT_SUBGAME(beta_r)``    ``build_turn_tree(public, depth_limit)``
``INITIALIZE_POLICY(G, th_pi)``  :func:`initialize_policy`
``SET_LEAF_VALUES``              ``SubgameSolver._evaluate_leaves``, run
                                 once per CFR iteration against pi_t
``COMPUTE_EV`` / ``v(beta_r)``   ``SubgameSolver.root_values``
``t_sample ~ linear{..}``        :func:`sample_iteration`
``UPDATE_POLICY`` / ``pi_bar``   regret matching + the linear weighting in
                                 :meth:`CFRConfig.linear_cfr_d`
``SAMPLE_LEAF(G, pi_{t-1})``     :func:`sample_leaf`
``Add {beta_r, v} to D_v``       the returned :class:`Example` values
``Add {beta, pi_bar} to D_pi``   the returned :class:`Example` policies
===============================  =======================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from paradigm_b.core.cfr.tabular_cfr import CFRConfig
from paradigm_b.holdem.engine.combos import NUM_COMBOS, board_mask, compatible_mass
from paradigm_b.holdem.net.features import INPUT_DIM, encode_pbs
from paradigm_b.holdem.net.value_net import HoldemValueNet, HoldemValueNetConfig
from paradigm_b.holdem.net.policy import (
    MAX_ACTIONS,
    HoldemPolicyNet,
    HoldemPolicyNetConfig,
    PolicyExample,
    PolicyReplayBuffer,
    query_policy,
    train_policy_net,
)
from paradigm_b.holdem.engine.public_tree import PublicState, build_turn_tree
from paradigm_b.holdem.engine.space import TurnEndgameSpace
from paradigm_b.holdem.net.leaf_values import NetLeafValues
from paradigm_b.core.search.evaluate import decision_reaches, leaf_reaches
from paradigm_b.core.search.policy import StrategyMap
from paradigm_b.core.search.subgame import SubgameSolver, WarmStart

NUM_PLAYERS = 2


@dataclass
class HoldemSelfPlayConfig:
    trajectories_per_iteration: int = 8
    # ``T``: the iteration the CFR loop finishes at, counting from t_warm.
    search_iterations: int = 40
    river_iterations: int = 60
    depth_limit: int = 1
    # ``t_warm``.  Zero means the algorithm's own no-warm-start branch: pi_0 is
    # uniform and the loop starts at t = 1.  Anything higher only takes effect
    # when a policy network is passed to :func:`collect_trajectory` — there is
    # nothing to warm start *from* otherwise — and then INITIALIZE_POLICY seeds
    # every decision node in the subgame from theta_pi.
    warm_start_iterations: int = 0
    # ``epsilon``.  Appendix E of the ReBeL paper: *"for all experiments we set
    # the probability to explore a random action to eps = 25%"*, so this is the
    # paper-faithful value and the default here.  It is applied the way
    # SAMPLE_LEAF applies it: to one uniformly chosen player, at every node of
    # the descent, mixing eps of uniform into that player's action
    # probabilities.  The other player follows the CFR iterate untouched.
    #
    # One measured caveat, so it is not rediscovered the hard way: at a 2,000
    # label budget a coarser form of this (both players uniform for the whole
    # descent) made the online arm *worse*, not better — held-out aggregate
    # exploitability 35.7 -> 38.5 (``runs/labels-02`` vs ``runs/fixes-u4000``).
    # Exploration widens the belief-state distribution, which pays only once
    # there is enough data to cover the wider space, and ReBeL's 12M-example
    # buffer has ~6,000x more of it than those runs did.  Set it to 0.0 for
    # deliberately small-budget experiments; it is also 0.0 at test time, which
    # is what evaluation through ``ContinualResolver`` already gives.
    exploration: float = 0.25
    # The "(optional)" line of Algorithm 2: record pi_bar at every public state
    # in G for D_pi.  Costs one extra tree descent plus a PBS encode per
    # decision node, so callers that train no policy network turn it off rather
    # than build examples nothing will consume.
    policy_targets: bool = True
    # Algorithm 2 is *Linear* CFR-D; see :meth:`CFRConfig.linear_cfr_d`.
    cfr: CFRConfig = field(default_factory=CFRConfig.linear_cfr_d)


@dataclass
class HoldemReBeLConfig:
    iterations: int = 30
    buffer_size: int = 20_000
    updates_per_iteration: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-3
    self_play: HoldemSelfPlayConfig = field(default_factory=HoldemSelfPlayConfig)
    value_net: HoldemValueNetConfig = field(default_factory=HoldemValueNetConfig)
    policy_net: HoldemPolicyNetConfig = field(default_factory=HoldemPolicyNetConfig)
    policy_buffer_size: int = 2_000
    policy_updates_per_iteration: int = 40
    seed: int = 0
    device: str = "cpu"


@dataclass
class Example:
    """One solved belief state: its D_v label, and its D_pi labels beside it."""

    features: np.ndarray
    mask: np.ndarray
    values: np.ndarray
    # ``for beta in G: add {beta, pi_bar(beta)} to D_pi`` — one entry per
    # decision node of the subgame that was solved here, not just its root.
    policies: Tuple[PolicyExample, ...] = ()
    # The board this example was solved on.  Carried so a caller can tell which
    # street produced it — the encoded features contain the board, but recovering
    # it from them is needless work when the sampler already knows.
    board: Tuple[int, ...] = ()


class Buffer:
    """Circular store of encoded belief states, their masks and their values."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.features = np.zeros((capacity, INPUT_DIM), dtype=np.float32)
        self.masks = np.zeros((capacity, NUM_COMBOS), dtype=np.float32)
        self.targets = np.zeros((capacity, NUM_PLAYERS, NUM_COMBOS), dtype=np.float32)
        self.size = 0
        self._next = 0

    def __len__(self) -> int:
        return self.size

    def add(self, examples: Sequence[Example]) -> None:
        for example in examples:
            self.features[self._next] = example.features
            self.masks[self._next] = example.mask
            self.targets[self._next] = example.values
            self._next = (self._next + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        index = rng.integers(0, self.size, size=min(batch_size, self.size))
        return self.features[index], self.masks[index], self.targets[index]

    def purge_oldest(self, fraction: float = 0.5) -> int:
        """Drop the oldest ``fraction`` of the buffer; return how many went.

        ReBeL's appendix E: *"As initial data is produced with a random value
        network, we remove half of the data from the replay buffer after 20
        epochs."*  Without this, labels written by an essentially random
        network keep being sampled at full weight for the rest of the run —
        and in a buffer that never fills (2,000 labels into 60,000 slots)
        nothing is ever evicted by the ordinary circular churn either, so
        *every* early mistake survives to the last gradient step.

        Order is reconstructed from the write pointer, so this is correct
        whether or not the buffer has wrapped.
        """
        if not 0.0 < fraction < 1.0 or self.size == 0:
            return 0
        keep = max(self.size - int(self.size * fraction), 1)
        if keep >= self.size:
            return 0
        if self.size < self.capacity:
            order = np.arange(self.size)
        else:  # wrapped: oldest sits at the write pointer
            order = np.concatenate(
                [np.arange(self._next, self.capacity), np.arange(0, self._next)]
            )
        newest = order[-keep:]
        # Fancy indexing copies, so this cannot alias the destination.
        self.features[:keep] = self.features[newest]
        self.masks[:keep] = self.masks[newest]
        self.targets[:keep] = self.targets[newest]
        dropped = self.size - keep
        self.size = keep
        self._next = keep % self.capacity
        return dropped


def normalise(values: np.ndarray, reach: np.ndarray, correction: float):
    masses = reach.sum(axis=1)
    if masses.min() <= 0.0:
        return None
    out = np.empty_like(values)
    out[0] = values[0] / (correction * masses[1])
    out[1] = values[1] / (correction * masses[0])
    return out


def collect_trajectory(
    leaf_value_fn,
    space: TurnEndgameSpace,
    root: PublicState,
    config: HoldemSelfPlayConfig,
    rng: np.random.Generator,
    reach: Optional[np.ndarray] = None,
    policy_net: Optional[HoldemPolicyNet] = None,
    device: str | torch.device = "cpu",
) -> List[Example]:
    """``REBEL-LINEAR-CFR-D``: one trajectory from ``root`` through the river.

    Each pass of the loop is one ``beta_r``: build its subgame, initialise the
    policy, run Linear CFR-D to ``T`` with the value network at the depth
    limit, emit ``{beta_r, v(beta_r)}`` for D_v and ``{beta, pi_bar(beta)}`` for
    D_pi, then step to the leaf ``SAMPLE_LEAF`` picked out of iteration
    ``t_sample``.  It ends when the subgame runs to real terminals — on the
    river there is no depth limit, so ``IS_TERMINAL(beta_r)`` is reached with
    exact values rather than predicted ones.
    """
    public = root
    reach = space.initial_reach() if reach is None else np.asarray(reach, float)
    examples: List[Example] = []

    while True:
        tree = build_turn_tree(public, depth_limit=config.depth_limit)
        has_leaves = bool(tree.leaves())
        solver = SubgameSolver(
            tree,
            leaf_value_fn=leaf_value_fn if has_leaves else None,
            config=config.cfr,
            space=space,
        )

        # pi, pi_bar, t_warm = INITIALIZE_POLICY(G, theta_pi)
        warm_start = initialize_policy(tree, space, reach, config, policy_net, device)
        warm_iterations = 0 if warm_start is None else warm_start.iterations
        total = config.search_iterations if has_leaves else config.river_iterations
        total = max(total, warm_iterations + 1)

        # t_sample ~ linear{t_warm + 1, ..., T}, drawn before the loop so that
        # SAMPLE_LEAF is called once, at that iteration, against pi_{t-1}.
        sampled = sample_iteration(warm_iterations, total, rng) if has_leaves else None

        solver.solve(
            reach=reach,
            iterations=total,
            warm_start=warm_start,
            capture_strategy_at=sampled,
            seed_root_value=True,
        )

        values = normalise(solver.root_values(), reach, space.pair_correction)
        if values is not None:
            examples.append(
                Example(
                    features=encode_pbs(space.pbs(public, reach)),
                    mask=board_mask(tuple(public.board)),
                    values=values,
                    policies=(
                        policy_targets(solver, tree, space, reach)
                        if config.policy_targets
                        else ()
                    ),
                    board=tuple(public.board),
                )
            )

        if not has_leaves:
            return examples

        chosen = sample_leaf(tree, space, solver.captured_strategy, reach, config, rng)
        if chosen is None:
            return examples
        leaf, reach = chosen
        public = leaf.public


def sample_iteration(warm_iterations: int, total: int, rng: np.random.Generator) -> int:
    """``t_sample ~ linear{t_warm + 1, ..., T}``: probability proportional to t.

    The linear weighting is not decoration.  Linear CFR-D weights iteration
    ``t`` by ``t`` in everything it averages, so late iterates describe the
    strategy far better than early ones do; sampling the descent uniformly (as
    Algorithm 1 does, and as this used to) would send a quarter of the
    trajectory through belief states produced by the first few, near-uniform
    iterates, and spend the value network's capacity on them.
    """
    candidates = np.arange(warm_iterations + 1, total + 1)
    return int(rng.choice(candidates, p=candidates / candidates.sum()))


def sample_leaf(
    tree,
    space: TurnEndgameSpace,
    strategies: Optional[List[Optional[np.ndarray]]],
    reach: np.ndarray,
    config: HoldemSelfPlayConfig,
    rng: np.random.Generator,
):
    """``SAMPLE_LEAF(G, pi_{t-1})``: the leaf the next belief state is rooted at.

    The pseudocode draws one player ``i* ~ unif{1, N}`` and one history
    ``h ~ beta_r``, then walks down: at each step ``i*`` takes a uniform-random
    action with probability ``eps`` and both players otherwise follow ``pi``.
    Because only one player acts at a node, that walk is playing the profile

        i*      :  (1 - eps) * pi + eps * uniform
        1 - i*  :  pi

    and the leaf it stops at is distributed exactly as that profile's arrival
    mass.  So the frontier is enumerated under that profile and one leaf drawn
    in proportion to arrival probability: the same distribution as the walk,
    with none of its sampling noise.  The ranges are already being propagated
    for the solve, so there is nothing to gain from following a single history
    and a lot of variance to lose.

    The returned reach vectors are the ones this profile produces, which is
    what makes ``beta_h`` the belief state of the play that actually reached
    it.  ``eps = 0`` recovers pure on-policy descent, which is what evaluation
    uses.
    """
    explorer = int(rng.integers(NUM_PLAYERS))  # i* ~ unif{1, N}
    profile = exploration_profile(
        tree, space, strategies, explorer, config.exploration
    )
    return _draw_leaf(leaf_reaches(tree, profile, reach, space=space), space, rng)


def exploration_profile(
    tree,
    space: TurnEndgameSpace,
    strategies: Optional[List[Optional[np.ndarray]]],
    explorer: int,
    epsilon: float,
) -> StrategyMap:
    """The profile ``SAMPLE_LEAF`` walks: ``eps`` of uniform, for ``i*`` only."""
    if strategies is None:  # solve stopped before t_sample; fall back to pi_0
        strategies = [None] * tree.num_nodes
    profile: StrategyMap = {}
    for node in tree.decision_nodes():
        strategy = strategies[node.index]
        if strategy is None:
            strategy = np.full(
                (space.num_hands, node.num_actions), 1.0 / node.num_actions
            )
        if node.player == explorer and epsilon > 0.0:
            strategy = (1.0 - epsilon) * strategy + epsilon / node.num_actions
        profile[node.public] = strategy
    return profile


def policy_targets(
    solver: SubgameSolver, tree, space: TurnEndgameSpace, reach: np.ndarray
) -> Tuple[PolicyExample, ...]:
    """``for beta in G: add {beta, pi_bar(beta)} to D_pi``.

    Every decision node of the solved subgame, each paired with the belief
    state the average strategy itself induces there — not only the root, which
    would train theta_pi on one public state per solve and leave it unable to
    warm start anything below the first action.
    """
    averages = {
        node.public: solver.average_strategy(node) for node in tree.decision_nodes()
    }
    out: List[PolicyExample] = []
    for node, node_reach in decision_reaches(tree, averages, reach, space=space):
        if node_reach.sum(axis=1).min() <= 0.0:
            continue  # a belief state neither player can reach
        target = np.zeros((space.num_hands, MAX_ACTIONS), dtype=np.float32)
        target[:, node.actions] = averages[node.public]
        legal = np.zeros(MAX_ACTIONS, dtype=np.float32)
        legal[list(node.actions)] = 1.0
        out.append(
            PolicyExample(
                features=encode_pbs(space.pbs(node.public, node_reach)),
                agent_index=node.player,
                legal_mask=legal,
                target=target,
            )
        )
    return tuple(out)


def initialize_policy(
    tree,
    space: TurnEndgameSpace,
    reach: np.ndarray,
    config: HoldemSelfPlayConfig,
    policy_net: Optional[HoldemPolicyNet],
    device: str | torch.device = "cpu",
) -> Optional[WarmStart]:
    """``INITIALIZE_POLICY(G, theta_pi)``.

    With no policy network — or with ``t_warm = 0`` — this is the branch the
    pseudocode spells out in its own comment: pi_0 is uniform, t_warm is 0, and
    there is nothing to hand the solver, so it returns ``None``.

    With one, every decision node in ``G`` is initialised to theta_pi's
    prediction for its *own* belief state.  That has to be a top-down pass: the
    ranges arriving at a node depend on the policy above it, so the network
    cannot be asked about a node until its ancestors have been set.  Levels are
    batched, which costs one forward pass per depth of the public tree rather
    than one per node.
    """
    if policy_net is None or config.warm_start_iterations <= 0:
        return None

    strategies: List[Optional[np.ndarray]] = [None] * tree.num_nodes
    reaches: List[Optional[np.ndarray]] = [None] * tree.num_nodes
    level = [(tree.root, np.asarray(reach, dtype=np.float64).copy())]

    while level:
        decisions = [(node, r) for node, r in level if node.is_decision]
        if decisions:
            legal = np.zeros((len(decisions), MAX_ACTIONS), dtype=np.float32)
            for row, (node, _) in enumerate(decisions):
                legal[row, list(node.actions)] = 1.0
            predicted = query_policy(
                policy_net,
                np.stack(
                    [encode_pbs(space.pbs(node.public, r)) for node, r in decisions]
                ),
                np.array([node.player for node, _ in decisions], dtype=np.int64),
                legal,
                device,
            )
            for (node, node_reach), probabilities in zip(decisions, predicted):
                strategy = probabilities[:, list(node.actions)]
                totals = strategy.sum(axis=1, keepdims=True)
                strategies[node.index] = np.where(
                    totals > 0.0,
                    strategy / np.where(totals > 0.0, totals, 1.0),
                    1.0 / node.num_actions,
                )
                reaches[node.index] = node_reach

        following = []
        for node, node_reach in level:
            if node.is_terminal or node.is_leaf:
                continue
            if node.is_chance:
                for board, child in zip(node.boards, node.children):
                    child_reach = node_reach * space.deal_mask(board)
                    if child_reach.sum() > 0.0:
                        following.append((child, child_reach))
                continue
            strategy = strategies[node.index]
            for j, child in enumerate(node.children):
                child_reach = node_reach.copy()
                child_reach[node.player] = node_reach[node.player] * strategy[:, j]
                following.append((child, child_reach))
        level = following

    return WarmStart(
        strategies=strategies,
        reaches=reaches,
        iterations=config.warm_start_iterations,
    )


def _draw_leaf(frontier, space: TurnEndgameSpace, rng):
    candidates, weights = [], []
    for leaf, reach in frontier:
        weight = _arrival_probability(reach, space.pair_correction)
        if weight <= 0.0:
            continue
        candidates.append((leaf, reach))
        weights.append(weight)
    if not candidates:
        return None
    probabilities = np.asarray(weights) / float(np.sum(weights))
    return candidates[int(rng.choice(len(candidates), p=probabilities))]


def _arrival_probability(reach: np.ndarray, correction: float = 1.0) -> float:
    """Joint reach mass over legal, non-overlapping private-hand deals."""
    return float(correction * np.dot(reach[0], compatible_mass(reach[1])))


def train(
    space: TurnEndgameSpace,
    root: PublicState,
    config: HoldemReBeLConfig | None = None,
    verbose: bool = False,
    evaluate=None,
):
    """Run the loop; ``evaluate(net, iteration)`` may report progress."""
    config = config or HoldemReBeLConfig()
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    net = HoldemValueNet(config.value_net).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    policy_net = HoldemPolicyNet(config.policy_net).to(device)
    policy_optimiser = torch.optim.Adam(policy_net.parameters(), lr=config.learning_rate)
    loss_fn = nn.HuberLoss(reduction="mean")
    buffer = Buffer(config.buffer_size)
    policy_buffer = PolicyReplayBuffer(config.policy_buffer_size)
    history: List[Dict[str, float]] = []

    for iteration in range(1, config.iterations + 1):
        leaf_values = NetLeafValues(net, space, device=device)
        for _ in range(config.self_play.trajectories_per_iteration):
            # theta_pi is fed back in, so a run with ``warm_start_iterations``
            # set warm-starts each solve from the policy it has learned so far.
            examples = collect_trajectory(
                leaf_values,
                space,
                root,
                config.self_play,
                rng,
                policy_net=policy_net,
                device=device,
            )
            buffer.add(examples)
            policy_buffer.add(
                [policy for example in examples for policy in example.policies]
            )

        net.train()
        total, updates = 0.0, 0
        for _ in range(config.updates_per_iteration):
            features, masks, targets = buffer.sample(config.batch_size, rng)
            batch = torch.as_tensor(features, device=device)
            mask = torch.as_tensor(masks, device=device)
            wanted = torch.as_tensor(targets, device=device)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(net(batch, mask), wanted)
            loss.backward()
            optimiser.step()
            total += float(loss.item())
            updates += 1

        record = {
            "iteration": float(iteration),
            "buffer": float(len(buffer)),
            "loss": total / max(updates, 1),
            "policy_loss": train_policy_net(
                policy_net,
                policy_buffer,
                config.policy_updates_per_iteration,
                config.batch_size,
                config.learning_rate,
                rng,
                device,
                policy_optimiser,
            ),
        }
        if evaluate is not None:
            record.update(evaluate(net, iteration))
        history.append(record)
        if verbose:
            print("  ".join(f"{k}={v:.5g}" for k, v in record.items()), flush=True)
    net.policy_net = policy_net
    return net, history
