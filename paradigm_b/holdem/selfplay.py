"""ReBeL self play on a postflop hold'em endgame.

The same loop as ``rebel/`` on Leduc — search at a belief state, train the
network on what search concluded, descend into a leaf reached by a random CFR
iterate — with the differences hold'em forces:

* a trajectory starts on any postflop street and follows one sampled CFR
  iterate's reach distribution through every later street; the river solve runs
  to real showdowns, so its values are exact;
* every example carries its own hand mask, because which of the 1,326 combos
  are possible depends on all five board cards;
* optional off-policy exploration can broaden the range distribution beyond
  the narrow slice equilibrium play visits, without changing the paper-aligned
  default trajectory sampler.
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
    train_policy_net,
)
from paradigm_b.holdem.engine.public_tree import PublicState, build_turn_tree
from paradigm_b.holdem.engine.space import TurnEndgameSpace
from paradigm_b.holdem.net.leaf_values import NetLeafValues
from paradigm_b.core.search.evaluate import leaf_reaches
from paradigm_b.core.search.policy import StrategyMap
from paradigm_b.core.search.subgame import SubgameSolver

NUM_PLAYERS = 2


@dataclass
class HoldemSelfPlayConfig:
    trajectories_per_iteration: int = 8
    search_iterations: int = 40
    river_iterations: int = 60
    depth_limit: int = 1
    # Algorithm 1 samples uniformly from every CFR iterate when there is no
    # policy warm start.  These remain configurable for deliberate off-policy
    # experiments, but the defaults follow the paper.
    warmup_fraction: float = 0.0
    exploration: float = 0.0
    cfr: CFRConfig = field(default_factory=CFRConfig.dcfr)


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
    features: np.ndarray
    mask: np.ndarray
    values: np.ndarray
    policy: PolicyExample | None = None
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
) -> List[Example]:
    """Sample one Algorithm 1 trajectory from ``root`` through the river."""
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
        iterations = (
            config.search_iterations if has_leaves else config.river_iterations
        )
        solver.solve(
            reach=reach, iterations=iterations, store_iteration_leaves=has_leaves
        )

        values = normalise(solver.root_values(), reach, space.pair_correction)
        if values is not None:
            pbs = space.pbs(public, reach)
            root = tree.root
            policy = None
            if root.is_decision:
                target = np.zeros((space.num_hands, MAX_ACTIONS), dtype=np.float32)
                target[:, root.actions] = solver.average_strategy(root)
                legal = np.zeros(MAX_ACTIONS, dtype=np.float32)
                legal[list(root.actions)] = 1.0
                policy = PolicyExample(
                    features=encode_pbs(pbs),
                    agent_index=root.player,
                    legal_mask=legal,
                    target=target,
                )
            examples.append(
                Example(
                    features=encode_pbs(pbs),
                    mask=board_mask(tuple(public.board)),
                    values=values,
                    policy=policy,
                    board=tuple(public.board),
                )
            )

        if not has_leaves:
            return examples

        frontier = _frontier(solver, tree, reach, space, config, rng)
        chosen = _sample_leaf(frontier, space, rng)
        if chosen is None:
            return examples
        leaf, reach = chosen
        public = leaf.public


def _frontier(solver, tree, reach, space, config: HoldemSelfPlayConfig, rng):
    if rng.random() < config.exploration:
        uniform: StrategyMap = {
            node.public: np.full((space.num_hands, node.num_actions), 1.0 / node.num_actions)
            for node in tree.decision_nodes()
        }
        return leaf_reaches(tree, uniform, reach, space=space)
    stored = solver.iteration_leaves
    first = int(len(stored) * config.warmup_fraction)
    index = int(rng.integers(first, len(stored))) if len(stored) > first else -1
    return stored[index]


def _sample_leaf(frontier, space: TurnEndgameSpace, rng):
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
            examples = collect_trajectory(leaf_values, space, root, config.self_play, rng)
            buffer.add(examples)
            policy_buffer.add([example.policy for example in examples if example.policy is not None])

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
