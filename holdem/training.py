"""Randomised ReBeL training on hold'em turn endgames, with held-out testing.

Every trajectory starts from a freshly sampled situation — board, pot, stack and
both ranges — which is how DeepStack and ReBeL generate their data and the fix
for the first run's flaw, where every trajectory began from a byte-identical
belief state and half the training set was one input repeated.

The evaluation is deliberately on **boards the network has never trained on**.
That is the only question worth asking of a value network: an endgame solver can
answer one board, and the network's whole purpose is to answer the 270,725 it
cannot afford to solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from holdem.combos import board_mask
from holdem.net import HoldemValueNet, HoldemValueNetConfig
from holdem.public_tree import PublicState, build_turn_tree
from holdem.rebel import Buffer, HoldemSelfPlayConfig, collect_trajectory
from holdem.sampling import SituationConfig, held_out_boards, sample_situation
from holdem.space import TurnEndgameSpace
from holdem.values import NetLeafValues
from search import ContinualResolver, ResolveConfig
from search.best_response import subgame_exploitability


@dataclass
class RandomisedReBeLConfig:
    iterations: int = 60
    trajectories_per_iteration: int = 16
    buffer_size: int = 60_000
    updates_per_iteration: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    self_play: HoldemSelfPlayConfig = field(default_factory=HoldemSelfPlayConfig)
    value_net: HoldemValueNetConfig = field(default_factory=HoldemValueNetConfig)
    situations: SituationConfig = field(default_factory=SituationConfig)
    # Evaluation: fixed held-out situations, re-scored at the same settings.
    num_test_boards: int = 3
    eval_every: int = 10
    eval_search_iterations: int = 40
    eval_safe_resolving: bool = False
    seed: int = 0
    device: str = "cpu"


@dataclass
class TestSituation:
    space: TurnEndgameSpace
    root: PublicState
    reach: np.ndarray

    def full_tree(self):
        return build_turn_tree(self.root, depth_limit=None)


def make_test_situations(
    rng: np.random.Generator, count: int, config: SituationConfig
) -> List[TestSituation]:
    """Fixed evaluation situations on boards excluded from training."""
    boards = held_out_boards(rng, count, config.board_cards)
    held_out = SituationConfig(**{**config.__dict__, "excluded_boards": ()})
    situations = []
    for board in boards:
        # Reuse the sampler for pot/stack/ranges, then pin the board.
        space, root, reach = sample_situation(rng, held_out)
        space = TurnEndgameSpace(board)
        root = PublicState(betting=root.betting, board=board)
        reach = np.stack(
            [r * board_mask(board) for r in reach]
        )
        reach /= reach.sum(axis=1, keepdims=True)
        situations.append(TestSituation(space=space, root=root, reach=reach))
    return situations


def exploitability_on(
    net: HoldemValueNet,
    situation: TestSituation,
    search_iterations: int,
    safe_resolving: bool,
    device: str = "cpu",
) -> float:
    """Exploitability of the searching agent on one endgame, exactly."""
    resolver = ContinualResolver(
        NetLeafValues(net, device=device),
        ResolveConfig(
            iterations=search_iterations, depth_limit=1, safe_resolving=safe_resolving
        ),
        space=situation.space,
        tree_builder=build_turn_tree,
    )
    strategies, _ = resolver.run(root=situation.root, reach=situation.reach)
    tree = situation.full_tree()
    filled = {
        node.public: strategies.get(
            node.public,
            np.full(
                (situation.space.num_hands, node.num_actions), 1.0 / node.num_actions
            ),
        )
        for node in tree.decision_nodes()
    }
    total, _ = subgame_exploitability(
        tree, filled, situation.reach, space=situation.space
    )
    return float(total)


def train(
    config: RandomisedReBeLConfig | None = None, verbose: bool = False
) -> Tuple[HoldemValueNet, List[Dict[str, float]]]:
    config = config or RandomisedReBeLConfig()
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    tests = make_test_situations(rng, config.num_test_boards, config.situations)
    situations = SituationConfig(
        **{
            **config.situations.__dict__,
            "excluded_boards": tuple(t.root.board for t in tests),
        }
    )

    net = HoldemValueNet(config.value_net).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    buffer = Buffer(config.buffer_size)
    history: List[Dict[str, float]] = []

    for iteration in range(1, config.iterations + 1):
        leaf_values = NetLeafValues(net, device=device)
        for _ in range(config.trajectories_per_iteration):
            space, root, reach = sample_situation(rng, situations)
            buffer.add(
                collect_trajectory(
                    leaf_values, space, root, config.self_play, rng, reach=reach
                )
            )

        net.train()
        total, updates = 0.0, 0
        for _ in range(config.updates_per_iteration):
            features, masks, targets = buffer.sample(config.batch_size, rng)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(
                net(
                    torch.as_tensor(features, device=device),
                    torch.as_tensor(masks, device=device),
                ),
                torch.as_tensor(targets, device=device),
            )
            loss.backward()
            optimiser.step()
            total += float(loss.item())
            updates += 1

        record = {
            "iteration": float(iteration),
            "buffer": float(len(buffer)),
            "loss": total / max(updates, 1),
        }
        if iteration % config.eval_every == 0 or iteration == config.iterations:
            net.eval()
            scores = [
                exploitability_on(
                    net,
                    test,
                    config.eval_search_iterations,
                    config.eval_safe_resolving,
                    config.device,
                )
                for test in tests
            ]
            record["held_out_exploitability"] = float(np.mean(scores))
            record["held_out_worst"] = float(np.max(scores))
        history.append(record)
        if verbose:
            print("  ".join(f"{k}={v:.5g}" for k, v in record.items()), flush=True)
    return net, history
