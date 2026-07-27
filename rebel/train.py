"""The ReBeL training loop.

Alternate two things until the network stops changing:

* **generate** — play trajectories through belief space, searching at every
  belief state with the current network and recording what search concluded;
* **learn** — regress the network onto those conclusions.

The metric is exploitability of the *searching agent*, not of the network: the
network is only ever a component of a search procedure, and predicting values
accurately is worth nothing if the resulting play is exploitable.  Measuring it
means walking every reachable belief state, re-solving at each, projecting the
result onto the world tree and running an exact best response — which is exactly
what stages 0 and 1 were built to make possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from cfr.best_response import expected_values, exploitability
from game.leduc import LeducHoldem
from game.tree import GameTree, build_tree
from rebel.buffer import ValueReplayBuffer
from rebel.config import ReBeLConfig
from rebel.selfplay import collect_trajectory
from search.resolve import ContinualResolver
from value_net.net import PBSValueNet
from value_net.values import NetLeafValues


@dataclass
class ReBeLRun:
    """Everything a completed run produces."""

    net: PBSValueNet
    history: List[Dict[str, float]] = field(default_factory=list)
    buffer: Optional[ValueReplayBuffer] = None


def train_rebel(
    config: ReBeLConfig | None = None,
    world_tree: Optional[GameTree] = None,
    verbose: bool = False,
) -> ReBeLRun:
    """Run ReBeL self play on Leduc and return the trained value network."""
    config = config or ReBeLConfig()
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    net = PBSValueNet(config.value_net).to(device)
    optimiser = torch.optim.Adam(
        net.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    loss_fn = nn.HuberLoss(reduction="mean")
    buffer = ValueReplayBuffer(config.buffer_size)
    world_tree = world_tree if world_tree is not None else build_tree(LeducHoldem())
    run = ReBeLRun(net=net, buffer=buffer)

    for iteration in range(1, config.iterations + 1):
        leaf_values = NetLeafValues(net, device=device)
        for _ in range(config.self_play.trajectories_per_iteration):
            buffer.add(collect_trajectory(leaf_values, config.self_play, rng))

        net.train()
        total, updates = 0.0, 0
        for _ in range(config.updates_per_iteration):
            features, targets = buffer.sample(config.batch_size, rng)
            batch = torch.as_tensor(features, dtype=torch.float32, device=device)
            wanted = torch.as_tensor(targets, dtype=torch.float32, device=device)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(net(batch), wanted)
            loss.backward()
            optimiser.step()
            total += float(loss.item())
            updates += 1

        record = {
            "iteration": float(iteration),
            "buffer": float(len(buffer)),
            "loss": total / max(updates, 1),
        }
        if iteration % config.evaluate_every == 0 or iteration == config.iterations:
            record.update(evaluate_agent(net, config, world_tree, device))
        run.history.append(record)
        if verbose:
            print(
                "  ".join(f"{k}={v:.5g}" for k, v in record.items()),
                flush=True,
            )
    return run


def evaluate_agent(
    net: PBSValueNet,
    config: ReBeLConfig,
    world_tree: GameTree,
    device: str | torch.device = "cpu",
) -> Dict[str, float]:
    """Exploitability of the agent that searches with ``net``."""
    resolver = ContinualResolver(NetLeafValues(net, device=device), config.evaluation)
    policy = resolver.policy(world_tree)
    return {
        "exploitability": exploitability(world_tree, policy),
        "value": float(expected_values(world_tree, policy)[0]),
    }
