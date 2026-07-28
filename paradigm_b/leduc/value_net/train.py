"""Supervised training for the value network.

Pointwise Huber regression: features in, per-hand values out, against the solver.
Nothing exotic is needed — the interesting part is not the optimiser, it is
whether a network that fits these values well is *good enough to search with*,
which only the exploitability of the resulting agent can answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from paradigm_b.leduc.value_net.dataset import ValueDataset
from paradigm_b.leduc.value_net.net import PBSValueNet


@dataclass
class ValueTrainConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    batch_size: int = 256
    epochs: int = 60
    device: str = "cpu"
    seed: int = 0
    log_every: int = 10


def train_value_net(
    net: PBSValueNet,
    train: ValueDataset,
    validation: Optional[ValueDataset] = None,
    config: ValueTrainConfig | None = None,
) -> List[Dict[str, float]]:
    """Fit ``net`` to ``train``; returns a per-epoch history."""
    config = config or ValueTrainConfig()
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    net.to(device)

    features = torch.as_tensor(train.features, dtype=torch.float32, device=device)
    targets = torch.as_tensor(train.targets, dtype=torch.float32, device=device)
    optimiser = torch.optim.Adam(
        net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.HuberLoss(reduction="mean")
    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        net.train()
        order = torch.randperm(len(features), device=device)
        total, batches = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(net(features[batch]), targets[batch])
            loss.backward()
            optimiser.step()
            total += float(loss.item())
            batches += 1
        record = {"epoch": epoch + 1, "train_huber": total / max(batches, 1)}
        if validation is not None:
            record.update(
                {f"val_{k}": v for k, v in evaluate_net(net, validation, device).items()}
            )
        history.append(record)
    return history


@torch.no_grad()
def evaluate_net(
    net: PBSValueNet, dataset: ValueDataset, device: str | torch.device = "cpu"
) -> Dict[str, float]:
    """Regression quality: MSE, mean absolute error, and R^2 in chips."""
    device = torch.device(device)
    net.to(device)
    net.eval()
    features = torch.as_tensor(dataset.features, dtype=torch.float32, device=device)
    predicted = net(features).cpu().numpy().astype(np.float64)
    targets = dataset.targets
    errors = predicted - targets
    variance = float(((targets - targets.mean()) ** 2).mean())
    mse = float((errors**2).mean())
    return {
        "mse": mse,
        "mae": float(np.abs(errors).mean()),
        "max_error": float(np.abs(errors).max()),
        "r2": 1.0 - mse / variance if variance > 0 else float("nan"),
    }
