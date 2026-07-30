"""Paper-style policy network for Hold'em ReBeL search warm starts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import torch
import torch.nn as nn

from paradigm_b.holdem.engine.combos import NUM_COMBOS
from paradigm_b.holdem.net.features import (
    BOARD_DIM,
    BOARD_OFFSET,
    CARD_SET_DIM,
    INPUT_DIM,
    NUM_CARD_SETS,
    PUBLIC_DIM,
    RANGE_DIM,
    SCALAR_OFFSET,
)
from common.nets import CardEmbedding, ReBeLMLP

MAX_ACTIONS = 9


@dataclass
class HoldemPolicyNetConfig:
    hidden_dim: int = 1536
    num_hidden_layers: int = 6
    card_embedding_dim: int = 128
    agent_embedding_dim: int = 16
    max_actions: int = MAX_ACTIONS


class HoldemPolicyNet(nn.Module):
    """Maps a PBS and indexed agent to legal action probabilities per combo."""

    def __init__(self, config: HoldemPolicyNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or HoldemPolicyNetConfig()
        self.board_embedding = CardEmbedding(52, self.config.card_embedding_dim)
        self.agent_embedding = nn.Embedding(2, self.config.agent_embedding_dim)
        input_dim = RANGE_DIM + (PUBLIC_DIM - BOARD_DIM)
        input_dim += NUM_CARD_SETS * self.config.card_embedding_dim
        input_dim += self.config.agent_embedding_dim
        self.trunk = ReBeLMLP(input_dim, self.config.hidden_dim, self.config.num_hidden_layers)
        self.head = nn.Linear(self.config.hidden_dim, NUM_COMBOS * self.config.max_actions)

    def forward(
        self, features: torch.Tensor, agent_index: torch.Tensor, legal_mask: torch.Tensor
    ) -> torch.Tensor:
        batch = features.shape[0]
        agents = agent_index.to(device=features.device, dtype=torch.long).reshape(-1)
        if agents.numel() == 1:
            agents = agents.expand(batch)
        board = features[..., BOARD_OFFSET:SCALAR_OFFSET].reshape(
            batch, NUM_CARD_SETS, CARD_SET_DIM
        )
        public_features = features[..., SCALAR_OFFSET:]
        embedded_board = self.board_embedding(board).reshape(batch, -1)
        encoded = torch.cat(
            (
                features[..., :RANGE_DIM],
                embedded_board,
                public_features,
                self.agent_embedding(agents),
            ),
            dim=-1,
        )
        logits = self.head(self.trunk(encoded)).reshape(batch, NUM_COMBOS, self.config.max_actions)
        mask = legal_mask.to(device=features.device, dtype=torch.bool).reshape(batch, 1, -1)
        return torch.softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=-1)


def policy_mse_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The paper's probability-regression objective, not cross entropy."""
    return torch.nn.functional.mse_loss(predicted, target)


@torch.no_grad()
def query_policy(
    net: HoldemPolicyNet,
    features: np.ndarray,
    agent_indices: np.ndarray,
    legal_masks: np.ndarray,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """``(N, NUM_COMBOS, MAX_ACTIONS)`` probabilities for a batch of belief states.

    The read side of the network, used by ``INITIALIZE_POLICY`` to warm-start a
    subgame.  Batched because the initialiser queries a whole level of the
    public tree at once.
    """
    device = torch.device(device)
    net.to(device).eval()
    probabilities = net(
        torch.as_tensor(features, dtype=torch.float32, device=device),
        torch.as_tensor(agent_indices, dtype=torch.long, device=device),
        torch.as_tensor(legal_masks, dtype=torch.float32, device=device),
    )
    return probabilities.cpu().numpy().astype(np.float64)


@dataclass
class PolicyExample:
    """One searched PBS policy target, stored with linear 8-bit quantisation."""

    features: np.ndarray
    agent_index: int
    legal_mask: np.ndarray
    target: np.ndarray

    def quantised(self) -> "PolicyExample":
        """The same example with its target already in the buffer's 8-bit form.

        Actors send policy targets to the learner through a queue, and a
        ``(1326, 9)`` float32 target is 47KB against 12KB quantised — worth the
        round trip through uint8 when a trajectory carries one of these per
        decision node.  :meth:`PolicyReplayBuffer.add` accepts either form.
        """
        if self.target.dtype == np.uint8:
            return self
        quantised = np.rint(np.clip(self.target, 0.0, 1.0) * 255.0).astype(np.uint8)
        return PolicyExample(
            features=self.features,
            agent_index=self.agent_index,
            legal_mask=self.legal_mask,
            target=quantised,
        )


class PolicyReplayBuffer:
    """Circular policy buffer using paper-style linear probability quantisation."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.features = np.zeros((capacity, INPUT_DIM), dtype=np.float32)
        self.agent_indices = np.zeros(capacity, dtype=np.int64)
        self.legal_masks = np.zeros((capacity, MAX_ACTIONS), dtype=np.uint8)
        self.targets = np.zeros((capacity, NUM_COMBOS, MAX_ACTIONS), dtype=np.uint8)
        self.size = 0
        self._next = 0

    def __len__(self) -> int:
        return self.size

    def add(self, examples: Sequence[PolicyExample]) -> None:
        for example in examples:
            self.features[self._next] = example.features
            self.agent_indices[self._next] = example.agent_index
            self.legal_masks[self._next] = example.legal_mask
            self.targets[self._next] = (
                example.target
                if example.target.dtype == np.uint8
                else np.rint(np.clip(example.target, 0.0, 1.0) * 255.0)
            )
            self._next = (self._next + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        indices = rng.integers(0, self.size, size=min(batch_size, self.size))
        return (
            self.features[indices],
            self.agent_indices[indices],
            self.legal_masks[indices].astype(np.float32),
            self.targets[indices].astype(np.float32) / 255.0,
        )


def train_policy_net(
    net: HoldemPolicyNet,
    buffer: PolicyReplayBuffer,
    updates: int,
    batch_size: int,
    learning_rate: float,
    rng: np.random.Generator,
    device: str | torch.device = "cpu",
    optimiser: torch.optim.Optimizer | None = None,
) -> float:
    """Train policy probabilities with the paper's MSE objective."""
    if not len(buffer):
        return 0.0
    device = torch.device(device)
    net.to(device).train()
    optimiser = optimiser or torch.optim.Adam(net.parameters(), lr=learning_rate)
    total = 0.0
    for _ in range(updates):
        features, agents, legal, targets = buffer.sample(batch_size, rng)
        optimiser.zero_grad(set_to_none=True)
        prediction = net(
            torch.as_tensor(features, device=device),
            torch.as_tensor(agents, device=device),
            torch.as_tensor(legal, device=device),
        )
        loss = policy_mse_loss(prediction, torch.as_tensor(targets, device=device))
        loss.backward()
        optimiser.step()
        total += float(loss.item())
    return total / updates