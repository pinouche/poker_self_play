"""The hold'em counterfactual value network.

Structurally the same as the Leduc one — encode the ranges, encode the public
state, residual trunk, per-hand outputs, masked and projected to zero sum — but
an order of magnitude wider, because a hold'em belief state is 2,652 numbers
rather than twelve.

Two things the Leduc version could ignore:

**The hand mask has to be passed in, not read off the input.** With one board
card, "which hands are impossible" was a single index; with 1,326 combos it is a
1,326-vector that depends on all five board cards, and recomputing it inside the
forward pass would mean re-deriving combinatorics from one-hot features.

**Values are scaled by the pot.** A turn endgame's pot ranges over an order of
magnitude, and without normalising, the large pots dominate the loss exactly as
they did in Leduc — only more so.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from holdem.combos import NUM_COMBOS
from holdem.features import (
    BOARD_DIM,
    BOARD_OFFSET,
    CARD_SET_DIM,
    GLOBAL_CHIP_SCALE,
    INPUT_DIM,
    NUM_CARD_SETS,
    POT_CHIPS_INDEX,
    PUBLIC_DIM,
    RANGE_DIM,
    SCALAR_OFFSET,
)
from model.rebel import CardEmbedding, ReBeLMLP

NUM_PLAYERS = 2


@dataclass
class HoldemValueNetConfig:
    hidden_dim: int = 1536
    num_residual_blocks: int = 6
    num_hidden_layers: int | None = None
    card_embedding_dim: int = 128
    agent_embedding_dim: int = 16
    # Accepted for checkpoint/config compatibility with the old grouped encoder.
    embed_ranges: int | None = None
    embed_public: int | None = None
    dropout: float = 0.0
    # Chips corresponding to a scaled output of 1.0; see the forward pass.
    pot_scale: float = GLOBAL_CHIP_SCALE

    @property
    def hidden_layers(self) -> int:
        return self.num_residual_blocks if self.num_hidden_layers is None else self.num_hidden_layers


class HoldemValueNet(nn.Module):
    """``(batch, INPUT_DIM)`` -> ``(batch, 2, 1326)`` counterfactual values."""

    def __init__(self, config: HoldemValueNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or HoldemValueNetConfig()
        self.board_embedding = CardEmbedding(52, self.config.card_embedding_dim)
        self.agent_embedding = nn.Embedding(NUM_PLAYERS, self.config.agent_embedding_dim)
        input_dim = RANGE_DIM + (PUBLIC_DIM - BOARD_DIM)
        input_dim += NUM_CARD_SETS * self.config.card_embedding_dim
        input_dim += self.config.agent_embedding_dim
        self.trunk = ReBeLMLP(input_dim, self.config.hidden_dim, self.config.hidden_layers)
        self.head = nn.Linear(self.config.hidden_dim, NUM_COMBOS)
        self.pot_scale = self.config.pot_scale

    def forward(self, features: torch.Tensor, possible: torch.Tensor) -> torch.Tensor:
        ranges = features[..., :RANGE_DIM].reshape(-1, NUM_PLAYERS, NUM_COMBOS)
        # In *chips*, not as a fraction: the head works in pot-sized units, so
        # this is what converts its output back to money.  Scaling by the
        # fraction instead leaves the head having to produce numbers two orders
        # of magnitude larger than its inputs, which it learns badly.
        pot = features[..., POT_CHIPS_INDEX].reshape(-1, 1, 1) * self.pot_scale

        raw = torch.stack(
            [self._indexed_raw(features, player) for player in range(NUM_PLAYERS)], dim=1
        )
        values = raw
        values = values * possible.unsqueeze(1) * pot
        excess = (ranges * values).sum(dim=(-2, -1), keepdim=True)
        return values - 0.5 * excess * possible.unsqueeze(1)

    def _indexed_raw(self, features: torch.Tensor, agent_index: int | torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        if not torch.is_tensor(agent_index):
            agent_index = torch.full((batch,), agent_index, device=features.device, dtype=torch.long)
        else:
            agent_index = agent_index.to(device=features.device, dtype=torch.long).reshape(-1)
            if agent_index.numel() == 1:
                agent_index = agent_index.expand(batch)
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
                self.agent_embedding(agent_index),
            ),
            dim=-1,
        )
        return self.head(self.trunk(encoded))

    def forward_indexed(
        self, features: torch.Tensor, possible: torch.Tensor, agent_index: torch.Tensor
    ) -> torch.Tensor:
        values = self(features, possible)
        indices = agent_index.to(device=features.device, dtype=torch.long).reshape(-1)
        if indices.numel() == 1:
            indices = indices.expand(features.shape[0])
        return values[torch.arange(features.shape[0], device=features.device), indices]


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
