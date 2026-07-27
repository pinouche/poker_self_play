"""The counterfactual value network: V(PBS) -> a value for every hand.

This is the piece that replaces search below the depth limit.  Given the public
state and both ranges it predicts, for each player and each hand, what that hand
is worth in chips if play continued from here under equilibrium — the quantity
CFR would have computed by searching to the end of the game.

Two structural constraints are built into the module rather than left to the
optimiser:

**Impossible hands are exactly zero.**  Nobody can hold the board card, so its
value is masked out instead of being learned as "approximately irrelevant".

**Values are zero-sum.**  The reach-weighted values of the two players must sum
to zero, because one player's chips are the other's.  The network's raw output
will not satisfy that, so the excess is measured and half of it subtracted from
each side — DeepStack's zero-sum layer.  It costs nothing, it is differentiable,
and it removes an entire class of drift where both players' predicted values
creep upward together.

The trunk is the same ``FeatureGroupEncoder`` + ``ResidualTrunk`` pair paradigm A
uses; only the input and output layout differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from belief.ranges import NUM_HANDS, NUM_PLAYERS
from model.rebel import CardEmbedding, ReBeLMLP
from value_net.features import (
    BOARD_SLICE,
    MAX_POT,
    POT_INDEX,
    PUBLIC_DIM,
    PUBLIC_SLICE,
    RANGE_DIM,
    RANGE_SLICE,
)


@dataclass
class ValueNetConfig:
    hidden_dim: int = 1536
    num_residual_blocks: int = 6
    num_hidden_layers: int | None = None
    card_embedding_dim: int = 64
    agent_embedding_dim: int = 16
    # Accepted for checkpoint/config compatibility with the old grouped encoder.
    embed_ranges: int | None = None
    embed_public: int | None = None
    dropout: float = 0.0

    @property
    def hidden_layers(self) -> int:
        return self.num_residual_blocks if self.num_hidden_layers is None else self.num_hidden_layers


class PBSValueNet(nn.Module):
    """``(batch, INPUT_DIM)`` features -> ``(batch, 2, NUM_HANDS)`` values."""

    def __init__(self, config: ValueNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or ValueNetConfig()
        self.board_embedding = CardEmbedding(
            NUM_HANDS, self.config.card_embedding_dim, num_suits=2
        )
        self.agent_embedding = nn.Embedding(NUM_PLAYERS, self.config.agent_embedding_dim)
        input_dim = RANGE_DIM + (PUBLIC_DIM - NUM_HANDS) + self.config.card_embedding_dim
        input_dim += self.config.agent_embedding_dim
        self.trunk = ReBeLMLP(input_dim, self.config.hidden_dim, self.config.hidden_layers)
        self.head = nn.Linear(self.config.hidden_dim, NUM_HANDS)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        ranges = features[..., RANGE_SLICE].reshape(-1, NUM_PLAYERS, NUM_HANDS)
        possible = 1.0 - features[..., BOARD_SLICE]
        # The head works in pot-sized units and is scaled back to chips here.
        # Otherwise the same belief state costs the network a different amount
        # to get wrong depending on how much money happens to be in front of it,
        # and the large pots dominate the loss.  (DeepStack does the same.)
        pot = features[..., POT_INDEX].unsqueeze(-1).unsqueeze(-1) * MAX_POT

        raw = torch.stack(
            [self._indexed_raw(features, player) for player in range(NUM_PLAYERS)], dim=1
        )
        values = raw * possible.unsqueeze(1) * pot
        return zero_sum_projection(values, ranges, possible)

    def _indexed_raw(self, features: torch.Tensor, agent_index: int | torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        if not torch.is_tensor(agent_index):
            agent_index = torch.full((batch,), agent_index, device=features.device, dtype=torch.long)
        else:
            agent_index = agent_index.to(device=features.device, dtype=torch.long).reshape(-1)
            if agent_index.numel() == 1:
                agent_index = agent_index.expand(batch)
        public = features[..., PUBLIC_SLICE]
        board = features[..., BOARD_SLICE]
        non_board = torch.cat((public[..., :0], public[..., NUM_HANDS:]), dim=-1)
        encoded = torch.cat(
            (features[..., RANGE_SLICE], self.board_embedding(board), non_board, self.agent_embedding(agent_index)),
            dim=-1,
        )
        return self.head(self.trunk(encoded))

    def forward_indexed(self, features: torch.Tensor, agent_index: torch.Tensor) -> torch.Tensor:
        """Paper-style query for the indexed agent's infostate values."""
        values = self(features)
        indices = agent_index.to(device=features.device, dtype=torch.long).reshape(-1)
        if indices.numel() == 1:
            indices = indices.expand(features.shape[0])
        return values[torch.arange(features.shape[0], device=features.device), indices]


def zero_sum_projection(
    values: torch.Tensor, ranges: torch.Tensor, possible: torch.Tensor
) -> torch.Tensor:
    """Shift both players' values so the reach-weighted total is zero.

    ``sum_h range_0[h] v_0[h] + sum_h range_1[h] v_1[h]`` is what one player
    wins and the other loses, so it must vanish.  Subtracting half the measured
    excess from every (possible) hand of both players achieves that exactly,
    since each range sums to one.
    """
    excess = (ranges * values).sum(dim=(-2, -1), keepdim=True)
    return values - 0.5 * excess * possible.unsqueeze(1)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
