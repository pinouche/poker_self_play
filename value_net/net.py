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
from model.residual_blocks import FeatureGroupEncoder, ResidualTrunk
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
    embed_ranges: int = 128
    embed_public: int = 64
    hidden_dim: int = 128
    num_residual_blocks: int = 3
    dropout: float = 0.0


class PBSValueNet(nn.Module):
    """``(batch, INPUT_DIM)`` features -> ``(batch, 2, NUM_HANDS)`` values."""

    def __init__(self, config: ValueNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or ValueNetConfig()
        self.range_encoder = FeatureGroupEncoder(RANGE_DIM, self.config.embed_ranges)
        self.public_encoder = FeatureGroupEncoder(PUBLIC_DIM, self.config.embed_public)
        self.project = nn.Linear(
            self.config.embed_ranges + self.config.embed_public, self.config.hidden_dim
        )
        self.trunk = ResidualTrunk(
            self.config.hidden_dim,
            self.config.num_residual_blocks,
            dropout=self.config.dropout,
        )
        self.head = nn.Linear(self.config.hidden_dim, NUM_PLAYERS * NUM_HANDS)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        ranges = features[..., RANGE_SLICE].reshape(-1, NUM_PLAYERS, NUM_HANDS)
        possible = 1.0 - features[..., BOARD_SLICE]
        # The head works in pot-sized units and is scaled back to chips here.
        # Otherwise the same belief state costs the network a different amount
        # to get wrong depending on how much money happens to be in front of it,
        # and the large pots dominate the loss.  (DeepStack does the same.)
        pot = features[..., POT_INDEX].unsqueeze(-1).unsqueeze(-1) * MAX_POT

        encoded = torch.cat(
            [
                self.range_encoder(features[..., RANGE_SLICE]),
                self.public_encoder(features[..., PUBLIC_SLICE]),
            ],
            dim=-1,
        )
        raw = self.head(self.trunk(self.project(encoded)))
        values = raw.reshape(-1, NUM_PLAYERS, NUM_HANDS) * possible.unsqueeze(1) * pot
        return zero_sum_projection(values, ranges, possible)


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
