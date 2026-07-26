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
    BOARD_OFFSET,
    GLOBAL_CHIP_SCALE,
    INPUT_DIM,
    POT_CHIPS_INDEX,
    PUBLIC_DIM,
    RANGE_DIM,
)
from model.residual_blocks import FeatureGroupEncoder, ResidualTrunk

NUM_PLAYERS = 2


@dataclass
class HoldemValueNetConfig:
    embed_ranges: int = 512
    embed_public: int = 128
    hidden_dim: int = 512
    num_residual_blocks: int = 4
    dropout: float = 0.0
    # Chips corresponding to a scaled output of 1.0; see the forward pass.
    pot_scale: float = GLOBAL_CHIP_SCALE


class HoldemValueNet(nn.Module):
    """``(batch, INPUT_DIM)`` -> ``(batch, 2, 1326)`` counterfactual values."""

    def __init__(self, config: HoldemValueNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or HoldemValueNetConfig()
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
        self.head = nn.Linear(self.config.hidden_dim, NUM_PLAYERS * NUM_COMBOS)
        self.pot_scale = self.config.pot_scale

    def forward(self, features: torch.Tensor, possible: torch.Tensor) -> torch.Tensor:
        ranges = features[..., :RANGE_DIM].reshape(-1, NUM_PLAYERS, NUM_COMBOS)
        # In *chips*, not as a fraction: the head works in pot-sized units, so
        # this is what converts its output back to money.  Scaling by the
        # fraction instead leaves the head having to produce numbers two orders
        # of magnitude larger than its inputs, which it learns badly.
        pot = features[..., POT_CHIPS_INDEX].reshape(-1, 1, 1) * self.pot_scale

        encoded = torch.cat(
            [
                self.range_encoder(features[..., :RANGE_DIM]),
                self.public_encoder(features[..., BOARD_OFFSET:]),
            ],
            dim=-1,
        )
        raw = self.head(self.trunk(self.project(encoded)))
        values = raw.reshape(-1, NUM_PLAYERS, NUM_COMBOS)
        values = values * possible.unsqueeze(1) * pot
        excess = (ranges * values).sum(dim=(-2, -1), keepdim=True)
        return values - 0.5 * excess * possible.unsqueeze(1)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
