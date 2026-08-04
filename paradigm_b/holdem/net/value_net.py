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
import torch.nn.functional as F

from paradigm_b.holdem.engine.combos import NUM_COMBOS
from paradigm_b.holdem.net.features import (
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
from common.nets import CardEmbedding, ReBeLMLP

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
        """``(batch, INPUT_DIM)`` -> ``(batch, 2, 1326)``, in one trunk pass.

        Both players' values come from the same trunk with only the agent
        embedding differing, so this used to be written as the obvious two
        passes — one per player, batch rows each.  That recomputed the board
        embedding twice and, more importantly, submitted two half-sized batches
        to the GPU where one full-sized batch would do.  Fusing them is 1.13x
        faster on both CPU and MPS.

        The fused form assembles *byte-identical* trunk inputs — the tests check
        that — but its output is not bit-identical in float32: torch picks a
        different GEMM tiling for a 2B-row matrix than for a B-row one, and
        float32 addition is not associative.  The gap is ~3e-7 absolute on the
        raw head, and vanishes to ~7e-16 in float64, which is how the tests
        establish that the algebra rather than the rounding is unchanged.  For
        scale, a swapped player pair would show ~1e-1.

        It matters more than 1.13x suggests, because during data generation the
        network is the binding constraint: eight actor processes at batch ~224
        already saturate this machine's GPU, so the forward pass is what caps
        label throughput.
        """
        batch = features.shape[0]
        ranges = features[..., :RANGE_DIM].reshape(-1, NUM_PLAYERS, NUM_COMBOS)
        # In *chips*, not as a fraction: the head works in pot-sized units, so
        # this is what converts its output back to money.  Scaling by the
        # fraction instead leaves the head having to produce numbers two orders
        # of magnitude larger than its inputs, which it learns badly.
        pot = features[..., POT_CHIPS_INDEX].reshape(-1, 1, 1) * self.pot_scale

        board = features[..., BOARD_OFFSET:SCALAR_OFFSET].reshape(
            batch, NUM_CARD_SETS, CARD_SET_DIM
        )
        # Everything except the agent embedding is shared between the two
        # players, so it is built once.
        shared = torch.cat(
            (
                features[..., :RANGE_DIM],
                self.board_embedding(board).reshape(batch, -1),
                features[..., SCALAR_OFFSET:],
            ),
            dim=-1,
        )
        agents = torch.arange(NUM_PLAYERS, device=features.device, dtype=torch.long)
        # Rows ``[0, batch)`` are player 0, ``[batch, 2 * batch)`` player 1.
        #
        # The obvious way to build that is to tile ``shared`` and hand the
        # trunk a 2B-row matrix, which is what this did.  But the trunk's first
        # layer is affine and its input is a concatenation, so it splits:
        #
        #     W [shared ; agent] + b  ==  W[:, :s] shared + b  +  W[:, s:] agent
        #
        # and the shared half — the wide one, 3,080 of the 3,096 input columns
        # — only has B *distinct* rows.  Computing it once at B rows and adding
        # each player's (2-row) agent term is the same arithmetic for half the
        # work, and it drops the tiled copy of the input as well.  Measured on
        # the first layer alone at batch 224: **1.93x**.
        #
        # The column slices are non-contiguous views, deliberately: BLAS takes
        # an arbitrary leading dimension, so torch passes them straight through
        # without materialising a copy, and a slice costs nothing to take.
        # Splitting the module into two ``Linear``s instead measured the same
        # (1.91x) and would have changed every checkpoint's state dict.
        layers = self.trunk.layers
        first = layers[0]
        width = shared.shape[-1]
        hidden = F.linear(shared, first.weight[..., :width], first.bias)
        offsets = F.linear(self.agent_embedding(agents), first.weight[..., width:])
        hidden = torch.cat(
            [hidden + offsets[player] for player in range(NUM_PLAYERS)], dim=0
        )
        # ``layers[0]`` is done; run the rest.  Indexed rather than sliced
        # because ``Sequential[1:]`` builds a fresh module on every call.
        for index in range(1, len(layers)):
            hidden = layers[index](hidden)
        raw = (
            self.head(hidden)
            .reshape(NUM_PLAYERS, batch, NUM_COMBOS)
            .transpose(0, 1)
        )

        values = raw * possible.unsqueeze(1) * pot
        excess = (ranges * values).sum(dim=(-2, -1), keepdim=True)
        return values - 0.5 * excess * possible.unsqueeze(1)

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
