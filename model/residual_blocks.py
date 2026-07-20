"""Residual building blocks for the shared trunk.

The observation is structured rather than spatial, so the trunk is a residual
*MLP* stack: same ResNet topology (two normalised linear layers plus an
identity skip), without convolutions.  LayerNorm is used instead of BatchNorm
so that a batch of one behaves identically to a batch of many — inference
serves single hands.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    """x -> Linear -> Norm -> ReLU -> Linear -> Norm -> (+x) -> ReLU"""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.norm1(self.fc1(x)))
        out = self.dropout(out)
        out = self.norm2(self.fc2(out))
        return self.act(out + residual)


class FeatureGroupEncoder(nn.Module):
    """Per-group input projection: Linear -> LayerNorm -> ReLU.

    The LayerNorm also absorbs the differing scales of the raw features (chip
    counts in big blinds versus one-hot indicators).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.fc(x)))


class ResidualTrunk(nn.Module):
    """A stack of residual MLP blocks."""

    def __init__(self, dim: int, num_blocks: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)
