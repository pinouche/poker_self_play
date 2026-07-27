"""Neural primitives shared by ReBeL's poker value and policy networks."""

from __future__ import annotations

import torch
import torch.nn as nn


class CardEmbedding(nn.Module):
    """Deep CFR card encoding: rank + suit + identity, summed as a set."""

    def __init__(self, num_cards: int, embedding_dim: int, num_suits: int = 4) -> None:
        super().__init__()
        if num_cards % num_suits:
            raise ValueError("num_cards must be divisible by num_suits")
        self.rank_embedding = nn.Embedding(num_cards // num_suits, embedding_dim)
        self.suit_embedding = nn.Embedding(num_suits, embedding_dim)
        self.card_embedding = nn.Embedding(num_cards, embedding_dim)
        cards = torch.arange(num_cards)
        self.register_buffer("rank_indices", cards // num_suits, persistent=False)
        self.register_buffer("suit_indices", cards % num_suits, persistent=False)

    def forward(self, card_indicators: torch.Tensor) -> torch.Tensor:
        weights = (
            self.rank_embedding(self.rank_indices)
            + self.suit_embedding(self.suit_indices)
            + self.card_embedding.weight
        )
        return card_indicators @ weights


class ReBeLMLP(nn.Module):
    """Fully connected LayerNorm/GeLU stack used in the ReBeL paper."""

    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.extend((nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()))
            in_dim = hidden_dim
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)