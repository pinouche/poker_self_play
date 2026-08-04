"""A circular replay buffer of belief states and their searched values.

Same role as paradigm A's replay buffer, but the samples are supervised
regression targets rather than transitions: the network is not learning from
outcomes, it is learning to reproduce what search computed.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from paradigm_b.core.belief.ranges import NUM_HANDS, NUM_PLAYERS
from paradigm_b.leduc.rebel.selfplay import TrainingExample
from paradigm_b.leduc.value_net.dataset import ValueDataset
from paradigm_b.leduc.value_net.features import INPUT_DIM


class ValueReplayBuffer:
    """Fixed-capacity, uniformly sampled, oldest-out."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.features = np.zeros((capacity, INPUT_DIM))
        self.targets = np.zeros((capacity, NUM_PLAYERS, NUM_HANDS))
        self.size = 0
        self._next = 0

    def __len__(self) -> int:
        return self.size

    def add(self, examples: Iterable[TrainingExample]) -> None:
        for example in examples:
            features, targets = example.encode()
            self.features[self._next] = features
            self.targets[self._next] = targets
            self._next = (self._next + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        indices = rng.integers(0, self.size, size=min(batch_size, self.size))
        return self.features[indices], self.targets[indices]

    def dataset(self) -> ValueDataset:
        return ValueDataset(
            features=self.features[: self.size], targets=self.targets[: self.size]
        )
