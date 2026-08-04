"""Uniform replay buffer over self-play transitions.

Storage is a set of preallocated ring buffers.  Observations dominate memory
and are kept in float16 (every feature is a normalised scalar or a one-hot, so
half precision is lossless enough); everything else is stored at full width.

Correctness note: ``player_perspective`` records which *seat* generated the
transition.  It is metadata for auditing only -- it is deliberately not fed to
the network, because the observation is already canonicalised and the network
must stay seat-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclass
class Transition:
    observation: np.ndarray        # [obs_dim]
    legal_action_mask: np.ndarray  # [num_actions]
    action: int
    reward: float                  # immediate reward (terminal payout on the last step)
    next_observation: Optional[np.ndarray]
    next_legal_action_mask: Optional[np.ndarray]
    done: bool
    player_perspective: int        # seat that acted (audit metadata only)
    old_policy: np.ndarray         # behaviour policy over actions
    q_target: float                # lambda-return target for the taken action
    value: float                   # V(s) under the behaviour policy


@dataclass
class Batch:
    observations: np.ndarray
    legal_action_masks: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: Optional[np.ndarray]
    next_legal_action_masks: np.ndarray
    dones: np.ndarray
    player_perspectives: np.ndarray
    old_policies: np.ndarray
    q_targets: np.ndarray
    values: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        num_actions: int,
        store_next_obs: bool = True,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.num_actions = int(num_actions)
        self.store_next_obs = store_next_obs

        self._observations = np.zeros((capacity, observation_dim), dtype=np.float16)
        self._legal_masks = np.zeros((capacity, num_actions), dtype=np.uint8)
        self._actions = np.zeros(capacity, dtype=np.int16)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._next_observations = (
            np.zeros((capacity, observation_dim), dtype=np.float16) if store_next_obs else None
        )
        self._next_legal_masks = np.zeros((capacity, num_actions), dtype=np.uint8)
        self._dones = np.zeros(capacity, dtype=np.uint8)
        self._perspectives = np.zeros(capacity, dtype=np.int8)
        self._old_policies = np.zeros((capacity, num_actions), dtype=np.float32)
        self._q_targets = np.zeros(capacity, dtype=np.float32)
        self._values = np.zeros(capacity, dtype=np.float32)

        self._cursor = 0
        self._size = 0
        self._total_added = 0

    # --- writing -----------------------------------------------------------
    def add(self, transition: Transition) -> None:
        i = self._cursor
        self._observations[i] = transition.observation
        self._legal_masks[i] = transition.legal_action_mask
        self._actions[i] = transition.action
        self._rewards[i] = transition.reward
        self._dones[i] = 1 if transition.done else 0
        self._perspectives[i] = transition.player_perspective
        self._old_policies[i] = transition.old_policy
        self._q_targets[i] = transition.q_target
        self._values[i] = transition.value

        if transition.next_observation is not None:
            if self._next_observations is not None:
                self._next_observations[i] = transition.next_observation
            self._next_legal_masks[i] = transition.next_legal_action_mask
        else:
            if self._next_observations is not None:
                self._next_observations[i] = 0
            self._next_legal_masks[i] = 0

        self._cursor = (self._cursor + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._total_added += 1

    def extend(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    # --- reading -----------------------------------------------------------
    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> Batch:
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        rng = rng or np.random.default_rng()
        size = min(batch_size, self._size)
        indices = rng.integers(0, self._size, size=size)
        return self.get(indices)

    def get(self, indices: Sequence[int]) -> Batch:
        indices = np.asarray(indices, dtype=np.int64)
        return Batch(
            observations=self._observations[indices].astype(np.float32),
            legal_action_masks=self._legal_masks[indices].astype(np.float32),
            actions=self._actions[indices].astype(np.int64),
            rewards=self._rewards[indices].copy(),
            next_observations=(
                self._next_observations[indices].astype(np.float32)
                if self._next_observations is not None
                else None
            ),
            next_legal_action_masks=self._next_legal_masks[indices].astype(np.float32),
            dones=self._dones[indices].astype(np.float32),
            player_perspectives=self._perspectives[indices].astype(np.int64),
            old_policies=self._old_policies[indices].copy(),
            q_targets=self._q_targets[indices].copy(),
            values=self._values[indices].copy(),
        )

    def recent(self, n: int) -> Batch:
        """The ``n`` most recently written transitions (diagnostics)."""
        n = min(n, self._size)
        start = (self._cursor - n) % self.capacity
        indices = [(start + i) % self.capacity for i in range(n)]
        return self.get(indices)

    # --- introspection -----------------------------------------------------
    def __len__(self) -> int:
        return self._size

    @property
    def total_added(self) -> int:
        return self._total_added

    def is_ready(self, minimum: int) -> bool:
        return self._size >= minimum

    def stats(self) -> dict:
        if self._size == 0:
            return {"size": 0}
        targets = self._q_targets[: self._size]
        return {
            "size": self._size,
            "total_added": self._total_added,
            "q_target_mean": float(targets.mean()),
            "q_target_std": float(targets.std()),
            "terminal_fraction": float(self._dones[: self._size].mean()),
        }

    def memory_bytes(self) -> int:
        arrays = [
            self._observations,
            self._legal_masks,
            self._actions,
            self._rewards,
            self._next_legal_masks,
            self._dones,
            self._perspectives,
            self._old_policies,
            self._q_targets,
            self._values,
        ]
        if self._next_observations is not None:
            arrays.append(self._next_observations)
        return sum(a.nbytes for a in arrays)
