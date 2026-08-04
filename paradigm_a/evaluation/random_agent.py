"""Non-learning baselines.

A uniform-over-legal-actions policy *is* the random policy for a masked
discrete action space, so ``UniformLegalAgent`` is an alias of ``RandomAgent``
rather than a second implementation.  The other two bots are trivially
exploitable but useful as fixed reference points: a learner that cannot beat a
calling station is broken.
"""

from __future__ import annotations

import random
from typing import Dict

import numpy as np

from paradigm_a.environment.state import CALL, CHECK, FOLD
from paradigm_a.training.self_play import ActionChoice, Agent


def _uniform_over_legal(legal_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(legal_mask, dtype=np.float64)
    total = mask.sum()
    if total <= 0:
        raise ValueError("no legal actions")
    return (mask / total).astype(np.float32)


def _choose(probs: np.ndarray, rng: random.Random) -> int:
    return int(rng.choices(range(len(probs)), weights=probs.tolist(), k=1)[0])


class RandomAgent(Agent):
    """Uniform over the legal actions."""

    name = "random"

    def act(self, observation, flat_observation, legal_mask, rng) -> ActionChoice:
        probs = _uniform_over_legal(legal_mask)
        return ActionChoice(action=_choose(probs, rng), policy=probs, value=0.0)


#: The "uniform legal-action policy" baseline: identical by construction.
UniformLegalAgent = RandomAgent


class CallingStationAgent(Agent):
    """Never folds, never raises: checks when it can, otherwise calls."""

    name = "calling_station"

    def act(self, observation, flat_observation, legal_mask, rng) -> ActionChoice:
        probs = np.zeros(len(legal_mask), dtype=np.float32)
        for action in (CHECK, CALL):
            if legal_mask[action]:
                probs[action] = 1.0
                return ActionChoice(action=action, policy=probs, value=0.0)
        # Only reachable if neither check nor call is available (all-in spots).
        return RandomAgent().act(observation, flat_observation, legal_mask, rng)


class AlwaysFoldAgent(Agent):
    """Folds whenever folding is legal; checks otherwise. A lower bound."""

    name = "always_fold"

    def act(self, observation, flat_observation, legal_mask, rng) -> ActionChoice:
        probs = np.zeros(len(legal_mask), dtype=np.float32)
        all_in = len(legal_mask) - 1  # ALL_IN is always the last action
        for action in (FOLD, CHECK, CALL, all_in):
            if legal_mask[action]:
                probs[action] = 1.0
                return ActionChoice(action=action, policy=probs, value=0.0)
        return RandomAgent().act(observation, flat_observation, legal_mask, rng)


BASELINE_AGENTS: Dict[str, type] = {
    "random": RandomAgent,
    "uniform": UniformLegalAgent,
    "calling_station": CallingStationAgent,
    "always_fold": AlwaysFoldAgent,
}
