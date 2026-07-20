"""A fixed rule-based poker bot used as an evaluation opponent.

It plays a simple equity-versus-pot-odds strategy: estimate win probability by
Monte-Carlo rollout against random opponent holdings, then bet, call or fold
according to fixed thresholds.  It only ever reads its own hole cards and the
public board -- the same information the network gets -- so beating it is a
meaningful signal rather than an artefact of asymmetric information.

It is not a strong bot.  It does not bluff, does not adjust to opponents and
ignores position beyond the pot odds it faces.
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Sequence

import numpy as np

from environment.state import (
    ALL_IN,
    BET_LARGE,
    BET_MEDIUM,
    BET_SMALL,
    CALL,
    CHECK,
    FOLD,
    NUM_ACTIONS,
    RAISE_LARGE,
    RAISE_MEDIUM,
    RAISE_SMALL,
)
from representation.canonicalizer import estimate_equity
from training.self_play import ActionChoice, Agent

VALUE_RAISE = (RAISE_MEDIUM, RAISE_SMALL, BET_MEDIUM, BET_SMALL, CALL, CHECK)
BIG_RAISE = (RAISE_LARGE, RAISE_MEDIUM, BET_LARGE, BET_MEDIUM, CALL, CHECK)
SMALL_BET = (BET_SMALL, BET_MEDIUM, CHECK, CALL)
PASSIVE = (CHECK, CALL, FOLD)
GIVE_UP = (CHECK, FOLD, CALL)


class HeuristicAgent(Agent):
    """Equity-threshold bot.

    ``tightness`` shifts every calling/betting threshold upward (more folding);
    ``aggression`` shifts value bets toward larger sizings.
    """

    name = "heuristic"

    def __init__(
        self,
        samples: int = 60,
        tightness: float = 0.0,
        aggression: float = 0.0,
        seed: int = 0,
        name: Optional[str] = None,
    ) -> None:
        self.samples = samples
        self.tightness = tightness
        self.aggression = aggression
        self.rng = random.Random(seed)
        if name:
            self.name = name

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def _first_legal(preferences: Sequence[int], legal_mask: np.ndarray) -> Optional[int]:
        for action in preferences:
            if legal_mask[action]:
                return int(action)
        return None

    def _pick(self, preferences: Sequence[int], legal_mask: np.ndarray) -> int:
        action = self._first_legal(preferences, legal_mask)
        if action is not None:
            return action
        # Fall back to anything legal; ALL_IN last so we do not stack off blindly.
        for action in (CHECK, CALL, FOLD, BET_SMALL, RAISE_SMALL, ALL_IN):
            if legal_mask[action]:
                return int(action)
        return int(np.flatnonzero(legal_mask)[0])

    # --- policy ------------------------------------------------------------
    def act(self, observation, flat_observation, legal_mask, rng) -> ActionChoice:
        meta = observation["meta"]
        hole = meta["hole_cards"]
        board = meta["board"]
        opponents = max(1, meta["num_active_opponents"])
        to_call = meta["to_call"]
        pot = meta["pot"]

        equity = estimate_equity(hole, board, opponents, self.samples, self.rng)
        pot_odds = to_call / float(pot + to_call) if to_call > 0 else 0.0

        if to_call <= 0:
            action = self._choose_unraised(equity, legal_mask)
        else:
            action = self._choose_facing_bet(equity, pot_odds, legal_mask)

        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
        policy[action] = 1.0
        return ActionChoice(action=action, policy=policy, value=float(2.0 * equity - 1.0))

    def _choose_unraised(self, equity: float, legal_mask: np.ndarray) -> int:
        strong = 0.70 + self.tightness - self.aggression
        decent = 0.55 + self.tightness - self.aggression
        if equity >= strong:
            return self._pick(BIG_RAISE, legal_mask)
        if equity >= decent:
            return self._pick(VALUE_RAISE, legal_mask)
        if equity >= 0.45 + self.tightness and self.rng.random() < 0.2 + self.aggression:
            return self._pick(SMALL_BET, legal_mask)
        return self._pick(PASSIVE, legal_mask)

    def _choose_facing_bet(
        self, equity: float, pot_odds: float, legal_mask: np.ndarray
    ) -> int:
        if equity >= 0.80 + self.tightness - self.aggression:
            return self._pick(BIG_RAISE, legal_mask)
        if equity >= 0.65 + self.tightness - self.aggression:
            return self._pick(VALUE_RAISE, legal_mask)
        # Call whenever the price is right, with a margin scaled by tightness.
        if equity >= pot_odds + 0.02 + self.tightness:
            return self._pick((CALL, CHECK, FOLD), legal_mask)
        return self._pick(GIVE_UP, legal_mask)


def tight_aggressive(seed: int = 0, samples: int = 60) -> HeuristicAgent:
    return HeuristicAgent(
        samples=samples, tightness=0.05, aggression=0.05, seed=seed, name="tight_aggressive"
    )


def loose_passive(seed: int = 0, samples: int = 60) -> HeuristicAgent:
    return HeuristicAgent(
        samples=samples, tightness=-0.10, aggression=-0.10, seed=seed, name="loose_passive"
    )


HEURISTIC_AGENTS: Dict[str, callable] = {
    "heuristic": lambda seed=0: HeuristicAgent(seed=seed),
    "tight_aggressive": tight_aggressive,
    "loose_passive": loose_passive,
}
