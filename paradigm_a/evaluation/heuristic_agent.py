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

from paradigm_a.environment.state import (
    CALL,
    CHECK,
    DEFAULT_ACTION_SPACE,
    FOLD,
    ActionSpace,
)
from paradigm_a.representation.canonicalizer import estimate_equity
from paradigm_a.training.self_play import ActionChoice, Agent


def _preferences(space: ActionSpace) -> Dict[str, Sequence[int]]:
    """Preference orders expressed over whatever sizings the space provides.

    The bot thinks in terms of "biggest available raise" rather than fixed
    action ids, so it keeps working when the bet abstraction is made finer.
    """
    bets, raises = space.bet_ids, space.raise_ids
    big_bets = tuple(reversed(bets))          # largest first
    big_raises = tuple(reversed(raises))
    mid = lambda ids: ids[len(ids) // 2 :][::-1] + ids[: len(ids) // 2][::-1]
    return {
        "big_raise": (*big_raises, *big_bets, CALL, CHECK),
        "value_raise": (*mid(raises), *mid(bets), CALL, CHECK),
        "small_bet": (*bets, CHECK, CALL),
        "passive": (CHECK, CALL, FOLD),
        "give_up": (CHECK, FOLD, CALL),
        "fallback": (CHECK, CALL, FOLD, *bets, *raises, space.all_in),
    }


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
        action_space: Optional[ActionSpace] = None,
    ) -> None:
        self.space = action_space or DEFAULT_ACTION_SPACE
        self.preferences = _preferences(self.space)
        self.samples = samples
        self.tightness = tightness
        self.aggression = aggression
        self.seed = seed
        self.rng = random.Random(seed)
        if name:
            self.name = name

    def reset(self) -> None:
        """Restore the internal RNG so a hand replays identically.

        This agent draws from its own RNG (bluff frequency and the Monte-Carlo
        equity rollouts), so without a reset the same deal played at different
        points in a sequence gives different decisions.  Duplicate-deal scoring
        depends on the opponent being reproducible, otherwise the pairing it
        buys is destroyed.
        """
        self.rng = random.Random(self.seed)

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
        for action in self.preferences["fallback"]:
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

        # No rng argument: this takes the memoised, suit-isomorphic path.
        # Bluff frequency below still uses self.rng.
        equity = estimate_equity(hole, board, opponents, self.samples)
        pot_odds = to_call / float(pot + to_call) if to_call > 0 else 0.0

        if to_call <= 0:
            action = self._choose_unraised(equity, legal_mask)
        else:
            action = self._choose_facing_bet(equity, pot_odds, legal_mask)

        policy = np.zeros(len(legal_mask), dtype=np.float32)
        policy[action] = 1.0
        return ActionChoice(action=action, policy=policy, value=float(2.0 * equity - 1.0))

    def _choose_unraised(self, equity: float, legal_mask: np.ndarray) -> int:
        strong = 0.70 + self.tightness - self.aggression
        decent = 0.55 + self.tightness - self.aggression
        if equity >= strong:
            return self._pick(self.preferences["big_raise"], legal_mask)
        if equity >= decent:
            return self._pick(self.preferences["value_raise"], legal_mask)
        if equity >= 0.45 + self.tightness and self.rng.random() < 0.2 + self.aggression:
            return self._pick(self.preferences["small_bet"], legal_mask)
        return self._pick(self.preferences["passive"], legal_mask)

    def _choose_facing_bet(
        self, equity: float, pot_odds: float, legal_mask: np.ndarray
    ) -> int:
        if equity >= 0.80 + self.tightness - self.aggression:
            return self._pick(self.preferences["big_raise"], legal_mask)
        if equity >= 0.65 + self.tightness - self.aggression:
            return self._pick(self.preferences["value_raise"], legal_mask)
        # Call whenever the price is right, with a margin scaled by tightness.
        if equity >= pot_odds + 0.02 + self.tightness:
            return self._pick((CALL, CHECK, FOLD), legal_mask)
        return self._pick(self.preferences["give_up"], legal_mask)


def tight_aggressive(seed: int = 0, samples: int = 60, action_space=None) -> HeuristicAgent:
    return HeuristicAgent(
        samples=samples, tightness=0.05, aggression=0.05, seed=seed,
        name="tight_aggressive", action_space=action_space,
    )


def loose_passive(seed: int = 0, samples: int = 60, action_space=None) -> HeuristicAgent:
    return HeuristicAgent(
        samples=samples, tightness=-0.10, aggression=-0.10, seed=seed,
        name="loose_passive", action_space=action_space,
    )


HEURISTIC_AGENTS: Dict[str, callable] = {
    "heuristic": lambda seed=0, samples=60, action_space=None: HeuristicAgent(
        seed=seed, samples=samples, action_space=action_space
    ),
    "tight_aggressive": tight_aggressive,
    "loose_passive": loose_passive,
}
