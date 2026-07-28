"""Drawing training situations across all three postflop streets.

The offline artifact spans river, turn and flop, and both arms are scored on
all three.  The online arm therefore has to sample across streets too — a
:class:`~holdem.sampling.SituationConfig` pins ``board_cards`` to one street, so
on its own it would train an agent on turn endgames and then test it on flops.
That is not a fair fight, and it would show up as an "iterative is worse"
result that is really just "iterative was never shown a flop".

:class:`StreetMix` is the fix: proportions over the three streets, which the
experiment sets to match the artifact's own composition so both arms see the
same distribution over streets and differ only in *when their labels were made*.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Tuple

import numpy as np

from holdem.public_tree import PublicState
from holdem.sampling import SituationConfig, sample_situation
from holdem.space import TurnEndgameSpace

STREET_NAMES: Dict[int, str] = {3: "flop", 4: "turn", 5: "river"}
STREET_CARDS: Dict[str, int] = {name: cards for cards, name in STREET_NAMES.items()}


@dataclass
class StreetMix:
    """How often each street is drawn.  Weights need not sum to one."""

    river: float = 0.5
    turn: float = 0.3
    flop: float = 0.2

    def normalised(self) -> Dict[str, float]:
        weights = {"river": self.river, "turn": self.turn, "flop": self.flop}
        positive = {k: float(v) for k, v in weights.items() if v > 0}
        if not positive:
            raise ValueError("at least one street must have positive weight")
        total = sum(positive.values())
        return {k: v / total for k, v in positive.items()}

    @classmethod
    def from_counts(cls, counts: Dict[str, float]) -> "StreetMix":
        """The mix implied by an artifact's per-source example counts.

        This is how the online arm is made to match the offline one: whatever
        proportion of the frozen dataset is turn data, the same proportion of
        online trajectories starts on the turn.
        """
        return cls(
            river=float(counts.get("river", 0.0)),
            turn=float(counts.get("turn", 0.0)),
            flop=float(counts.get("flop", 0.0)),
        )

    def to_dict(self) -> Dict[str, float]:
        return {"river": self.river, "turn": self.turn, "flop": self.flop}


def sample_street(rng: np.random.Generator, mix: StreetMix) -> int:
    """Board-card count for one draw from ``mix``."""
    weights = mix.normalised()
    names = tuple(weights)
    chosen = names[int(rng.choice(len(names), p=[weights[n] for n in names]))]
    return STREET_CARDS[chosen]


def sample_mixed_situation(
    rng: np.random.Generator, config: SituationConfig, mix: StreetMix
) -> Tuple[TurnEndgameSpace, PublicState, np.ndarray, int]:
    """One random situation whose street is itself drawn from ``mix``.

    Returns the usual ``(space, root, reach)`` triple plus the street, so a
    caller can account for what it drew without re-deriving it from the board.
    """
    board_cards = sample_street(rng, mix)
    space, root, reach = sample_situation(
        rng, replace(config, board_cards=board_cards)
    )
    return space, root, reach, board_cards
