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

from paradigm_b.holdem.engine.public_tree import PublicState
from paradigm_b.holdem.data.sampling import (
    SituationConfig,
    initial_situation,
    sample_situation,
)
from paradigm_b.holdem.engine.space import TurnEndgameSpace

STREET_NAMES: Dict[int, str] = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}
STREET_CARDS: Dict[str, int] = {name: cards for cards, name in STREET_NAMES.items()}

# How many labels a street collects per trajectory that reaches it: a root
# label always, plus the pre-deal label recorded on the way *out* of it.  The
# river has no deal to stop in front of, so it collects only the one.
LAYERS_PER_STREET: Dict[str, int] = {"preflop": 2, "flop": 2, "turn": 2, "river": 1}


def rounds_from_street(board_cards: int) -> int:
    """Betting rounds still to play from a street showing ``board_cards``."""
    return 4 if board_cards == 0 else 6 - board_cards


def labels_from_street(board_cards: int) -> int:
    """Labels one trajectory yields from that street: a root and a pre-deal one."""
    return 2 * rounds_from_street(board_cards) - 1


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
    def from_label_counts(cls, counts: Dict[str, float]) -> "StreetMix":
        """Start-street weights that make the *labels* match ``counts``.

        These are two different things, and conflating them silently skews the
        online arm's data toward the river.  A trajectory produces one label at
        every street it passes through on the way down: a flop start yields
        flop, turn *and* river labels; a turn start yields turn and river; a
        river start yields one.  So copying an artifact's label proportions
        straight onto the starting street over-produces river labels, because
        turn and flop starts pass through the river as well.

        Inverting it: the share of labels at a street is the total start
        probability at or above it, so the start probabilities are the
        *differences* of the cumulative label shares, deepest street first.

        Note the constraint this exposes — descending trajectories force
        ``river >= turn >= flop`` in label counts, so a target with more flop
        labels than river ones is unreachable.  Such a target is clamped rather
        than silently approximated by something else.
        """
        shares = {
            street: max(float(counts.get(street, 0.0)), 0.0)
            / LAYERS_PER_STREET[street]
            for street in ("flop", "turn", "river")
        }
        return cls(
            flop=shares["flop"],
            turn=max(shares["turn"] - shares["flop"], 0.0),
            river=max(shares["river"] - shares["turn"], 0.0),
        )

    def label_shares(self) -> Dict[str, float]:
        """The label mixture this start-street mixture actually produces.

        A street's share is the chance of starting at or above it — every such
        trajectory passes through — times the number of layers it collects
        there, which is two everywhere but the river.
        """
        starts = self.normalised()
        cumulative, running = {}, 0.0
        for street in ("flop", "turn", "river"):
            running += starts.get(street, 0.0)
            cumulative[street] = running * LAYERS_PER_STREET[street]
        total = sum(cumulative.values())
        return {k: v / total for k, v in cumulative.items()}

    @property
    def labels_per_trajectory(self) -> float:
        """Expected labels one trajectory yields — what budget sizing needs.

        ``2n - 1`` for a start with ``n`` betting rounds below it, not ``n``.
        Each subgame emits a label at its root *and* one at the pre-deal belief
        state it stopped in front of (all but the last, which runs to
        showdown), which is what ReBeL's two layers of values cost in data.
        """
        starts = self.normalised()
        return sum(
            probability * labels_from_street(STREET_CARDS[street])
            for street, probability in starts.items()
        )

    def to_dict(self) -> Dict[str, float]:
        return {"river": self.river, "turn": self.turn, "flop": self.flop}


def sample_street(rng: np.random.Generator, mix: StreetMix) -> int:
    """Board-card count for one draw from ``mix``."""
    weights = mix.normalised()
    names = tuple(weights)
    chosen = names[int(rng.choice(len(names), p=[weights[n] for n in names]))]
    return STREET_CARDS[chosen]


def sample_trajectory_start(
    rng: np.random.Generator,
    config: SituationConfig,
    mix: StreetMix,
    preflop: bool,
) -> Tuple[TurnEndgameSpace, PublicState, np.ndarray, int]:
    """Where one trajectory begins, under either regime.

    ``preflop=True`` is ReBeL's: every hand starts at the blinds and the
    training distribution is whatever self play reaches from there, so ``mix``
    is not consulted at all.  ``False`` keeps the DeepStack-style sampler, which
    is what a street-wise dataset and the fixed-vs-iterative comparison still
    need — those score each street separately and cannot do that if every
    trajectory enters the flop through the same preflop funnel.
    """
    if preflop:
        space, root, reach = initial_situation(rng, config)
        return space, root, reach, 0
    return sample_mixed_situation(rng, config, mix)


def preflop_labels_per_trajectory() -> int:
    """Labels a preflop-rooted trajectory yields: four roots, three pre-deal."""
    return labels_from_street(0)


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
