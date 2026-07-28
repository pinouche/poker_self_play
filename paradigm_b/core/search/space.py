"""What the solver needs to know about a game's private information.

Everything in ``search/`` — vector CFR, the re-solving gadget, continual
re-solving, the range best response — is really about *reach vectors and
counterfactual values*, and none of it cares whether a "hand" is one Leduc card
or two hold'em cards.  What differs between games is small and lives here:

* how many hands there are, and which the board rules out,
* how a newly dealt card blocks hands,
* what a terminal node is worth per hand,
* the constant that turns a product of two independent ranges into the true
  joint deal, given that the players hold disjoint cards.

Implementations exist for Leduc (six one-card hands) and for hold'em endgames
(1,326 two-card combinations).  Adding a game means writing one of these, not
touching the solver.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from paradigm_b.core.belief.ranges import PBS

NUM_PLAYERS = 2


class HandSpace(ABC):
    """The private-information model of a game."""

    num_hands: int
    # (hands available) / (hands available to an opponent who is not me).
    pair_correction: float

    def initial_reach(self) -> np.ndarray:
        """Uniform reach over the hands the root board allows."""
        mask = self.root_mask()
        row = mask / mask.sum()
        return np.stack([row, row])

    @abstractmethod
    def root_mask(self) -> np.ndarray: ...

    @abstractmethod
    def possible_mask(self, public) -> np.ndarray:
        """Hands not ruled out by the board at ``public``."""

    @abstractmethod
    def deal_mask(self, card: int) -> np.ndarray:
        """Hands ruled out once ``card`` appears on the board."""

    @abstractmethod
    def chance_weight(self, public) -> float:
        """Probability of one chance outcome, given both players' hands."""

    @abstractmethod
    def terminal_values(self, public, reach: np.ndarray) -> np.ndarray:
        """``(2, num_hands)`` counterfactual values at a terminal node."""

    def pbs(self, public, reach: np.ndarray) -> PBS:
        return PBS.from_reach(public, reach, mask=self.possible_mask(public))


class LeducHandSpace(HandSpace):
    """Six hands, one card each — the game everything was validated on."""

    def __init__(self) -> None:
        from paradigm_b.core.belief.ranges import NUM_HANDS, PAIR_CORRECTION

        self.num_hands = NUM_HANDS
        self.pair_correction = PAIR_CORRECTION

    def root_mask(self) -> np.ndarray:
        return np.ones(self.num_hands)

    def possible_mask(self, public) -> np.ndarray:
        from paradigm_b.core.belief.ranges import board_mask

        return board_mask(public.board)

    def deal_mask(self, card: int) -> np.ndarray:
        from paradigm_b.core.belief.ranges import board_mask

        return board_mask(card)

    def chance_weight(self, public) -> float:
        return 1.0 / (self.num_hands - 2)

    def terminal_values(self, public, reach: np.ndarray) -> np.ndarray:
        from paradigm_b.core.belief.ranges import opponent_mass_excluding_self, showdown_matrix

        betting = public.betting
        values = np.empty((NUM_PLAYERS, self.num_hands))
        if betting.folder >= 0:
            winner = 1 - betting.folder
            stake = float(betting.contributions[betting.folder])
            for player in range(NUM_PLAYERS):
                sign = 1.0 if player == winner else -1.0
                values[player] = sign * stake * opponent_mass_excluding_self(
                    reach[1 - player]
                )
            return values
        stake = betting.showdown_stake()
        matrix = showdown_matrix(public.board)
        values[0] = self.pair_correction * stake * (matrix @ reach[1])
        values[1] = -self.pair_correction * stake * (matrix.T @ reach[0])
        return values


LEDUC_SPACE = LeducHandSpace()
