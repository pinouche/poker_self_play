"""The hold'em endgame hand space: 1,326 two-card combinations.

The solver's view of hold'em.  Everything game-specific about private
information is here — how many hands exist, which the board blocks, what a
terminal node pays each of them — and nothing else in ``search/`` changes.
"""

from __future__ import annotations

import numpy as np

from paradigm_b.holdem.engine.combos import (
    CARD_IN_COMBO,
    NUM_CARDS,
    NUM_COMBOS,
    board_mask,
    compatible_mass,
    pair_correction,
)
from paradigm_b.holdem.engine.showdown import showdown_values
from paradigm_b.core.search.space import NUM_PLAYERS, HandSpace


class EndgameSpace(HandSpace):
    """Two-card hands over a fixed starting board, with more cards to come.

    The starting board may be a flop or a turn; ``pair_correction`` is fixed by
    its size, because it describes how the *hands were dealt*, not what the board
    later becomes.
    """

    def __init__(self, turn_board) -> None:
        self.turn_board = tuple(turn_board)
        self.num_hands = NUM_COMBOS
        # Fixed at the turn: it converts the product of the two (independent)
        # ranges into the true joint deal of two disjoint hands.
        self.pair_correction = pair_correction(len(self.turn_board))

    def root_mask(self) -> np.ndarray:
        return board_mask(self.turn_board)

    def possible_mask(self, public) -> np.ndarray:
        return board_mask(tuple(public.board))

    def deal_mask(self, card: int) -> np.ndarray:
        return 1.0 - CARD_IN_COMBO[card]

    def chance_weight(self, public) -> float:
        """One river out of the cards neither the board nor four hole cards use."""
        return 1.0 / (NUM_CARDS - len(public.board) - 4)

    def terminal_values(self, public, reach: np.ndarray) -> np.ndarray:
        betting = public.betting
        values = np.empty((NUM_PLAYERS, self.num_hands))
        mask = self.possible_mask(public)
        if betting.folder >= 0:
            winner = 1 - betting.folder
            stake = betting.fold_returns()[winner]
            for player in range(NUM_PLAYERS):
                sign = 1.0 if player == winner else -1.0
                values[player] = (
                    sign
                    * abs(stake)
                    * self.pair_correction
                    * compatible_mass(reach[1 - player])
                    * mask
                )
            return values
        stake = betting.showdown_stake()
        board = tuple(public.board)
        values[0] = self.pair_correction * showdown_values(board, reach[1], stake)
        values[1] = self.pair_correction * showdown_values(board, reach[0], stake)
        return values


# Named for the case it was written for; it serves any starting street.
TurnEndgameSpace = EndgameSpace
