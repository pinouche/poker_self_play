"""The hold'em endgame hand space: 1,326 two-card combinations.

The solver's view of hold'em.  Everything game-specific about private
information is here — how many hands exist, which the board blocks, what a
terminal node pays each of them — and nothing else in ``search/`` changes.
"""

from __future__ import annotations

from functools import lru_cache

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


@lru_cache(maxsize=NUM_CARDS)
def _deal_mask(card: int) -> np.ndarray:
    """``(1326,)`` 1.0 for combos not using ``card``.

    There are only 52 of these and they never change, but the obvious
    ``1.0 - CARD_IN_COMBO[card]`` allocates and fills a fresh 1,326-float array
    on every call — and a chance node asks for one per river per CFR iteration,
    which came to 80,000 allocations per label.  Read-only so a caller cannot
    corrupt the shared copy.
    """
    mask = 1.0 - CARD_IN_COMBO[card]
    mask.setflags(write=False)
    return mask


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
        return _deal_mask(card)

    def chance_weight(self, public) -> float:
        """One river out of the cards neither the board nor four hole cards use."""
        return 1.0 / (NUM_CARDS - len(public.board) - 4)

    def terminal_values(self, public, reach: np.ndarray) -> np.ndarray:
        """Counterfactual values for both players at a real terminal node.

        Two scalar calls, deliberately, and not the two-row batch that the
        shape invites.  A player's value depends on the opponent's reach, so the
        pair *is* a K=2 batch over one shared board, and routing it through
        :func:`showdown_values_batch` looked like a free halving of the numpy
        call count.  Measured, it was 1.7x **slower** on a river label: the
        batched form's gathers are two-dimensional, which costs more per element
        than the 1-D ones, and stacking the two reaches allocates a copy on
        every one of the ~25,000 calls a label makes.  The batching in
        ``data/batched.py`` pays off because K is 64 there; at K=2 the overhead
        is all there is.
        """
        betting = public.betting
        values = np.empty((NUM_PLAYERS, self.num_hands))
        if betting.folder >= 0:
            mask = self.possible_mask(public)
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
        # The showdown branch never wanted the mask: ``showdown_values`` skips
        # blocked combos in its prefix sums, so its answer is already zero
        # there.  Computing it up front for both branches cost a board-tuple
        # hash and a cache lookup on every one of the ~40,000 showdowns a label
        # walks, for a value only the fold branch reads.
        stake = betting.showdown_stake()
        board = tuple(public.board)
        # Written straight into the rows that are about to be returned, then
        # scaled in place — same arithmetic, four fewer 1,326-vectors per call.
        showdown_values(board, reach[1], stake, out=values[0])
        showdown_values(board, reach[0], stake, out=values[1])
        values *= self.pair_correction
        return values


# Named for the case it was written for; it serves any starting street.
TurnEndgameSpace = EndgameSpace
