"""Kuhn poker: the smallest interesting imperfect-information poker game.

Three cards (J < Q < K), one each to two players, one ante of 1 chip each, a
single betting round with a bet size of 1.  The equilibrium is known in closed
form — player 1's value is -1/18 and the first player bluffs the jack with
probability alpha in [0, 1/3] — which makes it the first sanity check for any
regret-minimisation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence, Tuple

from paradigm_b.core.game.actions import CALL, FOLD, RAISE, history_str
from paradigm_b.core.game.base import CHANCE, Game, State

NUM_CARDS = 3
CARD_NAMES = ("J", "Q", "K")
ANTE = 1
BET = 1

# Betting histories that end the hand, mapped to nothing here -- terminality is
# derived in ``is_terminal`` -- but listed for reference: cc, cbf, cbc, bf, bc.


@dataclass(frozen=True)
class KuhnState(State):
    cards: Tuple[int, ...] = ()
    history: Tuple[int, ...] = ()

    # --- structure ---------------------------------------------------------
    def is_terminal(self) -> bool:
        if len(self.cards) < 2:
            return False
        h = self.history
        if not h:
            return False
        if h[-1] == FOLD:
            return True
        if h[-1] == CALL:
            return len(h) >= 2  # check-check, or a call closing a bet
        return False

    def current_player(self) -> int:
        if len(self.cards) < 2:
            return CHANCE
        return len(self.history) % 2

    def legal_actions(self) -> Sequence[int]:
        facing_bet = bool(self.history) and self.history[-1] == RAISE
        if facing_bet:
            return (FOLD, CALL)
        return (CALL, RAISE)

    def chance_outcomes(self) -> Sequence[Tuple[int, float]]:
        remaining = [c for c in range(NUM_CARDS) if c not in self.cards]
        p = 1.0 / len(remaining)
        return tuple((c, p) for c in remaining)

    def apply(self, action: int) -> "KuhnState":
        if len(self.cards) < 2:
            return KuhnState(cards=self.cards + (action,), history=self.history)
        return KuhnState(cards=self.cards, history=self.history + (action,))

    # --- payoffs -----------------------------------------------------------
    def returns(self) -> Sequence[float]:
        h = self.history
        if h[-1] == FOLD:
            folder = (len(h) - 1) % 2
            # The folder loses their ante; a bet that is folded to is returned.
            win = ANTE
            return (-win, win) if folder == 0 else (win, -win)
        pot_each = ANTE + (BET if RAISE in h else 0)
        winner = 0 if self.cards[0] > self.cards[1] else 1
        return (pot_each, -pot_each) if winner == 0 else (-pot_each, pot_each)

    # --- information -------------------------------------------------------
    def infoset_key(self) -> Hashable:
        return (self.cards[self.current_player()], history_str(self.history))

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        cards = "".join(CARD_NAMES[c] for c in self.cards)
        return f"{cards}:{history_str(self.history)}"


class KuhnPoker(Game):
    num_players = 2
    max_actions = 3

    def new_initial_state(self) -> KuhnState:
        return KuhnState()

    def action_name(self, action: int) -> str:
        return ("fold", "check/call", "bet/raise")[action]
