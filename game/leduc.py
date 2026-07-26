"""Leduc hold'em: the standard benchmark for imperfect-information solvers.

Six cards (J, Q, K in two suits), one private card each, two betting rounds
with a public card revealed between them, an ante of 1, fixed raises of 2 then
4, and at most two raises per round.  A private card that pairs the board wins;
otherwise the higher rank wins.

The game is small enough to solve exactly (its value to player 0 under any
Nash equilibrium is -0.0856 chips) and large enough to contain everything that
makes poker hard: hidden information, a public card that reshapes both ranges,
and bluffing.  Every stage of paradigm B is validated here before anything is
pointed at hold'em.

Cards are physical ids in [0, 6) with ``rank = card // 2`` and
``suit = card % 2``.  Suits are strategically irrelevant, so information sets
are keyed by *rank* — a lossless abstraction — while the tree still deals
physical cards so that card removal stays exact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Hashable, Sequence, Tuple

from game.actions import CALL, FOLD, RAISE, history_str
from game.base import CHANCE, Game, State

NUM_CARDS = 6
NUM_RANKS = 3
NUM_SUITS = 2
RANK_NAMES = ("J", "Q", "K")
SUIT_NAMES = ("a", "b")

ANTE = 1
RAISE_SIZES = (2, 4)  # per betting round
MAX_RAISES_PER_ROUND = 2
NUM_ROUNDS = 2


def rank_of(card: int) -> int:
    return card // NUM_SUITS


def card_name(card: int) -> str:
    return RANK_NAMES[rank_of(card)] + SUIT_NAMES[card % NUM_SUITS]


@dataclass(frozen=True)
class Betting:
    """The betting state, which is entirely public.

    Nothing here depends on anyone's private card, which is exactly why it can
    be shared: the world tree (:class:`LeducState`) and the public belief tree
    used by search both advance the betting through this one implementation, so
    the rules cannot drift apart between them.
    """

    betting_round: int = 0
    history: Tuple[Tuple[int, ...], ...] = ((), ())  # actions, per round
    contributions: Tuple[int, int] = (ANTE, ANTE)
    folder: int = -1
    showdown: bool = False
    awaiting_board: bool = False  # round one closed, the board is not out yet

    @property
    def is_terminal(self) -> bool:
        return self.folder >= 0 or self.showdown

    @property
    def pot(self) -> int:
        return sum(self.contributions)

    def to_move(self) -> int:
        """Seat to act.  Player 0 acts first in both rounds."""
        return len(self.history[self.betting_round]) % 2

    def legal_actions(self) -> Tuple[int, ...]:
        me = self.to_move()
        facing_bet = self.contributions[me] < self.contributions[1 - me]
        capped = self.history[self.betting_round].count(RAISE) >= MAX_RAISES_PER_ROUND
        actions = [CALL]
        if facing_bet:
            actions.insert(0, FOLD)
        if not capped:
            actions.append(RAISE)
        return tuple(actions)

    def apply(self, action: int) -> "Betting":
        me = self.to_move()
        opp = 1 - me
        rnd = self.betting_round
        history = _append(self.history, rnd, action)

        if action == FOLD:
            return replace(self, history=history, folder=me)

        contributions = list(self.contributions)
        if action == CALL:
            contributions[me] = contributions[opp]
        elif action == RAISE:
            contributions[me] = contributions[opp] + RAISE_SIZES[rnd]
        else:
            raise ValueError(f"illegal action: {action}")
        contributions = tuple(contributions)

        round_closed = action == CALL and len(history[rnd]) >= 2
        if not round_closed:
            return replace(self, history=history, contributions=contributions)
        if rnd + 1 < NUM_ROUNDS:
            return replace(
                self,
                history=history,
                contributions=contributions,
                betting_round=rnd + 1,
                awaiting_board=True,
            )
        return replace(
            self, history=history, contributions=contributions, showdown=True
        )

    def deal_board(self) -> "Betting":
        return replace(self, awaiting_board=False)

    def fold_returns(self) -> Tuple[float, float]:
        """Payoffs when someone folded: the folder loses what they put in."""
        loss = float(self.contributions[self.folder])
        return (-loss, loss) if self.folder == 0 else (loss, -loss)

    def showdown_stake(self) -> float:
        """What the loser of a showdown pays (contributions are level by then)."""
        return float(min(self.contributions))

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return "/".join(history_str(h) for h in self.history)


@dataclass(frozen=True)
class LeducState(State):
    """A history: the cards dealt so far plus the public betting state."""

    cards: Tuple[int, ...] = ()  # (player 0's card, player 1's card)
    board: int = -1  # -1 while undealt
    betting: Betting = Betting()

    # Convenience forwarders; the betting state is the single source of truth.
    @property
    def betting_round(self) -> int:
        return self.betting.betting_round

    @property
    def contributions(self) -> Tuple[int, int]:
        return self.betting.contributions

    @property
    def history(self) -> Tuple[Tuple[int, ...], ...]:
        return self.betting.history

    # --- structure ---------------------------------------------------------
    def is_terminal(self) -> bool:
        return self.betting.is_terminal

    def _needs_deal(self) -> bool:
        return len(self.cards) < 2 or self.betting.awaiting_board

    def current_player(self) -> int:
        if self._needs_deal():
            return CHANCE
        return self.betting.to_move()

    def legal_actions(self) -> Sequence[int]:
        return self.betting.legal_actions()

    def chance_outcomes(self) -> Sequence[Tuple[int, float]]:
        dealt = set(self.cards)
        if self.board >= 0:
            dealt.add(self.board)
        remaining = [c for c in range(NUM_CARDS) if c not in dealt]
        p = 1.0 / len(remaining)
        return tuple((c, p) for c in remaining)

    def apply(self, action: int) -> "LeducState":
        if self._needs_deal():
            if len(self.cards) < 2:
                return replace(self, cards=self.cards + (action,))
            return replace(self, board=action, betting=self.betting.deal_board())
        return replace(self, betting=self.betting.apply(action))

    # --- payoffs -----------------------------------------------------------
    def returns(self) -> Sequence[float]:
        if self.betting.folder >= 0:
            return self.betting.fold_returns()
        winner = showdown_winner(self.cards[0], self.cards[1], self.board)
        if winner < 0:
            return (0.0, 0.0)
        stake = self.betting.showdown_stake()
        return (stake, -stake) if winner == 0 else (-stake, stake)

    # --- information -------------------------------------------------------
    def infoset_key(self) -> Hashable:
        me = self.current_player()
        board_rank = rank_of(self.board) if self.board >= 0 else -1
        return (
            rank_of(self.cards[me]),
            board_rank,
            history_str(self.history[0]),
            history_str(self.history[1]),
        )

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        hands = " ".join(card_name(c) for c in self.cards)
        board = card_name(self.board) if self.board >= 0 else "--"
        return f"[{hands}|{board}] {self.betting}"


def _append(history: Tuple[Tuple[int, ...], ...], rnd: int, action: int):
    rounds = list(history)
    rounds[rnd] = rounds[rnd] + (action,)
    return tuple(rounds)


def showdown_winner(card0: int, card1: int, board: int) -> int:
    """Seat that wins the showdown, or -1 for a split pot."""
    r0, r1, rb = rank_of(card0), rank_of(card1), rank_of(board)
    if r0 == rb:
        return 0 if r1 != rb else -1
    if r1 == rb:
        return 1
    if r0 == r1:
        return -1
    return 0 if r0 > r1 else 1


class LeducHoldem(Game):
    num_players = 2
    max_actions = 3

    def new_initial_state(self) -> LeducState:
        return LeducState()

    def action_name(self, action: int) -> str:
        return ("fold", "check/call", "bet/raise")[action]
