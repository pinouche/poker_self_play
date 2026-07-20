"""Game state containers and the fixed action space.

The action space is defined here because it is a property of the *rules*, not
of the encoder.  ``representation.action_encoder`` re-exports these constants
and adds the tensor-encoding helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

from .cards import Card


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4  # terminal marker, never an acting street


NUM_BETTING_STREETS = 4  # PREFLOP..RIVER, used for one-hot sizing

STREET_NAMES = {
    Street.PREFLOP: "preflop",
    Street.FLOP: "flop",
    Street.TURN: "turn",
    Street.RIVER: "river",
    Street.SHOWDOWN: "showdown",
}

STREET_FROM_NAME = {v: k for k, v in STREET_NAMES.items()}

# Number of board cards visible on each street.
BOARD_CARDS_BY_STREET = {
    Street.PREFLOP: 0,
    Street.FLOP: 3,
    Street.TURN: 4,
    Street.RIVER: 5,
    Street.SHOWDOWN: 5,
}

MAX_BOARD_CARDS = 5
NUM_HOLE_CARDS = 2

# --- fixed action space ---------------------------------------------------
FOLD = 0
CHECK = 1
CALL = 2
BET_SMALL = 3
BET_MEDIUM = 4
BET_LARGE = 5
RAISE_SMALL = 6
RAISE_MEDIUM = 7
RAISE_LARGE = 8
ALL_IN = 9

NUM_ACTIONS = 10

ACTION_NAMES = [
    "FOLD",
    "CHECK",
    "CALL",
    "BET_SMALL",
    "BET_MEDIUM",
    "BET_LARGE",
    "RAISE_SMALL",
    "RAISE_MEDIUM",
    "RAISE_LARGE",
    "ALL_IN",
]

ACTION_ID_FROM_NAME = {name: i for i, name in enumerate(ACTION_NAMES)}

BET_ACTIONS = (BET_SMALL, BET_MEDIUM, BET_LARGE)
RAISE_ACTIONS = (RAISE_SMALL, RAISE_MEDIUM, RAISE_LARGE)
AGGRESSIVE_ACTIONS = BET_ACTIONS + RAISE_ACTIONS + (ALL_IN,)


@dataclass
class PlayerState:
    """Per-seat state.  ``hole`` is hidden information."""

    seat: int
    stack: int
    hole: Tuple[Card, ...] = ()
    street_bet: int = 0          # chips committed on the current street
    contributed: int = 0         # chips committed across the whole hand
    folded: bool = False
    all_in: bool = False
    has_acted_this_round: bool = False

    @property
    def active(self) -> bool:
        """Still contesting the pot."""
        return not self.folded

    @property
    def can_act(self) -> bool:
        """Able to make a further betting decision this hand."""
        return not self.folded and not self.all_in and self.stack > 0

    def reset_for_street(self) -> None:
        self.street_bet = 0
        self.has_acted_this_round = False


@dataclass
class ActionRecord:
    """One entry of the chronological action history."""

    seat: int
    action_id: int
    amount: int          # chips added to the pot by this action
    to_amount: int       # resulting total street bet for this seat
    street: Street
    pot_before: int
    pot_after: int


@dataclass
class GameState:
    """Complete (perfect-information) state of one hand."""

    players: List[PlayerState]
    board: List[Card] = field(default_factory=list)
    street: Street = Street.PREFLOP
    dealer: int = 0
    small_blind: int = 10
    big_blind: int = 20
    current_bet: int = 0             # highest street_bet this round
    min_raise_increment: int = 0     # size of the last full raise
    last_full_raise_level: int = 0   # street_bet level set by the last full raise
    to_act: Optional[int] = None
    history: List[ActionRecord] = field(default_factory=list)
    initial_stacks: List[int] = field(default_factory=list)
    hand_over: bool = False
    went_to_showdown: bool = False
    payouts: List[int] = field(default_factory=list)

    # --- derived quantities ------------------------------------------------
    @property
    def num_players(self) -> int:
        return len(self.players)

    @property
    def pot(self) -> int:
        """Total chips committed this hand, including the current street."""
        return sum(p.contributed for p in self.players)

    @property
    def sb_seat(self) -> int:
        return (self.dealer + 1) % self.num_players

    @property
    def bb_seat(self) -> int:
        return (self.dealer + 2) % self.num_players

    def active_players(self) -> List[PlayerState]:
        return [p for p in self.players if p.active]

    def actable_players(self) -> List[PlayerState]:
        return [p for p in self.players if p.can_act]

    def to_call(self, seat: int) -> int:
        return max(0, self.current_bet - self.players[seat].street_bet)

    def min_raise_to(self) -> int:
        """Smallest legal total street bet for a raise."""
        return self.current_bet + max(self.min_raise_increment, self.big_blind)

    def effective_stack(self, seat: int) -> int:
        """Largest amount that can still be won or lost by ``seat``.

        Defined as the smallest of the seat's own remaining commitment
        capacity and the largest opponent capacity; guarded to stay positive.
        """
        me = self.players[seat]
        mine = me.stack + me.contributed
        others = [p.stack + p.contributed for p in self.players if p.seat != seat and p.active]
        if not others:
            return max(1, mine)
        return max(1, min(mine, max(others)))

    def position_of(self, seat: int) -> int:
        """Distance from the button, 0 = dealer, 1 = SB, 2 = BB."""
        return (seat - self.dealer) % self.num_players

    def clone_public(self) -> "GameState":
        """A copy with all hole cards stripped (debug/serialisation helper)."""
        import copy

        state = copy.deepcopy(self)
        for p in state.players:
            p.hole = ()
        return state
