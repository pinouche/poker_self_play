"""Game state containers and the fixed action space.

The action space is defined here because it is a property of the *rules*, not
of the encoder.  ``representation.action_encoder`` re-exports these constants
and adds the tensor-encoding helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from common.cards import Card


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

# --- action space ---------------------------------------------------------
# The space is fixed for a given configuration -- the network needs a constant
# output width -- but its *size* is derived from the number of bet and raise
# sizings in EnvConfig rather than hardcoded.  Layout:
#
#     0 FOLD, 1 CHECK, 2 CALL, then the bet sizings, then the raise sizings,
#     then ALL_IN last.
#
# With the default three bets and three raises this reproduces the documented
# ten-action space exactly, ids included.
FOLD = 0
CHECK = 1
CALL = 2
NUM_FIXED_LEADING_ACTIONS = 3

#: Names used when there are exactly three sizings, so the public API is stable.
LEGACY_SIZING_NAMES = ("SMALL", "MEDIUM", "LARGE")


@dataclass(frozen=True)
class ActionSpace:
    """Concrete action ids and names for one betting abstraction."""

    bet_names: Tuple[str, ...]
    raise_names: Tuple[str, ...]

    @property
    def num_bets(self) -> int:
        return len(self.bet_names)

    @property
    def num_raises(self) -> int:
        return len(self.raise_names)

    @property
    def num_actions(self) -> int:
        return NUM_FIXED_LEADING_ACTIONS + self.num_bets + self.num_raises + 1

    @property
    def bet_ids(self) -> Tuple[int, ...]:
        start = NUM_FIXED_LEADING_ACTIONS
        return tuple(range(start, start + self.num_bets))

    @property
    def raise_ids(self) -> Tuple[int, ...]:
        start = NUM_FIXED_LEADING_ACTIONS + self.num_bets
        return tuple(range(start, start + self.num_raises))

    @property
    def all_in(self) -> int:
        return self.num_actions - 1

    @property
    def aggressive_ids(self) -> Tuple[int, ...]:
        return self.bet_ids + self.raise_ids + (self.all_in,)

    @property
    def names(self) -> Tuple[str, ...]:
        return (
            "FOLD",
            "CHECK",
            "CALL",
            *self.bet_names,
            *self.raise_names,
            "ALL_IN",
        )

    def name(self, action_id: int) -> str:
        return self.names[action_id]

    def id_from_name(self, name: str) -> int:
        return self.names.index(name)

    @property
    def name_to_id(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.names)}


def _sizing_names(prefix: str, sizings: Sequence[float], percent: bool) -> Tuple[str, ...]:
    """SMALL/MEDIUM/LARGE for the classic three, otherwise size-derived names."""
    if len(sizings) == 3:
        return tuple(f"{prefix}_{suffix}" for suffix in LEGACY_SIZING_NAMES)
    if percent:
        return tuple(f"{prefix}_{int(round(value * 100))}" for value in sizings)
    return tuple(f"{prefix}_{value:g}X".replace(".", "_") for value in sizings)


def build_action_space(
    bet_fractions: Sequence[float], raise_multipliers: Sequence[float]
) -> ActionSpace:
    return ActionSpace(
        bet_names=_sizing_names("BET", bet_fractions, percent=True),
        raise_names=_sizing_names("RAISE", raise_multipliers, percent=False),
    )


@lru_cache(maxsize=32)
def _action_space_cached(bet_fractions: tuple, raise_multipliers: tuple) -> ActionSpace:
    return build_action_space(bet_fractions, raise_multipliers)


def action_space_for(env_cfg) -> ActionSpace:
    """Action space implied by an :class:`~config.EnvConfig` (memoised)."""
    return _action_space_cached(
        tuple(env_cfg.bet_fractions), tuple(env_cfg.raise_multipliers)
    )


#: The default space, used wherever no configuration is in scope.  Identical to
#: the documented ten-action layout.
DEFAULT_ACTION_SPACE = build_action_space((0.33, 0.66, 1.00), (2.0, 3.0, 4.0))

NUM_ACTIONS = DEFAULT_ACTION_SPACE.num_actions
ACTION_NAMES = list(DEFAULT_ACTION_SPACE.names)
ACTION_ID_FROM_NAME = DEFAULT_ACTION_SPACE.name_to_id

BET_SMALL, BET_MEDIUM, BET_LARGE = DEFAULT_ACTION_SPACE.bet_ids
RAISE_SMALL, RAISE_MEDIUM, RAISE_LARGE = DEFAULT_ACTION_SPACE.raise_ids
ALL_IN = DEFAULT_ACTION_SPACE.all_in

BET_ACTIONS = DEFAULT_ACTION_SPACE.bet_ids
RAISE_ACTIONS = DEFAULT_ACTION_SPACE.raise_ids
AGGRESSIVE_ACTIONS = DEFAULT_ACTION_SPACE.aggressive_ids


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
    payouts: List[float] = field(default_factory=list)
    # True when the pot was settled on the expected value over remaining
    # boards rather than on the single board that was dealt.
    expected_value_runout: bool = False

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
