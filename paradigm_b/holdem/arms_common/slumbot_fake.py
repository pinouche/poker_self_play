"""A local stand-in for the Slumbot server, speaking the same protocol.

The real bridge cannot be tested by using it: every check would be a request to
someone else's machine, the answers would depend on a bot that is not ours, and
a test suite that needs the network is a test suite that fails on a train.  So
the protocol gets a second implementation here, deliberately written from the
*observed* behaviour recorded in :mod:`.slumbot`'s docstring rather than from
the bridge's code — a shared misreading would otherwise agree with itself.

It is a real dealer: it shuffles, runs the betting, and settles at showdown
with the engine's own hand ranks.  What it is not is a poker player — the
built-in opponents are the same trivial policies the calibration runs used, so
the number this produces measures the *bridge*, not the agent.

``chips`` throughout are Slumbot's, 50/100 blinds and a 20,000 stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from paradigm_b.holdem.arms_common.slumbot_agent import (
    CHIPS_PER_ENGINE_UNIT,
    SLUMBOT_STACK,
    format_card,
    tokenize_street,
)
from paradigm_b.holdem.engine.combos import combo_index
from paradigm_b.holdem.engine.strength import hand_ranks

SMALL_BLIND_CHIPS = 50
BIG_BLIND_CHIPS = 100
BOARD_AT_STREET = (0, 3, 4, 5)


def _street_actor(street_index: int) -> int:
    """Seat 0 is the button/small blind: first preflop, second after it."""
    return 0 if street_index == 0 else 1


class FakeSlumbotClient:
    """Implements ``new_hand`` / ``act`` / ``token`` the way the real one does.

    Seats alternate across hands exactly as a carried session does, because
    seat alternation is the thing the real client got wrong first and a fake
    that seats you the same way every hand would hide the same bug again.
    """

    def __init__(
        self,
        opponent: Optional[Callable[[str, Sequence[str], int], str]] = None,
        seed: int = 0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.opponent = opponent or call_opponent
        self.token = "fake-session"
        self.hands_played = 0
        self._hand: Optional[_Hand] = None

    def new_hand(self) -> Dict:
        deck = list(self.rng.permutation(52))
        self._hand = _Hand(
            deck=deck,
            client_pos=self.hands_played % 2,
            opponent=self.opponent,
        )
        self.hands_played += 1
        return self._hand.advance_to_client()

    def act(self, token: str, incr: str) -> Dict:
        assert self._hand is not None, "act before new_hand"
        return self._hand.apply_client(incr)

    def reset(self) -> None:
        self._hand = None


def call_opponent(action: str, board: Sequence[str], seat: int) -> str:
    """Call anything, check otherwise — the calling station."""
    street = action.split("/")[-1]
    return "c" if _facing_bet(street, action) else "k"


def fold_opponent(action: str, board: Sequence[str], seat: int) -> str:
    street = action.split("/")[-1]
    return "f" if _facing_bet(street, action) else "k"


def _facing_bet(street: str, action: str) -> bool:
    tokens = tokenize_street(street)
    if not tokens:
        # A fresh postflop street has nothing outstanding; preflop the blinds
        # always leave the button owing.
        return "/" not in action
    return tokens[-1].startswith("b")


@dataclass
class _Hand:
    deck: List[int]
    client_pos: int
    opponent: Callable[[str, Sequence[str], int], str]
    street: int = 0
    total: List[int] = field(default_factory=lambda: [SMALL_BLIND_CHIPS, BIG_BLIND_CHIPS])
    base: List[int] = field(default_factory=lambda: [0, 0])
    tokens: List[List[str]] = field(default_factory=lambda: [[]])
    folder: Optional[int] = None
    finished: bool = False

    def __post_init__(self) -> None:
        # client_pos 1 means the client has the button, i.e. seat 0.
        self.client_seat = 0 if self.client_pos == 1 else 1
        self.hole = {
            0: (self.deck[0], self.deck[1]),
            1: (self.deck[2], self.deck[3]),
        }
        self.full_board = tuple(self.deck[4:9])

    # --- protocol surface --------------------------------------------------
    @property
    def action(self) -> str:
        return "/".join("".join(street) for street in self.tokens)

    @property
    def board(self) -> List[str]:
        return [format_card(c) for c in self.full_board[: BOARD_AT_STREET[self.street]]]

    def _payload(self, extra: Optional[Dict] = None) -> Dict:
        payload = {
            "client_pos": self.client_pos,
            "hole_cards": [format_card(c) for c in self.hole[self.client_seat]],
            "board": self.board,
            "action": self.action,
        }
        payload.update(extra or {})
        return payload

    def _to_move(self) -> int:
        return (_street_actor(self.street) + len(self.tokens[self.street])) % 2

    # --- driving -----------------------------------------------------------
    def advance_to_client(self) -> Dict:
        """Let the opponent act until it is the client's turn, or the hand ends."""
        while not self.finished and self._to_move() != self.client_seat:
            self._apply(self.opponent(self.action, self.board, 1 - self.client_seat))
        return self._payload(self._settlement() if self.finished else None)

    def apply_client(self, token: str) -> Dict:
        if self.finished:
            return self._payload(self._settlement())
        try:
            self._apply(token)
        except ValueError as error:
            return {"error_msg": str(error)}
        return self.advance_to_client()

    def _apply(self, token: str) -> None:
        actor = self._to_move()
        tokens = tokenize_street(token)
        if len(tokens) != 1:
            raise ValueError(f"expected one action, got {token!r}")
        token = tokens[0]
        owed = self.total[1 - actor] - self.total[actor]
        if token == "f":
            if owed <= 0:
                raise ValueError("Illegal fold")
            self.folder = actor
            self.tokens[self.street].append(token)
            self.finished = True
            return
        if token == "k":
            if owed > 0:
                raise ValueError("Illegal check")
        elif token == "c":
            if owed <= 0:
                raise ValueError("Illegal call")
            self.total[actor] = self.total[1 - actor]
        else:
            size = int(token[1:])
            target = self.base[actor] + size
            if target <= self.total[1 - actor]:
                raise ValueError(f"Illegal bet size {size}")
            if target > SLUMBOT_STACK:
                raise ValueError("Bet over stack")
            self.total[actor] = target
        self.tokens[self.street].append(token)
        self._maybe_close_street()

    def _maybe_close_street(self) -> None:
        tokens = self.tokens[self.street]
        if len(tokens) < 2:
            return
        if self.total[0] != self.total[1]:
            return
        # Matched contributions and at least two actions: the round is over.
        # ``kb100c`` is three tokens and closes; ``kk`` and ``bXc`` are two.
        if tokens[-1] not in ("c", "k"):
            return
        if self.street == 3 or max(self.total) >= SLUMBOT_STACK:
            self.finished = True
            return
        self.street += 1
        self.base = list(self.total)
        self.tokens.append([])

    # --- settlement --------------------------------------------------------
    def _settlement(self) -> Dict:
        me = self.client_seat
        if self.folder is not None:
            winnings = self.total[1 - me] if self.folder != me else -self.total[me]
        else:
            board = tuple(self.full_board)
            ranks = hand_ranks(board)
            mine = ranks[combo_index(*self.hole[me])]
            theirs = ranks[combo_index(*self.hole[1 - me])]
            if mine == theirs:
                winnings = 0
            elif mine > theirs:
                winnings = self.total[1 - me]
            else:
                winnings = -self.total[me]
        return {
            "winnings": int(winnings),
            "bot_hole_cards": [format_card(c) for c in self.hole[1 - me]],
            "board": [format_card(c) for c in self.full_board],
        }
