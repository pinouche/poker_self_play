"""Playing hands against Slumbot, and counting the result honestly.

Slumbot is the standard opponent: ReBeL and its successors all report mbb/g
against it, so it is the one number on this project that is comparable to
published work — unlike exploitability on a postflop endgame, which is not.

**The protocol, established by probing rather than assumed.**  Every one of
these was wrong or unknown in the obvious reading, so they are written down:

``https://slumbot.com``  and *not* ``www.slumbot.com``.  The ``www`` host issues
    a 301, which turns a POST into a GET, and the API answers 400.  The snippet
    in most examples uses ``www`` and fails.
``new_hand`` returns the token; ``act`` does **not**.  Carry the token from the
    first response through the whole hand.
``bX``  raises the street's contribution **to** X chips — per street, not
    cumulative over the hand.  Verified: preflop call (200) then flop ``b400``
    then folding cost exactly 600, not 400.
``k`` / ``c`` / ``f``  check, call, fold.  ``c`` is strictly a *call* and is
    rejected with "Illegal call" when there is nothing to call, so a checked-to
    street needs ``k``.
``/``  separates streets inside the ``action`` string.
``client_pos``  0 means we are the big blind and the bot opens; 1 means we
    have the button.  **It alternates only if the session token is carried into
    the next ``new_hand``.**  Passing ``{"token": ""}`` every hand -- which is
    what every example snippet does -- starts a fresh session each time and
    seats you as the big blind *every* hand, silently measuring the agent
    exclusively out of position.  Asking for a ``client_pos`` explicitly is
    ignored.  Carrying the token gives the strict 0,1,0,1 alternation that
    makes a reported mbb/g comparable to published ones.
``winnings``  appears on the terminal response, in chips, positive when we win.

Stack is 20,000 with blinds 50/100 — 200 big blinds, confirmed by an accepted
``b20000`` shove.

**One big blind is 100 chips**, so mbb/g is ``chips / 100 * 1000`` per hand.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

BASE = "https://slumbot.com/api"
BIG_BLIND = 100
SMALL_BLIND = 50
STACK = 20_000


@dataclass
class HandState:
    """What we know part-way through a hand."""

    token: str
    hole_cards: Tuple[str, ...]
    board: Tuple[str, ...]
    action: str
    client_pos: int

    @property
    def streets(self) -> List[str]:
        return self.action.split("/")

    @property
    def street_index(self) -> int:
        return len(self.streets) - 1


@dataclass
class HandResult:
    hand: int
    client_pos: int
    hole_cards: Tuple[str, ...]
    board: Tuple[str, ...]
    action: str
    winnings: int
    bot_hole_cards: Optional[Tuple[str, ...]]
    seconds: float

    def to_dict(self) -> Dict:
        return {
            "hand": self.hand,
            "client_pos": self.client_pos,
            "hole_cards": list(self.hole_cards),
            "board": list(self.board),
            "action": self.action,
            "winnings": self.winnings,
            "bot_hole_cards": list(self.bot_hole_cards or ()),
            "seconds": round(self.seconds, 3),
        }


class Policy(Protocol):
    """Anything that can choose a Slumbot action string from a hand state."""

    def __call__(self, state: HandState) -> str: ...


def fold_policy(state: HandState) -> str:
    """Fold whenever legal, else check.  The calibration opponent.

    Its expected loss is known in advance -- as the big blind, folding every
    hand to the bot's open surrenders exactly one big blind, so the harness must
    measure close to **-1000 mbb/g**.  Anything else means the accounting is
    wrong, and that is much easier to see against a policy with a known answer
    than against an agent whose true strength is the thing being measured.
    """
    return "f" if _facing_bet(state) else "k"


def call_policy(state: HandState) -> str:
    """Call anything, check otherwise: the calling station."""
    return "c" if _facing_bet(state) else "k"


def _facing_bet(state: HandState) -> bool:
    """Is there an outstanding bet on the current street?"""
    street = state.streets[-1]
    if not street:
        # A fresh postflop street with no action yet; preflop always has the
        # blinds outstanding, and the bot has always acted by the time we see it.
        return state.street_index == 0
    # Trailing token decides: a bet or raise leaves something to answer.
    index = len(street) - 1
    while index >= 0 and street[index].isdigit():
        index -= 1
    return street[index] == "b"


class SlumbotClient:
    """Thin, retrying HTTP client for the two endpoints that exist."""

    def __init__(self, base: str = BASE, timeout: float = 20.0, retries: int = 3) -> None:
        self.base = base
        self.timeout = timeout
        self.retries = retries
        # The session, which is what makes the seat alternate.  Held here so a
        # caller cannot forget to thread it through and quietly play every hand
        # from the big blind.
        self.token: str = ""

    def _post(self, path: str, payload: Dict) -> Dict:
        last: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(
                    f"{self.base}/{path}",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode())
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last = error
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"slumbot {path} failed after {self.retries} attempts: {last}")

    def new_hand(self) -> Dict:
        """Deal the next hand *of this session*, so the seat alternates."""
        payload = self._post("new_hand", {"token": self.token})
        self.token = payload.get("token", self.token)
        return payload

    def reset(self) -> None:
        """Abandon the session; the next hand is dealt from the big blind."""
        self.token = ""

    def act(self, token: str, incr: str) -> Dict:
        return self._post("act", {"token": token, "incr": incr})


@dataclass
class SessionSummary:
    """Aggregate result, with the caveats that decide whether it means anything."""

    hands: int
    total_chips: int
    mbb_per_game: float
    stderr_mbb: float
    ci95_mbb: Tuple[float, float]
    seconds: float

    def to_dict(self) -> Dict:
        return {
            "hands": self.hands,
            "total_chips": self.total_chips,
            "mbb_per_game": round(self.mbb_per_game, 1),
            "stderr_mbb": round(self.stderr_mbb, 1),
            "ci95_mbb": [round(self.ci95_mbb[0], 1), round(self.ci95_mbb[1], 1)],
            "seconds": round(self.seconds, 1),
            "hands_per_second": round(self.hands / max(self.seconds, 1e-9), 2),
        }


def summarise(winnings: Sequence[int], seconds: float) -> SessionSummary:
    """mbb/g with a 95% interval, because the point estimate alone misleads.

    Heads-up no-limit has enormous per-hand variance, so a few hundred hands
    cannot separate "loses badly" from "loses catastrophically", let alone
    resolve a small edge.  Reporting the interval alongside is what stops a
    noisy number being read as a result.
    """
    import math

    count = len(winnings)
    if count == 0:
        return SessionSummary(0, 0, 0.0, 0.0, (0.0, 0.0), seconds)
    per_hand_mbb = [w / BIG_BLIND * 1000.0 for w in winnings]
    mean = sum(per_hand_mbb) / count
    if count > 1:
        variance = sum((x - mean) ** 2 for x in per_hand_mbb) / (count - 1)
        stderr = math.sqrt(variance / count)
    else:
        stderr = 0.0
    return SessionSummary(
        hands=count,
        total_chips=int(sum(winnings)),
        mbb_per_game=mean,
        stderr_mbb=stderr,
        ci95_mbb=(mean - 1.96 * stderr, mean + 1.96 * stderr),
        seconds=seconds,
    )


def by_seat(results: Sequence[HandResult]) -> Dict[str, Dict]:
    """mbb/g split by position.

    Worth reporting always: the two seats are not symmetric in hold'em, so a
    single mean hides both an unbalanced sample and an agent that is broken in
    one seat only.  With the session token carried the seats alternate exactly,
    so these counts should differ by at most one.
    """
    out: Dict[str, Dict] = {}
    for seat, name in ((0, "big_blind"), (1, "button")):
        chosen = [r.winnings for r in results if r.client_pos == seat]
        if chosen:
            out[name] = summarise(chosen, 0.0).to_dict()
    return out


def play_hand(client: SlumbotClient, policy: Policy, index: int) -> HandResult:
    """One hand, from deal to terminal."""
    started = time.perf_counter()
    payload = client.new_hand()
    # Not ``payload["token"]``: a continuing session reuses the token it was
    # given and the response omits it, so only the client knows it.
    token = client.token
    seat = payload.get("client_pos", 0)
    while "winnings" not in payload:
        state = HandState(
            token=token,
            hole_cards=tuple(payload.get("hole_cards", ())),
            board=tuple(payload.get("board", ())),
            action=payload.get("action", ""),
            client_pos=payload.get("client_pos", 0),
        )
        payload_next = client.act(token, policy(state))
        if "error_msg" in payload_next:
            raise RuntimeError(
                f"slumbot rejected our action on hand {index}: "
                f"{payload_next['error_msg']} (action so far {state.action!r})"
            )
        payload = {**payload_next, "token": token}
    return HandResult(
        hand=index,
        client_pos=seat,
        hole_cards=tuple(payload.get("hole_cards", ())),
        board=tuple(payload.get("board", ())),
        action=payload.get("action", ""),
        winnings=int(payload["winnings"]),
        bot_hole_cards=tuple(payload.get("bot_hole_cards", ()) or ()),
        seconds=time.perf_counter() - started,
    )


def play_session(
    hands: int,
    policy: Policy,
    log_path: Optional[Path] = None,
    client: Optional[SlumbotClient] = None,
    on_hand: Optional[Callable[[HandResult, SessionSummary], None]] = None,
) -> SessionSummary:
    """Play ``hands`` hands, logging each one as it finishes.

    Written as append-as-you-go rather than collect-then-write: a session of any
    useful length runs for hours, and a crash at hour three should leave three
    hours of hands on disk rather than nothing.
    """
    client = client or SlumbotClient()
    winnings: List[int] = []
    started = time.perf_counter()
    handle = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handle = open(log_path, "a", buffering=1)
    try:
        for index in range(hands):
            result = play_hand(client, policy, index)
            winnings.append(result.winnings)
            if handle is not None:
                handle.write(json.dumps(result.to_dict()) + "\n")
            if on_hand is not None:
                on_hand(result, summarise(winnings, time.perf_counter() - started))
    finally:
        if handle is not None:
            handle.close()
    return summarise(winnings, time.perf_counter() - started)
