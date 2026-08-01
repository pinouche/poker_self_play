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

**Sessions are independent**, which is what makes
:func:`play_session_parallel` sound: several tokens playing at once are several
players, and the server has always had those.  One token played concurrently
would be one player answering their own hand twice, and no API here offers it.
"""

from __future__ import annotations

import json
import threading
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


def split_hands(hands: int, workers: int) -> List[Tuple[int, int]]:
    """``(first index, count)`` per worker, in **pairs** wherever it can be.

    Seat alternation is per session — every token starts you in the big blind
    and flips from there — so a worker that plays an odd number of hands plays
    one more from the blind than from the button.  Splitting in pairs keeps each
    worker's own seats balanced, which keeps the whole session's seats balanced
    to within the single leftover hand an odd ``hands`` leaves over.  Splitting
    naively (``hands // workers``) would instead give four workers four extra
    big blinds, which is a real bias in the reported mbb/g and an invisible one.
    """
    workers = max(1, min(int(workers), max(1, hands)))
    pairs, leftover = divmod(max(0, hands), 2)
    counts = [2 * (pairs // workers) for _ in range(workers)]
    for index in range(pairs % workers):
        counts[index] += 2
    if leftover:
        counts[0] += 1
    split: List[Tuple[int, int]] = []
    start = 0
    for count in counts:
        if count:
            split.append((start, count))
            start += count
    return split


class _Session:
    """The bookkeeping every worker shares: log handle, winnings, first error."""

    def __init__(
        self,
        log_path: Optional[Path],
        on_hand: Optional[Callable[[HandResult, SessionSummary], None]],
    ) -> None:
        self.lock = threading.Lock()
        # Set by the first worker to fail, and checked by all of them between
        # hands.  A session that has already gone wrong should stop talking to
        # the server rather than run the remaining hands into the same error.
        self.stop = threading.Event()
        self.error: Optional[BaseException] = None
        self.winnings: List[int] = []
        self.started = time.perf_counter()
        self.on_hand = on_hand
        self.handle = None
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self.handle = open(log_path, "a", buffering=1)

    def record(self, result: HandResult) -> None:
        """One finished hand.  Held under the lock so callers need no locking.

        The log is append-as-you-go rather than collect-then-write: a session of
        any useful length runs for hours, and a crash at hour three should leave
        three hours of hands on disk rather than nothing.  With workers the
        lines interleave, so they arrive out of index order — every line carries
        its own ``hand`` and ``client_pos``, so nothing downstream needs them
        ordered.
        """
        with self.lock:
            self.winnings.append(result.winnings)
            if self.handle is not None:
                self.handle.write(json.dumps(result.to_dict()) + "\n")
            if self.on_hand is not None:
                self.on_hand(result, summarise(self.winnings, self.elapsed))

    def fail(self, error: BaseException) -> None:
        with self.lock:
            if self.error is None:
                self.error = error
        self.stop.set()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def _play_chunk(
    worker: int,
    start: int,
    count: int,
    policy_factory: Callable[[int], Policy],
    client_factory: Callable[[int], SlumbotClient],
    session: _Session,
) -> None:
    """One worker's hands, on its own client and its own policy."""
    try:
        client = client_factory(worker)
        policy = policy_factory(worker)
        for index in range(start, start + count):
            if session.stop.is_set():
                return
            session.record(play_hand(client, policy, index))
    except BaseException as error:  # noqa: BLE001 - re-raised on the caller's thread
        session.fail(error)


def play_session(
    hands: int,
    policy: Policy,
    log_path: Optional[Path] = None,
    client: Optional[SlumbotClient] = None,
    on_hand: Optional[Callable[[HandResult, SessionSummary], None]] = None,
) -> SessionSummary:
    """Play ``hands`` hands in a single session, logging each one as it finishes."""
    return play_session_parallel(
        hands,
        policy_factory=lambda _: policy,
        workers=1,
        log_path=log_path,
        client_factory=(lambda _: client) if client is not None else None,
        on_hand=on_hand,
    )


def play_session_parallel(
    hands: int,
    policy_factory: Callable[[int], Policy],
    workers: int = 4,
    log_path: Optional[Path] = None,
    client_factory: Optional[Callable[[int], SlumbotClient]] = None,
    on_hand: Optional[Callable[[HandResult, SessionSummary], None]] = None,
) -> SessionSummary:
    """The same session, played by ``workers`` concurrent clients.

    **Why this exists.**  A hand is a handful of request/response round trips
    with a decision between each, and the two halves do not overlap: measured
    here, a round trip to ``slumbot.com`` costs ~0.35s and a re-solving agent
    spends ~1.2s per hand thinking, so a sequential 1000-hand session is roughly
    half wall clock spent watching a socket.  Workers fill that gap with each
    other's thinking, and the session becomes bounded by the CPU it can actually
    use instead of by the round trip.

    **Why it is legitimate.**  Each worker holds its own client and therefore
    its own session token, which is exactly the "several people playing Slumbot
    at once" case the server already serves; nothing is shared between them and
    no hand is played faster or with more information than it would be alone.
    What is *not* legitimate is playing a single session concurrently, and the
    factory signature is what prevents it: there is no way to hand this function
    one client to share.  Keep ``workers`` modest for the same reason — it is
    someone else's machine, and four is the tested default.

    ``policy_factory`` gets one call per worker and must return a *fresh* policy
    each time.  A re-solving policy carries a hand's belief state across
    decisions, so two workers sharing one would interleave two hands into the
    same agent and quietly corrupt both.  It receives the worker index so the
    caller can seed each one differently.

    The result is the same ``SessionSummary`` the sequential path returns —
    mean and variance do not care what order the hands arrived in — and seats
    stay balanced because :func:`split_hands` hands out even-sized chunks.
    """
    split = split_hands(hands, workers)
    client_factory = client_factory or (lambda _: SlumbotClient())
    session = _Session(Path(log_path) if log_path is not None else None, on_hand)
    try:
        if len(split) <= 1:
            # Sequential stays genuinely sequential: no thread, so a failure
            # arrives with the stack that produced it.
            for worker, (start, count) in enumerate(split):
                _play_chunk(worker, start, count, policy_factory, client_factory, session)
                if session.error is not None:
                    raise session.error
        else:
            threads = [
                threading.Thread(
                    target=_play_chunk,
                    args=(worker, start, count, policy_factory, client_factory, session),
                    name=f"slumbot-{worker}",
                    daemon=True,
                )
                for worker, (start, count) in enumerate(split)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if session.error is not None:
                # Every worker has stopped by now, and the hands they did
                # finish are on disk and in the summary the caller will not
                # get.  Raising is still right: a session that lost a worker
                # is a session with an unplanned number of hands in it.
                raise session.error
    finally:
        session.close()
    return summarise(session.winnings, session.elapsed)
