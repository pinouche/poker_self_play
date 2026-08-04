"""Wiring the value network into a Slumbot session.

:mod:`.slumbot` speaks the protocol; :mod:`.play` holds two cards and re-solves.
This is the translation layer between them, and translation is the whole
difficulty — the two do not agree about chips, about bet sizes, or about how
many times a round may be raised.

**Chips.**  Slumbot is 50/100 with a 20,000 stack.  The engine is 1/2, so one
engine chip is 50 Slumbot chips and the conversion is exact with no rounding:
the stack is 400 engine chips and a Slumbot ``b300`` preflop is a raise to 6.

**Sizes.**  Slumbot's ``bX`` gives the *street* contribution; the engine's
totals are cumulative over the hand.  Converting needs the contributions each
player carried into the current street, which is what ``_street_base`` records
at every deal.

**The abstraction.**  Slumbot bets whatever it likes.  The agent holds a handful
of pot fractions, so an incoming bet is priced at its real chips via
``apply_amount`` — the pot stays honest — while the *action id* used to condition
the opponent's range is the nearest size the abstraction does hold.  That is
standard action translation and it is the same real/seen split
:mod:`.lbr` makes, with one difference: here the real chips win, because
Slumbot is the authority on what is in the pot.

**Two mismatches worth stating rather than hiding**, because they bound what a
number from this harness means:

*Stack depth.*  ``SituationConfig.preflop_stack`` is 200 engine chips — 100 big
blinds — so the network was trained on hands half as deep as the ones Slumbot
deals.  Playing at Slumbot's true depth queries the network well outside its
training distribution; playing at 100bb would measure a different game.  This
plays at Slumbot's depth, because the alternative is not a Slumbot number, and
reports the gap instead.

*Raise caps.*  Training used ``max_raises=1`` per round.  Slumbot 3-bets and
4-bets.  A tree that cannot express the opponent's raise cannot condition their
range on it, so the cap is configurable here and every time it still binds the
policy counts a ``translation_fallback`` — a number to read alongside the
mbb/g, since a session full of them is measuring the abstraction, not the net.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from paradigm_b.holdem.arms_common.play import PlayConfig, ResolvingAgent
from paradigm_b.holdem.arms_common.slumbot import HandState
from paradigm_b.holdem.engine.betting import (
    CALL,
    DEFAULT_BET_FRACTIONS,
    FOLD,
    SMALL_BLIND,
    BIG_BLIND,
    preflop_betting,
)

# One engine chip is this many Slumbot chips: their small blind is 50, ours 1.
CHIPS_PER_ENGINE_UNIT = 50
SLUMBOT_STACK = 20_000
RANKS = "23456789TJQKA"
SUITS = "cdhs"
# Board cards showing once each street has been dealt.
BOARD_AT_STREET = (0, 3, 4, 5)


def parse_card(text: str) -> int:
    """``"Ac"`` -> 48.  The engine's ``rank * 4 + suit``, same notation."""
    if len(text) != 2 or text[0] not in RANKS or text[1] not in SUITS:
        raise ValueError(f"not a card: {text!r}")
    return RANKS.index(text[0]) * 4 + SUITS.index(text[1])


def format_card(card: int) -> str:
    return f"{RANKS[card // 4]}{SUITS[card % 4]}"


def tokenize_street(street: str) -> List[str]:
    """``"b200c"`` -> ``["b200", "c"]``."""
    tokens: List[str] = []
    index = 0
    while index < len(street):
        char = street[index]
        if char in "kcf":
            tokens.append(char)
            index += 1
        elif char == "b":
            end = index + 1
            while end < len(street) and street[end].isdigit():
                end += 1
            if end == index + 1:
                raise ValueError(f"bet with no size in {street!r}")
            tokens.append(street[index:end])
            index = end
        else:
            raise ValueError(f"unexpected character {char!r} in {street!r}")
    return tokens


def events_of(action: str) -> List[Tuple[str, object]]:
    """The action string as an ordered replay: deals and actions, interleaved.

    Recomputed from scratch on every decision and replayed from a cursor, so
    the agent's state is always a function of what Slumbot says happened rather
    than of what we remember doing.
    """
    events: List[Tuple[str, object]] = []
    for index, street in enumerate(action.split("/")):
        if index > 0:
            events.append(("deal", index))
        for token in tokenize_street(street):
            events.append(("act", token))
    return events


@dataclass
class SlumbotAgentConfig:
    """How the agent is set up for a Slumbot seat."""

    play: PlayConfig = field(default_factory=PlayConfig)
    bet_fractions: Tuple[float, ...] = DEFAULT_BET_FRACTIONS
    # Raises per round the agent's own tree may express.  Training used 1; two
    # lets the agent follow a 3-bet, at the cost of a bigger tree per solve.
    max_raises: int = 2
    # Engine chips.  400 is Slumbot's real 200bb; see the module docstring for
    # why this is not the 200 the network trained at.
    stack: int = SLUMBOT_STACK // CHIPS_PER_ENGINE_UNIT


class SlumbotAgentPolicy:
    """A :class:`~.slumbot.Policy` backed by a re-solving value network.

    Stateful across the hand on purpose: rebuilding the agent at every decision
    would re-solve every previous round, and a solve is the expensive thing
    here.  The cursor into ``events_of`` is what keeps that incremental state
    honest — if Slumbot ever sends an action string that does not extend the one
    we have already consumed, the hand is rebuilt from scratch rather than
    patched.
    """

    def __init__(self, net, config: Optional[SlumbotAgentConfig] = None, rng=None) -> None:
        self.net = net
        self.config = config or SlumbotAgentConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.translation_fallbacks = 0
        self.hands = 0
        self._reset()

    def _reset(self) -> None:
        self._agent: Optional[ResolvingAgent] = None
        self._key: Optional[tuple] = None
        self._cursor = 0
        self._street_base = (0, 0)

    # --- setup -------------------------------------------------------------
    def _start_hand(self, state: HandState) -> None:
        cards = tuple(parse_card(c) for c in state.hole_cards)
        if len(cards) != 2:
            raise ValueError(f"expected two hole cards, got {state.hole_cards}")
        betting = preflop_betting(
            stack=self.config.stack,
            blinds=(SMALL_BLIND, BIG_BLIND),
            bet_fractions=self.config.bet_fractions,
            max_raises=self.config.max_raises,
        )
        # ``client_pos`` 1 is the button, which acts first preflop and is the
        # small blind heads-up; the engine calls that seat 0.
        seat = 0 if state.client_pos == 1 else 1
        self._agent = ResolvingAgent(
            self.net,
            seat=seat,
            hole_cards=cards,
            betting=betting,
            config=self.config.play,
            rng=self.rng,
        )
        self._key = (state.token, state.hole_cards, state.client_pos)
        self._cursor = 0
        self._street_base = (0, 0)
        self.hands += 1

    # --- replay ------------------------------------------------------------
    def _apply(self, events: Sequence[Tuple[str, object]], board: Sequence[str]) -> None:
        agent = self._agent
        assert agent is not None
        for kind, payload in events[self._cursor :]:
            if kind == "deal":
                street = int(payload)
                target = BOARD_AT_STREET[street]
                shown = [parse_card(c) for c in board[:target]]
                new = shown[len(agent.public.board) :]
                if new:
                    agent.observe_board(new)
                self._street_base = tuple(agent.public.betting.contributions)
            else:
                self._apply_action(str(payload))
            self._cursor += 1

    def _apply_action(self, token: str) -> None:
        """One opponent action, priced in real chips, translated for the range."""
        agent = self._agent
        assert agent is not None
        betting = agent.public.betting
        mover = betting.to_move()
        if mover == agent.seat:
            # Our own action, echoed back by Slumbot.  We advanced the agent
            # when we chose it, so there is nothing to replay.
            return
        legal = betting.legal_actions()
        if token == "f":
            agent.observe(FOLD)
            return
        if token in ("c", "k"):
            agent.observe(CALL)
            return
        total = self._engine_total(mover, int(token[1:]))
        action = self._nearest_action(betting, mover, total, legal)
        agent.observe(action, total)

    def _engine_total(self, player: int, street_chips: int) -> int:
        """Slumbot's street contribution -> the engine's cumulative total."""
        if street_chips % CHIPS_PER_ENGINE_UNIT:
            raise ValueError(
                f"{street_chips} is not a whole number of engine chips; "
                f"Slumbot should never bet off a {CHIPS_PER_ENGINE_UNIT}-chip grid"
            )
        return self._street_base[player] + street_chips // CHIPS_PER_ENGINE_UNIT

    def _nearest_action(self, betting, player: int, total: int, legal) -> int:
        """The abstraction's closest raise to ``total`` chips.

        When the round's raise cap has already bound there is no aggressive
        action to pick, and the agent simply cannot represent what happened.
        Counting that is the point: the alternative is to silently fold the
        opponent's raise into ``call`` and report a number that looks fine.
        """
        aggressive = [a for a in legal if betting.is_aggressive(a) or a == betting.all_in_action]
        if not aggressive:
            self.translation_fallbacks += 1
            return CALL
        return min(aggressive, key=lambda a: abs(betting.raise_total(player, a) - total))

    # --- the Policy protocol ----------------------------------------------
    def __call__(self, state: HandState) -> str:
        key = (state.token, state.hole_cards, state.client_pos)
        events = events_of(state.action)
        # A hand is identified by the token *and* the cards, so a new deal on a
        # carried session rebuilds.  ``_cursor > len(events)`` catches the case
        # the incremental state cannot be right about: an action string shorter
        # than what we have already replayed is not an extension of it.
        if self._agent is None or key != self._key or self._cursor > len(events):
            self._start_hand(state)
        self._apply(events, state.board)

        agent = self._agent
        assert agent is not None
        if agent.is_terminal:
            raise RuntimeError("asked to act in a finished hand")
        betting = agent.public.betting
        owed = betting.to_call(agent.seat)
        action, total = agent.act()
        self._cursor += 1  # our own action, when Slumbot echoes it back
        if action == FOLD:
            return "f"
        if action == CALL:
            return "c" if owed > 0 else "k"
        street_chips = (total - self._street_base[agent.seat]) * CHIPS_PER_ENGINE_UNIT
        return f"b{street_chips}"
