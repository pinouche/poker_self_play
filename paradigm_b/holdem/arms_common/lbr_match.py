"""Local best response as an *opponent that plays hands*, not as a tree walk.

:mod:`.lbr` already answers "how much can this responder win against this
strategy" exactly: it walks the whole tree against full 1,326-combo ranges,
enumerates every runout, and returns a number that is the same every time it is
computed.  That is the stronger measurement and it stays the headline.

This module answers a different question, the one ReBeL's Table 1 reports:
**play LBR against the agent for N dealt hands and count the chips.**  881 ± 94
mbb/g there is a sample mean over hands, and the ± is the sampling error of a
match that could have gone otherwise.  Three things make it worth having
alongside the exact number rather than instead of it:

* it runs **from the blinds**, through all four betting rounds, where the exact
  metric scores sampled postflop endgames one at a time;
* it puts the agent in the same code path it uses against Slumbot — one hand,
  two cards, a decision at a time — so the two mbb/g figures are comparable;
* it is the form the literature reports, so it can be quoted next to a
  published number without a conversion nobody can check.

**The responder is the paper's, including its handicap.**  Table 1: *"the LBR
agent must call for the first two betting rounds, and can either fold, call, bet
1x pot, or bet all-in on the last two rounds."*  That is
:attr:`LbrMatchConfig.passive_rounds` and the probe set, and it is a real
restriction — this responder is weaker than the tree one, which probes four
sizes on every round.

**How it prices an action, and why that is not duplicated here.**  Exactly as
:func:`~.lbr.checkdown_value` does: assume the hand checks down, and split the
agent's range into what folds, calls and raises.  The one thing a dealt hand
needs that a tree walk does not is a way to ask the agent what it *would* do —
answered by :meth:`~.play.ResolvingAgent.clone`, advancing a copy through the
candidate bet and reading the strategy it re-solves.  So the two LBRs share
their arithmetic and differ only in how the response distribution is obtained.

**What LBR is allowed to know.**  The agent's *range* — never its cards.  LBR is
defined against a known strategy, and ``agent.reach[agent.seat]`` is precisely
the agent's range given the public history, updated with the agent's own
strategy as it acts.  Reading it is the access LBR assumes; reading
``agent.hand`` would be cheating and nothing here does.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from paradigm_b.holdem.arms_common.lbr import checkdown_value, split_response
from paradigm_b.holdem.arms_common.play import PlayConfig, ResolvingAgent
from paradigm_b.holdem.arms_common.slumbot import (
    SessionSummary,
    split_hands,
    summarise,
)
from paradigm_b.holdem.arms_common.slumbot_agent import CHIPS_PER_ENGINE_UNIT
from paradigm_b.holdem.engine.betting import (
    BIG_BLIND,
    CALL,
    DEFAULT_BET_FRACTIONS,
    FOLD,
    SMALL_BLIND,
    Betting,
    preflop_betting,
)
from paradigm_b.holdem.engine.combos import combo_index
from paradigm_b.holdem.engine.public_tree import PublicState
from paradigm_b.holdem.engine.space import TurnEndgameSpace
from paradigm_b.holdem.engine.strength import hand_ranks

# Board cards showing at the start of each betting round.
BOARD_AT_ROUND = (0, 3, 4, 5)


@dataclass
class LbrMatchConfig:
    """The match, the responder's handicap, and the agent's own settings."""

    play: PlayConfig = field(default_factory=PlayConfig)
    bet_fractions: Tuple[float, ...] = DEFAULT_BET_FRACTIONS
    max_raises: int = 2
    # Engine chips.  200 is the 100 big blinds ``SituationConfig`` trains at,
    # which is the depth the network has actually seen; ReBeL's HUNL is 200bb.
    # Playing deeper than training queries the network off its distribution, so
    # the default stays at the trained depth and the number says which it was.
    stack: int = 200
    # ReBeL Table 1's restriction: call the first two betting rounds, then
    # fold / call / pot / all-in.
    passive_rounds: int = 2
    probe_fractions: Tuple[float, ...] = (1.0,)
    include_all_in: bool = True
    # Runouts to average when pricing a check-down; ``None`` enumerates them.
    # LBR only ever acts on the last two rounds here, where that is 44 runouts
    # and then 1 — cheap enough that sampling would trade exactness for nothing.
    rollout_samples: Optional[int] = None


class LbrPlayer:
    """One seat, one hand, choosing by local best response."""

    def __init__(
        self,
        seat: int,
        hole_cards: Tuple[int, int],
        space: TurnEndgameSpace,
        config: Optional[LbrMatchConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = config or LbrMatchConfig()
        self.seat = seat
        self.space = space
        self.hand = combo_index(*hole_cards)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.probes = 0

    def act(
        self, real: Betting, board: Tuple[int, ...], agent: ResolvingAgent
    ) -> Tuple[int, int]:
        """Choose an action against ``agent``.  Returns ``(action, total)``.

        ``total`` is the chip total the action commits, so an off-abstraction
        bet is expressible: the action id is the nearest one the agent's tree
        holds and the chips are what really goes in, which is the same split
        :mod:`.slumbot_agent` makes against the real server.
        """
        if real.betting_round < self.config.passive_rounds:
            # The paper's handicap: no folding and no betting early, so the
            # responder cannot win a pot before the streets it is allowed to
            # attack.  ``CALL`` is a check when nothing is owed.
            return CALL, min(real.contributions[1 - self.seat], real.stack)

        # Solve this decision point once, on the live agent, before any clone is
        # taken: every clone starts from the map this produces, so the round's
        # solve is paid for once instead of once per candidate.  It is the same
        # solve the agent would do when its turn comes, cached.
        agent.strategy_for()

        candidates = [(CALL, min(real.contributions[1 - self.seat], real.stack))]
        if real.to_call(self.seat) > 0:
            candidates.append((FOLD, real.contributions[self.seat]))
        candidates.extend((None, total) for total in self._probe_totals(real))

        best, best_value = candidates[0], -np.inf
        for action, total in candidates:
            value = self._value_of(real, board, agent, action, total)
            if value > best_value:
                best, best_value = (action, total), value
        action, total = best
        if action is None:
            action = _nearest_aggressive(real, self.seat, total)
        return action, total

    def _probe_totals(self, real: Betting) -> List[int]:
        """Chip totals the responder may bet to, deduplicated.

        The raise cap binds the responder as it binds anyone: it is a rule of
        the game as modelled, so a responder allowed past it would be scored on
        lines the agent could never face.
        """
        owed = real.to_call(self.seat)
        room = real.stack - real.contributions[self.seat] - owed
        if room <= 0:
            return []
        aggressive = sum(
            real.is_aggressive(action) for action in real.history[real.betting_round]
        )
        if aggressive >= real.max_raises:
            return []
        pot_after_call = real.pot + owed
        totals: List[int] = []
        for fraction in self.config.probe_fractions:
            total = (
                real.contributions[self.seat]
                + owed
                + int(round(fraction * pot_after_call))
            )
            if real.contributions[1 - self.seat] < total < real.stack:
                totals.append(total)
        if self.config.include_all_in:
            totals.append(real.stack)
        return sorted(set(totals))

    def _value_of(
        self,
        real: Betting,
        board: Tuple[int, ...],
        agent: ResolvingAgent,
        action: Optional[int],
        total: int,
    ) -> float:
        """What one candidate is worth to the hand we hold."""
        if action == FOLD:
            child = real.apply(FOLD)
            public = PublicState(betting=child, board=board)
            return float(self.space.terminal_values(public, agent.reach)[self.seat][self.hand])

        if action is None:
            action = _nearest_aggressive(real, self.seat, total)
        child = real.apply_amount(action, total)
        folded, called, raised = self._response_to(agent, action, total)
        values = checkdown_value(
            self.space,
            child,
            board,
            folded,
            called,
            raised,
            self.seat,
            self.rng,
            self.config.rollout_samples,
        )
        return float(values[self.hand])

    def _response_to(
        self, agent: ResolvingAgent, action: int, total: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """How the agent's range splits if we take this action.

        The clone is what makes this honest: the live agent is left exactly as
        it was, and the copy is advanced through our bet so the strategy read
        back is the one the agent would really re-solve — conditioned on our
        action, at the pot our action really makes — rather than a strategy from
        a state that has not happened.
        """
        self.probes += 1
        clone = agent.clone()
        clone.observe(action, total)
        decides = (
            not (clone.public.is_terminal or clone.public.awaiting_board)
            and clone.public.to_move() == clone.seat
        )
        strategy = clone.strategy_for() if decides else None
        return split_response(
            strategy, clone.reach[clone.seat], clone.public.betting, self.space.num_hands
        )


def _nearest_aggressive(betting: Betting, player: int, total: int) -> int:
    """The abstraction's closest raise to ``total`` chips.

    The same rule :class:`~.slumbot_agent.SlumbotAgentPolicy` uses on incoming
    Slumbot bets, for the same reason: the action id is what conditions a range,
    the chips are what the pot gets, and when the raise cap has already bound
    there is no aggressive action left to name.
    """
    legal = betting.legal_actions()
    aggressive = [
        a for a in legal if betting.is_aggressive(a) or a == betting.all_in_action
    ]
    if not aggressive:
        return CALL
    return min(aggressive, key=lambda a: abs(betting.raise_total(player, a) - total))


@dataclass
class HandOutcome:
    """One dealt hand, from the agent's side of the table."""

    winnings: int  # engine chips, positive when the agent wins
    agent_seat: int
    showdown: bool
    board: Tuple[int, ...]
    solves: int
    probes: int


def play_hand(
    net,
    agent_seat: int,
    rng: np.random.Generator,
    config: Optional[LbrMatchConfig] = None,
) -> HandOutcome:
    """Deal one hand and play it out, agent against LBR.

    The dealer is here rather than borrowed from :mod:`.slumbot_fake` because
    nothing needs translating: both players speak the engine's own ``Betting``,
    so a hand is the betting object plus a deck and there is no protocol in the
    middle to get wrong.
    """
    config = config or LbrMatchConfig()
    deck = rng.permutation(52)
    hole = {0: (int(deck[0]), int(deck[1])), 1: (int(deck[2]), int(deck[3]))}
    runout = tuple(int(c) for c in deck[4:9])

    betting = preflop_betting(
        stack=config.stack,
        blinds=(SMALL_BLIND, BIG_BLIND),
        bet_fractions=config.bet_fractions,
        max_raises=config.max_raises,
    )
    agent = ResolvingAgent(
        net,
        seat=agent_seat,
        hole_cards=hole[agent_seat],
        betting=betting,
        config=config.play,
        rng=rng,
    )
    lbr = LbrPlayer(
        seat=1 - agent_seat,
        hole_cards=hole[1 - agent_seat],
        space=agent.space,
        config=config,
        rng=rng,
    )

    real = betting
    board: Tuple[int, ...] = ()
    while not real.is_terminal:
        if real.awaiting_board:
            # ``_close_round`` has already advanced ``betting_round``, so the
            # round we are waiting on is the one whose cards to show.
            shown = runout[: BOARD_AT_ROUND[real.betting_round]]
            new = shown[len(board) :]
            real = real.deal_board()
            board = shown
            agent.observe_board(new)
            continue
        if real.to_move() == agent_seat:
            action, total = agent.act()
            real = real.apply_amount(action, total)
        else:
            action, total = lbr.act(real, board, agent)
            real = real.apply_amount(action, total)
            agent.observe(action, total)

    return HandOutcome(
        winnings=_settle(real, hole, runout, agent_seat),
        agent_seat=agent_seat,
        showdown=real.folder < 0,
        board=runout,
        solves=agent.solves,
        probes=lbr.probes,
    )


def _settle(
    real: Betting,
    hole: Dict[int, Tuple[int, int]],
    runout: Tuple[int, ...],
    agent_seat: int,
) -> int:
    """Chips the agent won, in engine units.

    Both cases come from the engine's own payoff accounting rather than a second
    reading of the contributions: ``fold_returns`` for a hand that ended to a
    fold, ``showdown_stake`` for one that reached the river — the same functions
    the exploitability and LBR tree walks settle with, so a match and a tree
    agree about what a pot was worth.
    """
    if real.folder >= 0:
        return int(real.fold_returns()[agent_seat])

    ranks = hand_ranks(runout)
    mine = ranks[combo_index(*hole[agent_seat])]
    theirs = ranks[combo_index(*hole[1 - agent_seat])]
    if mine == theirs:
        return 0
    stake = int(real.showdown_stake())
    return stake if mine > theirs else -stake


def play_lbr_session(
    net,
    hands: int,
    config: Optional[LbrMatchConfig] = None,
    seed: int = 0,
    workers: int = 1,
) -> SessionSummary:
    """Play ``hands`` and report mbb/g with its standard error.

    Seats alternate hand by hand, so the blinds cannot bias the number, and the
    summary is :func:`~.slumbot.summarise` — the same mbb/g and interval the
    Slumbot sessions report, computed the same way, because a number meant to
    sit next to another one should not be arithmetic of its own.  Engine chips
    are converted at ``CHIPS_PER_ENGINE_UNIT``, which makes the big blind 100
    exactly as it is there.

    Positive is **the responder winning**, matching how the paper reports LBR:
    881 is LBR beating the agent for 881 mbb/g.
    """
    config = config or LbrMatchConfig()
    started = time.perf_counter()
    winnings: List[int] = []
    lock = threading.Lock()

    def run(worker: int, start: int, count: int) -> None:
        rng = np.random.default_rng([seed, worker])
        local: List[int] = []
        for index in range(count):
            outcome = play_hand(net, (start + index) % 2, rng, config)
            # Reported from the responder's side, and in Slumbot's chips.
            local.append(-outcome.winnings * CHIPS_PER_ENGINE_UNIT)
        with lock:
            winnings.extend(local)

    split = split_hands(hands, max(workers, 1))
    if len(split) <= 1:
        for worker, (start, count) in enumerate(split):
            run(worker, start, count)
    else:
        threads = [
            threading.Thread(target=run, args=(worker, start, count), daemon=True)
            for worker, (start, count) in enumerate(split)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    return summarise(winnings, time.perf_counter() - started)
