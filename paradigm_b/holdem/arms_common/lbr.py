"""Local best response: exploitability the agent's own abstraction cannot hide.

The existing metric (``arms_common/evaluation.py``) computes an *exact* best
response over the agent's public tree.  That is the right number for "did search
find a good strategy", and it is blind to the thing action abstraction costs:
the responder is confined to the same four bet sizes the agent has, so it can
never ask a question the agent has to translate.  A real opponent bets 63% of
the pot, the agent maps that onto 50% or 100%, and answers a question slightly
different from the one it was asked.  That gap is invisible to a responder that
only ever bets 50% or 100% itself.

So this responder bets sizes the agent does not have, and the agent's state
advances by the *translated* action while the pot advances by the chips actually
put in.  The two diverge from that point on — the agent sizes its own later bets
off a pot that is not the real one — and the divergence is exactly what is being
measured.  Everything hangs off :func:`translate_bet`'s pseudo-harmonic mapping,
which is the agent's stated defence; this is the attack on it.

**Two responders, one tree.**

``classic``   Lisy & Bowling's LBR.  At each of its decisions the responder
              assumes the hand checks down after the candidate action, prices
              that with the showdown equity of its own hand against the agent's
              range, and takes the best.  One action of lookahead, then a
              rollout.  It is a deliberately weak responder, and its value is
              therefore a *lower bound* on exploitability — the number the
              literature reports, and the reason a published LBR figure is
              always described as a bound rather than a measurement.

``full``      The same probe sizes, but the responder prices each action by
              actually playing on.  A tighter lower bound, and affordable here
              only because these are endgames — on full hold'em it is the thing
              LBR exists to avoid.

Running both is worth the ~1.3x: the gap between them is how much the myopic
heuristic gives up, which is the one number that says whether a published LBR
figure on this game is worth taking seriously.

.. warning::

   **This is not comparable to the ReBeL paper's HUNL figures.**  Those are for
   the whole game from the blinds, in mbb/g, against Libratus, Slumbot and
   humans.  This measures a postflop endgame whose pot, stacks and starting
   ranges were invented by a sampler.  The numbers are the same *kind* of
   quantity and are comparable across runs and abstractions here; they are not
   convertible to anything in the paper, in either direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from paradigm_b.core.search.policy import StrategyMap
from paradigm_b.holdem.engine.betting import CALL, FOLD, Betting
from paradigm_b.holdem.engine.combos import NUM_CARDS, board_mask, compatible_mass
from paradigm_b.holdem.engine.public_tree import PublicState
from paradigm_b.holdem.engine.showdown import showdown_values
from paradigm_b.holdem.engine.space import EndgameSpace
from paradigm_b.holdem.engine.translation import translate_bet

NUM_PLAYERS = 2


@dataclass
class LBRConfig:
    """The responder's own action abstraction — deliberately not the agent's.

    ``probe_fractions`` defaults to sizes sitting *between* the shipped
    abstraction's (0.25, 0.5, 1.0, 2.0), because a probe that coincides with one
    of the agent's sizes translates to it exactly and measures nothing about
    translation.  The midpoints are where the pseudo-harmonic mapping has to
    split its weight, and therefore where it can be wrong.
    """

    probe_fractions: Tuple[float, ...] = (0.375, 0.75, 1.5, 3.0)
    include_all_in: bool = True
    # Runouts to average over when pricing a check-down.  ``None`` enumerates
    # them exactly: 1 on the river, 44 on the turn, 1,176 on the flop.  A flop
    # situation with every runout enumerated is ~1,176 showdown solves per
    # responder node, which is affordable only because there are few such nodes.
    rollout_samples: Optional[int] = None


def _remaining_cards(board: Tuple[int, ...]) -> List[int]:
    used = set(board)
    return [c for c in range(NUM_CARDS) if c not in used]


def checkdown_equity(
    space: EndgameSpace,
    board: Tuple[int, ...],
    villain_reach: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    samples: Optional[int] = None,
) -> np.ndarray:
    """Per-hand showdown value if the hand is checked down from ``board``.

    ``showdown_values`` already answers "opponent mass I beat minus mass I lose
    to" on a *complete* board, with blocked combinations excluded, so a
    check-down is that quantity averaged over every way the remaining cards can
    fall.  Returned per unit of stake: the caller multiplies by what is actually
    at risk.

    The averaging is over board runouts, and the reach is masked by each runout
    as it is dealt — a card on the board is a card neither player can hold, and
    forgetting that inflates the equity of exactly the hands the runout kills.
    """
    missing = 5 - len(board)
    if missing <= 0:
        return showdown_values(board, np.ascontiguousarray(villain_reach), 1.0)

    remaining = _remaining_cards(board)
    runouts: List[Tuple[int, ...]] = []
    if missing == 1:
        runouts = [(c,) for c in remaining]
    else:
        for i, first in enumerate(remaining):
            for second in remaining[i + 1 :]:
                runouts.append((first, second))
    if samples is not None and rng is not None and samples < len(runouts):
        picked = rng.choice(len(runouts), size=samples, replace=False)
        runouts = [runouts[int(i)] for i in picked]

    total = np.zeros(space.num_hands)
    for runout in runouts:
        full = tuple(sorted(board + runout))
        mask = board_mask(full)
        masked = np.ascontiguousarray(villain_reach * mask)
        total += showdown_values(full, masked, 1.0) * mask
    return total / max(len(runouts), 1)


def split_response(
    strategy: Optional[np.ndarray],
    agent_reach: np.ndarray,
    betting: Betting,
    num_hands: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split the agent's range into the mass that folds, calls and raises.

    ``strategy`` is the agent's per-hand action probabilities at this state, or
    ``None`` where nothing produced behaviour for it — uniform, matching what
    the scoring path fills in, so an agent measured here plays as it does there.

    **A round the agent cannot act in is all "called".**  Once a call closes the
    betting, the next thing that happens is a card, not a decision — but a
    ``Betting`` that is awaiting a board still answers ``legal_actions`` and
    ``to_move``, so asking it who folds returns a strategy over actions nobody
    is going to take.  Left unguarded, the myopic responder priced its own
    round-closing call as though the agent might still raise over it, subtracted
    the concession term for that imaginary raise, and reported a *lower* bound
    than it should have.
    """
    zero = np.zeros(num_hands)
    if betting.is_terminal or betting.awaiting_board:
        return zero.copy(), agent_reach.copy(), zero.copy()

    actions = betting.legal_actions()
    if strategy is None:
        strategy = np.full((num_hands, len(actions)), 1.0 / len(actions))
    folded, called, raised = zero.copy(), zero.copy(), zero.copy()
    for index, action in enumerate(actions):
        share = agent_reach * strategy[:, index]
        if action == FOLD:
            folded += share
        elif betting.is_aggressive(action):
            raised += share
        else:
            called += share
    return folded, called, raised


def checkdown_value(
    space: EndgameSpace,
    real: Betting,
    board: Tuple[int, ...],
    folded: np.ndarray,
    called: np.ndarray,
    raised: np.ndarray,
    hero: int,
    rng: Optional[np.random.Generator] = None,
    rollout_samples: Optional[int] = None,
) -> np.ndarray:
    """Classic LBR's per-hand value of a state, assuming the hand checks down.

    Three terms, priced at ``real``'s chips, against the agent's range split by
    :func:`split_response`.  This is the whole of the myopic responder's
    arithmetic and it is shared: :class:`_Walker` uses it to score a tree, and
    :mod:`.lbr_match` uses it to choose an action in a dealt hand, so the two
    can disagree about what LBR *does* but never about what an action is worth.
    """
    correction = space.pair_correction
    mask = board_mask(board) if len(board) == 5 else space.root_mask()

    # They fold: the responder takes the pot as it stood, against exactly the
    # mass of agent hands that folded.
    pot_if_folded = float(real.contributions[1 - hero]) + real.starting_pot / 2.0
    fold_term = correction * pot_if_folded * compatible_mass(folded) * mask

    # They call: showdown, at the stake that would then be at risk.
    #
    # ``min(contributions)`` is the right stake at a *settled* terminal, and
    # exactly wrong here: this state is the responder's bet, before the agent
    # has matched it, so the minimum is still the agent's old contribution and
    # the responder's own money silently drops out of the showdown.  It got the
    # fold equity for free and could only lose the dead pot, which is why the
    # myopic responder was scoring four times the searching one.  A call levels
    # the contributions at the larger of the two, so that is what is at risk.
    stake = float(max(real.contributions)) + real.starting_pot / 2.0
    equity = checkdown_equity(space, board, called, rng, rollout_samples)
    call_term = correction * stake * equity

    # They raise, and the responder gives up.  Pricing a raise as though it were
    # a call is what made this responder score *above* the fully searching one —
    # it collected a free check-down at a stake it would have had to pay again
    # to reach, which is not a bound on anything.  Conceding is the conservative
    # reading, and conservative is the whole point: LBR's number is only
    # meaningful as a lower bound.
    committed = float(real.contributions[hero]) + real.starting_pot / 2.0
    raise_term = -correction * committed * compatible_mass(raised) * mask

    return fold_term + call_term + raise_term


class _Walker:
    """One responder, walking the real game while the agent walks its own.

    ``real`` is the betting that decides payoffs; ``seen`` is what the agent
    believes, which is the same object until the responder bets a size the
    abstraction lacks and translation forces them apart.
    """

    def __init__(
        self,
        space: EndgameSpace,
        strategies: StrategyMap,
        hero: int,
        config: LBRConfig,
        myopic: bool,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.space = space
        self.strategies = strategies
        self.hero = hero
        self.agent = 1 - hero
        self.config = config
        self.myopic = myopic
        self.rng = rng

    # --- entry -------------------------------------------------------------
    def value(self, root: PublicState, reach: np.ndarray) -> np.ndarray:
        return self._walk(root.betting, root.betting, tuple(root.board), reach.copy())

    # --- the walk ----------------------------------------------------------
    def _walk(
        self,
        real: Betting,
        seen: Betting,
        board: Tuple[int, ...],
        reach: np.ndarray,
    ) -> np.ndarray:
        if real.is_terminal:
            public = PublicState(betting=real, board=board)
            return self.space.terminal_values(public, reach)[self.hero]
        if real.awaiting_board:
            return self._chance(real, seen, board, reach)
        if real.to_move() == self.hero:
            return self._responder(real, seen, board, reach)
        return self._agent(real, seen, board, reach)

    def _chance(self, real, seen, board, reach):
        remaining = _remaining_cards(board)
        weight = 1.0 / (NUM_CARDS - len(board) - 4)
        values = np.zeros(self.space.num_hands)
        for card in remaining:
            mask = self.space.deal_mask(card)
            child_reach = reach * mask
            if child_reach.sum() == 0.0:
                continue
            child = self._walk(
                real.deal_board(),
                seen.deal_board(),
                tuple(sorted(board + (card,))),
                child_reach,
            )
            values += weight * child * mask
        return values

    def _agent(self, real, seen, board, reach):
        """The agent acts, in its own accounting, paying real chips."""
        public = PublicState(betting=seen, board=board)
        actions = seen.legal_actions()
        strategy = self.strategies.get(public)
        if strategy is None:  # a state the resolver never produced behaviour for
            strategy = np.full((self.space.num_hands, len(actions)), 1.0 / len(actions))

        values = np.zeros(self.space.num_hands)
        for index, action in enumerate(actions):
            child_reach = reach.copy()
            child_reach[self.agent] = reach[self.agent] * strategy[:, index]
            if child_reach[self.agent].sum() <= 0.0:
                continue
            seen_child = seen.apply(action)
            # The agent computes its size off the pot it *believes* it is in,
            # then puts that many chips into the real one.  Carrying the
            # increment rather than the total is what keeps the two accountings
            # from silently re-syncing after an off-tree bet.
            increment = seen_child.contributions[self.agent] - seen.contributions[self.agent]
            real_total = real.contributions[self.agent] + increment
            real_child = real.apply_amount(action, real_total)
            values += self._walk(real_child, seen_child, board, child_reach)
        return values

    # --- the responder -----------------------------------------------------
    def _responder(self, real, seen, board, reach):
        """Every probe priced, best taken per hand."""
        candidates: List[np.ndarray] = []
        owed = real.to_call(self.hero)

        if owed > 0:
            # Priced by walking into the fold, not by writing down what it
            # costs.  A fold's counterfactual value is not a flat "what I put
            # in": like every terminal it is weighted by the opponent mass that
            # is actually compatible with each of the responder's hands, and
            # hand-rolling it dropped that factor.
            candidates.append(self._price(real, seen, board, reach, FOLD, None))

        candidates.append(self._price(real, seen, board, reach, CALL, None))

        for total in self._probe_totals(real):
            candidates.append(self._price(real, seen, board, reach, None, total))

        stacked = np.stack(candidates)
        return stacked.max(axis=0)

    def _probe_totals(self, real: Betting) -> List[int]:
        """Chip totals the responder may raise to, deduplicated."""
        me = real.to_move()
        owed = real.to_call(me)
        room = real.stack - real.contributions[me] - owed
        if room <= 0:
            return []
        # The raise cap binds the responder too.  It is a rule of the game as
        # modelled, not a feature of the agent's abstraction, so a responder
        # allowed to re-raise past it would be scored on lines that do not
        # exist and would report exploitability the agent could never suffer.
        aggressive = sum(
            real.is_aggressive(action) for action in real.history[real.betting_round]
        )
        if aggressive >= real.max_raises:
            return []
        pot_after_call = real.pot + owed
        totals: List[int] = []
        for fraction in self.config.probe_fractions:
            total = real.contributions[me] + owed + int(round(fraction * pot_after_call))
            if real.contributions[1 - me] < total < real.stack:
                totals.append(total)
        if self.config.include_all_in:
            totals.append(real.stack)
        return sorted(set(totals))

    def _price(self, real, seen, board, reach, action, total):
        """Value of one candidate, per responder hand.

        A bet the abstraction lacks becomes a *distribution* over the two sizes
        it sits between, so the agent's continuation is the weighted mixture of
        what it would do at each.  That is not an approximation of translation —
        it is what translation does, evaluated in expectation instead of sampled.
        """
        if action == FOLD:
            real_total = real.contributions[self.hero]
            branches = {FOLD: 1.0}
        elif action == CALL:
            real_total = min(real.contributions[1 - self.hero], real.stack)
            branches = {CALL: 1.0}
        else:
            real_total = int(total)
            branches = translate_bet(seen, self._seen_equivalent(real, seen, real_total))

        values = np.zeros(self.space.num_hands)
        for translated, weight in branches.items():
            if weight <= 0.0:
                continue
            seen_child = seen.apply(translated)
            real_child = real.apply_amount(translated, real_total)
            if self.myopic:
                values += weight * self._rollout(
                    real_child, seen_child, board, reach
                )
            else:
                values += weight * self._walk(real_child, seen_child, board, reach)
        return values

    def _seen_equivalent(self, real: Betting, seen: Betting, real_total: int) -> int:
        """The responder's real raise, expressed in the agent's accounting.

        Translation reads a bet as a fraction of the pot, and the agent's pot is
        not the real one once they have diverged.  Rescaling by the pot ratio is
        what keeps "63% of the pot" meaning 63% to the agent as well, rather
        than silently shrinking as the real pot outgrows the believed one.
        """
        owed_real = real.to_call(self.hero)
        extra = real_total - real.contributions[self.hero] - owed_real
        pot_real = max(real.pot + owed_real, 1)
        owed_seen = seen.to_call(self.hero)
        pot_seen = max(seen.pot + owed_seen, 1)
        scaled = extra * (pot_seen / pot_real)
        return int(round(seen.contributions[self.hero] + owed_seen + scaled))

    def _rollout(self, real, seen, board, reach):
        """Classic LBR's estimate: assume the hand is checked down from here.

        Two terms, and the fold term is the one that makes LBR find anything at
        all: a responder that never bets cannot win a pot the agent would have
        surrendered.  The agent's *continuing* range is what the showdown term
        is priced against, so a bet that folds out the weak part of the range
        makes the remainder harder to beat, and the heuristic sees that.
        """
        if real.is_terminal:
            public = PublicState(betting=real, board=board)
            return self.space.terminal_values(public, reach)[self.hero]

        folded, called, raised = self._agent_response(real, seen, board, reach)
        return checkdown_value(
            self.space,
            real,
            board,
            folded,
            called,
            raised,
            self.hero,
            self.rng,
            self.config.rollout_samples,
        )

    def _agent_response(self, real, seen, board, reach):
        """Split the agent's range three ways: folds, calls, and raises."""
        public = PublicState(betting=seen, board=board)
        return split_response(
            self.strategies.get(public),
            reach[self.agent],
            seen,
            self.space.num_hands,
        )


def lbr_values(
    space: EndgameSpace,
    strategies: StrategyMap,
    root: PublicState,
    reach: np.ndarray,
    config: Optional[LBRConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Both responders, both seats, against one agent strategy.

    Reported the way :func:`subgame_exploitability` reports: the sum over seats,
    which is zero for a strategy no responder can beat and grows with how much
    it can be beaten.  ``lbr_full >= lbr_classic`` always, because the full
    responder can reproduce any myopic choice and is free to do better.
    """
    config = config or LBRConfig()
    scores: Dict[str, float] = {}
    for name, myopic in (("lbr_classic", True), ("lbr_full", False)):
        total = 0.0
        for hero in range(NUM_PLAYERS):
            walker = _Walker(space, strategies, hero, config, myopic, rng)
            per_hand = walker.value(root, reach)
            total += float((reach[hero] * per_hand).sum())
        scores[name] = total
    scores["lbr_myopia_gap"] = scores["lbr_full"] - scores["lbr_classic"]
    return scores
