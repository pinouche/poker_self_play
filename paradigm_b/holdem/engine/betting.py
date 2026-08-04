"""No-limit betting, from the blinds to the river.

Leduc's betting had one legal raise size; no-limit has a continuum, so the tree
only exists once an *action abstraction* is chosen.  This one keeps four sizes —
half pot, pot, all-in, plus check/call and fold — with a cap on raises per
round.  That is coarse next to a real solver, and coarseness here is a source of
exploitability no amount of search recovers: a real opponent can bet sizes the
abstraction cannot represent.  It is the honest cost of leaving Leduc behind,
and it is why stage 4 needs action translation before it means anything against
a human.

Chips are integers.  ``starting_pot`` is what is already in the middle when the
round begins; ``stack`` is what each player can still put in.

**A whole hand and a postflop endgame are the same dataclass.**  A four-round
state from :func:`preflop_betting` starts at the blinds with an empty board; a
two-round state built directly is a turn endgame that assumes the earlier
streets happened and left ``starting_pot`` behind.  One rule has to know the
difference: heads-up, the small blind opens preflop and the big blind is first
to act on every later street, whereas an endgame with no blinds posted simply
lets player 0 open.  :meth:`Betting.first_actor` carries that, and ``blinds`` is
what tells it which of the two games it is in.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple

FOLD = 0
CALL = 1
FIRST_BET = 2

# Action ids are FOLD, CALL, then one per bet fraction, then all-in last.  The
# module-level names below describe the *default* abstraction; a state with a
# different ``bet_fractions`` has its own layout, which ``bet_action_ids`` and
# ``all_in_action`` report.
#
# Four sizes rather than two.  Every size the abstraction lacks is a size a real
# opponent can bet and the agent must *translate* — answering a question
# slightly different from the one it was asked — and that gap is a floor on
# exploitability no amount of search removes (see ``engine/translation.py``).
# Going from (0.5, 1.0) to (0.25, 0.5, 1.0, 2.0) roughly halves the widest
# untranslated gap, for 1.57x the depth-limited turn tree (357 -> 561 nodes).
#
# The raise cap stays at one.  Lifting it to two is the other axis and costs
# far more — 6.4x the tree with these sizes — so it is left as a config knob.
# Be aware of what that means when reading an LBR number: the agent genuinely
# cannot re-raise, and a best responder will collect that as real
# exploitability rather than as an artifact of measurement.
DEFAULT_BET_FRACTIONS = (0.25, 0.5, 1.0, 2.0)
BET_QUARTER = 2
BET_HALF = 3
BET_POT = 4
BET_DOUBLE = 5
ALL_IN = 6

MAX_RAISES_PER_ROUND = 2
NUM_ROUNDS = 2
# preflop, flop, turn, river — a whole hand rather than a postflop endgame.
NUM_ROUNDS_FULL = 4
SMALL_BLIND = 1
BIG_BLIND = 2

# ReBeL appendix D: *"The bet sizes are hand-chosen based on conventional poker
# wisdom and are fixed fractions of the pot, though each bet size is perturbed
# by +/-0.1x pot during training to ensure diversity in the training data."*
BET_PERTURBATION = 0.1
# A floor, so that jitter can never turn a small size into a non-bet.  With the
# default sizes it never binds; it exists for abstractions that include one.
MIN_BET_FRACTION = 0.05

# Distinct action ids the abstraction needs: fold, call, the bets, and all-in.
# ``HoldemPolicyNet`` has a fixed-width head, so an abstraction wider than it
# would silently drop its largest sizes off the end of the policy target.
MAX_SUPPORTED_BET_FRACTIONS = 6


def num_action_ids(bet_fractions: Tuple[float, ...] = DEFAULT_BET_FRACTIONS) -> int:
    """How many distinct action ids ``bet_fractions`` gives rise to."""
    return FIRST_BET + len(bet_fractions) + 1


def validate_bet_fractions(bet_fractions: Tuple[float, ...]) -> Tuple[float, ...]:
    """Check an abstraction is usable, and return it in canonical order.

    Called where the abstraction is *chosen* — a config or a CLI argument —
    rather than in ``Betting.__post_init__``, because a frozen ``Betting`` is
    constructed hundreds of thousands of times per label and must stay free.
    """
    fractions = tuple(float(f) for f in bet_fractions)
    if not fractions:
        raise ValueError("at least one bet fraction is required")
    if any(f <= 0.0 for f in fractions):
        raise ValueError(f"bet fractions must be positive: {fractions}")
    if len(set(fractions)) != len(fractions):
        raise ValueError(f"duplicate bet fractions: {fractions}")
    if len(fractions) > MAX_SUPPORTED_BET_FRACTIONS:
        raise ValueError(
            f"{len(fractions)} bet fractions needs "
            f"{num_action_ids(fractions)} action ids, but the policy network's "
            f"head is {FIRST_BET + MAX_SUPPORTED_BET_FRACTIONS + 1} wide; "
            f"raise MAX_ACTIONS in holdem/net/policy.py to go wider"
        )
    return tuple(sorted(fractions))


def perturbed_bet_fractions(
    rng,
    bet_fractions: Tuple[float, ...] = DEFAULT_BET_FRACTIONS,
    magnitude: float = BET_PERTURBATION,
) -> Tuple[float, ...]:
    """ReBeL's ``+/-0.1x pot`` jitter, one independent draw per size.

    **Drawn once per situation, not once per decision node.**  ``bet_fractions``
    rides on the frozen :class:`Betting` and therefore propagates to every node
    of the tree built from it, which is what makes the abstraction well defined:
    action id ``FIRST_BET + i`` has to mean the same size at a node and at its
    children, or a policy target recorded at one is unreadable at the other.
    Re-drawing per node would break that contract, and it is not what the paper
    describes either — the sizes are perturbed to diversify *training
    situations*, not to randomise within a single solve.

    Without this every situation in the buffer shares one action abstraction, so
    the network only ever sees pots split at exactly 0.25x, 0.5x, 1x and 2x and
    has no reason to learn what the neighbourhood of those sizes is worth.  That
    matters at test time precisely because the abstraction is coarse: action
    translation maps a real opponent's off-tree bet onto these sizes, so the
    values either side of them are the ones being interpolated.

    The result is re-validated, so an abstraction whose perturbation ranges
    overlap is rejected rather than silently reordered — action ids are
    positional, and swapping two sizes would relabel them.
    """
    jittered = tuple(
        max(float(f) + float(rng.uniform(-magnitude, magnitude)), MIN_BET_FRACTION)
        for f in bet_fractions
    )
    return validate_bet_fractions(jittered)


def preflop_betting(
    stack: int = 200,
    blinds: Tuple[int, int] = (SMALL_BLIND, BIG_BLIND),
    bet_fractions: Tuple[float, ...] = DEFAULT_BET_FRACTIONS,
    max_raises: int = 1,
) -> "Betting":
    """The betting state a hand actually starts in: blinds posted, nothing dealt.

    This is what makes ReBeL's "collect training data purely from self play"
    reachable.  A postflop-rooted situation has to *invent* the pot, the stacks
    and both ranges, and inventing them is the handcrafted PBS sampling the
    paper argues against; a hand that starts here invents nothing, because the
    initial belief state is common knowledge — uniform ranges over all 1,326
    combinations, and the blinds in the middle.

    The blinds are posted as *contributions* rather than as ``starting_pot``,
    which is what makes them behave correctly downstream without a special case:
    the small blind is owed one chip and may fold, the big blind is owed nothing
    and keeps the option to raise, and ``fold_returns`` already charges a folder
    exactly what they have put in.
    """
    return Betting(
        starting_pot=0,
        stack=stack,
        max_raises=max_raises,
        bet_fractions=bet_fractions,
        num_rounds=NUM_ROUNDS_FULL,
        blinds=blinds,
        contributions=blinds,
    )


def action_name(action: int, num_bets: int = len(DEFAULT_BET_FRACTIONS)) -> str:
    if action == FOLD:
        return "f"
    if action == CALL:
        return "c"
    if action == FIRST_BET + num_bets:
        return "a"
    return f"b{action - FIRST_BET}"


def history_str(actions) -> str:
    return "".join(str(a) for a in actions)


@dataclass(frozen=True)
class Betting:
    """The public betting state of a two-round no-limit endgame."""

    starting_pot: int = 20
    stack: int = 100
    max_raises: int = 1  # per round; 2 doubles the tree and the memory
    bet_fractions: Tuple[float, ...] = DEFAULT_BET_FRACTIONS
    # 2 = turn endgame (turn, river); 3 = flop endgame; 4 = a whole hand.
    num_rounds: int = NUM_ROUNDS
    betting_round: int = 0
    # Always four slots so a preflop-rooted hand has somewhere to record every
    # street; a two-round endgame simply never touches the last two.
    history: Tuple[Tuple[int, ...], ...] = ((), (), (), ())
    contributions: Tuple[int, int] = (0, 0)
    # Chips forced in before anyone acts, as (small, big).  Non-zero only for a
    # hand that starts preflop, and the flag ``first_actor`` reads to know that.
    blinds: Tuple[int, int] = (0, 0)
    folder: int = -1
    showdown: bool = False
    awaiting_board: bool = False

    # --- structure ---------------------------------------------------------
    @property
    def is_terminal(self) -> bool:
        return self.folder >= 0 or self.showdown

    @property
    def pot(self) -> int:
        return self.starting_pot + sum(self.contributions)

    @property
    def all_in(self) -> bool:
        return (
            self.contributions[0] == self.contributions[1] == self.stack
        )

    @property
    def all_in_action(self) -> int:
        return FIRST_BET + len(self.bet_fractions)

    def bet_action_ids(self) -> Tuple[int, ...]:
        return tuple(FIRST_BET + i for i in range(len(self.bet_fractions)))

    def fraction_of(self, action: int) -> float:
        return self.bet_fractions[action - FIRST_BET]

    def is_aggressive(self, action: int) -> bool:
        return action >= FIRST_BET

    @property
    def first_actor(self) -> int:
        """Who opens the current betting round.

        Heads-up, position flips after the deal: the small blind (player 0, on
        the button) acts first preflop and last on every street after it.  A
        postflop endgame with no blinds posted has no such asymmetry to
        reproduce — it begins mid-hand with the pot already built — so player 0
        simply opens, which is what every existing endgame assumed.
        """
        if self.blinds == (0, 0):
            return 0
        return 0 if self.betting_round == 0 else 1

    def to_move(self) -> int:
        return (self.first_actor + len(self.history[self.betting_round])) % 2

    def to_call(self, player: int) -> int:
        return self.contributions[1 - player] - self.contributions[player]

    def legal_actions(self) -> Tuple[int, ...]:
        me = self.to_move()
        owed = self.to_call(me)
        actions = [CALL]
        if owed > 0:
            actions.insert(0, FOLD)
        capped = (
            sum(self.is_aggressive(a) for a in self.history[self.betting_round])
            >= self.max_raises
        )
        room = self.stack - self.contributions[me] - owed
        if not capped and room > 0:
            all_in_total = self.stack
            sizes = []
            for action in self.bet_action_ids():
                total = self._raise_total(me, self.fraction_of(action))
                # Drop a size that is not a real raise, or that is just all-in
                # under another name; duplicate actions only inflate the tree.
                if total > self.contributions[1 - me] and total < all_in_total:
                    sizes.append((action, total))
            seen = set()
            for action, total in sizes:
                if total not in seen:
                    seen.add(total)
                    actions.append(action)
            actions.append(self.all_in_action)
        return tuple(actions)

    def _raise_total(self, player: int, fraction: float) -> int:
        owed = self.to_call(player)
        pot_after_call = self.pot + owed
        total = self.contributions[player] + owed + int(round(fraction * pot_after_call))
        return min(total, self.stack)

    # --- transitions -------------------------------------------------------
    def apply_amount(self, action: int, total: int) -> "Betting":
        """Advance as if the mover committed exactly ``total`` chips.

        The abstraction-free transition, and the thing that makes a
        translation-aware measurement possible.  ``apply`` derives the chips
        from the action id, which is right whenever both the actor and the
        state agree on what that id means.  They stop agreeing the moment a
        responder bets a size the abstraction does not contain: the *real* pot
        then follows the chips actually put in, while the agent's own state
        follows the size it translated the bet to, and the two diverge for the
        rest of the hand.

        ``action`` is still recorded in the history, because that is what the
        feature encoder and the round-closing rules read; only the chips come
        from ``total``.  Callers own the correctness of that pairing — this is
        deliberately a lower-level door than :meth:`apply`.
        """
        return self._advance(action, min(max(int(total), 0), self.stack))

    def apply(self, action: int) -> "Betting":
        me = self.to_move()
        if action == FOLD:
            history = _append(self.history, self.betting_round, action)
            return replace(self, history=history, folder=me)
        if action == CALL:
            total = min(self.contributions[1 - me], self.stack)
        elif action == self.all_in_action:
            total = self.stack
        elif action in self.bet_action_ids():
            total = self._raise_total(me, self.fraction_of(action))
        else:
            raise ValueError(f"illegal action: {action}")
        return self._advance(action, total)

    def _advance(self, action: int, total: int) -> "Betting":
        """Record ``action`` in the history with ``total`` chips behind it."""
        me = self.to_move()
        history = _append(self.history, self.betting_round, action)

        if action == FOLD:
            return replace(self, history=history, folder=me)

        contributions = list(self.contributions)
        contributions[me] = total
        contributions = (contributions[0], contributions[1])

        state = replace(self, history=history, contributions=contributions)
        matched = contributions[0] == contributions[1]
        closed = action == CALL and len(history[state.betting_round]) >= 2
        if not (closed or (matched and state.all_in and len(history[state.betting_round]) >= 2)):
            return state
        return state._close_round()

    def _close_round(self) -> "Betting":
        if self.betting_round + 1 < self.num_rounds:
            return replace(self, betting_round=self.betting_round + 1, awaiting_board=True)
        return replace(self, showdown=True)

    def deal_board(self) -> "Betting":
        state = replace(self, awaiting_board=False)
        if not self.all_in:
            return state
        # With nobody left to act, skip directly through any remaining board
        # deals.  A flop all-in still needs both the turn and river before the
        # hand can be evaluated.
        if self.betting_round + 1 < self.num_rounds:
            return replace(
                state, betting_round=self.betting_round + 1, awaiting_board=True
            )
        return replace(state, showdown=True)

    # --- payoffs -----------------------------------------------------------
    def fold_returns(self) -> Tuple[float, float]:
        """The folder loses what they put in; the pot itself is not at stake."""
        loss = float(self.contributions[self.folder]) + self.starting_pot / 2.0
        return (-loss, loss) if self.folder == 0 else (loss, -loss)

    def showdown_stake(self) -> float:
        return float(min(self.contributions)) + self.starting_pot / 2.0

    def raise_total(self, player: int, action: int) -> int:
        """Chips ``player`` would have in after taking ``action``."""
        if action == self.all_in_action:
            return self.stack
        return self._raise_total(player, self.fraction_of(action))

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return "/".join(history_str(h) for h in self.history) + f" pot={self.pot}"


def _append(history: Tuple[Tuple[int, ...], ...], rnd: int, action: int):
    rounds = list(history)
    rounds[rnd] = rounds[rnd] + (action,)
    return tuple(rounds)
