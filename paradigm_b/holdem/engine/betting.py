"""No-limit betting for a hold'em endgame.

Leduc's betting had one legal raise size; no-limit has a continuum, so the tree
only exists once an *action abstraction* is chosen.  This one keeps four sizes —
half pot, pot, all-in, plus check/call and fold — with a cap on raises per
round.  That is coarse next to a real solver, and coarseness here is a source of
exploitability no amount of search recovers: a real opponent can bet sizes the
abstraction cannot represent.  It is the honest cost of leaving Leduc behind,
and it is why stage 4 needs action translation before it means anything against
a human.

Chips are integers.  ``starting_pot`` is what is already in the middle when the
endgame begins; ``stack`` is what each player can still put in.
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
DEFAULT_BET_FRACTIONS = (0.5, 1.0)
BET_HALF = 2
BET_POT = 3
ALL_IN = 4

MAX_RAISES_PER_ROUND = 2
NUM_ROUNDS = 2


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
    # 2 = turn endgame (turn, river); 3 = flop endgame (flop, turn, river).
    num_rounds: int = NUM_ROUNDS
    betting_round: int = 0
    # Always three slots so a flop-rooted game has somewhere to record its
    # third round; a two-round endgame simply never touches the last one.
    history: Tuple[Tuple[int, ...], ...] = ((), (), ())
    contributions: Tuple[int, int] = (0, 0)
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

    def to_move(self) -> int:
        return len(self.history[self.betting_round]) % 2

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
    def apply(self, action: int) -> "Betting":
        me = self.to_move()
        history = _append(self.history, self.betting_round, action)

        if action == FOLD:
            return replace(self, history=history, folder=me)

        contributions = list(self.contributions)
        if action == CALL:
            contributions[me] = min(self.contributions[1 - me], self.stack)
        elif action == self.all_in_action:
            contributions[me] = self.stack
        elif action in self.bet_action_ids():
            contributions[me] = self._raise_total(me, self.fraction_of(action))
        else:
            raise ValueError(f"illegal action: {action}")
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
