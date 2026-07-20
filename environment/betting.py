"""Betting rules: legal-action generation, action resolution and pot splitting.

These are pure functions over :class:`GameState` so they can be unit-tested in
isolation from dealing and street progression.

Two rules deserve a note because they are frequently omitted:

* **Minimum raise.**  A raise must increase the outstanding bet by at least the
  size of the last full raise (or one big blind if there has not been one).
* **Reopening.**  An all-in that raises by *less* than a full raise does not
  reopen the betting.  A player who has already acted and is already in for the
  last full-raise level may then only call or fold.  This is tracked with
  ``last_full_raise_level``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .state import (
    ALL_IN,
    BET_ACTIONS,
    CALL,
    CHECK,
    FOLD,
    NUM_ACTIONS,
    RAISE_ACTIONS,
    ActionRecord,
    GameState,
)


@dataclass
class LegalActions:
    """Legality mask plus the concrete chip amount each action resolves to."""

    mask: np.ndarray                 # shape [NUM_ACTIONS], float32 in {0, 1}
    to_amounts: Dict[int, int]       # action id -> resulting total street bet

    def is_legal(self, action_id: int) -> bool:
        return bool(self.mask[action_id])

    def legal_ids(self) -> List[int]:
        return [int(i) for i in np.flatnonzero(self.mask)]

    def any_legal(self) -> bool:
        return bool(self.mask.any())


def legal_actions(state: GameState, seat: int, cfg) -> LegalActions:
    """Compute the legal-action mask for ``seat``.

    ``cfg`` is an :class:`~config.EnvConfig`.
    """
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    to_amounts: Dict[int, int] = {}

    player = state.players[seat]
    if player.folded or player.all_in or player.stack <= 0:
        return LegalActions(mask, to_amounts)

    to_call = state.to_call(seat)
    facing_bet = to_call > 0
    bet_exists = state.current_bet > 0
    max_to = player.street_bet + player.stack  # street bet level when all-in

    # An aggressive action is only meaningful if somebody can still respond.
    has_live_opponent = any(
        p.seat != seat and p.can_act for p in state.players
    )

    # Betting is reopened for a player who has not acted yet, or who is not
    # already in for the last full-raise level.
    can_reopen = (not player.has_acted_this_round) or (
        player.street_bet < state.last_full_raise_level
    )

    # --- fold / check / call ----------------------------------------------
    if facing_bet or cfg.allow_fold_when_check_available:
        mask[FOLD] = 1.0
        to_amounts[FOLD] = player.street_bet

    if not facing_bet:
        mask[CHECK] = 1.0
        to_amounts[CHECK] = player.street_bet

    if facing_bet:
        mask[CALL] = 1.0
        # Calling for less than the full amount when short-stacked is a call,
        # not an all-in raise; ALL_IN is masked out below to avoid a duplicate.
        to_amounts[CALL] = min(state.current_bet, max_to)

    # --- sizing actions ----------------------------------------------------
    # `bet_exists` (not `facing_bet`) selects the sizing family, so the big
    # blind's option preflop is correctly treated as a raise.
    if has_live_opponent:
        if bet_exists:
            sizing_ids = RAISE_ACTIONS
            min_legal_to = state.min_raise_to()
            candidates = [
                int(round(mult * state.current_bet)) for mult in cfg.raise_multipliers
            ]
            allowed = can_reopen and max_to > state.current_bet
        else:
            sizing_ids = BET_ACTIONS
            min_legal_to = player.street_bet + cfg.big_blind
            pot = state.pot
            candidates = [
                player.street_bet + int(round(frac * pot)) for frac in cfg.bet_fractions
            ]
            allowed = True

        if allowed:
            chosen: List[int] = []
            for action_id, raw_to in zip(sizing_ids, candidates):
                target = max(raw_to, min_legal_to)
                # Anything that reaches the stack is ALL_IN, which keeps the
                # abstract actions mutually exclusive.
                if target >= max_to:
                    continue
                if target in chosen:
                    continue
                chosen.append(target)
                mask[action_id] = 1.0
                to_amounts[action_id] = target

        # ALL_IN.  When facing a bet it is only distinct from CALL if it puts
        # in more than the call, and it is only permitted if raising is.
        if bet_exists:
            if max_to > state.current_bet and can_reopen:
                mask[ALL_IN] = 1.0
                to_amounts[ALL_IN] = max_to
        else:
            mask[ALL_IN] = 1.0
            to_amounts[ALL_IN] = max_to

    return LegalActions(mask, to_amounts)


def apply_action(state: GameState, seat: int, action_id: int, legal: LegalActions) -> ActionRecord:
    """Mutate ``state`` by applying ``action_id`` for ``seat``.

    The action must already have been validated against ``legal``.
    """
    if not legal.is_legal(action_id):
        raise ValueError(f"illegal action {action_id} for seat {seat}")

    player = state.players[seat]
    pot_before = state.pot

    if action_id == FOLD:
        player.folded = True
        added = 0
    else:
        target = legal.to_amounts[action_id]
        added = target - player.street_bet
        if added < 0:
            raise AssertionError("negative contribution")
        if added > player.stack:
            raise AssertionError("contribution exceeds stack")

        player.stack -= added
        player.street_bet += added
        player.contributed += added
        if player.stack == 0:
            player.all_in = True

        if player.street_bet > state.current_bet:
            increment = player.street_bet - state.current_bet
            min_full = max(state.min_raise_increment, state.big_blind)
            state.current_bet = player.street_bet
            if increment >= min_full:
                # A full raise: betting reopens for everyone else.
                state.min_raise_increment = increment
                state.last_full_raise_level = player.street_bet

    player.has_acted_this_round = True

    record = ActionRecord(
        seat=seat,
        action_id=action_id,
        amount=added,
        to_amount=player.street_bet,
        street=state.street,
        pot_before=pot_before,
        pot_after=state.pot,
    )
    state.history.append(record)
    return record


def player_needs_to_act(state: GameState, seat: int) -> bool:
    """Whether ``seat`` still owes an action in the current betting round."""
    player = state.players[seat]
    if player.folded or player.all_in or player.stack <= 0:
        return False
    return (not player.has_acted_this_round) or (player.street_bet < state.current_bet)


def betting_round_complete(state: GameState) -> bool:
    if len(state.active_players()) <= 1:
        return True
    return not any(player_needs_to_act(state, s) for s in range(state.num_players))


def next_to_act(state: GameState, start_from: int, inclusive: bool = False) -> Optional[int]:
    """First seat at or after ``start_from`` (clockwise) that owes an action."""
    n = state.num_players
    offset = 0 if inclusive else 1
    for i in range(n):
        seat = (start_from + offset + i) % n
        if player_needs_to_act(state, seat):
            return seat
    return None


# --- pot resolution -------------------------------------------------------
@dataclass
class Pot:
    """One (main or side) pot layer."""

    amount: int
    eligible: List[int]     # seats that may win it
    contributors: List[int]  # seats whose chips formed it


def compute_side_pots(contributions: Sequence[int], folded: Sequence[bool]) -> List[Pot]:
    """Split total contributions into main and side pots.

    Layers are cut at each distinct all-in level.  Folded players contribute
    chips but are never eligible to win them.
    """
    n = len(contributions)
    levels = sorted({c for c in contributions if c > 0})
    pots: List[Pot] = []
    previous = 0

    for level in levels:
        amount = 0
        contributors: List[int] = []
        for seat in range(n):
            share = min(contributions[seat], level) - min(contributions[seat], previous)
            if share > 0:
                amount += share
                contributors.append(seat)
        if amount > 0:
            eligible = [
                seat
                for seat in range(n)
                if not folded[seat] and contributions[seat] >= level
            ]
            pots.append(Pot(amount=amount, eligible=eligible, contributors=contributors))
        previous = level

    return pots


def distribute_pot(
    state: GameState,
    hand_ranks: Dict[int, tuple],
) -> List[int]:
    """Award every pot layer and return per-seat payouts (chips won).

    ``hand_ranks`` maps seat -> comparable hand rank and only needs to contain
    entries for players still in the hand.
    """
    n = state.num_players
    contributions = [p.contributed for p in state.players]
    folded = [p.folded for p in state.players]
    payouts = [0] * n

    for pot in compute_side_pots(contributions, folded):
        if not pot.eligible:
            # Only reachable when the sole remaining players folded above this
            # layer; the chips are uncalled and go back to whoever put them in.
            total_in_layer = sum(1 for _ in pot.contributors)
            if total_in_layer == 0:
                continue
            share, remainder = divmod(pot.amount, total_in_layer)
            for seat in pot.contributors:
                payouts[seat] += share
            if remainder:
                payouts[_first_clockwise(state, pot.contributors)] += remainder
            continue

        best = max(hand_ranks[seat] for seat in pot.eligible)
        winners = [seat for seat in pot.eligible if hand_ranks[seat] == best]
        share, remainder = divmod(pot.amount, len(winners))
        for seat in winners:
            payouts[seat] += share
        if remainder:
            # Odd chips go to the first winner clockwise from the button.
            payouts[_first_clockwise(state, winners)] += remainder

    return payouts


def _first_clockwise(state: GameState, seats: Sequence[int]) -> int:
    return min(seats, key=lambda s: (s - state.dealer - 1) % state.num_players)


def current_pot_split(state: GameState) -> Tuple[int, int]:
    """(main pot, total side pots) for the *current* contributions."""
    pots = compute_side_pots(
        [p.contributed for p in state.players], [p.folded for p in state.players]
    )
    if not pots:
        return 0, 0
    return pots[0].amount, sum(p.amount for p in pots[1:])
