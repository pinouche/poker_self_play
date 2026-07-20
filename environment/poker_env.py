"""Three-player No-Limit Texas Hold'em environment.

Seat layout (fixed, matching the public API naming):

    seat 0 = hero, seat 1 = villain_left, seat 2 = villain_right

Positions rotate with the button: SB = dealer + 1, BB = dealer + 2.  Three
handed, the button acts first preflop and the small blind acts first on every
later street.

The environment holds perfect information (it must, to evaluate showdowns) but
:meth:`PokerEnv.get_observation` never exposes another seat's hole cards.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

from config import EnvConfig, ObsConfig, reward_bound

from .betting import (
    LegalActions,
    apply_action,
    betting_round_complete,
    distribute_pot,
    legal_actions,
    next_to_act,
)
from .cards import Card, Deck
from .hand_evaluator import evaluate_hand
from .state import (
    BOARD_CARDS_BY_STREET,
    NUM_HOLE_CARDS,
    ActionRecord,
    GameState,
    PlayerState,
    Street,
)

SEAT_NAMES = ["hero", "villain_left", "villain_right"]
SEAT_FROM_NAME = {name: i for i, name in enumerate(SEAT_NAMES)}


class PokerEnv:
    """A single hand of 3-handed no-limit hold'em."""

    def __init__(
        self,
        env_cfg: Optional[EnvConfig] = None,
        obs_cfg: Optional[ObsConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.cfg = env_cfg or EnvConfig()
        self.obs_cfg = obs_cfg or ObsConfig()
        self.rng = random.Random(seed)
        self.deck = Deck(self.rng)
        self.state: Optional[GameState] = None
        self._hand_index = 0

    # --- lifecycle ---------------------------------------------------------
    def reset(
        self,
        dealer: Optional[int] = None,
        stacks: Optional[Sequence[int]] = None,
        hole_cards: Optional[Sequence[Sequence[Card]]] = None,
        board: Optional[Sequence[Card]] = None,
    ) -> GameState:
        """Start a new hand.

        ``hole_cards`` and ``board`` allow deterministic setups in tests; any
        cards supplied are removed from the deck before dealing the rest.
        """
        n = self.cfg.num_players
        if dealer is None:
            dealer = self._hand_index % n if self.cfg.rotate_dealer else 0
        if stacks is None:
            stacks = [self.cfg.starting_stack] * n

        preset: List[Card] = []
        if hole_cards:
            for hand in hole_cards:
                preset.extend(hand)
        if board:
            preset.extend(board)
        if len(set(preset)) != len(preset):
            raise ValueError("duplicate cards in preset")
        self.deck.reset(exclude=preset)

        players = [PlayerState(seat=i, stack=int(stacks[i])) for i in range(n)]
        state = GameState(
            players=players,
            dealer=dealer % n,
            small_blind=self.cfg.small_blind,
            big_blind=self.cfg.big_blind,
            initial_stacks=[int(s) for s in stacks],
        )
        self.state = state

        # Deal hole cards.
        for i in range(n):
            if hole_cards and i < len(hole_cards) and hole_cards[i]:
                players[i].hole = tuple(hole_cards[i])
            else:
                players[i].hole = tuple(self.deck.deal(NUM_HOLE_CARDS))

        self._preset_board = list(board) if board else []

        self._post_blinds()
        self._open_round(first=(state.bb_seat + 1) % n)
        self._hand_index += 1
        return state

    def _post_blinds(self) -> None:
        state = self.state
        assert state is not None
        for seat, amount in ((state.sb_seat, self.cfg.small_blind), (state.bb_seat, self.cfg.big_blind)):
            player = state.players[seat]
            posted = min(amount, player.stack)
            player.stack -= posted
            player.street_bet += posted
            player.contributed += posted
            if player.stack == 0:
                player.all_in = True
        state.current_bet = max(p.street_bet for p in state.players)
        state.min_raise_increment = self.cfg.big_blind
        # Preflop, the big blind is the standing full-raise level.
        state.last_full_raise_level = state.current_bet

    # --- round / street progression ---------------------------------------
    def _open_round(self, first: int) -> None:
        state = self.state
        assert state is not None
        if len(state.active_players()) <= 1:
            self._end_hand(showdown=False)
            return
        if len(state.actable_players()) <= 1 or betting_round_complete(state):
            self._close_round()
            return
        seat = next_to_act(state, first, inclusive=True)
        if seat is None:
            self._close_round()
            return
        state.to_act = seat

    def _close_round(self) -> None:
        state = self.state
        assert state is not None
        state.to_act = None

        if len(state.active_players()) <= 1:
            self._end_hand(showdown=False)
            return
        if len(state.actable_players()) <= 1:
            self._runout()
            return
        if state.street == Street.RIVER:
            self._end_hand(showdown=True)
            return
        self._deal_next_street()
        self._open_round(first=(state.dealer + 1) % state.num_players)

    def _deal_next_street(self) -> None:
        state = self.state
        assert state is not None
        next_street = Street(int(state.street) + 1)
        needed = BOARD_CARDS_BY_STREET[next_street] - len(state.board)
        for _ in range(needed):
            if self._preset_board:
                state.board.append(self._preset_board.pop(0))
            else:
                state.board.append(self.deck.deal_one())

        state.street = next_street
        state.current_bet = 0
        state.min_raise_increment = self.cfg.big_blind
        state.last_full_raise_level = 0
        for player in state.players:
            player.reset_for_street()

    def _runout(self) -> None:
        """No further betting is possible: deal the rest of the board."""
        state = self.state
        assert state is not None
        while state.street != Street.RIVER:
            self._deal_next_street()
        self._end_hand(showdown=True)

    def _end_hand(self, showdown: bool) -> None:
        state = self.state
        assert state is not None
        state.to_act = None

        active = state.active_players()
        hand_ranks: Dict[int, tuple] = {}
        if len(active) >= 2:
            for player in active:
                hand_ranks[player.seat] = evaluate_hand(list(player.hole) + state.board)
            state.went_to_showdown = True
        else:
            # Everyone else folded; the single survivor wins without showing.
            for player in active:
                hand_ranks[player.seat] = (0,)
            state.went_to_showdown = False

        payouts = distribute_pot(state, hand_ranks)
        for seat, amount in enumerate(payouts):
            state.players[seat].stack += amount
        state.payouts = payouts
        state.hand_over = True

    # --- interaction -------------------------------------------------------
    @property
    def to_act(self) -> Optional[int]:
        return None if self.state is None else self.state.to_act

    @property
    def is_terminal(self) -> bool:
        return self.state is None or self.state.hand_over

    def legal_actions(self, seat: Optional[int] = None) -> LegalActions:
        state = self.state
        assert state is not None, "call reset() first"
        if seat is None:
            seat = state.to_act
        if seat is None:
            return LegalActions(mask=_zeros_mask(), to_amounts={})
        return legal_actions(state, seat, self.cfg)

    def step(self, action_id: int) -> ActionRecord:
        """Apply ``action_id`` for the seat to act and advance the hand."""
        state = self.state
        assert state is not None, "call reset() first"
        if state.hand_over:
            raise RuntimeError("hand is already over")
        seat = state.to_act
        if seat is None:
            raise RuntimeError("no player to act")

        legal = legal_actions(state, seat, self.cfg)
        if not legal.is_legal(action_id):
            raise ValueError(
                f"illegal action {action_id} for seat {seat}; legal={legal.legal_ids()}"
            )
        record = apply_action(state, seat, action_id, legal)

        if len(state.active_players()) <= 1:
            self._end_hand(showdown=False)
        elif betting_round_complete(state):
            self._close_round()
        else:
            state.to_act = next_to_act(state, seat, inclusive=False)
            if state.to_act is None:  # defensive: should be unreachable
                self._close_round()
        return record

    # --- observation -------------------------------------------------------
    def get_observation(self, acting_player_id: int) -> dict:
        """Canonical observation from ``acting_player_id``'s point of view.

        The legal-action mask is all-zero when the seat is not the one to act.
        """
        from representation.canonicalizer import build_observation

        state = self.state
        assert state is not None, "call reset() first"
        legal = (
            self.legal_actions(acting_player_id)
            if state.to_act == acting_player_id
            else LegalActions(mask=_zeros_mask(), to_amounts={})
        )
        return build_observation(state, acting_player_id, legal, self.cfg, self.obs_cfg)

    # --- results -----------------------------------------------------------
    def chip_deltas(self) -> List[int]:
        state = self.state
        assert state is not None
        return [p.stack - state.initial_stacks[i] for i, p in enumerate(state.players)]

    def terminal_rewards(self) -> List[float]:
        """Per-seat terminal reward under the configured reward mode.

        Rewards are terminal-only: no shaping for winning pots, betting,
        surviving streets or reaching showdown.  Those invite the agent to farm
        the reward function instead of learning profitable play.

        * ``normalized_chip_return`` (default) -- net chips as a fraction of
          *that seat's own stack at the start of the hand*, optionally clipped.
          Money is the reward, and it is scaled to the risk the seat actually
          took.
        * ``bb_normalized`` -- net chips in big blinds.  The conventional poker
          unit, but unbounded; pair it with a linear Q head.
        * ``chip_return`` -- raw chips.  Unbounded.
        * ``binary`` -- ``sign(net chips)``.  Retained for comparison; it
          maximises win rate rather than chip EV.  See the README.
        """
        if not self.is_terminal:
            raise RuntimeError("hand is not over")
        deltas = self.chip_deltas()
        mode = self.cfg.reward_mode

        if mode == "normalized_chip_return":
            return [
                self._clip(d / max(float(self.state.initial_stacks[seat]), 1.0))
                for seat, d in enumerate(deltas)
            ]
        if mode == "bb_normalized":
            return [float(d) / max(float(self.cfg.big_blind), 1.0) for d in deltas]
        if mode == "chip_return":
            return [float(d) for d in deltas]
        if mode == "binary":
            return [float((d > 0) - (d < 0)) for d in deltas]
        raise ValueError(f"unknown reward_mode: {mode}")

    def _clip(self, reward: float) -> float:
        """Apply the symmetric reward clip.

        The default limit is the tightest bound the rules allow
        (``num_players - 1``), so it acts purely as a guard against rule or
        side-pot bugs and never truncates a legitimate result.  Clipping tighter
        than that is not free: it breaks the zero-sum property of the reward and
        biases the agent toward folding.
        """
        limit = reward_bound(self.cfg)
        if limit is None:
            return float(reward)
        return float(min(max(reward, -limit), limit))

    def showdown_summary(self) -> dict:
        """Human-readable result of the finished hand (debugging/eval only)."""
        state = self.state
        assert state is not None and state.hand_over
        return {
            "board": [str(c) for c in state.board],
            "went_to_showdown": state.went_to_showdown,
            "payouts": list(state.payouts),
            "chip_deltas": self.chip_deltas(),
            "hands": {
                SEAT_NAMES[p.seat]: [str(c) for c in p.hole] for p in state.players
            },
            "folded": [p.folded for p in state.players],
        }


def _zeros_mask():
    import numpy as np

    from .state import NUM_ACTIONS

    return np.zeros(NUM_ACTIONS, dtype=np.float32)
