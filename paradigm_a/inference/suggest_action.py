"""Public inference API: raw table state in, play suggestion out.

The raw format is the one in the specification::

    {"hero": {...}, "villain_left": {...}, "villain_right": {...},
     "board": [...], "pot": 60, "small_blind": 10, "big_blind": 20,
     "dealer": "hero", "street": "flop", "showdown": false}

Three things the raw format does not carry, and how they are handled:

* **Whose turn it is.**  Defaults to ``hero``; override with a ``"to_act"``
  key ("hero" / "villain_left" / "villain_right", or a seat index).
* **Action history.**  Absent, so the history block is zero-padded and its mask
  is all zeros.  The network sees a legal, well-formed observation, but a
  suggestion made from a bare snapshot is weaker than one made inside a live
  hand.  Pass ``"action_history"`` (a list of ``{"player", "action", "amount",
  "street"}`` entries) to fill it in.
* **How much of the pot each player already contributed on earlier streets.**
  Only the total is given, so the dead portion is split evenly between the
  players still in the hand.  This affects normalised features only, never
  legality.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from paradigm_a.config import Config, EnvConfig, reward_scale
from paradigm_a.environment.betting import legal_actions
from common.cards import Card, parse_cards
from paradigm_a.environment.poker_env import SEAT_FROM_NAME, SEAT_NAMES
from paradigm_a.environment.state import (
    BOARD_CARDS_BY_STREET,
    ActionRecord,
    GameState,
    PlayerState,
    STREET_FROM_NAME,
    action_space_for,
)
from paradigm_a.model.network import PokerNet, load_checkpoint
from paradigm_a.representation.action_encoder import (
    legal_action_names,
    masked_softmax,
    probs_to_dict,
    q_values_to_dict,
    sample_action,
)
from paradigm_a.representation.canonicalizer import build_observation
from paradigm_a.representation.observation_encoder import ObservationEncoder
from paradigm_a.training.policy_improvement import expected_value, improved_policy_np


class TableStateError(ValueError):
    """The supplied table state is malformed or internally inconsistent."""


# --- parsing ---------------------------------------------------------------
def parse_table_state(
    table_state: Dict,
    env_cfg: Optional[EnvConfig] = None,
    action_space=None,
) -> Tuple[GameState, int, EnvConfig]:
    """Convert the public table-state JSON into an internal ``GameState``."""
    env_cfg = env_cfg or EnvConfig()
    action_space = action_space or action_space_for(env_cfg)
    if not isinstance(table_state, dict):
        raise TableStateError("table state must be a JSON object")

    small_blind = int(table_state.get("small_blind", env_cfg.small_blind))
    big_blind = int(table_state.get("big_blind", env_cfg.big_blind))
    if big_blind <= 0:
        raise TableStateError("big_blind must be positive")

    street_name = str(table_state.get("street", "preflop")).lower()
    if street_name not in STREET_FROM_NAME:
        raise TableStateError(f"unknown street: {street_name}")
    street = STREET_FROM_NAME[street_name]

    board = parse_cards(table_state.get("board", []))
    expected_board = BOARD_CARDS_BY_STREET[street]
    if len(board) != expected_board:
        raise TableStateError(
            f"street '{street_name}' expects {expected_board} board cards, got {len(board)}"
        )

    players: List[PlayerState] = []
    for seat, name in enumerate(SEAT_NAMES):
        raw = table_state.get(name)
        if raw is None:
            raise TableStateError(f"missing player entry: {name}")
        cards = parse_cards(raw.get("cards", []))
        stack = int(raw.get("stack", 0))
        bet = int(raw.get("bet", 0))
        active = bool(raw.get("active", True))
        players.append(
            PlayerState(
                seat=seat,
                stack=stack,
                hole=tuple(cards),
                street_bet=bet,
                contributed=bet,
                folded=not active,
                all_in=(stack <= 0 and active),
                has_acted_this_round=False,
            )
        )

    _check_duplicate_cards(players, board)

    dealer_raw = table_state.get("dealer", "hero")
    dealer = _resolve_seat(dealer_raw, "dealer")

    acting_seat = _resolve_seat(table_state.get("to_act", "hero"), "to_act")
    if players[acting_seat].folded:
        raise TableStateError(f"seat to act ({SEAT_NAMES[acting_seat]}) is not active")
    if len(players[acting_seat].hole) != 2:
        raise TableStateError(
            f"acting player ({SEAT_NAMES[acting_seat]}) must have two hole cards"
        )

    # Spread the pot that predates this street evenly over the live players.
    pot_total = int(table_state.get("pot", sum(p.street_bet for p in players)))
    dead = pot_total - sum(p.street_bet for p in players)
    if dead < 0:
        raise TableStateError("pot is smaller than the sum of the current bets")
    live = [p for p in players if not p.folded] or players
    share, remainder = divmod(dead, len(live))
    for i, player in enumerate(live):
        player.contributed += share + (1 if i < remainder else 0)

    current_bet = max((p.street_bet for p in players), default=0)
    state = GameState(
        players=players,
        board=board,
        street=street,
        dealer=dealer,
        small_blind=small_blind,
        big_blind=big_blind,
        current_bet=current_bet,
        min_raise_increment=big_blind,
        # Unknown from a snapshot; assuming the standing bet was a full raise
        # keeps raising available to everyone, which is the permissive default.
        last_full_raise_level=current_bet,
        to_act=acting_seat,
        initial_stacks=[p.stack + p.contributed for p in players],
    )
    state.history = _parse_history(
        table_state.get("action_history", []), state, action_space
    )

    env_cfg = EnvConfig(**{**env_cfg.__dict__, "small_blind": small_blind, "big_blind": big_blind})
    return state, acting_seat, env_cfg


def _resolve_seat(value, field: str) -> int:
    if isinstance(value, int):
        if 0 <= value < len(SEAT_NAMES):
            return value
        raise TableStateError(f"{field} seat index out of range: {value}")
    key = str(value).lower()
    if key not in SEAT_FROM_NAME:
        raise TableStateError(f"unknown {field}: {value!r}")
    return SEAT_FROM_NAME[key]


def _check_duplicate_cards(players: Sequence[PlayerState], board: Sequence[Card]) -> None:
    seen: Dict[Card, str] = {}
    for player in players:
        for card in player.hole:
            if card in seen:
                raise TableStateError(f"duplicate card {card} ({seen[card]} and {SEAT_NAMES[player.seat]})")
            seen[card] = SEAT_NAMES[player.seat]
    for card in board:
        if card in seen:
            raise TableStateError(f"duplicate card {card} ({seen[card]} and board)")
        seen[card] = "board"


def _parse_history(
    raw_history: Sequence[Dict], state: GameState, action_space
) -> List[ActionRecord]:
    records: List[ActionRecord] = []
    name_to_id = action_space.name_to_id
    running_pot = 0
    for entry in raw_history:
        action_name = str(entry.get("action", "")).upper()
        if action_name not in name_to_id:
            raise TableStateError(f"unknown action in history: {action_name!r}")
        seat = _resolve_seat(entry.get("player", "hero"), "action_history.player")
        amount = int(entry.get("amount", 0))
        street_name = str(entry.get("street", "preflop")).lower()
        if street_name not in STREET_FROM_NAME:
            raise TableStateError(f"unknown street in history: {street_name!r}")
        pot_before = running_pot
        running_pot += amount
        records.append(
            ActionRecord(
                seat=seat,
                action_id=name_to_id[action_name],
                amount=amount,
                to_amount=amount,
                street=STREET_FROM_NAME[street_name],
                pot_before=pot_before,
                pot_after=running_pot,
            )
        )
    return records


# --- suggestion ------------------------------------------------------------
class SuggestionEngine:
    """Holds a loaded network so repeated calls do not re-read the checkpoint."""

    def __init__(
        self,
        network: PokerNet,
        cfg: Config,
        device: str = "cpu",
    ) -> None:
        self.network = network
        self.cfg = cfg
        self.device = device
        self.action_space = action_space_for(cfg.env)
        self.encoder = ObservationEncoder.from_config(cfg)
        self.network.to(device)
        self.network.eval()

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "SuggestionEngine":
        network, cfg, _ = load_checkpoint(path, device=device)
        return cls(network, cfg, device=device)

    @classmethod
    def untrained(cls, cfg: Optional[Config] = None, device: str = "cpu") -> "SuggestionEngine":
        """A randomly initialised engine, for wiring up the API before training."""
        from paradigm_a.model.network import build_network

        cfg = cfg or Config()
        return cls(build_network(cfg), cfg, device=device)

    def suggest(
        self,
        table_state: Dict,
        temperature: float = 0.0,
        mode: str = "argmax",
        rng=None,
    ) -> Dict:
        """Return a play suggestion for the acting player.

        ``mode`` is ``"argmax"`` (deterministic) or ``"sample"``.  ``temperature``
        sharpens (``< 1``) or flattens (``> 1``) the distribution that is
        reported and, in sample mode, drawn from; ``temperature = 0`` collapses
        it onto the best action.
        """
        state, acting_seat, env_cfg = parse_table_state(
            table_state, self.cfg.env, self.action_space
        )
        legal = legal_actions(state, acting_seat, env_cfg)
        if not legal.any_legal():
            raise TableStateError(
                f"no legal actions for {SEAT_NAMES[acting_seat]} in this state"
            )

        observation = build_observation(
            state, acting_seat, legal, env_cfg, self.cfg.obs
        )
        flat = self.encoder.encode_flat(observation)
        logits, q_values = self.network.infer(flat, device=self.device)

        mask = legal.mask
        reference = masked_softmax(logits, mask, temperature=1.0)
        improved = improved_policy_np(
            q_values,
            reference,
            mask,
            self.cfg.train.alpha,
            self.cfg.train.beta,
            temperature=1.0,
            q_scale=reward_scale(self.cfg.env),
        )

        # Reported distribution honours the requested temperature.
        if temperature == 1.0:
            probabilities = improved
        else:
            from paradigm_a.representation.action_encoder import apply_temperature

            probabilities = apply_temperature(improved, mask, temperature)

        if mode == "sample":
            import random as _random

            rng = rng or _random.Random()
            action = sample_action(probabilities, rng)
        elif mode == "argmax":
            action = int(np.argmax(np.where(mask > 0, probabilities, -np.inf)))
        else:
            raise ValueError(f"unknown mode: {mode!r} (expected 'argmax' or 'sample')")

        if not mask[action]:  # defensive; masking above makes this unreachable
            raise RuntimeError("selected an illegal action")

        to_amount = legal.to_amounts.get(action)
        chips_to_put_in = (
            None if to_amount is None else int(to_amount - state.players[acting_seat].street_bet)
        )

        names = self.action_space.names
        return {
            "action": names[action],
            "probabilities": probs_to_dict(probabilities, mask, names),
            "q_values": q_values_to_dict(q_values, mask, names),
            "legal_actions": legal_action_names(mask, names),
            # Extras beyond the required contract.
            "acting_player": SEAT_NAMES[acting_seat],
            "bet_to": to_amount,
            "chips_to_put_in": chips_to_put_in,
            "state_value": expected_value(probabilities, q_values, mask),
        }


def suggest_action(
    table_state: Dict,
    engine: Optional[SuggestionEngine] = None,
    checkpoint: Optional[str] = None,
    temperature: float = 0.0,
    mode: str = "argmax",
    device: str = "cpu",
) -> Dict:
    """One-shot convenience wrapper around :class:`SuggestionEngine`."""
    if engine is None:
        engine = (
            SuggestionEngine.from_checkpoint(checkpoint, device=device)
            if checkpoint
            else SuggestionEngine.untrained(device=device)
        )
    return engine.suggest(table_state, temperature=temperature, mode=mode)


def load_table_state(path: str) -> Dict:
    with open(path) as fh:
        return json.load(fh)
