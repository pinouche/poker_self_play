"""Perspective canonicalisation: raw game state -> acting player's view.

Every observation is expressed in *relative* seat indices:

    0 = SELF, 1 = OPPONENT_LEFT, 2 = OPPONENT_RIGHT

where OPPONENT_LEFT is the next seat clockwise from SELF.  The network never
receives an absolute seat identity, so a single set of weights plays all three
seats.  Absolute position still matters in poker, so it is supplied explicitly
as position *relative to the button* (dealer / SB / BB).

Nothing in this module reads another seat's hole cards, except the explicitly
opt-in ``reveal_at_showdown`` path.
"""

from __future__ import annotations

import random
from functools import lru_cache
from itertools import permutations
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from environment.betting import LegalActions, current_pot_split
from environment.cards import NUM_CARDS, NUM_RANKS, NUM_SUITS, Card
from environment.hand_evaluator import NUM_CATEGORIES, evaluate_hand
from environment.state import (
    MAX_BOARD_CARDS,
    NUM_ACTIONS,
    NUM_BETTING_STREETS,
    NUM_HOLE_CARDS,
    GameState,
    Street,
    action_space_for,
)

# --- relative seat indices -------------------------------------------------
REL_SELF = 0
REL_OPPONENT_LEFT = 1
REL_OPPONENT_RIGHT = 2
NUM_REL_PLAYERS = 3
REL_NAMES = ["SELF", "OPPONENT_LEFT", "OPPONENT_RIGHT"]

# --- block sizes (single source of truth for the encoder) ------------------
CARD_DIM = NUM_RANKS + NUM_SUITS + 1        # 18
PLAYER_FEATURE_DIM = 14
POT_FEATURE_DIM = 12
POSITION_FEATURE_DIM = 6
DERIVED_FEATURE_DIM = 38
EQUITY_FEATURE_DIM = 2
def history_event_dim(num_actions: int = NUM_ACTIONS) -> int:
    """Width of one action-history slot; grows with the action abstraction."""
    return NUM_REL_PLAYERS + num_actions + 2 + NUM_BETTING_STREETS + 1 + 1


#: Width for the default ten-action space (21).
HISTORY_EVENT_DIM = history_event_dim()

_EPS = 1e-6


def relative_index(seat: int, self_seat: int, num_players: int = NUM_REL_PLAYERS) -> int:
    """Relative index of ``seat`` seen from ``self_seat``."""
    return (seat - self_seat) % num_players


def seat_from_relative(rel: int, self_seat: int, num_players: int = NUM_REL_PLAYERS) -> int:
    return (self_seat + rel) % num_players


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / (float(denominator) + _EPS)


def encode_card(card: Optional[Card]) -> np.ndarray:
    """Rank one-hot, suit one-hot and a present mask."""
    vec = np.zeros(CARD_DIM, dtype=np.float32)
    if card is None:
        return vec
    vec[card.rank_index] = 1.0
    vec[NUM_RANKS + card.suit] = 1.0
    vec[NUM_RANKS + NUM_SUITS] = 1.0
    return vec


def encode_cards(cards: Sequence[Optional[Card]], slots: int) -> np.ndarray:
    out = np.zeros((slots, CARD_DIM), dtype=np.float32)
    for i in range(min(slots, len(cards))):
        out[i] = encode_card(cards[i])
    return out


def _street_one_hot(street: Street) -> np.ndarray:
    vec = np.zeros(NUM_BETTING_STREETS, dtype=np.float32)
    vec[min(int(street), NUM_BETTING_STREETS - 1)] = 1.0
    return vec


def build_observation(
    state: GameState,
    self_seat: int,
    legal: LegalActions,
    env_cfg,
    obs_cfg,
) -> Dict[str, np.ndarray]:
    """Build the canonical observation dictionary for ``self_seat``."""
    n = state.num_players
    bb = float(env_cfg.big_blind)
    pot = state.pot
    me = state.players[self_seat]
    eff = float(state.effective_stack(self_seat))

    # --- private cards (SELF only) ----------------------------------------
    hole = encode_cards(list(me.hole), NUM_HOLE_CARDS)

    # --- opponent card slots: always masked unless explicitly revealed -----
    opponent_cards = np.zeros(
        (NUM_REL_PLAYERS - 1, NUM_HOLE_CARDS, CARD_DIM), dtype=np.float32
    )
    if obs_cfg.include_opponent_card_slots and obs_cfg.reveal_at_showdown and state.went_to_showdown:
        for rel in (REL_OPPONENT_LEFT, REL_OPPONENT_RIGHT):
            seat = seat_from_relative(rel, self_seat, n)
            player = state.players[seat]
            if player.active:
                opponent_cards[rel - 1] = encode_cards(list(player.hole), NUM_HOLE_CARDS)

    # --- public board ------------------------------------------------------
    board = encode_cards(list(state.board), MAX_BOARD_CARDS)

    # --- street ------------------------------------------------------------
    street = _street_one_hot(state.street)

    # --- per-player features, ordered SELF / LEFT / RIGHT ------------------
    players = np.zeros((NUM_REL_PLAYERS, PLAYER_FEATURE_DIM), dtype=np.float32)
    for rel in range(NUM_REL_PLAYERS):
        seat = seat_from_relative(rel, self_seat, n)
        p = state.players[seat]
        position = state.position_of(seat)  # 0 = dealer, 1 = SB, 2 = BB
        players[rel] = np.array(
            [
                _safe_div(p.stack, bb),
                _safe_div(p.street_bet, bb),
                _safe_div(p.contributed, bb),
                _safe_div(p.stack, eff),
                _safe_div(p.contributed, max(pot, 1)),
                float(p.active),
                float(p.folded),
                float(p.all_in),
                float(p.has_acted_this_round),
                float(state.to_act == seat),
                float(position == 0),
                float(position == 1),
                float(position == 2),
                position / float(n),
            ],
            dtype=np.float32,
        )

    # --- pot features ------------------------------------------------------
    to_call = state.to_call(self_seat)
    main_pot, side_pot = current_pot_split(state)
    min_raise_size = state.min_raise_to() - state.current_bet
    pot_features = np.array(
        [
            _safe_div(pot, bb),
            _safe_div(main_pot, bb),
            _safe_div(side_pot, bb),
            float(side_pot > 0),
            _safe_div(to_call, bb),
            _safe_div(min_raise_size, bb),
            _safe_div(pot, eff),
            _safe_div(to_call, max(pot, 1)),
            _safe_div(to_call, max(me.stack, 1)),
            _safe_div(to_call, max(pot + to_call, 1)),  # pot odds
            _safe_div(eff, bb),
            _safe_div(me.stack, max(pot, 1)),           # stack-to-pot ratio
        ],
        dtype=np.float32,
    )

    # --- position features -------------------------------------------------
    my_position = state.position_of(self_seat)
    opponents_active = sum(1 for p in state.players if p.seat != self_seat and p.active)
    acted_this_street = any(rec.street == state.street for rec in state.history)
    players_after = sum(
        1
        for rel in (REL_OPPONENT_LEFT, REL_OPPONENT_RIGHT)
        if state.players[seat_from_relative(rel, self_seat, n)].can_act
    )
    position_features = np.array(
        [
            float(my_position == 0),
            float(my_position == 1),
            float(my_position == 2),
            len(state.active_players()) / float(n),
            float(not acted_this_street),
            players_after / float(n),
        ],
        dtype=np.float32,
    )

    obs: Dict[str, np.ndarray] = {
        "hole_cards": hole,
        "opponent_cards": opponent_cards,
        "board": board,
        "street": street,
        "players": players,
        "pot_features": pot_features,
        "position_features": position_features,
        "action_history": encode_action_history(state, self_seat, env_cfg, obs_cfg),
        "legal_action_mask": legal.mask.astype(np.float32).copy(),
    }

    if obs_cfg.use_derived_card_features:
        obs["derived_features"] = derived_card_features(list(me.hole), list(state.board))

    if obs_cfg.use_equity_feature:
        obs["equity_features"] = equity_features(state, self_seat, obs_cfg)

    # Metadata is *not* encoded into the tensor.  It carries only information
    # the acting player legitimately has (own cards, public board, public
    # betting state) so that rule-based agents and the inference API can read
    # it without re-deriving it from the feature vector.
    obs["meta"] = {
        "self_seat": self_seat,
        "street": int(state.street),
        "hole_cards": tuple(me.hole),
        "board": tuple(state.board),
        "pot": pot,
        "to_call": to_call,
        "big_blind": env_cfg.big_blind,
        "stack": me.stack,
        "position": my_position,
        "num_active_opponents": opponents_active,
        "legal_to_amounts": dict(legal.to_amounts),
        "relative_seats": {
            REL_NAMES[rel]: seat_from_relative(rel, self_seat, n)
            for rel in range(NUM_REL_PLAYERS)
        },
    }
    return obs


def encode_action_history(
    state: GameState, self_seat: int, env_cfg, obs_cfg
) -> np.ndarray:
    """Fixed-size chronological history, zero-padded, with a validity mask.

    Slot 0 holds the oldest retained action.  When a hand exceeds the window
    the oldest events are dropped.
    """
    length = obs_cfg.action_history_length
    bb = float(env_cfg.big_blind)
    num_actions = action_space_for(env_cfg).num_actions
    out = np.zeros((length, history_event_dim(num_actions)), dtype=np.float32)

    events = state.history[-length:]
    for i, rec in enumerate(events):
        rel = relative_index(rec.seat, self_seat, state.num_players)
        offset = 0
        out[i, offset + rel] = 1.0
        offset += NUM_REL_PLAYERS
        out[i, offset + rec.action_id] = 1.0
        offset += num_actions
        out[i, offset] = _safe_div(rec.amount, bb)
        out[i, offset + 1] = _safe_div(rec.amount, max(rec.pot_before, 1))
        offset += 2
        out[i, offset + min(int(rec.street), NUM_BETTING_STREETS - 1)] = 1.0
        offset += NUM_BETTING_STREETS
        out[i, offset] = _safe_div(rec.pot_after, bb)
        offset += 1
        out[i, offset] = 1.0  # history mask: this slot holds a real event
    return out


def derived_card_features(hole: Sequence[Card], board: Sequence[Card]) -> np.ndarray:
    """Texture features computed from public board + own hole cards only."""
    visible = list(hole) + list(board)

    rank_counts = np.zeros(NUM_RANKS, dtype=np.float32)
    suit_counts = np.zeros(NUM_SUITS, dtype=np.float32)
    for card in visible:
        rank_counts[card.rank_index] += 1.0
        suit_counts[card.suit] += 1.0

    board_rank_counts: Dict[int, int] = {}
    board_suit_counts = np.zeros(NUM_SUITS, dtype=np.float32)
    for card in board:
        board_rank_counts[card.rank] = board_rank_counts.get(card.rank, 0) + 1
        board_suit_counts[card.suit] += 1.0

    board_paired = float(any(c >= 2 for c in board_rank_counts.values()))
    board_trips = float(any(c >= 3 for c in board_rank_counts.values()))
    max_board_suit = float(board_suit_counts.max()) if len(board) else 0.0
    board_texture = np.array(
        [
            board_paired,
            board_trips,
            max_board_suit / 5.0,
            float(max_board_suit >= 3),
            float(max_board_suit >= 4),
            _max_consecutive_run(sorted({c.rank for c in board})) / 5.0,
        ],
        dtype=np.float32,
    )

    if len(hole) >= 2:
        r1, r2 = hole[0].rank, hole[1].rank
        hole_features = np.array(
            [
                float(r1 == r2),
                float(hole[0].suit == hole[1].suit),
                abs(r1 - r2) / 12.0,
                max(r1, r2) / 14.0,
                min(r1, r2) / 14.0,
            ],
            dtype=np.float32,
        )
    else:
        hole_features = np.zeros(5, dtype=np.float32)

    category = np.zeros(NUM_CATEGORIES + 1, dtype=np.float32)
    if len(visible) >= 5:
        rank = evaluate_hand(visible)
        category[rank[0]] = 1.0
        category[NUM_CATEGORIES] = 1.0  # "category is valid" mask

    return np.concatenate(
        [rank_counts / 4.0, suit_counts / 7.0, board_texture, hole_features, category]
    ).astype(np.float32)


def _max_consecutive_run(sorted_unique_ranks: Sequence[int]) -> int:
    if not sorted_unique_ranks:
        return 0
    best = run = 1
    for i in range(1, len(sorted_unique_ranks)):
        if sorted_unique_ranks[i] == sorted_unique_ranks[i - 1] + 1:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def equity_features(state: GameState, self_seat: int, obs_cfg) -> np.ndarray:
    """Monte-Carlo equity against random opponent ranges.

    Opponent holdings are sampled from the cards *not visible to the acting
    player*, i.e. the full deck minus own hole cards and the board.  The actual
    dealt opponent cards are never consulted, so this leaks nothing.
    """
    me = state.players[self_seat]
    opponents = sum(1 for p in state.players if p.seat != self_seat and p.active)
    if opponents == 0 or len(me.hole) < NUM_HOLE_CARDS:
        return np.zeros(EQUITY_FEATURE_DIM, dtype=np.float32)

    equity = estimate_equity(
        list(me.hole), list(state.board), opponents, obs_cfg.equity_samples
    )
    return np.array([equity, 1.0], dtype=np.float32)


_SUIT_PERMUTATIONS = list(permutations(range(NUM_SUITS)))


def canonical_equity_key(
    hole: Sequence[Card], board: Sequence[Card], num_opponents: int
) -> Tuple:
    """Suit-isomorphic key for an equity query.

    Equity is invariant under any relabelling of suits, so hands that differ
    only by suit share an answer.  Taking the lexicographically smallest image
    over all 24 suit permutations gives a canonical form: preflop this collapses
    1326 hole combinations to 169, and it merges every board with the same
    suit *pattern*.  The 24 relabellings cost far less than one rollout.
    """
    best = None
    for perm in _SUIT_PERMUTATIONS:
        candidate = (
            tuple(sorted((c.rank, perm[c.suit]) for c in hole)),
            tuple(sorted((c.rank, perm[c.suit]) for c in board)),
        )
        if best is None or candidate < best:
            best = candidate
    return (*best, num_opponents)


@lru_cache(maxsize=500_000)
def _equity_from_key(key: Tuple, samples: int) -> float:
    hole_key, board_key, num_opponents = key
    hole = [Card(rank=r, suit=s) for r, s in hole_key]
    board = [Card(rank=r, suit=s) for r, s in board_key]
    return _estimate_equity_uncached(hole, board, num_opponents, samples, random.Random(0))


def equity_cache_info():
    """Hit/miss statistics for the equity memo (diagnostics)."""
    return _equity_from_key.cache_info()


def clear_equity_cache() -> None:
    _equity_from_key.cache_clear()


def estimate_equity(
    hole: Sequence[Card],
    board: Sequence[Card],
    num_opponents: int,
    samples: int,
    rng: Optional[random.Random] = None,
) -> float:
    """Win probability (ties counted fractionally) by Monte-Carlo rollout.

    Memoised on a suit-isomorphic key.  Passing an explicit ``rng`` bypasses the
    cache, since the caller is then asking for a specific random draw.
    """
    if rng is None:
        return _equity_from_key(
            canonical_equity_key(hole, board, num_opponents), samples
        )
    return _estimate_equity_uncached(hole, board, num_opponents, samples, rng)


def _estimate_equity_uncached(
    hole: Sequence[Card],
    board: Sequence[Card],
    num_opponents: int,
    samples: int,
    rng: random.Random,
) -> float:
    known = set(hole) | set(board)
    deck = [Card.from_id(i) for i in range(NUM_CARDS) if Card.from_id(i) not in known]
    needed_board = MAX_BOARD_CARDS - len(board)
    draw = needed_board + 2 * num_opponents
    if draw > len(deck):
        return 0.0

    score = 0.0
    for _ in range(samples):
        sampled = rng.sample(deck, draw)
        full_board = list(board) + sampled[:needed_board]
        mine = evaluate_hand(list(hole) + full_board)
        best_other = None
        cursor = needed_board
        for _ in range(num_opponents):
            other = evaluate_hand(list(sampled[cursor : cursor + 2]) + full_board)
            cursor += 2
            if best_other is None or other > best_other:
                best_other = other
        if mine > best_other:
            score += 1.0
        elif mine == best_other:
            score += 0.5
    return score / float(samples)
