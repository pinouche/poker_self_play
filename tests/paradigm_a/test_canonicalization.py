"""Perspective canonicalisation.

The guarantee under test: the observation is a function of (physical state,
observer), and changing the observer rotates every player-indexed block by the
same amount.
"""

import numpy as np
import pytest

from paradigm_a.config import EnvConfig, ObsConfig
from paradigm_a.environment.poker_env import PokerEnv
from paradigm_a.environment.state import RAISE_SMALL
from paradigm_a.representation.canonicalizer import (
    REL_OPPONENT_LEFT,
    REL_OPPONENT_RIGHT,
    REL_SELF,
    relative_index,
    seat_from_relative,
)
from paradigm_a.representation.observation_encoder import ObservationEncoder

# Column 3 of the player block is stack / effective_stack.  The effective stack
# is measured from the observer's seat, so this one column is observer-relative
# by design; every other column is a pure permutation.
OBSERVER_RELATIVE_COLUMNS = {3}


def make_env():
    return PokerEnv(EnvConfig(), ObsConfig(), seed=13)


# --- index helpers ---------------------------------------------------------
def test_relative_indices_rotate_as_specified():
    # Hero acting: SELF=hero, LEFT=villain_left, RIGHT=villain_right
    assert seat_from_relative(REL_SELF, 0) == 0
    assert seat_from_relative(REL_OPPONENT_LEFT, 0) == 1
    assert seat_from_relative(REL_OPPONENT_RIGHT, 0) == 2

    # Villain left acting: SELF=villain_left, LEFT=villain_right, RIGHT=hero
    assert seat_from_relative(REL_SELF, 1) == 1
    assert seat_from_relative(REL_OPPONENT_LEFT, 1) == 2
    assert seat_from_relative(REL_OPPONENT_RIGHT, 1) == 0

    # Villain right acting: SELF=villain_right, LEFT=hero, RIGHT=villain_left
    assert seat_from_relative(REL_SELF, 2) == 2
    assert seat_from_relative(REL_OPPONENT_LEFT, 2) == 0
    assert seat_from_relative(REL_OPPONENT_RIGHT, 2) == 1


def test_relative_index_is_the_inverse_of_seat_from_relative():
    for self_seat in range(3):
        for rel in range(3):
            assert relative_index(seat_from_relative(rel, self_seat), self_seat) == rel


# --- player block rotation -------------------------------------------------
def test_player_blocks_rotate_with_the_observer():
    env = make_env()
    env.reset(dealer=0)
    env.step(RAISE_SMALL)  # create some asymmetry between the seats

    observations = {seat: env.get_observation(seat) for seat in range(3)}

    for observer in range(3):
        for other in range(3):
            for seat in range(3):
                row_a = observations[observer]["players"][relative_index(seat, observer)]
                row_b = observations[other]["players"][relative_index(seat, other)]
                for column in range(len(row_a)):
                    if column in OBSERVER_RELATIVE_COLUMNS:
                        continue
                    assert row_a[column] == pytest.approx(row_b[column]), (
                        f"seat {seat} column {column} differs between observers "
                        f"{observer} and {other}"
                    )


def test_self_row_describes_the_observer():
    env = make_env()
    state = env.reset(dealer=0)
    env.step(RAISE_SMALL)

    big_blind = env.cfg.big_blind
    for seat in range(3):
        observation = env.get_observation(seat)
        self_row = observation["players"][REL_SELF]
        player = state.players[seat]
        assert self_row[0] == pytest.approx(player.stack / big_blind, abs=1e-3)
        assert self_row[1] == pytest.approx(player.street_bet / big_blind, abs=1e-3)
        assert self_row[2] == pytest.approx(player.contributed / big_blind, abs=1e-3)


def test_hole_cards_belong_to_the_observer():
    env = make_env()
    state = env.reset(dealer=0)
    encoder = ObservationEncoder(ObsConfig())

    for seat in range(3):
        observation = env.get_observation(seat)
        hole = observation["hole_cards"]
        for i, card in enumerate(state.players[seat].hole):
            assert hole[i][card.rank_index] == 1.0
            assert hole[i][13 + card.suit] == 1.0
            assert hole[i][17] == 1.0  # present mask
        # Different seats must produce different card blocks (with overwhelming
        # probability); at minimum the observation as a whole must differ.
        assert encoder.encode_flat(observation).shape == (encoder.observation_dim,)


def test_observations_differ_between_seats():
    env = make_env()
    env.reset(dealer=0)
    encoder = ObservationEncoder(ObsConfig())
    flats = [encoder.encode_flat(env.get_observation(seat)) for seat in range(3)]
    assert not np.array_equal(flats[0], flats[1])
    assert not np.array_equal(flats[1], flats[2])


# --- position and history --------------------------------------------------
def test_position_relative_to_the_button_is_encoded_per_observer():
    env = make_env()
    env.reset(dealer=1)  # seat 1 is the button, seat 2 the SB, seat 0 the BB
    expected = {1: 0, 2: 1, 0: 2}  # seat -> distance from the button
    for seat, distance in expected.items():
        position = env.get_observation(seat)["position_features"][:3]
        assert position[distance] == 1.0
        assert position.sum() == 1.0


def test_action_history_players_are_recorded_relative_to_the_observer():
    env = make_env()
    env.reset(dealer=0)
    env.step(RAISE_SMALL)  # seat 0 acts first

    for observer in range(3):
        event = env.get_observation(observer)["action_history"][0]
        expected_rel = relative_index(0, observer)  # the actor was seat 0
        assert event[expected_rel] == 1.0
        assert event[:3].sum() == 1.0
        assert event[-1] == 1.0  # history mask marks a real event


def test_history_padding_is_masked_out():
    env = make_env()
    env.reset(dealer=0)
    history = env.get_observation(0)["action_history"]
    assert history[0][-1] == 0.0  # no actions yet: every slot is padding
    assert np.count_nonzero(history) == 0


def test_the_same_network_input_is_produced_for_every_seat_shape_wise():
    env = make_env()
    env.reset(dealer=2)
    encoder = ObservationEncoder(ObsConfig())
    shapes = {encoder.encode_flat(env.get_observation(seat)).shape for seat in range(3)}
    assert len(shapes) == 1
