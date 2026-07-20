"""Imperfect information: the observation must not depend on hidden state.

The decisive tests here do not inspect feature indices.  They mutate the hidden
part of the world -- the opponents' hole cards and the undealt deck -- and
require the encoded observation to come out bit-for-bit identical.  Any leak,
however indirect, breaks that equality.
"""

import random

import numpy as np

from config import EnvConfig, ObsConfig
from environment.cards import NUM_CARDS, Card
from environment.poker_env import PokerEnv
from representation.observation_encoder import ObservationEncoder


def make_env(seed=21):
    return PokerEnv(EnvConfig(), ObsConfig(), seed=seed)


def unused_cards(state, count):
    """Cards held by nobody and not on the board."""
    used = set(state.board)
    for player in state.players:
        used.update(player.hole)
    free = [Card.from_id(i) for i in range(NUM_CARDS) if Card.from_id(i) not in used]
    return free[:count]


def test_opponent_card_slots_are_zero_during_normal_play():
    env = make_env()
    rng = random.Random(2)
    for _ in range(30):
        env.reset()
        while not env.is_terminal:
            seat = env.to_act
            observation = env.get_observation(seat)
            assert np.count_nonzero(observation["opponent_cards"]) == 0
            env.step(rng.choice(env.legal_actions().legal_ids()))


def test_changing_opponent_hole_cards_does_not_change_the_observation():
    env = make_env()
    encoder = ObservationEncoder(ObsConfig())
    rng = random.Random(4)

    for _ in range(30):
        state = env.reset()
        while not env.is_terminal:
            seat = env.to_act
            before = encoder.encode_flat(env.get_observation(seat))

            # Swap both opponents onto completely different holdings.
            opponents = [p for p in state.players if p.seat != seat]
            replacements = unused_cards(state, 4)
            original = [p.hole for p in opponents]
            opponents[0].hole = tuple(replacements[:2])
            opponents[1].hole = tuple(replacements[2:])

            after = encoder.encode_flat(env.get_observation(seat))
            np.testing.assert_array_equal(before, after)

            for player, hole in zip(opponents, original):
                player.hole = hole
            env.step(rng.choice(env.legal_actions().legal_ids()))


def test_changing_the_undealt_deck_does_not_change_the_observation():
    env = make_env()
    encoder = ObservationEncoder(ObsConfig())
    rng = random.Random(6)

    for _ in range(30):
        env.reset()
        while not env.is_terminal:
            seat = env.to_act
            before = encoder.encode_flat(env.get_observation(seat))

            # Reorder the remaining deck: the future board is now different.
            env.deck._cards.reverse()
            after = encoder.encode_flat(env.get_observation(seat))
            np.testing.assert_array_equal(before, after)

            env.step(rng.choice(env.legal_actions().legal_ids()))


def test_board_slots_beyond_the_current_street_are_masked():
    env = make_env()
    env.reset(dealer=0)
    observation = env.get_observation(0)
    board = observation["board"]
    assert np.count_nonzero(board) == 0  # preflop: no board cards at all

    from environment.state import CALL, CHECK

    env.step(CALL)
    env.step(CALL)
    env.step(CHECK)  # -> flop

    board = env.get_observation(1)["board"]
    for slot in range(3):
        assert board[slot][17] == 1.0  # present mask set
    for slot in (3, 4):
        assert np.count_nonzero(board[slot]) == 0  # turn and river still unknown


def test_observation_never_encodes_the_terminal_winner():
    """A losing and a winning continuation from the same decision look identical."""
    env = make_env()
    encoder = ObservationEncoder(ObsConfig())

    from environment.cards import Card as C

    def observe_with_opponent_cards(left, right):
        env.reset(
            dealer=0,
            hole_cards=[
                [C.from_str("Ac"), C.from_str("Ad")],
                left,
                right,
            ],
            board=[C.from_str("2h"), C.from_str("7d"), C.from_str("9s"),
                   C.from_str("Jc"), C.from_str("4h")],
        )
        return encoder.encode_flat(env.get_observation(0))

    # Hero holds aces either way; the opponents hold a losing or a winning hand.
    losing = observe_with_opponent_cards(
        [C.from_str("3c"), C.from_str("5d")], [C.from_str("6c"), C.from_str("8d")]
    )
    winning = observe_with_opponent_cards(
        [C.from_str("2c"), C.from_str("2d")], [C.from_str("9c"), C.from_str("9d")]
    )
    np.testing.assert_array_equal(losing, winning)


def test_derived_features_use_only_own_cards_and_the_board():
    """Derived features are the most likely place for an accidental leak."""
    cfg = ObsConfig(use_derived_card_features=True)
    env = PokerEnv(EnvConfig(), cfg, seed=31)
    encoder = ObservationEncoder(cfg)
    state = env.reset(dealer=0)

    before = encoder.encode_flat(env.get_observation(0))
    derived_before = encoder.field(before, "derived_features").copy()

    replacements = unused_cards(state, 4)
    state.players[1].hole = tuple(replacements[:2])
    state.players[2].hole = tuple(replacements[2:])

    after = encoder.encode_flat(env.get_observation(0))
    np.testing.assert_array_equal(derived_before, encoder.field(after, "derived_features"))


def test_equity_feature_does_not_consult_opponent_cards():
    cfg = ObsConfig(use_equity_feature=True, equity_samples=40)
    env = PokerEnv(EnvConfig(), cfg, seed=33)
    encoder = ObservationEncoder(cfg)
    state = env.reset(dealer=0)

    first = encoder.field(encoder.encode_flat(env.get_observation(0)), "equity_features").copy()
    replacements = unused_cards(state, 4)
    state.players[1].hole = tuple(replacements[:2])
    state.players[2].hole = tuple(replacements[2:])
    second = encoder.field(encoder.encode_flat(env.get_observation(0)), "equity_features")

    np.testing.assert_array_equal(first, second)
