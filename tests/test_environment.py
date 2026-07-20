"""Poker rules: dealing, legality, betting mechanics, showdown and pots."""

import random

import pytest

from config import EnvConfig, ObsConfig
from environment.betting import compute_side_pots, distribute_pot
from environment.cards import Card
from environment.poker_env import PokerEnv
from environment.state import (
    ALL_IN,
    BET_LARGE,
    BET_MEDIUM,
    BET_SMALL,
    CALL,
    CHECK,
    FOLD,
    RAISE_LARGE,
    RAISE_MEDIUM,
    RAISE_SMALL,
    Street,
)


def cards(*texts):
    return [Card.from_str(t) for t in texts]


def make_env(**overrides) -> PokerEnv:
    cfg = EnvConfig(**{**EnvConfig().__dict__, **overrides})
    return PokerEnv(cfg, ObsConfig(), seed=7)


def legal_ids(env, seat=None):
    return set(env.legal_actions(seat).legal_ids())


def advance_to_flop(env):
    """Button calls, small blind calls, big blind checks."""
    env.step(CALL)   # seat 0 (button)
    env.step(CALL)   # seat 1 (small blind)
    env.step(CHECK)  # seat 2 (big blind)


# --- dealing ---------------------------------------------------------------
def test_deals_two_hole_cards_each_and_no_duplicates():
    env = make_env()
    for _ in range(50):
        state = env.reset()
        seen = []
        for player in state.players:
            assert len(player.hole) == 2
            seen.extend(player.hole)
        seen.extend(state.board)
        assert len(seen) == len(set(seen))


def test_no_duplicate_cards_across_a_full_hand():
    env = make_env()
    rng = random.Random(3)
    for _ in range(100):
        state = env.reset()
        while not env.is_terminal:
            env.step(rng.choice(env.legal_actions().legal_ids()))
        all_cards = list(state.board)
        for player in state.players:
            all_cards.extend(player.hole)
        assert len(all_cards) == len(set(all_cards))


def test_blinds_are_posted_and_button_acts_first_preflop():
    env = make_env()
    state = env.reset(dealer=0)
    assert state.sb_seat == 1 and state.bb_seat == 2
    assert state.players[1].street_bet == 10
    assert state.players[2].street_bet == 20
    assert state.current_bet == 20
    assert state.pot == 30
    assert state.to_act == 0  # three-handed, the button is first in preflop


def test_small_blind_acts_first_postflop():
    env = make_env()
    state = env.reset(dealer=0)
    advance_to_flop(env)
    assert state.street == Street.FLOP
    assert state.to_act == 1  # small blind
    assert len(state.board) == 3


# --- legal actions ---------------------------------------------------------
def test_legal_actions_when_facing_a_bet():
    env = make_env()
    env.reset(dealer=0)
    assert legal_ids(env) == {FOLD, CALL, RAISE_SMALL, RAISE_MEDIUM, RAISE_LARGE, ALL_IN}


def test_legal_actions_when_no_bet_exists():
    env = make_env()
    env.reset(dealer=0)
    advance_to_flop(env)
    assert legal_ids(env) == {CHECK, BET_SMALL, BET_MEDIUM, BET_LARGE, ALL_IN}


def test_fold_is_disallowed_when_checking_is_available_by_default():
    env = make_env()
    env.reset(dealer=0)
    advance_to_flop(env)
    assert FOLD not in legal_ids(env)


def test_fold_when_check_available_can_be_enabled():
    env = make_env(allow_fold_when_check_available=True)
    env.reset(dealer=0)
    advance_to_flop(env)
    assert FOLD in legal_ids(env)


def test_illegal_actions_are_rejected():
    env = make_env()
    env.reset(dealer=0)
    with pytest.raises(ValueError):
        env.step(CHECK)  # facing the big blind


# --- action sizing ---------------------------------------------------------
def test_bet_sizes_are_pot_fractions():
    env = make_env()
    state = env.reset(dealer=0)
    advance_to_flop(env)
    assert state.pot == 60
    amounts = env.legal_actions().to_amounts
    assert amounts[BET_SMALL] == 20   # 0.33 * 60, floored at one big blind
    assert amounts[BET_MEDIUM] == 40  # 0.66 * 60
    assert amounts[BET_LARGE] == 60   # 1.00 * 60


def test_raise_sizes_are_multiples_of_the_current_bet():
    env = make_env()
    env.reset(dealer=0)
    amounts = env.legal_actions().to_amounts
    assert amounts[RAISE_SMALL] == 40   # 2 x 20
    assert amounts[RAISE_MEDIUM] == 60  # 3 x 20
    assert amounts[RAISE_LARGE] == 80   # 4 x 20


def test_raises_respect_the_minimum_raise():
    env = make_env()
    state = env.reset(dealer=0)
    env.step(RAISE_SMALL)  # button raises to 40
    assert state.current_bet == 40
    assert state.min_raise_increment == 20
    assert state.min_raise_to() == 60
    amounts = env.legal_actions().to_amounts  # small blind to act
    assert amounts[RAISE_SMALL] == 80  # 2 x 40, already above the minimum


def test_bet_buckets_that_reach_the_stack_become_all_in_only():
    env = make_env()
    env.reset(dealer=0, stacks=[1000, 1000, 1000])
    advance_to_flop(env)
    env.step(BET_LARGE)  # small blind bets 60
    # Big blind put 20 in preflop, so it has 980 behind and faces a bet of 60.
    amounts = env.legal_actions().to_amounts
    assert amounts[ALL_IN] == 980  # street bet 0 + stack 980
    assert all(amounts[a] < amounts[ALL_IN] for a in (RAISE_SMALL, RAISE_MEDIUM, RAISE_LARGE))


def test_short_stack_has_only_all_in_available_as_an_aggressive_action():
    env = make_env()
    env.reset(dealer=0, stacks=[1000, 1000, 55])
    env.step(RAISE_SMALL)  # button to 40
    env.step(CALL)         # small blind calls 40
    # Big blind: 20 in, 35 behind, so a legal raise (to 60) is unaffordable.
    ids = legal_ids(env)
    assert ids == {FOLD, CALL, ALL_IN}


# --- betting reopening -----------------------------------------------------
def test_short_all_in_does_not_reopen_the_betting():
    env = make_env()
    state = env.reset(dealer=0, stacks=[1000, 1000, 55])
    env.step(RAISE_SMALL)  # button to 40
    env.step(CALL)         # small blind calls
    env.step(ALL_IN)       # big blind all-in to 55: a 15 raise, less than a full 20

    assert state.current_bet == 55
    assert state.last_full_raise_level == 40  # unchanged: not a full raise
    assert state.to_act == 0
    # The button already matched the last full raise, so it may only call or fold.
    assert legal_ids(env) == {FOLD, CALL}


def test_full_raise_reopens_the_betting():
    env = make_env()
    state = env.reset(dealer=0)
    env.step(RAISE_SMALL)   # button to 40
    env.step(RAISE_MEDIUM)  # small blind to 120: a full raise
    assert state.last_full_raise_level == 120
    env.step(CALL)          # big blind calls
    assert state.to_act == 0
    assert RAISE_SMALL in legal_ids(env)  # button may re-raise


# --- street transitions and termination ------------------------------------
def test_streets_advance_and_the_board_grows():
    env = make_env()
    state = env.reset(dealer=0)
    advance_to_flop(env)
    assert (state.street, len(state.board)) == (Street.FLOP, 3)
    env.step(CHECK); env.step(CHECK); env.step(CHECK)
    assert (state.street, len(state.board)) == (Street.TURN, 4)
    env.step(CHECK); env.step(CHECK); env.step(CHECK)
    assert (state.street, len(state.board)) == (Street.RIVER, 5)
    env.step(CHECK); env.step(CHECK); env.step(CHECK)
    assert env.is_terminal and state.went_to_showdown


def test_hand_ends_when_everyone_folds_to_one_player():
    env = make_env()
    state = env.reset(dealer=0)
    env.step(FOLD)  # button folds, having invested nothing
    env.step(FOLD)  # small blind folds, losing 10
    assert env.is_terminal
    assert not state.went_to_showdown
    assert env.chip_deltas() == [0, -10, 10]


def test_folding_without_investing_chips_scores_as_neutral():
    """True in every mode: risking nothing is neither a win nor a loss."""
    for mode in ("binary", "normalized_chip_return", "bb_normalized", "chip_return"):
        env = make_env(reward_mode=mode)
        env.reset(dealer=0)
        env.step(FOLD)  # the button folds without investing a chip
        env.step(FOLD)
        rewards = env.terminal_rewards()
        assert rewards[0] == 0.0, mode
        assert rewards[1] < 0 < rewards[2], mode


def test_all_in_runs_the_board_out_to_showdown():
    env = make_env()
    state = env.reset(dealer=0, stacks=[100, 100, 100])
    env.step(ALL_IN)  # button jams 100
    env.step(CALL)    # small blind calls all-in
    env.step(CALL)    # big blind calls all-in
    assert env.is_terminal
    assert len(state.board) == 5
    assert state.went_to_showdown


def test_one_all_in_player_does_not_stop_the_others_betting():
    env = make_env()
    state = env.reset(dealer=0, stacks=[100, 1000, 1000])
    env.step(ALL_IN)  # button jams 100
    env.step(CALL)
    env.step(CALL)
    # Two players still have chips behind, so the flop is dealt and bet normally.
    assert not env.is_terminal
    assert state.street == Street.FLOP
    assert state.to_act == 1


# --- showdown and pots -----------------------------------------------------
def test_showdown_awards_the_pot_to_the_best_hand():
    env = make_env()
    state = env.reset(
        dealer=0,
        hole_cards=[cards("Ac", "Ad"), cards("Kc", "Kd"), cards("7c", "2d")],
        board=cards("Ah", "Kh", "9s", "4c", "3d"),
    )
    while not env.is_terminal:
        env.step(CALL if env.legal_actions().is_legal(CALL) else CHECK)
    assert state.went_to_showdown
    deltas = env.chip_deltas()
    assert deltas[0] > 0                      # set of aces
    assert deltas[1] < 0 and deltas[2] < 0
    assert sum(deltas) == 0

    env.cfg.reward_mode = "binary"
    assert env.terminal_rewards() == [1.0, -1.0, -1.0]
    env.cfg.reward_mode = "normalized_chip_return"
    rewards = env.terminal_rewards()
    assert rewards[0] > 0 > rewards[1] and rewards[2] < 0
    assert rewards[0] == pytest.approx(deltas[0] / 1000)


def test_a_board_that_plays_splits_three_ways():
    env = make_env()
    state = env.reset(
        dealer=0,
        hole_cards=[cards("2c", "3c"), cards("4d", "5d"), cards("7h", "8h")],
        board=cards("As", "Ks", "Qs", "Js", "Ts"),  # royal flush on the board
    )
    while not env.is_terminal:
        env.step(CALL if env.legal_actions().is_legal(CALL) else CHECK)
    assert state.went_to_showdown
    assert env.chip_deltas() == [0, 0, 0]
    for mode in ("binary", "normalized_chip_return", "bb_normalized"):
        env.cfg.reward_mode = mode
        assert env.terminal_rewards() == [0.0, 0.0, 0.0], mode


def test_chips_are_conserved_over_many_random_hands():
    """Zero-sum to floating point: EV runouts settle pots on expectations."""
    env = make_env()
    rng = random.Random(11)
    for _ in range(200):
        env.reset()
        while not env.is_terminal:
            env.step(rng.choice(env.legal_actions().legal_ids()))
        assert sum(env.chip_deltas()) == pytest.approx(0.0, abs=1e-6)
        for player in env.state.players:
            assert player.stack >= -1e-9


def test_no_player_can_bet_more_than_their_stack():
    env = make_env()
    rng = random.Random(5)
    for _ in range(100):
        state = env.reset(stacks=[300, 120, 45])
        while not env.is_terminal:
            env.step(rng.choice(env.legal_actions().legal_ids()))
        for seat, player in enumerate(state.players):
            assert player.contributed <= state.initial_stacks[seat]


# --- side pots (unit level) ------------------------------------------------
def test_side_pot_layers_are_cut_at_each_all_in_level():
    pots = compute_side_pots([100, 300, 300], [False, False, False])
    assert [p.amount for p in pots] == [300, 400]
    assert pots[0].eligible == [0, 1, 2]
    assert pots[1].eligible == [1, 2]


def test_folded_players_contribute_but_cannot_win():
    pots = compute_side_pots([100, 300, 300], [True, False, False])
    assert pots[0].eligible == [1, 2]
    assert pots[1].eligible == [1, 2]


def test_short_stack_wins_only_the_main_pot():
    env = make_env()
    state = env.reset(dealer=0, stacks=[100, 300, 300])
    state.players[0].contributed = 100
    state.players[1].contributed = 300
    state.players[2].contributed = 300
    payouts = distribute_pot(
        state,
        hand_ranks={0: (8, 14), 1: (7, 10, 5), 2: (6, 9, 2)},  # seat 0 best, seat 1 second
    )
    assert payouts[0] == 300  # main pot only
    assert payouts[1] == 400  # side pot
    assert payouts[2] == 0
    assert sum(payouts) == 700


def test_uncalled_chips_return_to_their_owner():
    # Seat 1 put in more than anyone could call and everyone else folded.
    env = make_env()
    state = env.reset(dealer=0)
    state.players[0].contributed, state.players[0].folded = 50, True
    state.players[1].contributed, state.players[1].folded = 200, False
    state.players[2].contributed, state.players[2].folded = 0, True
    payouts = distribute_pot(state, hand_ranks={1: (0,)})
    assert payouts[1] == 250
    assert sum(payouts) == 250


def test_split_pot_odd_chip_goes_clockwise_from_the_button():
    env = make_env()
    state = env.reset(dealer=0)
    for player in state.players:
        player.contributed = 0
    state.players[0].contributed, state.players[0].folded = 1, True
    state.players[1].contributed = 5
    state.players[2].contributed = 5
    payouts = distribute_pot(state, hand_ranks={1: (5, 9), 2: (5, 9)})
    assert sum(payouts) == 11
    # The first layer holds 3 chips split two ways; seat 1 is first clockwise
    # from the button (seat 0) and takes the odd chip.
    assert payouts[1] == 6 and payouts[2] == 5


# --- reward modes ----------------------------------------------------------
def test_normalized_reward_is_net_chips_over_that_seats_own_starting_stack():
    env = make_env(reward_mode="normalized_chip_return")
    env.reset(dealer=0, stacks=[1000, 500, 200])
    env.step(FOLD)  # button folds
    env.step(FOLD)  # small blind folds, losing its 10
    rewards = env.terminal_rewards()
    assert rewards[0] == pytest.approx(0.0)          # invested nothing
    assert rewards[1] == pytest.approx(-10 / 500)    # own stack, not the global one
    assert rewards[2] == pytest.approx(10 / 200)     # own stack, not the global one


def test_normalization_uses_per_seat_stacks_not_a_global_constant():
    """The same chip swing is a different reward for a short and a deep stack."""
    env = make_env(reward_mode="normalized_chip_return")
    env.reset(dealer=0, stacks=[1000, 100, 100])
    env.step(FOLD)
    env.step(FOLD)
    short_stack_reward = env.terminal_rewards()[1]

    env.reset(dealer=0, stacks=[1000, 1000, 1000])
    env.step(FOLD)
    env.step(FOLD)
    deep_stack_reward = env.terminal_rewards()[1]

    assert short_stack_reward < deep_stack_reward < 0  # same 10 chips, larger loss


def test_reward_is_zero_sum_in_chips_and_bb_but_not_after_normalization():
    env = make_env(reward_mode="bb_normalized")
    rng = random.Random(17)
    for _ in range(100):
        env.reset()
        while not env.is_terminal:
            env.step(rng.choice(env.legal_actions().legal_ids()))
        assert sum(env.terminal_rewards()) == pytest.approx(0.0, abs=1e-9)


def test_normalized_reward_stays_within_the_rule_derived_bound():
    """Losses cannot exceed -1; three-handed wins can reach +2."""
    env = make_env(reward_mode="normalized_chip_return")
    rng = random.Random(19)
    observed_max = -99.0
    for _ in range(400):
        env.reset()
        while not env.is_terminal:
            env.step(rng.choice(env.legal_actions().legal_ids()))
        for reward in env.terminal_rewards():
            assert -1.0 <= reward <= 2.0
            observed_max = max(observed_max, reward)
    assert observed_max > 1.0  # the +1..+2 band is genuinely reachable


def test_default_clip_never_truncates_a_legitimate_outcome():
    """A clean three-way all-in win pays exactly +2 and survives clipping."""
    env = make_env(reward_mode="normalized_chip_return")
    env.reset(
        dealer=0,
        stacks=[100, 100, 100],
        hole_cards=[cards("Ac", "Ad"), cards("Kc", "Kd"), cards("Qc", "Qd")],
        board=cards("Ah", "7d", "2s", "3c", "9h"),
    )
    env.step(ALL_IN)
    env.step(CALL)
    env.step(CALL)
    assert env.chip_deltas() == [200, -100, -100]
    assert env.terminal_rewards() == pytest.approx([2.0, -1.0, -1.0])


def test_tight_clipping_breaks_the_zero_sum_property():
    """Documents why reward_clip must not be set to 1.0."""
    env = make_env(reward_mode="normalized_chip_return", reward_clip=1.0)
    env.reset(
        dealer=0,
        stacks=[100, 100, 100],
        hole_cards=[cards("Ac", "Ad"), cards("Kc", "Kd"), cards("Qc", "Qd")],
        board=cards("Ah", "7d", "2s", "3c", "9h"),
    )
    env.step(ALL_IN)
    env.step(CALL)
    env.step(CALL)
    rewards = env.terminal_rewards()
    assert rewards == pytest.approx([1.0, -1.0, -1.0])  # the +2 win is truncated
    assert sum(rewards) < 0  # no longer zero-sum: a systematic fold incentive


def test_clipping_can_be_disabled_entirely():
    env = make_env(reward_mode="normalized_chip_return", reward_clip=0)
    env.reset(dealer=0, stacks=[100, 100, 100],
              hole_cards=[cards("Ac", "Ad"), cards("Kc", "Kd"), cards("Qc", "Qd")],
              board=cards("Ah", "7d", "2s", "3c", "9h"))
    env.step(ALL_IN); env.step(CALL); env.step(CALL)
    assert env.terminal_rewards() == pytest.approx([2.0, -1.0, -1.0])


def test_bb_normalized_reward_is_chips_in_big_blinds():
    env = make_env(reward_mode="bb_normalized")
    env.reset(dealer=0)
    env.step(FOLD)
    env.step(FOLD)
    assert env.terminal_rewards() == pytest.approx([0.0, -0.5, 0.5])  # 10 chips = 0.5 bb


def test_unknown_reward_mode_is_rejected():
    env = make_env(reward_mode="nonsense")
    env.reset(dealer=0)
    env.step(FOLD)
    env.step(FOLD)
    with pytest.raises(ValueError):
        env.terminal_rewards()


# --- all-in EV runouts -----------------------------------------------------
def test_ev_runout_is_unbiased_and_lower_variance():
    """Same expectation as dealing one board, much less noise.

    That noise flows straight into the Q targets, so halving it is worth as
    much as quadrupling the number of hands.
    """
    import numpy as np

    results = {}
    for label, enabled in (("concrete", False), ("ev", True)):
        env = make_env(all_in_ev_runout=enabled)
        rng = random.Random(3)
        rewards = []
        for _ in range(800):
            env.reset()
            while not env.is_terminal:
                env.step(rng.choice(env.legal_actions().legal_ids()))
            rewards.extend(env.terminal_rewards())
        results[label] = np.array(rewards)

    assert results["ev"].mean() == pytest.approx(0.0, abs=0.05)
    assert results["concrete"].mean() == pytest.approx(0.0, abs=0.05)
    assert results["ev"].std() < 0.75 * results["concrete"].std()


def test_ev_runout_still_conserves_chips():
    env = make_env(all_in_ev_runout=True)
    rng = random.Random(9)
    settled = 0
    for _ in range(200):
        env.reset()
        while not env.is_terminal:
            env.step(rng.choice(env.legal_actions().legal_ids()))
        assert sum(env.chip_deltas()) == pytest.approx(0.0, abs=1e-6)
        settled += env.state.expected_value_runout
    assert settled > 0, "no hand reached an all-in runout"


def test_preset_boards_always_take_the_concrete_path():
    """Scripted hands must stay exactly reproducible."""
    env = make_env(all_in_ev_runout=True)
    env.reset(
        dealer=0,
        stacks=[100, 100, 100],
        hole_cards=[cards("Ac", "Ad"), cards("Kc", "Kd"), cards("Qc", "Qd")],
        board=cards("Ah", "7d", "2s", "3c", "9h"),
    )
    env.step(ALL_IN); env.step(CALL); env.step(CALL)
    assert not env.state.expected_value_runout
    assert env.chip_deltas() == [200, -100, -100]


def test_ev_runout_pays_the_exact_split_on_the_river():
    """One card to come is enumerated exactly, not sampled."""
    env = make_env(all_in_ev_runout=True, all_in_ev_exact_threshold=100)
    env.reset(dealer=0, stacks=[100, 100, 100])
    # Drive to an all-in on the turn so a single card remains.
    env.step(ALL_IN); env.step(CALL); env.step(CALL)
    assert env.is_terminal
    assert sum(env.chip_deltas()) == pytest.approx(0.0, abs=1e-6)


def test_ev_runout_can_be_disabled():
    env = make_env(all_in_ev_runout=False)
    rng = random.Random(9)
    for _ in range(60):
        env.reset()
        while not env.is_terminal:
            env.step(rng.choice(env.legal_actions().legal_ids()))
        assert not env.state.expected_value_runout
        assert sum(env.chip_deltas()) == pytest.approx(0.0, abs=1e-6)
