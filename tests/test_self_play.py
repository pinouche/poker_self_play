"""Self-play stability and, above all, per-seat perspective correctness."""

import random

import numpy as np
import pytest

from config import Config, ModelConfig
from environment.poker_env import PokerEnv
from evaluation.random_agent import CallingStationAgent, RandomAgent
from model.network import build_network
from representation.observation_encoder import ObservationEncoder
from training.self_play import NetworkAgent, SelfPlayWorker, play_hand


def small_config() -> Config:
    cfg = Config()
    cfg.model = ModelConfig(hidden_dim=32, num_residual_blocks=2, head_hidden=32)
    return cfg


def make_pieces(seed=0):
    cfg = small_config()
    network = build_network(cfg)
    encoder = ObservationEncoder(cfg.obs)
    env = PokerEnv(cfg.env, cfg.obs, seed=seed)
    agent = NetworkAgent(network, alpha=cfg.train.alpha, beta=cfg.train.beta, temperature=1.0)
    return cfg, env, encoder, [agent] * 3


# --- stability -------------------------------------------------------------
def test_one_hundred_hands_of_self_play_complete_without_errors():
    cfg, env, encoder, agents = make_pieces(seed=1)
    rng = random.Random(1)

    hands = 0
    for _ in range(100):
        result = play_hand(env, agents, encoder, rng, gamma=cfg.train.gamma, lam=cfg.train.lam)
        assert result.num_decisions > 0
        assert sum(result.chip_deltas) == 0
        assert len(result.transitions) == result.num_decisions
        hands += 1
    assert hands == 100


def test_self_play_worker_produces_transitions_and_stats():
    cfg = small_config()
    network = build_network(cfg)
    encoder = ObservationEncoder(cfg.obs)
    worker = SelfPlayWorker(cfg, network, encoder, seed=2)

    transitions, stats = worker.generate(25)
    assert stats["hands"] == 25
    assert stats["transitions"] == len(transitions)
    assert 0.0 <= stats["showdown_rate"] <= 1.0
    # Three players share every pot, so mean chips across seats must cancel.
    assert sum(stats["mean_chips_per_seat"]) == pytest.approx(0.0, abs=1e-6)


def test_mixed_agents_can_share_a_table():
    cfg, env, encoder, agents = make_pieces(seed=3)
    rng = random.Random(3)
    table = [agents[0], RandomAgent(), CallingStationAgent()]
    for _ in range(25):
        result = play_hand(env, table, encoder, rng, collect=False)
        assert sum(result.chip_deltas) == 0


# --- perspective correctness ----------------------------------------------
def test_every_transition_carries_the_acting_seat_reward():
    cfg, env, encoder, agents = make_pieces(seed=4)
    rng = random.Random(4)

    for _ in range(60):
        result = play_hand(env, agents, encoder, rng, gamma=cfg.train.gamma, lam=cfg.train.lam)
        for transition in result.transitions:
            seat = transition.player_perspective
            if transition.done:
                assert transition.reward == result.rewards[seat]
            else:
                assert transition.reward == 0.0


def test_targets_equal_each_seats_own_outcome_under_monte_carlo_settings():
    """With gamma = lam = 1 every target must be that seat's terminal reward.

    This is the test that catches training all three seats from hero's point of
    view: a per-seat mix-up shows up immediately as a mismatched target.
    """
    cfg, env, encoder, agents = make_pieces(seed=5)
    rng = random.Random(5)

    for _ in range(60):
        result = play_hand(env, agents, encoder, rng, gamma=1.0, lam=1.0)
        for transition in result.transitions:
            seat = transition.player_perspective
            assert transition.q_target == pytest.approx(result.rewards[seat])


def test_seat_rewards_are_not_all_identical():
    """Guards against a collapse to a single shared reward for the table."""
    cfg, env, encoder, agents = make_pieces(seed=6)
    rng = random.Random(6)
    seen_distinct = False
    for _ in range(40):
        result = play_hand(env, agents, encoder, rng, collect=False)
        if len(set(result.rewards)) > 1:
            seen_distinct = True
            break
    assert seen_distinct


def test_exactly_one_transition_per_seat_is_terminal():
    cfg, env, encoder, agents = make_pieces(seed=7)
    rng = random.Random(7)

    for _ in range(40):
        result = play_hand(env, agents, encoder, rng, gamma=cfg.train.gamma, lam=cfg.train.lam)
        for seat in range(3):
            owned = [t for t in result.transitions if t.player_perspective == seat]
            if not owned:
                continue  # a seat can be all-in from the blinds and never act
            assert sum(t.done for t in owned) == 1
            assert owned[-1].done


def test_next_observation_points_at_the_same_seats_next_decision():
    cfg, env, encoder, agents = make_pieces(seed=8)
    rng = random.Random(8)

    for _ in range(30):
        result = play_hand(env, agents, encoder, rng, gamma=cfg.train.gamma, lam=cfg.train.lam)
        for seat in range(3):
            owned = [t for t in result.transitions if t.player_perspective == seat]
            for i, transition in enumerate(owned[:-1]):
                np.testing.assert_array_equal(
                    transition.next_observation, owned[i + 1].observation
                )
            assert owned[-1].next_observation is None if owned else True


def test_chosen_actions_are_always_legal():
    cfg, env, encoder, agents = make_pieces(seed=9)
    rng = random.Random(9)

    for _ in range(60):
        result = play_hand(env, agents, encoder, rng, gamma=cfg.train.gamma, lam=cfg.train.lam)
        for transition in result.transitions:
            assert transition.legal_action_mask[transition.action] == 1.0
            assert transition.old_policy[transition.legal_action_mask == 0].sum() == 0.0
            assert transition.old_policy.sum() == pytest.approx(1.0, abs=1e-5)


def test_binary_rewards_sum_correctly_for_a_three_player_pot():
    """One winner means (+1, -1, -1); a seat that risked nothing scores 0."""
    cfg, env, encoder, agents = make_pieces(seed=10)
    rng = random.Random(10)

    for _ in range(80):
        result = play_hand(env, agents, encoder, rng, collect=False)
        assert set(result.rewards) <= {-1.0, 0.0, 1.0}
        winners = [r for r in result.rewards if r > 0]
        assert len(winners) <= 3  # split pots can produce several
        # Reward sign must agree with the chip outcome for every seat.
        for reward, chips in zip(result.rewards, result.chip_deltas):
            assert reward == float((chips > 0) - (chips < 0))
