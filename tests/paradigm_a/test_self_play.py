"""Self-play stability and, above all, per-seat perspective correctness."""

import random

import numpy as np
import pytest

from paradigm_a.config import Config, ModelConfig
from paradigm_a.environment.poker_env import PokerEnv
from paradigm_a.evaluation.random_agent import CallingStationAgent, RandomAgent
from paradigm_a.model.network import build_network
from paradigm_a.representation.observation_encoder import ObservationEncoder
from paradigm_a.training.self_play import NetworkAgent, SelfPlayWorker, play_hand


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
        assert sum(result.chip_deltas) == pytest.approx(0.0, abs=1e-6)
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
        assert sum(result.chip_deltas) == pytest.approx(0.0, abs=1e-6)


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
    env.cfg.reward_mode = "binary"
    rng = random.Random(10)

    for _ in range(80):
        result = play_hand(env, agents, encoder, rng, collect=False)
        assert set(result.rewards) <= {-1.0, 0.0, 1.0}
        winners = [r for r in result.rewards if r > 0]
        assert len(winners) <= 3  # split pots can produce several
        # Reward sign must agree with the chip outcome for every seat.
        for reward, chips in zip(result.rewards, result.chip_deltas):
            assert reward == float((chips > 0) - (chips < 0))


def test_default_normalized_rewards_track_chips_and_stay_in_range():
    cfg, env, encoder, agents = make_pieces(seed=11)
    assert env.cfg.reward_mode == "normalized_chip_return"
    rng = random.Random(11)

    for _ in range(60):
        result = play_hand(env, agents, encoder, rng, collect=False)
        for seat, (reward, chips) in enumerate(zip(result.rewards, result.chip_deltas)):
            # Reward is a strictly increasing function of chips won.
            assert reward == pytest.approx(chips / env.state.initial_stacks[seat])
            assert -1.0 <= reward <= 2.0


# --- mixed-opponent training ----------------------------------------------
def test_collect_seats_restricts_which_transitions_are_stored():
    """Fixed opponents share the table but must never become training data."""
    cfg, env, encoder, agents = make_pieces(seed=20)
    rng = random.Random(20)
    table = [agents[0], RandomAgent(), CallingStationAgent()]

    for _ in range(30):
        result = play_hand(env, table, encoder, rng, collect_seats=[0])
        seats = {t.player_perspective for t in result.transitions}
        assert seats <= {0}


def test_opponent_pool_is_built_from_names():
    from paradigm_a.training.self_play import build_opponent_pool

    cfg = small_config()
    cfg.train.opponent_pool = ("random", "calling_station", "tight_aggressive", "checkpoint")
    pool = build_opponent_pool(cfg, build_network(cfg))
    # "checkpoint" is added by the training loop, not here.
    assert [a.name for a in pool] == ["random", "calling_station", "tight_aggressive"]


def test_unknown_opponent_name_is_rejected():
    from paradigm_a.training.self_play import build_opponent_pool

    cfg = small_config()
    cfg.train.opponent_pool = ("nonexistent_bot",)
    with pytest.raises(ValueError):
        build_opponent_pool(cfg, build_network(cfg))


def test_snapshot_agent_is_frozen_and_independent():
    import torch

    from paradigm_a.training.self_play import snapshot_agent

    cfg = small_config()
    network = build_network(cfg)
    frozen = snapshot_agent(cfg, network)

    assert all(not p.requires_grad for p in frozen.network.parameters())

    observation = np.zeros(network.spec.total_dim, dtype=np.float32)
    before = frozen.network.infer(observation)[1].copy()
    with torch.no_grad():  # mutate the live network
        for parameter in network.parameters():
            parameter.add_(1.0)
    after = frozen.network.infer(observation)[1]
    np.testing.assert_array_equal(before, after)


def test_worker_with_a_pool_only_collects_learner_transitions():
    from paradigm_a.training.self_play import build_opponent_pool

    cfg = small_config()
    cfg.train.opponent_pool = ("calling_station", "random")
    cfg.train.opponent_mix_prob = 1.0  # always seat opponents
    network = build_network(cfg)
    encoder = ObservationEncoder(cfg.obs)
    pool = build_opponent_pool(cfg, network)
    worker = SelfPlayWorker(cfg, network, encoder, seed=21, opponent_pool=pool)

    # Per hand, the learner holds only the seats not taken by the pool.  (Across
    # many hands it still visits every seat, so this must be checked per hand.)
    for _ in range(40):
        seated, learner_seats = worker._seat_agents()
        assert 1 <= len(learner_seats) <= 2
        assert learner_seats == [s for s, a in enumerate(seated) if a is worker.agent]
        for seat, agent in enumerate(seated):
            if seat not in learner_seats:
                assert agent in pool

    transitions, stats = worker.generate(30)
    assert stats["mixed_opponent_rate"] == 1.0
    assert transitions, "the learner still occupies at least one seat"


def test_pure_self_play_remains_the_default():
    cfg = small_config()
    network = build_network(cfg)
    worker = SelfPlayWorker(cfg, network, ObservationEncoder(cfg.obs), seed=22)
    _, stats = worker.generate(10)
    assert stats["mixed_opponent_rate"] == 0.0


# --- recency-weighted league ----------------------------------------------
def _snapshot(cfg, network, tag):
    from paradigm_a.training.self_play import snapshot_agent

    agent = snapshot_agent(cfg, network)
    agent.tag = tag
    return agent


def test_league_evicts_the_oldest_snapshots():
    cfg = small_config()
    cfg.train.league_size = 3
    network = build_network(cfg)
    worker = SelfPlayWorker(cfg, network, ObservationEncoder(cfg.obs), seed=0)

    for i in range(6):
        worker.add_snapshot(_snapshot(cfg, network, i))
    assert [a.tag for a in worker.opponent_pool] == [3, 4, 5]


def test_recent_snapshots_are_sampled_more_often():
    import collections

    cfg = small_config()
    cfg.train.opponent_pool = ("checkpoint",)
    cfg.train.league_size = 4
    cfg.train.league_recency_decay = 0.5
    network = build_network(cfg)
    worker = SelfPlayWorker(cfg, network, ObservationEncoder(cfg.obs), seed=0)
    for i in range(4):
        worker.add_snapshot(_snapshot(cfg, network, i))

    # Geometric in age: newest 1, then 1/2, 1/4, 1/8.
    assert worker._opponent_weights() == pytest.approx([0.125, 0.25, 0.5, 1.0])

    counts = collections.Counter(worker._choose_opponent().tag for _ in range(4000))
    shares = [counts[i] / 4000 for i in range(4)]
    assert shares == sorted(shares), "share must increase with recency"
    assert shares[3] > 0.45 and shares[0] < 0.12


def test_uniform_league_when_decay_is_one():
    cfg = small_config()
    cfg.train.league_size = 4
    cfg.train.league_recency_decay = 1.0
    network = build_network(cfg)
    worker = SelfPlayWorker(cfg, network, ObservationEncoder(cfg.obs), seed=0)
    for i in range(4):
        worker.add_snapshot(_snapshot(cfg, network, i))
    assert worker._opponent_weights() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_fixed_bots_keep_unit_weight_alongside_snapshots():
    from paradigm_a.evaluation.random_agent import RandomAgent
    from paradigm_a.training.self_play import build_opponent_pool

    cfg = small_config()
    cfg.train.opponent_pool = ("random", "calling_station", "checkpoint")
    cfg.train.league_recency_decay = 0.5
    network = build_network(cfg)
    worker = SelfPlayWorker(
        cfg, network, ObservationEncoder(cfg.obs), seed=0,
        opponent_pool=build_opponent_pool(cfg, network),
    )
    worker.add_snapshot(_snapshot(cfg, network, 0))
    worker.add_snapshot(_snapshot(cfg, network, 1))
    # random, calling_station stay at 1.0; snapshots decay by age.
    assert worker._opponent_weights() == pytest.approx([1.0, 1.0, 0.5, 1.0])
    assert isinstance(worker.opponent_pool[0], RandomAgent)


# --- batched self-play -----------------------------------------------------
def _fresh_env(cfg, seed):
    from paradigm_a.environment.poker_env import PokerEnv

    return PokerEnv(cfg.env, cfg.obs, seed=seed)


def test_batched_matches_sequential_under_a_deterministic_policy():
    """Batching must change throughput, never behaviour.

    At temperature 0 action selection is an argmax, so no RNG is consumed and
    the two code paths must agree transition for transition on the same deal.
    """
    from paradigm_a.training.self_play import BatchedSelfPlayWorker

    cfg = small_config()
    cfg.train.sampling_temperature = 0.0
    network = build_network(cfg)
    encoder = ObservationEncoder(cfg.obs)

    sequential = SelfPlayWorker(cfg, network, encoder, seed=0)
    sequential.env = _fresh_env(cfg, 0)
    expected, _ = sequential.generate(20)

    batched = BatchedSelfPlayWorker(cfg, network, encoder, seed=0, num_envs=1)
    batched.envs = [_fresh_env(cfg, 0)]
    actual, _ = batched.generate(20)

    assert len(actual) == len(expected)
    for a, b in zip(expected, actual):
        assert a.action == b.action
        assert a.player_perspective == b.player_perspective
        assert a.q_target == pytest.approx(b.q_target)
        np.testing.assert_allclose(a.observation, b.observation)


@pytest.mark.parametrize("num_envs", [1, 8, 32])
def test_batched_worker_produces_well_formed_transitions(num_envs):
    from paradigm_a.training.self_play import BatchedSelfPlayWorker

    cfg = small_config()
    network = build_network(cfg)
    worker = BatchedSelfPlayWorker(
        cfg, network, ObservationEncoder(cfg.obs), seed=1, num_envs=num_envs
    )
    transitions, stats = worker.generate(24)

    assert stats["hands"] == 24
    assert stats["num_envs"] == num_envs
    assert stats["transitions"] == len(transitions)
    assert sum(stats["mean_chips_per_seat"]) == pytest.approx(0.0, abs=1e-6)
    for transition in transitions:
        assert transition.legal_action_mask[transition.action] == 1.0
        assert transition.old_policy.sum() == pytest.approx(1.0, abs=1e-5)


def test_batched_worker_respects_the_requested_hand_count():
    from paradigm_a.training.self_play import BatchedSelfPlayWorker

    cfg = small_config()
    worker = BatchedSelfPlayWorker(
        cfg, build_network(cfg), ObservationEncoder(cfg.obs), seed=2, num_envs=16
    )
    # Fewer hands than envs must not overshoot.
    assert worker.generate(5)[1]["hands"] == 5


def test_batched_worker_supports_the_opponent_pool():
    from paradigm_a.training.self_play import BatchedSelfPlayWorker, build_opponent_pool

    cfg = small_config()
    cfg.train.opponent_pool = ("calling_station",)
    cfg.train.opponent_mix_prob = 1.0
    network = build_network(cfg)
    worker = BatchedSelfPlayWorker(
        cfg,
        network,
        ObservationEncoder(cfg.obs),
        seed=3,
        num_envs=8,
        opponent_pool=build_opponent_pool(cfg, network),
    )
    transitions, stats = worker.generate(16)
    assert stats["mixed_opponent_rate"] == 1.0
    assert transitions


# --- co-evolving population ------------------------------------------------
def _coevo_worker(cfg, num_policies, seed=0, num_envs=8):
    from paradigm_a.training.self_play import CoevolutionSelfPlayWorker

    networks = [build_network(cfg) for _ in range(num_policies)]
    worker = CoevolutionSelfPlayWorker(
        cfg, networks, ObservationEncoder(cfg.obs), seed=seed, num_envs=num_envs
    )
    return networks, worker


def test_coevolution_groups_transitions_by_owning_network():
    cfg = small_config()
    networks, worker = _coevo_worker(cfg, num_policies=3, seed=1)
    per_network, stats = worker.generate(150)

    assert len(per_network) == 3
    assert stats["num_policies"] == 3
    assert stats["hands"] == 150
    # The per-network groups partition the transitions with nothing lost.
    assert stats["transitions"] == sum(len(g) for g in per_network)
    assert stats["transitions_per_network"] == [len(g) for g in per_network]
    # Over 150 hands every network is seated many times, so each collects data.
    assert all(len(group) > 0 for group in per_network)
    for group in per_network:
        for transition in group:
            assert transition.legal_action_mask[transition.action] == 1.0
            assert transition.old_policy.sum() == pytest.approx(1.0, abs=1e-5)
            assert -1.0 <= transition.q_target <= 2.0


def test_coevolution_conserves_chips_across_seats():
    cfg = small_config()
    _, worker = _coevo_worker(cfg, num_policies=2, seed=2, num_envs=4)
    _, stats = worker.generate(40)
    assert sum(stats["mean_chips_per_seat"]) == pytest.approx(0.0, abs=1e-6)


def test_coevolution_samples_seat_owners_with_replacement():
    """The same network must be able to occupy several seats in one hand."""
    cfg = small_config()
    _, worker = _coevo_worker(cfg, num_policies=2, seed=5)
    saw_repeat = False
    for _ in range(200):
        owners = worker._sample_seat_owners()
        assert len(owners) == cfg.env.num_players
        assert all(0 <= o < 2 for o in owners)
        saw_repeat |= len(set(owners)) < len(owners)
    assert saw_repeat, "with replacement, a network should sometimes take >1 seat"


def test_coevolution_single_network_is_ordinary_self_play():
    cfg = small_config()
    _, worker = _coevo_worker(cfg, num_policies=1, seed=0, num_envs=4)
    per_network, stats = worker.generate(20)
    assert len(per_network) == 1
    assert stats["num_policies"] == 1
    assert stats["transitions"] == len(per_network[0])


def test_coevolution_optimizes_every_network():
    """Each network trains on the data it produced, and all of them move."""
    import torch

    from paradigm_a.training.replay_buffer import ReplayBuffer
    from paradigm_a.training.trainer import Trainer

    cfg = small_config()
    cfg.train.batch_size = 64
    n = 2
    networks, worker = _coevo_worker(cfg, num_policies=n, seed=3)
    buffers = [
        ReplayBuffer(2000, worker.encoder.observation_dim, worker.encoder.spec.num_actions)
        for _ in range(n)
    ]
    trainers = [Trainer(cfg, networks[k], device="cpu") for k in range(n)]
    before = [[p.detach().clone() for p in net.parameters()] for net in networks]

    while any(len(b) < 128 for b in buffers):
        per_network, _ = worker.generate(40)
        for k in range(n):
            buffers[k].extend(per_network[k])

    rng = np.random.default_rng(0)
    for k in range(n):
        for _ in range(20):
            metrics = trainers[k].train_step(buffers[k].sample(64, rng))
            assert np.isfinite(metrics["loss"])

    for k in range(n):
        moved = any(
            not torch.equal(old, new)
            for old, new in zip(before[k], networks[k].parameters())
        )
        assert moved, f"network {k} did not update"
