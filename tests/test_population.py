"""Population self-play: policy library, randomised games, scenario mixture."""

import random
import collections

import numpy as np
import pytest

from config import Config, ModelConfig
from environment.state import Street
from model.network import build_network
from representation.observation_encoder import ObservationEncoder
from training.population import (
    PolicyLibrary,
    freeze_policy,
    sample_entry_street,
    sample_game_setup,
)
from training.self_play import PopulationSelfPlayWorker


def small_config() -> Config:
    cfg = Config()
    cfg.model = ModelConfig(hidden_dim=32, num_residual_blocks=2, head_hidden=32)
    cfg.train.population_self_play = True
    cfg.train.league_weighting = "linear"
    return cfg


# --- policy library --------------------------------------------------------
def test_linear_weighting_declines_by_one_per_step():
    lib = PolicyLibrary(capacity=5, weighting="linear")
    for tag in range(4):
        lib.add(("policy", tag))
    # oldest -> newest gets 1, 2, 3, 4.
    assert lib.weights() == [1.0, 2.0, 3.0, 4.0]


def test_geometric_weighting_is_still_available():
    lib = PolicyLibrary(capacity=5, weighting="geometric", decay=0.5)
    for tag in range(3):
        lib.add(tag)
    # newest 1, then 0.5, then 0.25 (oldest).
    assert lib.weights() == pytest.approx([0.25, 0.5, 1.0])


def test_library_evicts_oldest_past_capacity():
    lib = PolicyLibrary(capacity=3, weighting="linear")
    for tag in range(6):
        lib.add(tag)
    assert lib._policies == [3, 4, 5]


def test_recent_policies_are_sampled_more_often_linearly():
    lib = PolicyLibrary(capacity=4, weighting="linear")
    for tag in range(4):
        lib.add(tag)
    rng = random.Random(0)
    counts = collections.Counter(lib.sample(rng) for _ in range(8000))
    shares = [counts[t] / 8000 for t in range(4)]
    # Weights 1:2:3:4 -> shares 0.1:0.2:0.3:0.4.
    assert shares == sorted(shares)
    assert shares[3] == pytest.approx(0.4, abs=0.04)
    assert shares[0] == pytest.approx(0.1, abs=0.04)


def test_sampling_an_empty_library_is_an_error():
    with pytest.raises(ValueError):
        PolicyLibrary().sample(random.Random(0))


def test_unknown_weighting_is_rejected():
    with pytest.raises(ValueError):
        PolicyLibrary(weighting="quadratic")


# --- randomised game setup -------------------------------------------------
def test_random_setup_varies_stacks_and_blinds():
    cfg = small_config()
    rng = random.Random(1)
    setups = [sample_game_setup(rng, cfg) for _ in range(200)]
    # Stack depths and blind levels both vary.
    assert len({tuple(s.stacks) for s in setups}) > 100
    assert len({s.big_blind for s in setups}) > 1
    for s in setups:
        assert s.small_blind == max(1, s.big_blind // 2)
        assert all(stack >= s.big_blind for stack in s.stacks)  # can post the blind
        assert 0 <= s.dealer < cfg.env.num_players


def test_random_setup_respects_the_stack_range():
    cfg = small_config()
    cfg.train.population_stack_min_bb = 30
    cfg.train.population_stack_max_bb = 40
    rng = random.Random(2)
    for _ in range(300):
        s = sample_game_setup(rng, cfg)
        for stack in s.stacks:
            depth = stack / s.big_blind
            assert 30 - 1 <= depth <= 40 + 1


def test_setup_can_be_made_deterministic():
    cfg = small_config()
    cfg.train.population_random_setup = False
    s = sample_game_setup(random.Random(3), cfg)
    assert s.big_blind == cfg.env.big_blind
    assert all(stack == cfg.env.starting_stack for stack in s.stacks)


# --- scenario mixture ------------------------------------------------------
def test_entry_street_distribution_matches_the_config():
    cfg = small_config()
    cfg.train.entry_street_probs = (0.70, 0.15, 0.10, 0.05)
    rng = random.Random(4)
    counts = collections.Counter(sample_entry_street(rng, cfg) for _ in range(6000))
    frac = {s: counts[s] / 6000 for s in counts}
    assert frac[Street.PREFLOP] == pytest.approx(0.70, abs=0.03)
    assert frac[Street.FLOP] == pytest.approx(0.15, abs=0.03)
    assert frac[Street.TURN] == pytest.approx(0.10, abs=0.03)
    assert frac[Street.RIVER] == pytest.approx(0.05, abs=0.02)


def test_unnormalised_entry_probs_are_normalised():
    cfg = small_config()
    cfg.train.entry_street_probs = (7, 1.5, 1.0, 0.5)  # sums to 10
    rng = random.Random(5)
    counts = collections.Counter(sample_entry_street(rng, cfg) for _ in range(4000))
    assert counts[Street.PREFLOP] / 4000 == pytest.approx(0.70, abs=0.04)


# --- the worker ------------------------------------------------------------
def make_worker(seed=0, snapshots=6):
    cfg = small_config()
    network = build_network(cfg)
    worker = PopulationSelfPlayWorker(
        cfg, network, ObservationEncoder.from_config(cfg), seed=seed
    )
    cfg.train.population_snapshot_every = 1
    for _ in range(snapshots):
        worker.maybe_snapshot()
    return cfg, network, worker


def test_worker_seeds_the_library_so_villains_exist_immediately():
    cfg = small_config()
    network = build_network(cfg)
    worker = PopulationSelfPlayWorker(cfg, network, ObservationEncoder.from_config(cfg), seed=0)
    assert len(worker.library) == 1  # seeded at construction
    transitions, stats = worker.generate(20)
    assert stats["hands"] == 20


def test_worker_produces_valid_learner_transitions():
    cfg, network, worker = make_worker()
    transitions, stats = worker.generate(150)
    assert transitions
    assert stats["transitions"] == len(transitions)
    for t in transitions:
        assert t.legal_action_mask[t.action] == 1.0
        assert t.old_policy.sum() == pytest.approx(1.0, abs=1e-5)
        assert -1.0 <= t.q_target <= 2.0


def test_every_transition_is_from_a_single_learner_seat_per_hand():
    """Population play collects the learner only, never the villains."""
    cfg, network, worker = make_worker(seed=3)
    # One hand at a time: all transitions must share one perspective seat.
    for _ in range(40):
        transitions, _ = worker.generate(1)
        seats = {t.player_perspective for t in transitions}
        assert len(seats) <= 1


def test_later_street_entries_actually_occur():
    cfg, network, worker = make_worker(seed=7)
    _, stats = worker.generate(400)
    fractions = stats["entry_street_fractions"]
    assert fractions[0] == pytest.approx(0.75, abs=0.08)   # preflop backbone
    assert sum(fractions[1:]) == pytest.approx(0.25, abs=0.08)  # later-street mix
    assert fractions[1] > 0 and fractions[2] > 0  # flop and turn entries happen


def test_snapshot_schedule_grows_the_library():
    cfg = small_config()
    cfg.train.population_snapshot_every = 5
    cfg.train.population_library_size = 4
    network = build_network(cfg)
    worker = PopulationSelfPlayWorker(cfg, network, ObservationEncoder.from_config(cfg), seed=0)
    sizes = []
    for _ in range(30):
        worker.maybe_snapshot()
        sizes.append(len(worker.library))
    assert max(sizes) == 4  # capped
    assert sizes[-1] == 4


def test_population_play_conserves_chips_and_runs_full_games():
    """Games start from a random setup and reach a terminal state every time."""
    cfg, network, worker = make_worker(seed=11)
    # A completed hand must have a non-trivial decision count on average.
    _, stats = worker.generate(100)
    assert stats["decisions_per_hand"] > 2
    assert 0.0 <= stats["showdown_rate"] <= 1.0


# --- heterogeneous co-evolving population ----------------------------------
def test_seat_reward_matches_each_mode():
    from config import EnvConfig, seat_reward

    env = EnvConfig()  # normalized_chip_return, 1000 stack, bb 20
    assert seat_reward(200, 1000, 20, env) == pytest.approx(0.2)
    env.reward_mode = "binary"
    assert [seat_reward(x, 1000, 20, env) for x in (200, -50, 0)] == [1.0, -1.0, 0.0]
    env.reward_mode = "bb_normalized"
    assert seat_reward(200, 1000, 20, env) == pytest.approx(10.0)
    env.reward_mode = "chip_return"
    assert seat_reward(200, 1000, 20, env) == 200.0


def test_build_population_configs_homogeneous_by_default():
    from training.population import build_population_configs

    cfg = Config()  # heterogeneous_population defaults False
    cfgs = build_population_configs(cfg, 5)
    assert len(cfgs) == 5
    assert all(c is cfg for c in cfgs)  # five clones = the base config


def test_build_population_configs_heterogeneous_is_diverse():
    from model.network import build_network
    from training.population import build_population_configs

    cfg = Config()
    cfg.train.heterogeneous_population = True
    cfgs = build_population_configs(cfg, 5)

    assert len(cfgs) == 5
    # Every member optimises the SAME objective -- mixing in win-rate members
    # poisons co-evolution (measured), so diversity is style-only.
    assert {c.env.reward_mode for c in cfgs} == {"normalized_chip_return"}
    # At least three distinct network sizes (tiny/medium/large).
    assert len({build_network(c).num_parameters() for c in cfgs}) >= 3
    # Exploration temperature and improvement sharpness vary across members.
    assert len({c.train.sampling_temperature for c in cfgs}) > 1
    assert len({c.train.alpha for c in cfgs}) > 1
    # Observation/action layout is shared, so members interoperate at one table.
    assert all(c.obs == cfgs[0].obs for c in cfgs)
    # The base config is left untouched (deep-copied).
    assert cfg.env.reward_mode == "normalized_chip_return"
