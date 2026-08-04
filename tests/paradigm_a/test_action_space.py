"""Configurable bet abstraction.

The action space stays *fixed* for a given configuration -- the network needs a
constant output width -- but its size is derived from the number of bet and
raise sizings rather than hardcoded.
"""

import random

import pytest

from paradigm_a.config import Config, EnvConfig, ObsConfig
from paradigm_a.environment.poker_env import PokerEnv
from paradigm_a.environment.state import (
    ALL_IN,
    BET_LARGE,
    BET_MEDIUM,
    BET_SMALL,
    CALL,
    CHECK,
    FOLD,
    NUM_ACTIONS,
    RAISE_LARGE,
    RAISE_MEDIUM,
    RAISE_SMALL,
    action_space_for,
    build_action_space,
)
from paradigm_a.model.network import build_network, load_checkpoint, save_checkpoint
from paradigm_a.representation.observation_encoder import ObservationEncoder

FINE_BETS = (0.25, 0.50, 0.75, 1.00, 1.50)
FINE_RAISES = (2.0, 2.5, 3.0, 4.0, 6.0)


def fine_config() -> Config:
    cfg = Config()
    cfg.env = EnvConfig(bet_fractions=FINE_BETS, raise_multipliers=FINE_RAISES)
    cfg.obs = ObsConfig(use_equity_feature=False)
    cfg.model.hidden_dim = 32
    cfg.model.num_residual_blocks = 2
    cfg.model.head_hidden = 32
    return cfg


# --- backwards compatibility ----------------------------------------------
def test_default_space_is_exactly_the_documented_ten_actions():
    space = action_space_for(EnvConfig())
    assert space.num_actions == NUM_ACTIONS == 10
    assert space.names == (
        "FOLD", "CHECK", "CALL",
        "BET_SMALL", "BET_MEDIUM", "BET_LARGE",
        "RAISE_SMALL", "RAISE_MEDIUM", "RAISE_LARGE",
        "ALL_IN",
    )
    assert (space.bet_ids, space.raise_ids, space.all_in) == ((3, 4, 5), (6, 7, 8), 9)
    # The legacy module constants must still line up.
    assert (FOLD, CHECK, CALL) == (0, 1, 2)
    assert (BET_SMALL, BET_MEDIUM, BET_LARGE) == (3, 4, 5)
    assert (RAISE_SMALL, RAISE_MEDIUM, RAISE_LARGE) == (6, 7, 8)
    assert ALL_IN == 9


# --- finer abstractions ----------------------------------------------------
def test_finer_space_extends_ids_and_keeps_all_in_last():
    space = build_action_space(FINE_BETS, FINE_RAISES)
    assert space.num_actions == 3 + 5 + 5 + 1 == 14
    assert space.bet_ids == (3, 4, 5, 6, 7)
    assert space.raise_ids == (8, 9, 10, 11, 12)
    assert space.all_in == 13
    assert space.names[3:8] == ("BET_25", "BET_50", "BET_75", "BET_100", "BET_150")
    assert space.names[8:13] == (
        "RAISE_2X", "RAISE_2_5X", "RAISE_3X", "RAISE_4X", "RAISE_6X"
    )


def test_asymmetric_sizing_counts_are_supported():
    space = build_action_space((0.5, 1.0), (2.0, 3.0, 4.0, 5.0))
    assert space.num_actions == 3 + 2 + 4 + 1
    assert space.bet_ids == (3, 4) and space.raise_ids == (5, 6, 7, 8)
    assert space.all_in == 9


def test_env_masks_and_amounts_follow_the_finer_space():
    cfg = fine_config()
    space = action_space_for(cfg.env)
    env = PokerEnv(cfg.env, cfg.obs, seed=1)
    env.reset(dealer=0)

    mask = env.get_observation(0)["legal_action_mask"]
    assert len(mask) == space.num_actions

    amounts = env.legal_actions().to_amounts
    raise_amounts = sorted({amounts[a] for a in space.raise_ids if a in amounts})
    # Five multipliers of the 20-chip blind, deduplicated against the minimum.
    assert raise_amounts == [40, 50, 60, 80, 120]


def test_finer_space_plays_legally_and_conserves_chips():
    cfg = fine_config()
    space = action_space_for(cfg.env)
    env = PokerEnv(cfg.env, cfg.obs, seed=2)
    rng = random.Random(2)
    for _ in range(200):
        env.reset()
        while not env.is_terminal:
            mask = env.get_observation(env.to_act)["legal_action_mask"]
            assert len(mask) == space.num_actions and mask.sum() > 0
            env.step(rng.choice(env.legal_actions().legal_ids()))
        assert sum(env.chip_deltas()) == pytest.approx(0.0, abs=1e-6)


# --- observation and network follow ---------------------------------------
def test_observation_and_network_widths_track_the_action_count():
    default_encoder = ObservationEncoder.from_config(Config())
    fine_cfg = fine_config()
    fine_encoder = ObservationEncoder.from_config(fine_cfg)

    assert fine_encoder.spec.num_actions == 14
    assert fine_encoder.observation_dim > default_encoder.observation_dim
    # The mask and every history slot widen by the extra actions.
    assert fine_encoder.spec.field_dims["legal_action_mask"] == 14

    network = build_network(fine_cfg)
    assert network.num_actions == 14
    import torch

    logits, q_values = network(torch.zeros(3, network.spec.total_dim))
    assert logits.shape == (3, 14) and q_values.shape == (3, 14)


def test_checkpoint_round_trips_a_finer_space():
    import os
    import tempfile

    cfg = fine_config()
    network = build_network(cfg)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "fine.pt")
        save_checkpoint(path, network, cfg)
        restored, restored_cfg, _ = load_checkpoint(path)
    assert restored.num_actions == 14
    assert tuple(restored_cfg.env.bet_fractions) == FINE_BETS


# --- agents and self-play --------------------------------------------------
def test_baseline_and_heuristic_agents_work_in_a_finer_space():
    from paradigm_a.evaluation.evaluate import evaluate_match
    from paradigm_a.evaluation.heuristic_agent import tight_aggressive
    from paradigm_a.evaluation.random_agent import CallingStationAgent, RandomAgent

    cfg = fine_config()
    space = action_space_for(cfg.env)
    table = [
        RandomAgent(),
        CallingStationAgent(),
        tight_aggressive(seed=0, samples=15, action_space=space),
    ]
    result = evaluate_match(table, cfg, num_hands=30, seed=0)
    assert result["hands"] > 0  # play_hand raises on an illegal choice


def test_self_play_runs_in_a_finer_space():
    from paradigm_a.training.self_play import BatchedSelfPlayWorker

    cfg = fine_config()
    network = build_network(cfg)
    worker = BatchedSelfPlayWorker(
        cfg, network, ObservationEncoder.from_config(cfg), seed=1, num_envs=8
    )
    transitions, stats = worker.generate(20)
    assert stats["hands"] == 20
    for transition in transitions:
        assert len(transition.legal_action_mask) == 14
        assert transition.legal_action_mask[transition.action] == 1.0


# --- public API ------------------------------------------------------------
def test_inference_reports_the_finer_action_names():
    import copy

    from paradigm_a.cli.infer import EXAMPLE_TABLE_STATE
    from paradigm_a.inference.suggest_action import SuggestionEngine

    cfg = fine_config()
    engine = SuggestionEngine.untrained(cfg)
    result = engine.suggest(copy.deepcopy(EXAMPLE_TABLE_STATE), temperature=1.0)

    assert set(result["probabilities"]) == set(action_space_for(cfg.env).names)
    assert "BET_150" in result["probabilities"]
    assert result["action"] in result["legal_actions"]
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
