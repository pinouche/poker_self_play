"""Baseline agents and the match-play harness."""

import pytest

from config import Config, ModelConfig
from evaluation.evaluate import evaluate_match, format_results, make_network_agent
from evaluation.heuristic_agent import HeuristicAgent, loose_passive, tight_aggressive
from evaluation.random_agent import (
    AlwaysFoldAgent,
    CallingStationAgent,
    RandomAgent,
    UniformLegalAgent,
)
from model.network import build_network


def small_config() -> Config:
    cfg = Config()
    cfg.model = ModelConfig(hidden_dim=32, num_residual_blocks=2, head_hidden=32)
    return cfg


def test_uniform_legal_agent_is_the_random_policy():
    assert UniformLegalAgent is RandomAgent


def test_three_identical_agents_split_the_pot_evenly_over_time():
    cfg = small_config()
    agent = RandomAgent()
    result = evaluate_match([agent, agent, agent], cfg, num_hands=150, seed=0)

    stats = result["per_agent"]["random"]
    assert stats["hands"] == result["hands"] * 3
    # Exactly one seat wins each pot (before split pots), so roughly a third.
    assert 0.2 < stats["win_rate"] < 0.5
    assert stats["avg_chips"] == pytest.approx(0.0, abs=1e-6)


def test_seat_statistics_cover_every_seat():
    cfg = small_config()
    agent = RandomAgent()
    result = evaluate_match([agent, agent, agent], cfg, num_hands=60, seed=1)
    assert set(result["per_seat"]) == {0, 1, 2}
    for stats in result["per_seat"].values():
        assert stats["hands"] == result["hands"]
        rates = stats["win_rate"] + stats["loss_rate"] + stats["tie_rate"]
        assert rates == pytest.approx(1.0, abs=1e-6)


def test_rotation_gives_each_agent_the_same_number_of_hands():
    cfg = small_config()
    agents = [RandomAgent(), CallingStationAgent(), AlwaysFoldAgent()]
    result = evaluate_match(agents, cfg, num_hands=90, seed=2)
    counts = {name: stats["hands"] for name, stats in result["per_agent"].items()}
    assert len(set(counts.values())) == 1


def test_always_folding_loses_to_players_who_contest_pots():
    cfg = small_config()
    agents = [AlwaysFoldAgent(), CallingStationAgent(), CallingStationAgent()]
    result = evaluate_match(agents, cfg, num_hands=150, seed=3)
    assert result["per_agent"]["always_fold"]["avg_chips"] < 0


def test_heuristic_agents_only_pick_legal_actions():
    cfg = small_config()
    agents = [tight_aggressive(seed=0, samples=20), loose_passive(seed=1, samples=20), RandomAgent()]
    result = evaluate_match(agents, cfg, num_hands=30, seed=4)
    assert result["hands"] > 0  # play_hand raises on an illegal choice


def test_heuristic_beats_a_player_that_never_contests():
    cfg = small_config()
    agents = [
        HeuristicAgent(samples=20, seed=0, name="heuristic"),
        AlwaysFoldAgent(),
        AlwaysFoldAgent(),
    ]
    result = evaluate_match(agents, cfg, num_hands=60, seed=5)
    assert result["per_agent"]["heuristic"]["avg_chips"] > 0


def test_network_agent_plays_a_full_match():
    cfg = small_config()
    network = build_network(cfg)
    hero = make_network_agent(network, cfg, temperature=0.0)
    result = evaluate_match([hero, RandomAgent(), RandomAgent()], cfg, num_hands=30, seed=6)
    assert "network" in result["per_agent"]
    assert 0.0 <= result["showdown_rate"] <= 1.0


def test_results_format_without_error():
    cfg = small_config()
    agent = RandomAgent()
    results = {"match": evaluate_match([agent, agent, agent], cfg, num_hands=15, seed=7)}
    text = format_results(results)
    assert "per seat" in text and "bb/100" in text


def test_wrong_number_of_agents_is_rejected():
    with pytest.raises(ValueError):
        evaluate_match([RandomAgent(), RandomAgent()], small_config(), num_hands=3)
