"""Baseline agents and the match-play harness."""

import numpy as np
import pytest

from paradigm_a.config import Config, ModelConfig
from paradigm_a.evaluation.evaluate import evaluate_match, format_results, make_network_agent
from paradigm_a.evaluation.heuristic_agent import HeuristicAgent, loose_passive, tight_aggressive
from paradigm_a.evaluation.random_agent import (
    AlwaysFoldAgent,
    CallingStationAgent,
    RandomAgent,
    UniformLegalAgent,
)
from paradigm_a.model.network import build_network


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


# --- duplicate-deal scoring ------------------------------------------------
def test_duplicate_deals_replay_the_same_cards_for_every_hero():
    """The whole point: two heroes are scored on identical deck orders."""
    from paradigm_a.evaluation.evaluate import duplicate_deal_scores

    cfg = small_config()
    seeds = list(range(1, 21))
    first = duplicate_deal_scores(RandomAgent(), [CallingStationAgent()] * 2, cfg, seeds)
    again = duplicate_deal_scores(RandomAgent(), [CallingStationAgent()] * 2, cfg, seeds)
    assert len(first) == len(seeds)
    # Same agent, same deals, same rngs -> bit-identical, so differences between
    # two *different* agents are attributable to the agents, not the cards.
    np.testing.assert_array_equal(first, again)


def test_duplicate_deal_scoring_reduces_variance():
    """Seat-averaging must shrink the spread versus scoring one seat only.

    Card luck is the dominant term in a poker result; rotating the hero through
    every seat on the same deck order cancels much of it.
    """
    import random

    from paradigm_a.environment.poker_env import PokerEnv
    from paradigm_a.evaluation.evaluate import duplicate_deal_scores
    from paradigm_a.representation.observation_encoder import ObservationEncoder
    from paradigm_a.training.self_play import play_hand

    cfg = small_config()
    seeds = list(range(1, 201))
    hero, opponents = CallingStationAgent(), [RandomAgent(), RandomAgent()]

    averaged = duplicate_deal_scores(hero, opponents, cfg, seeds)

    encoder = ObservationEncoder(cfg.obs)
    single = []
    for seed in seeds:
        env = PokerEnv(cfg.env, cfg.obs, seed=seed)
        result = play_hand(
            env, [hero] + opponents, encoder, random.Random(seed * 7), collect=False
        )
        single.append(result.chip_deltas[0])

    assert averaged.std() < np.std(single)


def test_confidence_interval_shrinks_with_more_deals():
    from paradigm_a.evaluation.evaluate import bb_per_100_interval

    rng = np.random.default_rng(0)
    small = bb_per_100_interval(rng.normal(0, 100, size=100), 20)
    large = bb_per_100_interval(rng.normal(0, 100, size=10_000), 20)
    assert large["ci_half_width"] < small["ci_half_width"] / 5
    assert small["deals"] == 100 and large["deals"] == 10_000


def test_paired_comparison_reports_a_difference():
    from paradigm_a.evaluation.evaluate import compare_agents_paired

    cfg = small_config()
    result = compare_agents_paired(
        CallingStationAgent(), AlwaysFoldAgent(), [RandomAgent()] * 2, cfg, num_deals=40
    )
    assert set(result) == {"calling_station", "always_fold", "difference"}
    expected = result["calling_station"]["bb_per_100"] - result["always_fold"]["bb_per_100"]
    assert result["difference"]["bb_per_100"] == pytest.approx(expected, abs=1e-6)
