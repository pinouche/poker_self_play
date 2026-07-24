"""Role-structured league population and its three management metrics."""

import numpy as np
import pytest

from config import Config, ModelConfig
from representation.observation_encoder import ObservationEncoder
from training.league import LeagueMember, LeaguePopulation, Role
from training.league_metrics import (
    diversity_from_kl,
    masked_softmax_rows,
    min_max_normalise,
    pairwise_kl,
    population_scores,
)


def small_league_config(learners=2, champions=1, explorers=1) -> Config:
    cfg = Config()
    cfg.model = ModelConfig(hidden_dim=32, num_residual_blocks=2, head_hidden=32)
    cfg.train.league_learners = learners
    cfg.train.league_champions = champions
    cfg.train.league_explorers = explorers
    cfg.train.league_strength_hands = 40
    cfg.train.league_diversity_states = 60
    return cfg


# --- diversity: action-distribution distance -------------------------------
def test_masked_softmax_zeroes_illegal_actions():
    probs = masked_softmax_rows(np.array([[1.0, 2.0, 3.0]]), np.array([[1, 0, 1]], dtype=np.float32))
    assert probs[0, 1] == 0.0
    assert probs.sum(axis=1)[0] == pytest.approx(1.0)


def test_pairwise_kl_is_zero_for_identical_policies_and_positive_otherwise():
    masks = np.ones((4, 3), dtype=np.float32)
    p = np.tile(np.array([0.7, 0.2, 0.1]), (4, 1))
    q = np.tile(np.array([0.1, 0.2, 0.7]), (4, 1))

    kl = pairwise_kl([p, p.copy(), q], masks)
    assert kl.shape == (3, 3)
    assert np.allclose(np.diag(kl), 0.0)
    assert kl[0, 1] == pytest.approx(0.0, abs=1e-9)  # identical policies
    assert kl[0, 2] > 0.1                            # genuinely different


def test_kl_ignores_illegal_actions():
    """Illegal actions carry zero mass and must not contribute divergence."""
    masks = np.array([[1, 1, 0]], dtype=np.float32)
    p = np.array([[0.5, 0.5, 0.0]])
    q = np.array([[0.5, 0.5, 0.0]])
    assert pairwise_kl([p, q], masks)[0, 1] == pytest.approx(0.0, abs=1e-9)


def test_diversity_is_mean_kl_to_every_other_member():
    kl = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 1.0], [3.0, 1.0, 0.0]])
    np.testing.assert_allclose(diversity_from_kl(kl), [2.0, 1.0, 2.0])


# --- scoring ---------------------------------------------------------------
def test_min_max_normalise_handles_an_all_equal_population():
    np.testing.assert_allclose(min_max_normalise(np.array([0.3, 0.3, 0.3])), [0.5, 0.5, 0.5])
    np.testing.assert_allclose(min_max_normalise(np.array([0.0, 1.0, 2.0])), [0.0, 0.5, 1.0])


def test_diversity_term_saves_a_weaker_but_different_network():
    """The whole point of the design: rank on score, not strength alone.

    A and B are near-identical; C is weaker but strategically distinct.  Ranked
    on strength C is last, so strength-only management would delete it and the
    league drifts toward copies of A.  The diversity term promotes C above B.
    """
    strength = np.array([1.00, 0.90, 0.85])
    diversity = np.array([0.00, 0.00, 1.00])

    score, _, _ = population_scores(strength, diversity, 0.6, 0.4)
    assert score[2] > score[1], "weaker-but-different must outrank slightly-stronger-but-same"

    strength_only, _, _ = population_scores(strength, diversity, 1.0, 0.0)
    assert strength_only[1] > strength_only[2], "on strength alone C would be culled"


def test_scores_are_normalised_before_weighting():
    """Chip EV is O(0.05) and KL is O(1); raw sums would ignore strength."""
    strength = np.array([0.01, 0.05])   # tiny magnitudes
    diversity = np.array([2.0, 0.0])    # large magnitudes
    score, ns, nd = population_scores(strength, diversity, 0.6, 0.4)
    np.testing.assert_allclose(ns, [0.0, 1.0])
    np.testing.assert_allclose(nd, [1.0, 0.0])
    np.testing.assert_allclose(score, [0.4, 0.6])  # strength weight decides


# --- population structure --------------------------------------------------
def test_default_league_is_sixteen_learners_eight_champions_eight_explorers():
    train = Config().train
    assert (train.league_learners, train.league_champions, train.league_explorers) == (16, 8, 8)
    assert train.league_learners + train.league_champions + train.league_explorers == 32
    assert (train.league_strength_weight, train.league_diversity_weight) == (0.6, 0.4)


def test_population_has_the_configured_roles():
    cfg = small_league_config(learners=3, champions=2, explorers=1)
    pop = LeaguePopulation(cfg, seed=0)
    assert len(pop) == 6
    assert pop.role_counts() == {"learner": 3, "champion": 2, "explorer": 1}
    # Champions are opponents only; learners and explorers are optimised.
    assert sorted(pop.trainable_indices()) == sorted(
        pop.indices(Role.LEARNER) + pop.indices(Role.EXPLORER)
    )


def test_champions_are_frozen():
    cfg = small_league_config()
    pop = LeaguePopulation(cfg, seed=0)
    for index in pop.indices(Role.CHAMPION):
        network = pop.members[index].network
        assert all(not p.requires_grad for p in network.parameters())


def test_sample_seats_returns_valid_members():
    cfg = small_league_config(learners=2, champions=1, explorers=1)
    pop = LeaguePopulation(cfg, seed=0)
    for _ in range(50):
        seats = pop.sample_seats(cfg.env.num_players)
        assert len(seats) == cfg.env.num_players
        assert all(0 <= s < len(pop) for s in seats)


# --- management ------------------------------------------------------------
def test_promoted_champion_is_frozen_and_independent_of_the_live_learner():
    import torch

    cfg = small_league_config(learners=2, champions=1, explorers=0)
    pop = LeaguePopulation(cfg, seed=0)
    best = pop.indices(Role.LEARNER)[1]
    scores = [0.0] * len(pop)
    scores[best] = 1.0

    promoted, slot = pop.promote(scores)
    assert promoted == best
    champion = pop.members[slot]
    assert champion.role is Role.CHAMPION
    assert all(not p.requires_grad for p in champion.network.parameters())

    # The snapshot must not track the learner it was copied from.
    observation = np.zeros(champion.network.spec.total_dim, dtype=np.float32)
    before = champion.network.infer(observation)[0].copy()
    with torch.no_grad():
        for parameter in pop.members[best].network.parameters():
            parameter.add_(1.0)
    np.testing.assert_array_equal(before, champion.network.infer(observation)[0])


def test_cull_reinitialises_the_lowest_scoring_learner_from_scratch():
    cfg = small_league_config(learners=3, champions=1, explorers=0)
    pop = LeaguePopulation(cfg, seed=0)
    learners = pop.indices(Role.LEARNER)
    scores = [1.0] * len(pop)
    scores[learners[1]] = -1.0
    original = pop.members[learners[1]].network

    culled = pop.cull(scores)
    assert culled == learners[1]
    assert pop.members[culled].role is Role.LEARNER
    assert pop.members[culled].network is not original  # a fresh network, not a clone


def test_explorers_reset_from_scratch_on_the_hand_schedule():
    cfg = small_league_config(learners=1, champions=1, explorers=2)
    cfg.train.league_explorer_reset_hands = 1_000
    pop = LeaguePopulation(cfg, seed=0)
    explorers = pop.indices(Role.EXPLORER)
    originals = [pop.members[i].network for i in explorers]

    assert pop.maybe_reset_explorers() == []      # window not elapsed
    pop.total_hands = 1_000
    assert sorted(pop.maybe_reset_explorers()) == sorted(explorers)
    for index, original in zip(explorers, originals):
        assert pop.members[index].network is not original
        assert pop.members[index].role is Role.EXPLORER
    assert pop.maybe_reset_explorers() == []      # not again until the next window


def test_manage_reports_slots_whose_network_was_replaced():
    cfg = small_league_config(learners=3, champions=2, explorers=0)
    pop = LeaguePopulation(cfg, seed=0)
    scores = list(np.linspace(0.0, 1.0, len(pop)))
    report = pop.manage(scores)
    assert report.promoted in pop.indices(Role.CHAMPION) + pop.indices(Role.LEARNER)
    # The caller must rebuild the optimiser/buffer for every replaced slot.
    assert report.champion_slot in report.rebuilt_slots
    if report.culled is not None:
        assert report.culled in report.rebuilt_slots


def test_manage_rejects_a_mismatched_score_vector():
    pop = LeaguePopulation(small_league_config(), seed=0)
    with pytest.raises(ValueError):
        pop.manage([0.0, 1.0])


# --- end-to-end metrics ----------------------------------------------------
def test_evaluate_produces_strength_diversity_coverage_and_score():
    cfg = small_league_config(learners=2, champions=1, explorers=1)
    pop = LeaguePopulation(cfg, seed=0)
    metrics = pop.evaluate(ObservationEncoder(cfg.obs), seed=0)
    n = len(pop)

    for key in ("strength", "diversity", "coverage", "score"):
        assert len(metrics[key]) == n
        assert np.all(np.isfinite(metrics[key]))
    assert metrics["kl_matrix"].shape == (n, n)
    assert np.allclose(np.diag(metrics["kl_matrix"]), 0.0)
    assert np.all(metrics["diversity"] >= 0.0)          # KL is non-negative
    assert np.all(metrics["coverage"] <= n - 1)         # never your own opponent
    assert np.all((metrics["score"] >= 0.0) & (metrics["score"] <= 1.0))
    assert metrics["hands_played"].sum() > 0
    assert pop.summary(metrics)                          # renders for logs
