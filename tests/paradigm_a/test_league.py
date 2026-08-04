"""Role-structured league population and its three management metrics."""

import collections

import numpy as np
import pytest
import torch

from paradigm_a.config import Config, ModelConfig
from paradigm_a.representation.observation_encoder import ObservationEncoder
from paradigm_a.training.league import LeaguePopulation, MatchType, Role
from paradigm_a.training.league_metrics import (
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
    cfg.train.league_gauntlet_min_hands = 1   # count small samples in tests
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
    counts = pop.role_counts()
    assert (counts["learner"], counts["champion"], counts["explorer"]) == (3, 2, 1)
    # Frozen is the default, so every champion is an anchor and none train.
    assert (counts["anchor"], counts["veteran"]) == (2, 0)
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


def test_sample_seats_returns_valid_members_and_a_match_type():
    cfg = small_league_config(learners=2, champions=1, explorers=1)
    pop = LeaguePopulation(cfg, seed=0)
    for _ in range(50):
        seats, match = pop.sample_seats(cfg.env.num_players)
        assert len(seats) == cfg.env.num_players
        assert all(0 <= s < len(pop) for s in seats)
        assert isinstance(match, MatchType)


# --- matchmaking -----------------------------------------------------------
def test_matchmaking_follows_the_configured_mix():
    cfg = small_league_config(learners=4, champions=2, explorers=2)
    cfg.train.league_match_learner_vs_learner = 0.5
    cfg.train.league_match_learner_vs_champion = 0.3
    cfg.train.league_match_mixed = 0.2
    pop = LeaguePopulation(cfg, seed=0)

    counts = collections.Counter(pop.sample_seats(3)[1] for _ in range(4000))
    assert counts[MatchType.LEARNER_VS_LEARNER] / 4000 == pytest.approx(0.5, abs=0.04)
    assert counts[MatchType.LEARNER_VS_CHAMPION] / 4000 == pytest.approx(0.3, abs=0.04)
    assert counts[MatchType.MIXED] / 4000 == pytest.approx(0.2, abs=0.04)


def test_learner_vs_champion_seats_one_learner_against_champions():
    cfg = small_league_config(learners=4, champions=2, explorers=1)
    cfg.train.league_match_learner_vs_learner = 0.0
    cfg.train.league_match_learner_vs_champion = 1.0
    cfg.train.league_match_mixed = 0.0
    pop = LeaguePopulation(cfg, seed=0)
    learners, champions = set(pop.indices(Role.LEARNER)), set(pop.indices(Role.CHAMPION))

    for _ in range(50):
        seats, match = pop.sample_seats(3)
        assert match is MatchType.LEARNER_VS_CHAMPION
        assert sum(1 for s in seats if s in learners) == 1
        assert sum(1 for s in seats if s in champions) == 2


def test_seat_order_is_shuffled_so_roles_do_not_inherit_position():
    """A learner pinned to one seat would inherit that seat's positional edge."""
    cfg = small_league_config(learners=4, champions=2, explorers=1)
    cfg.train.league_match_learner_vs_learner = 0.0
    cfg.train.league_match_learner_vs_champion = 1.0
    cfg.train.league_match_mixed = 0.0
    pop = LeaguePopulation(cfg, seed=0)
    learners = set(pop.indices(Role.LEARNER))

    positions = {seat_index for _ in range(200)
                 for seat_index, member in enumerate(pop.sample_seats(3)[0])
                 if member in learners}
    assert positions == {0, 1, 2}


# --- running record from real play -----------------------------------------
def test_record_hand_builds_running_strength_and_pair_results():
    cfg = small_league_config(learners=2, champions=1, explorers=0)
    pop = LeaguePopulation(cfg, seed=0)
    pop.record_hand([0, 1, 2], [0.5, -0.25, -0.25])
    pop.record_hand([0, 1, 2], [0.1, -0.05, -0.05])

    assert pop.total_hands == 2
    assert pop.members[0].hands_played == 2
    assert pop.members[0].strength == pytest.approx(0.3)
    pair = pop.running_pair_mean()
    assert pair[0, 1] == pytest.approx(0.3)    # member 0's mean when 1 also sat
    assert pair[1, 0] == pytest.approx(-0.15)
    assert pop.running_coverage()[0] == 2      # positive against both opponents


def test_replacing_a_slot_forgets_its_record():
    cfg = small_league_config(learners=2, champions=1, explorers=0)
    pop = LeaguePopulation(cfg, seed=0)
    pop.record_hand([0, 1, 2], [0.5, -0.25, -0.25])
    scores = [1.0] * len(pop)
    scores[0] = -1.0

    assert pop.cull(scores) == 0
    assert pop.members[0].hands_played == 0   # a new network did not earn that record
    assert pop.running_pair_mean()[0, 1] == 0.0


# --- management ------------------------------------------------------------
def test_promoted_champion_is_frozen_and_independent_of_the_live_learner():
    import torch

    cfg = small_league_config(learners=2, champions=1, explorers=0)
    pop = LeaguePopulation(cfg, seed=0)
    best = pop.indices(Role.LEARNER)[1]

    slot = pop.promote(best, "elite")
    assert slot in pop.indices(Role.CHAMPION)
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


# --- champion gauntlet and earned promotion --------------------------------
def _beat(pop, learner, opponent, filler, value, times=5):
    """Record ``times`` hands where ``learner`` scores ``value`` against ``opponent``."""
    for _ in range(times):
        pop.record_hand([learner, opponent, filler], [value, -value / 2, -value / 2])


def test_gauntlet_flags_a_learner_that_only_beats_the_newest_champion():
    """Beating the newest champion while losing to the oldest = over-specialised."""
    cfg = small_league_config(learners=2, champions=2, explorers=0)
    pop = LeaguePopulation(cfg, seed=0)
    l0, l1 = pop.indices(Role.LEARNER)
    champions = pop.indices(Role.CHAMPION)
    pop.members[champions[0]].trained_at_hands = 0      # oldest generation
    pop.members[champions[1]].trained_at_hands = 100    # newest generation
    old, new = pop.champion_generations()

    _beat(pop, l0, old, l1, -0.30)   # loses badly to the old guard
    _beat(pop, l0, new, l1, +0.30)   # beats the current meta

    gauntlet = pop.champion_gauntlet()
    row = list(gauntlet["learners"]).index(l0)
    assert gauntlet["generations"] == [old, new]
    assert gauntlet["results"][row, 0] < 0 < gauntlet["results"][row, 1]
    assert gauntlet["beats"][row] == 1
    assert gauntlet["over_specialised"][row]


def test_promotion_gate_is_the_relative_median_not_beat_all():
    """Fix 1: beat the median champion faced, so a strong ladder can't lock."""
    cfg = small_league_config(learners=2, champions=2, explorers=0)
    cfg.train.league_promotion_hands = 10
    cfg.train.league_promotion_mbb_per_100 = 0.0
    cfg.train.league_promotion_min_generations = 0.5   # median
    cfg.train.league_promotion_reject_over_specialised = False  # isolate the median gate
    pop = LeaguePopulation(cfg, seed=0)
    l0, l1 = pop.indices(Role.LEARNER)
    c_old, c_new = pop.champion_generations()

    assert pop.promotion_candidates() == []            # no hands played yet

    for c in (c_old, c_new):
        _beat(pop, l0, c, l1, +0.20, times=10)
    assert l0 in pop.promotion_candidates()            # beats both

    # Beat exactly the median (1 of 2) while a net winner -> still qualifies.
    # The old "beat all generations" gate would have rejected this.
    pop._clear_stats(l0)
    _beat(pop, l0, c_old, l1, -0.10, times=10)         # small loss to one
    _beat(pop, l0, c_new, l1, +0.40, times=10)         # big win over the other
    assert l0 in pop.promotion_candidates()

    # Beat none -> below the median, fails.
    pop._clear_stats(l0)
    _beat(pop, l0, c_old, l1, -0.20, times=10)
    _beat(pop, l0, c_new, l1, -0.20, times=10)
    assert l0 not in pop.promotion_candidates()


def test_over_specialised_learner_is_rejected():
    """Fix 1 (anti-over-spec): beats the newest but loses to the oldest -> no promotion."""
    cfg = small_league_config(learners=2, champions=2, explorers=0)
    cfg.train.league_promotion_hands = 10
    cfg.train.league_promotion_mbb_per_100 = 0.0
    cfg.train.league_promotion_min_generations = 0.5
    cfg.train.league_promotion_reject_over_specialised = True
    pop = LeaguePopulation(cfg, seed=0)
    l0, l1 = pop.indices(Role.LEARNER)
    c_old, c_new = pop.champion_generations()
    pop.members[c_old].trained_at_hands = 0
    pop.members[c_new].trained_at_hands = 100

    _beat(pop, l0, c_new, l1, +0.20, times=10)   # beats newest
    _beat(pop, l0, c_old, l1, -0.20, times=10)   # loses to oldest -> over-specialised
    assert l0 not in pop.promotion_candidates()

    cfg.train.league_promotion_reject_over_specialised = False
    assert l0 in pop.promotion_candidates()      # same learner, gate off -> qualifies


def test_cull_grace_protects_a_freshly_reset_learner():
    """Fix 2: a just-culled learner is not re-culled every pass while it develops."""
    cfg = small_league_config(learners=3, champions=1, explorers=0)
    cfg.train.league_cull_grace_passes = 2
    pop = LeaguePopulation(cfg, seed=0)
    learners = pop.indices(Role.LEARNER)
    scores = [1.0] * len(pop)
    scores[learners[0]] = -1.0                   # learners[0] is always the weakest

    r1 = pop.manage(scores)                       # pass 1: culls the weakest
    assert r1.culled == learners[0]
    assert pop.members[learners[0]].reset_at_pass == 1

    r2 = pop.manage(scores)                        # pass 2: age 1 < grace -> protected
    assert r2.culled != learners[0]

    r3 = pop.manage(scores)                        # pass 3: age 2 >= grace -> eligible again
    assert r3.culled == learners[0]


def test_champion_retired_on_the_external_yardstick():
    """Fix 3: a champion catastrophic vs the heuristic is stale even if fine in-league."""
    cfg = small_league_config(learners=1, champions=2, explorers=0)
    cfg.train.league_champion_retire_vs_heuristic = -100.0
    cfg.train.league_champion_retire_mbb_per_100 = -1e12   # disable the in-league gate
    pop = LeaguePopulation(cfg, seed=0)
    c0, c1 = pop.indices(Role.CHAMPION)

    external = [float("nan")] * len(pop)
    external[c0] = -300.0    # catastrophic against the held-out heuristic
    external[c1] = +20.0     # fine

    assert pop.mark_stale_champions() == []               # nothing stale in-league
    stale = pop.mark_stale_champions(external)
    assert c0 in stale and c1 not in stale


def test_most_different_fires_even_when_no_strict_candidate():
    """Fix 4: diversity injection continues when the strict gate admits nobody."""
    cfg = small_league_config(learners=3, champions=2, explorers=0)
    cfg.train.league_promotion_hands = 1
    cfg.train.league_most_different_top_k = 3
    cfg.train.league_promotion_min_generations = 2.0   # impossible strict gate
    pop = LeaguePopulation(cfg, seed=0)
    learners = pop.indices(Role.LEARNER)
    for c in pop.champion_generations():
        for offset, learner in enumerate(learners):
            _beat(pop, learner, c, learners[(offset + 1) % 3], +0.10, times=2)

    assert pop.promotion_candidates() == []            # strict gate unmeetable
    diversity = [0.0] * len(pop)
    diversity[learners[2]] = 1.0
    report = pop.manage([0.5] * len(pop), diversity=diversity, cull_worst=False)

    reasons = {reason: learner for learner, _, reason in report.promoted}
    assert "elite" not in reasons                      # nothing cleared the strict gate
    assert reasons.get("most_different") == learners[2]  # ...but diversity still entered


def test_most_different_pool_respects_a_strength_floor():
    """Refinement: a near-random (weak) learner is never frozen for diversity."""
    cfg = small_league_config(learners=3, champions=2, explorers=0)
    cfg.train.league_promotion_hands = 1
    cfg.train.league_most_different_top_k = 3
    cfg.train.league_most_different_min_mbb_per_100 = 0.0   # must be net-positive in-league
    pop = LeaguePopulation(cfg, seed=0)
    l_strong, l_mid, l_weak = pop.indices(Role.LEARNER)
    champ = pop.champion_generations()[0]
    # strong/mid win in-league; the weak one loses (a stand-in for a fresh reset).
    # l_weak is the filler for the winners so it is dragged down, not l_mid.
    for c in pop.champion_generations():
        _beat(pop, l_strong, c, l_weak, +0.30, times=5)
        _beat(pop, l_mid, c, l_weak, +0.20, times=5)
        _beat(pop, l_weak, c, l_mid, -0.40, times=5)

    scores = [0.0] * len(pop)
    diversity = [0.0] * len(pop)
    diversity[l_weak] = 1.0                    # weakest is also the most diverse
    report = pop.manage(scores, diversity=diversity, cull_worst=False)

    reasons = {reason: learner for learner, _, reason in report.promoted}
    # The weak-but-diverse learner is below the floor, so it is NOT promoted...
    assert reasons.get("most_different") != l_weak
    # ...but a competent learner still fills the diversity slot.
    assert "most_different" in reasons


def test_eviction_removes_the_most_redundant_champion_not_the_oldest():
    """Refinement: keep distinct strategies, evict the redundant one."""
    cfg = small_league_config(learners=1, champions=3, explorers=0)
    pop = LeaguePopulation(cfg, seed=0)
    learner = pop.indices(Role.LEARNER)[0]
    c0, c1, c2 = pop.champion_generations()     # oldest -> newest
    pop.members[c0].trained_at_hands = 0        # oldest but (below) unique
    pop.members[c1].trained_at_hands = 10
    pop.members[c2].trained_at_hands = 20

    # No champion is stale, so eviction falls to the redundancy rule.
    diversity = [1.0] * len(pop)
    diversity[c0] = 0.9                          # the oldest is distinct -> keep it
    diversity[c1] = 0.1                          # the most redundant -> evict this
    diversity[c2] = 0.8

    slot = pop.promote(learner, "elite", diversity=diversity)
    assert slot == c1                            # redundant, not oldest (c0)

    # With the rule off, eviction reverts to oldest.
    cfg.train.league_evict_most_redundant = False
    pop2 = LeaguePopulation(cfg, seed=1)
    ln = pop2.indices(Role.LEARNER)[0]
    a, b, c = pop2.champion_generations()
    pop2.members[a].trained_at_hands = 0
    pop2.members[b].trained_at_hands = 10
    pop2.members[c].trained_at_hands = 20
    assert pop2.promote(ln, "elite", diversity=[0.0, 0.0, 0.0, 0.0, 0.0][:len(pop2)]) == a


def test_stale_champions_are_evicted_before_redundant_healthy_ones():
    """Stale-first still holds: a bad champion goes even if it is distinct."""
    cfg = small_league_config(learners=1, champions=3, explorers=0)
    pop = LeaguePopulation(cfg, seed=0)
    learner = pop.indices(Role.LEARNER)[0]
    c0, c1, c2 = pop.champion_generations()
    pop.members[c1].stale = True                 # one stale champion, and it is distinct
    diversity = [1.0] * len(pop)
    diversity[c1] = 0.9                           # high diversity, but stale
    diversity[c0] = 0.1                           # most redundant, but healthy

    assert pop.promote(learner, "elite", diversity=diversity) == c1   # stale wins


def test_within_stale_diversity_does_not_buy_survival():
    """A distinct-but-catastrophic champion is still evicted: evict oldest stale."""
    cfg = small_league_config(learners=1, champions=3, explorers=0)
    pop = LeaguePopulation(cfg, seed=0)
    learner = pop.indices(Role.LEARNER)[0]
    c0, c1, c2 = pop.champion_generations()
    pop.members[c0].trained_at_hands = 0        # oldest stale
    pop.members[c1].trained_at_hands = 10
    pop.members[c0].stale = True
    pop.members[c1].stale = True                 # two stale champions
    diversity = [0.0] * len(pop)
    diversity[c0] = 1.0                          # oldest stale is also the most diverse
    diversity[c1] = 0.0

    # Redundancy would keep c0 (diverse) and evict c1; the correct rule evicts
    # the oldest stale (c0) -- diversity does not protect a bad champion.
    assert pop.promote(learner, "elite", diversity=diversity) == c0


def test_most_different_panel_filter_excludes_externally_weak():
    """Panel gate: a diverse learner catastrophic vs standard opponents is dropped."""
    cfg = small_league_config(learners=2, champions=1, explorers=0)
    cfg.train.league_promotion_hands = 1
    cfg.train.league_most_different_top_k = 2
    cfg.train.league_most_different_min_panel_bb_per_100 = -100.0
    pop = LeaguePopulation(cfg, seed=0)
    a, b = pop.indices(Role.LEARNER)
    champ = pop.champion_generations()[0]
    _beat(pop, a, champ, b, +0.30, times=5)          # both net-positive in-league
    _beat(pop, b, champ, a, +0.30, times=5)

    scores = [0.0] * len(pop)
    shortlist = pop.most_different_shortlist(scores)
    assert a in shortlist and b in shortlist

    panel = {a: -500.0, b: +20.0}                    # a catastrophic, b fine
    pool = pop._most_different_pool(scores, panel_scores=panel)
    assert a not in pool and b in pool
    # No panel scores -> no filtering (backward compatible).
    assert set(pop._most_different_pool(scores, panel_scores=None)) == set(shortlist)


def test_evaluate_vs_panel_returns_finite_worst_case():
    from paradigm_a.model.network import build_network
    from paradigm_a.training.league_metrics import evaluate_vs_panel

    cfg = small_league_config()
    scores = evaluate_vs_panel(
        [build_network(cfg)], cfg, ("loose_passive", "calling_station"),
        40, ObservationEncoder(cfg.obs), seed=0,
    )
    assert scores.shape == (1,) and np.isfinite(scores[0])


def test_estimate_exploitability_runs_and_returns_bb_per_100():
    from paradigm_a.model.network import build_network
    from paradigm_a.training.exploitability import estimate_exploitability

    cfg = small_league_config()
    cfg.train.batch_size = 32
    cfg.train.min_buffer_before_training = 32
    value = estimate_exploitability(
        build_network(cfg), cfg, ObservationEncoder(cfg.obs),
        iters=3, hands_per_iter=20, eval_hands=60, seed=0,
    )
    assert isinstance(value, float) and np.isfinite(value)


def test_population_evaluate_is_deterministic_for_a_fixed_seed():
    """Fix 5: a fixed eval seed gives paired (repeatable) metrics."""
    cfg = small_league_config(learners=2, champions=1, explorers=1)
    pop = LeaguePopulation(cfg, seed=0)
    encoder = ObservationEncoder(cfg.obs)
    m1 = pop.evaluate(encoder, seed=777)
    m2 = pop.evaluate(encoder, seed=777)
    for key in ("strength", "diversity", "coverage"):
        np.testing.assert_allclose(m1[key], m2[key])


def test_stale_champions_are_marked_and_evicted_before_the_oldest():
    cfg = small_league_config(learners=2, champions=2, explorers=0)
    cfg.train.league_champion_min_hands = 1
    cfg.train.league_champion_retire_mbb_per_100 = -1.0
    pop = LeaguePopulation(cfg, seed=0)
    l0, l1 = pop.indices(Role.LEARNER)
    oldest, newest = pop.champion_generations()

    _beat(pop, l0, newest, l1, +0.30)   # the newest champion is being crushed
    _beat(pop, l0, oldest, l1, -0.30)   # the oldest still holds up

    stale = pop.mark_stale_champions()
    assert newest in stale and oldest not in stale
    # Staleness beats age: the newest slot is recycled even though it is younger.
    assert pop.promote(l0, "elite") == newest


def test_manage_reports_slots_whose_network_was_replaced():
    cfg = small_league_config(learners=3, champions=2, explorers=0)
    cfg.train.league_promotion_hands = 1
    cfg.train.league_promotion_mbb_per_100 = 0.0
    cfg.train.league_promotion_min_generations = 0.0
    pop = LeaguePopulation(cfg, seed=0)
    learners = pop.indices(Role.LEARNER)
    for champion in pop.champion_generations():
        _beat(pop, learners[0], champion, learners[1], +0.20, times=3)

    scores = list(np.linspace(0.0, 1.0, len(pop)))
    diversity = list(np.linspace(1.0, 0.0, len(pop)))
    report = pop.manage(scores, diversity=diversity)

    assert report.promoted, "a qualifying learner should have been promoted"
    for learner, slot, reason in report.promoted:
        assert reason in {"elite", "most_different"}
        assert slot in report.rebuilt_slots      # caller rebuilds every replaced slot
    if report.culled is not None:
        assert report.culled in report.rebuilt_slots
    assert report.describe()


def test_manage_promotes_both_an_elite_and_a_most_different_member():
    cfg = small_league_config(learners=3, champions=2, explorers=0)
    cfg.train.league_promotion_hands = 1
    cfg.train.league_promotion_mbb_per_100 = 0.0
    cfg.train.league_promotion_min_generations = 0.5
    cfg.train.league_most_different_top_k = 3
    pop = LeaguePopulation(cfg, seed=0)
    learners = pop.indices(Role.LEARNER)
    # learners[0] beats champions hardest (highest mbb -> elite); all qualify.
    for champion in pop.champion_generations():
        _beat(pop, learners[0], champion, learners[2], +0.30, times=3)
        _beat(pop, learners[1], champion, learners[2], +0.10, times=3)
        _beat(pop, learners[2], champion, learners[0], +0.10, times=3)

    scores = [0.0] * len(pop)
    scores[learners[0]], scores[learners[1]], scores[learners[2]] = 1.0, 0.9, 0.8
    diversity = [0.0] * len(pop)
    diversity[learners[1]] = 1.0                 # most distinct among the top-K
    report = pop.manage(scores, diversity=diversity, cull_worst=False)

    reasons = {reason: learner for learner, _, reason in report.promoted}
    assert reasons.get("elite") == learners[0]           # strongest strict candidate
    assert reasons.get("most_different") == learners[1]   # most diverse of the top-K


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


# --- default config sanity -------------------------------------------------
# Every test above overrides `league_promotion_hands`, which is precisely how a
# 2_000_000-hand default (31x more than a full run generates) shipped unnoticed
# and froze the champion ladder for every real `--league` run.  These two guard
# the *defaults* instead.
def test_default_promotion_gate_is_reachable_in_a_default_run():
    cfg = Config()
    # Hands a single member accrues per iteration: every hand seats
    # `num_players` members out of the whole league.
    league_size = (
        cfg.train.league_learners + cfg.train.league_champions + cfg.train.league_explorers
    )
    per_member_per_iter = cfg.train.hands_per_iteration * cfg.env.num_players / league_size
    iterations_to_gate = cfg.train.league_promotion_hands / per_member_per_iter

    # Reachable well inside a default-length run, or no learner is ever promoted
    # and the champion pool stays at its random initialisation forever.
    assert iterations_to_gate <= cfg.train.iterations / 2, (
        f"promotion gate needs ~{iterations_to_gate:.0f} iterations but a run is "
        f"{cfg.train.iterations}"
    )
    # And reachable within a few management passes, so the ladder actually rolls.
    assert iterations_to_gate <= 5 * cfg.train.league_manage_every


def test_default_explorer_reset_fires_during_a_default_run():
    cfg = Config()
    total_hands = cfg.train.hands_per_iteration * cfg.train.iterations
    assert cfg.train.league_explorer_reset_hands <= total_hands / 2, (
        "explorers would never be reinitialised; the explorer role would be inert"
    )


# --- anchored champions: veterans that keep training ------------------------
def anchored_league_config(learners=2, champions=3, anchors=2, explorers=0) -> Config:
    cfg = small_league_config(learners=learners, champions=champions, explorers=explorers)
    cfg.train.champion_mode = "anchored"
    cfg.train.league_champion_anchors = anchors
    return cfg


def test_frozen_is_the_default_so_no_champion_trains():
    assert Config().train.champion_mode == "frozen"


def test_anchored_mode_splits_champions_into_frozen_anchors_and_veterans():
    cfg = anchored_league_config(learners=2, champions=3, anchors=2)
    pop = LeaguePopulation(cfg, seed=0)

    assert len(pop.anchor_indices()) == 2 and len(pop.veteran_indices()) == 1
    # Anchors are opponents only, exactly as in frozen mode.
    for index in pop.anchor_indices():
        assert not pop.members[index].trainable
        assert all(not p.requires_grad for p in pop.members[index].network.parameters())
    # Veterans train, and carry a frozen snapshot to be anchored to.
    for index in pop.veteran_indices():
        member = pop.members[index]
        assert member.trainable and member.role is Role.CHAMPION
        assert all(p.requires_grad for p in member.network.parameters())
        assert all(not p.requires_grad for p in member.snapshot.parameters())
    assert sorted(pop.trainable_indices()) == sorted(
        pop.indices(Role.LEARNER) + pop.veteran_indices()
    )


def test_a_veterans_snapshot_does_not_move_when_the_veteran_trains():
    cfg = anchored_league_config()
    pop = LeaguePopulation(cfg, seed=0)
    veteran = pop.members[pop.veteran_indices()[0]]

    observation = np.zeros(veteran.network.spec.total_dim, dtype=np.float32)
    before = veteran.snapshot.infer(observation)[0].copy()
    with torch.no_grad():
        for parameter in veteran.network.parameters():
            parameter.add_(0.5)

    np.testing.assert_array_equal(before, veteran.snapshot.infer(observation)[0])
    assert not np.allclose(before, veteran.network.infer(observation)[0])


def test_the_gauntlet_reads_anchors_only():
    """A moving measuring stick measures nothing, so veterans stay out of it."""
    cfg = anchored_league_config(learners=2, champions=3, anchors=2)
    pop = LeaguePopulation(cfg, seed=0)

    assert pop.gauntlet_generations() == sorted(pop.anchor_indices())
    gauntlet = pop.champion_gauntlet()
    assert list(gauntlet["generations"]) == sorted(pop.anchor_indices())
    # ...and a veteran is a champion, so it is never a *row* either: it cannot be
    # promoted, and scoring it against the ladder would be meaningless.
    assert not set(gauntlet["learners"]) & set(pop.veteran_indices())
    assert pop.contender_indices() == pop.indices(Role.LEARNER) + pop.indices(Role.EXPLORER)


def test_frozen_mode_leaves_the_gauntlet_exactly_as_it_was():
    cfg = small_league_config(learners=2, champions=3, explorers=1)
    pop = LeaguePopulation(cfg, seed=0)
    assert pop.gauntlet_generations() == pop.champion_generations()
    assert pop.contender_indices() == pop.trainable_indices()
    assert pop.veteran_indices() == []


def test_promoting_into_a_veteran_slot_yields_a_trainable_copy_and_a_snapshot():
    cfg = anchored_league_config(learners=2, champions=1, anchors=0)
    pop = LeaguePopulation(cfg, seed=0)
    slot = pop.veteran_indices()[0]
    learner = pop.indices(Role.LEARNER)[0]

    assert pop.promote(learner) == slot
    member = pop.members[slot]
    assert member.role is Role.CHAMPION and member.trainable
    # The copy is independent of the learner it came from: the learner keeps
    # training, and must not drag the champion along with it.
    observation = np.zeros(member.network.spec.total_dim, dtype=np.float32)
    before = member.network.infer(observation)[0].copy()
    with torch.no_grad():
        for parameter in pop.members[learner].network.parameters():
            parameter.add_(0.5)
    np.testing.assert_array_equal(before, member.network.infer(observation)[0])
    # ...and it starts life exactly at its own anchor.
    np.testing.assert_array_equal(before, member.snapshot.infer(observation)[0])


def test_promoting_into_an_anchor_slot_still_hard_freezes():
    cfg = anchored_league_config(learners=2, champions=1, anchors=1)
    pop = LeaguePopulation(cfg, seed=0)
    slot = pop.promote(pop.indices(Role.LEARNER)[0])
    assert pop.members[slot].is_anchor
    assert not pop.members[slot].trainable
    assert all(not p.requires_grad for p in pop.members[slot].network.parameters())


def _validation_bank(pop, cfg, encoder_seed=0):
    from paradigm_a.training.league_metrics import sample_validation_states

    encoder = ObservationEncoder.from_config(cfg)
    return sample_validation_states(
        pop.agents, cfg, cfg.train.league_diversity_states, encoder, seed=encoder_seed
    )


def test_veteran_drift_is_zero_at_the_snapshot_and_grows_with_training():
    cfg = anchored_league_config()
    pop = LeaguePopulation(cfg, seed=0)
    observations, masks = _validation_bank(pop, cfg)
    slot = pop.veteran_indices()[0]

    drift = pop.veteran_drift(observations, masks)
    assert np.isnan(drift[pop.anchor_indices()[0]])       # only veterans have one
    assert drift[slot] == pytest.approx(0.0, abs=1e-6)    # starts at its anchor

    # Random noise, not a constant: adding the same scalar to every parameter
    # barely moves a softmax over logits, so it would not test anything.
    torch.manual_seed(0)
    with torch.no_grad():
        for parameter in pop.members[slot].network.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    assert pop.veteran_drift(observations, masks)[slot] > 0.0


def test_a_veteran_past_the_budget_is_reset_to_its_snapshot():
    cfg = anchored_league_config()
    cfg.train.league_veteran_max_kl = 0.01
    pop = LeaguePopulation(cfg, seed=0)
    observations, masks = _validation_bank(pop, cfg)
    slot = pop.veteran_indices()[0]

    # Random noise, not a constant: adding the same scalar to every parameter
    # barely moves a softmax over logits, so it would not test anything.
    torch.manual_seed(0)
    with torch.no_grad():
        for parameter in pop.members[slot].network.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    drift = pop.veteran_drift(observations, masks)
    assert drift[slot] > cfg.train.league_veteran_max_kl

    assert pop.enforce_veteran_drift(drift) == [slot]
    assert pop.veteran_drift(observations, masks)[slot] == pytest.approx(0.0, abs=1e-6)


def test_drift_enforcement_skips_slots_replaced_this_pass():
    """Their drift reading predates the network now sitting in the slot."""
    cfg = anchored_league_config()
    cfg.train.league_veteran_max_kl = 0.01
    pop = LeaguePopulation(cfg, seed=0)
    slot = pop.veteran_indices()[0]
    drift = np.full(len(pop), np.nan)
    drift[slot] = 10.0
    assert pop.enforce_veteran_drift(drift, skip=[slot]) == []
    assert pop.enforce_veteran_drift(drift) == [slot]


def test_a_zero_drift_budget_disables_enforcement():
    cfg = anchored_league_config()
    cfg.train.league_veteran_max_kl = 0.0
    pop = LeaguePopulation(cfg, seed=0)
    drift = np.full(len(pop), 99.0)
    assert pop.enforce_veteran_drift(drift) == []


def test_an_unknown_champion_mode_is_rejected():
    cfg = small_league_config()
    cfg.train.champion_mode = "trainable"
    with pytest.raises(ValueError, match="champion_mode"):
        LeaguePopulation(cfg, seed=0)
