"""The benchmark matrix definition and its reporting."""

import json
import os
import shutil
import tempfile

import pytest

from benchmark import (
    CONFIG_ORDER,
    CONFIGS,
    build_config,
    build_report,
    cell_dir,
    curves_from_archive,
    histories_path,
    replot,
)


def test_every_config_builds_and_is_internally_consistent():
    for name in CONFIGS:
        cfg = build_config(name, seed=0, iterations=5)
        assert cfg.train.iterations == 5
        assert cfg.train.alpha > 0
        # A tanh critic is only allowed when the reward is genuinely bounded.
        from config import reward_bound, resolve_q_head

        bounded, scale = resolve_q_head(cfg)
        assert bounded == (reward_bound(cfg.env) is not None)
        if bounded:
            assert scale == reward_bound(cfg.env)


def test_unknown_config_is_rejected():
    with pytest.raises(SystemExit):
        build_config("no_such_config", seed=0, iterations=1)


def test_ablations_differ_from_full_in_exactly_one_respect():
    """Each ablation must be attributable: one change relative to `full`."""
    full = build_config("full", 0, 1)

    def differences(name):
        other = build_config(name, 0, 1)
        changed = set()
        for section in ("env", "obs", "train"):
            a, b = getattr(full, section).__dict__, getattr(other, section).__dict__
            changed |= {k for k in a if a[k] != b[k]}
        return changed

    assert differences("no_equity") == {"use_equity_feature"}
    assert differences("no_ev_runout") == {"all_in_ev_runout"}
    assert differences("no_pool") == {"opponent_pool", "opponent_mix_prob"}
    assert differences("alpha_0.10") == {"alpha"}
    assert differences("alpha_0.02") == {"alpha"}
    assert differences("fine_abstraction") == {"bet_fractions", "raise_multipliers"}


def test_fine_abstraction_actually_widens_the_action_space():
    from environment.state import action_space_for

    assert action_space_for(build_config("full", 0, 1).env).num_actions == 10
    assert action_space_for(build_config("fine_abstraction", 0, 1).env).num_actions == 14


def test_legacy_config_reproduces_the_original_settings():
    cfg = build_config("legacy", 0, 1)
    assert cfg.train.alpha == 0.5
    assert cfg.train.opponent_pool == ()
    assert cfg.obs.use_equity_feature is False
    assert cfg.env.all_in_ev_runout is False
    assert cfg.train.self_play_envs == 1


def test_config_order_covers_every_config():
    assert set(CONFIG_ORDER) == set(CONFIGS)
    assert CONFIG_ORDER[0] == "full", "ablations are reported as deltas from full"


def test_report_aggregates_seeds_and_computes_deltas():
    with tempfile.TemporaryDirectory() as root:
        for name, value in (("full", 100.0), ("no_equity", 40.0)):
            for seed in (0, 1):
                path = cell_dir(root, name, seed)
                os.makedirs(path)
                results = {
                    opponent: {
                        "bb_per_100": value,
                        "ci_half_width": 10.0,
                        "significant": True,
                        "deals": 600,
                    }
                    for opponent in ("random", "calling_station", "heuristic")
                }
                with open(os.path.join(path, "results.json"), "w") as fh:
                    json.dump(results, fh)

        out = os.path.join(root, "report")
        build_report(root, out)

        with open(f"{out}.json") as fh:
            payload = json.load(fh)
        with open(f"{out}.md") as fh:
            report_text = fh.read()

    assert payload["summary"]["full"]["random"] == pytest.approx(100.0)
    assert payload["summary"]["no_equity"]["random"] == pytest.approx(40.0)
    assert len(payload["cells"]["full"]) == 2
    # Ablations are reported as deltas from `full`.
    assert "-60" in report_text or "−60" in report_text


def test_report_survives_missing_cells():
    with tempfile.TemporaryDirectory() as root:
        path = cell_dir(root, "full", 0)
        os.makedirs(path)
        with open(os.path.join(path, "results.json"), "w") as fh:
            json.dump({o: {"bb_per_100": 1.0, "ci_half_width": 1.0} for o in
                       ("random", "calling_station", "heuristic")}, fh)
        build_report(root, os.path.join(root, "report"))  # must not raise


# --- history archiving -----------------------------------------------------
def write_cell(root, name, seed, iterations=12):
    """A cell directory with both of the files `report` reads."""
    path = cell_dir(root, name, seed)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "results.json"), "w") as fh:
        json.dump({o: {"bb_per_100": 10.0, "ci_half_width": 1.0, "significant": True,
                       "deals": 600} for o in
                   ("random", "calling_station", "heuristic")}, fh)
    history = [
        {"iteration": i, "entropy": 1.1, "q_loss": 0.5 * (0.8**i),
         "kl_target_vs_policy": 0.02 * (0.8**i), "showdown_rate": 0.5}
        for i in range(1, iterations + 1)
    ]
    with open(os.path.join(path, "history.json"), "w") as fh:
        json.dump(history, fh)


def test_report_archives_the_history_of_every_seed():
    with tempfile.TemporaryDirectory() as root:
        for seed in (0, 1):
            write_cell(root, "full", seed)
        out = os.path.join(root, "report")
        build_report(root, out)
        with open(histories_path(out)) as fh:
            archive = json.load(fh)

    # Only seed 0 is plotted, but every seed is kept -- the run directories are
    # deleted routinely and re-training a cell to recover a curve costs hours.
    assert set(archive) == {"full__seed0", "full__seed1"}
    assert len(archive["full__seed0"]) == 12
    assert curves_from_archive(archive) == {"full": archive["full__seed0"]}


def test_figures_can_be_redrawn_after_the_run_directories_are_gone():
    with tempfile.TemporaryDirectory() as keep:
        with tempfile.TemporaryDirectory() as root:
            write_cell(root, "full", 0)
            build_report(root, os.path.join(root, "report"))
            for suffix in (".json", ".md", "_histories.json"):
                shutil.copy(os.path.join(root, f"report{suffix}"),
                            os.path.join(keep, f"report{suffix}"))
        # `root` is gone here, and the figures went with it: only the tables
        # and the archived histories survive.
        prefix = os.path.join(keep, "report")
        assert not os.path.exists(os.path.join(keep, "report_curves.png"))
        replot(prefix, prefix)
        assert os.path.getsize(os.path.join(keep, "report_curves.png")) > 0
        assert os.path.getsize(os.path.join(keep, "report_results.png")) > 0


def test_replot_skips_the_curves_for_a_report_that_predates_archiving():
    with tempfile.TemporaryDirectory() as directory:
        prefix = os.path.join(directory, "report")
        with open(f"{prefix}.json", "w") as fh:
            json.dump({"summary": {"full": {"random": 1.0, "calling_station": 1.0,
                                            "heuristic": 1.0}},
                       "cells": {"full": [{"seed": 0, **{o: {"bb_per_100": 1.0,
                                                            "ci_half_width": 1.0}
                                                         for o in ("random",
                                                                   "calling_station",
                                                                   "heuristic")}}]}}, fh)
        replot(prefix, prefix)  # must not raise
        assert os.path.exists(f"{prefix}_results.png")
        assert not os.path.exists(f"{prefix}_curves.png")


def test_replot_reports_when_there_is_nothing_to_draw():
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(SystemExit):
            replot(os.path.join(directory, "report"), os.path.join(directory, "report"))


# --- alpha sweep reference -------------------------------------------------
def test_sweep_configs_hold_out_every_evaluation_baseline():
    """The pool must not contain an opponent we score against.

    The first benchmark seated `calling_station` and heuristic-family bots in
    the training pool while also using them as baselines, so those columns were
    partly training on the test opponent.
    """
    from benchmark import SWEEP_ALPHAS

    for alpha in SWEEP_ALPHAS:
        cfg = build_config(f"sweep_a{alpha:g}", 0, 1)
        assert cfg.train.opponent_pool == ("checkpoint",)
        for baseline in ("random", "calling_station", "heuristic", "tight_aggressive",
                         "loose_passive"):
            assert baseline not in cfg.train.opponent_pool


def test_sweep_configs_apply_the_benchmark_conclusions():
    from environment.state import action_space_for

    cfg = build_config("sweep_a0.03", 0, 1)
    assert cfg.obs.use_equity_feature is False       # not demonstrated, costly
    assert cfg.env.all_in_ev_runout is True          # clearest single win
    assert action_space_for(cfg.env).num_actions == 14   # finer abstraction kept
    assert cfg.train.alpha == 0.03


def test_sweep_alphas_differ_only_in_alpha():
    from benchmark import SWEEP_ALPHAS

    reference = build_config(f"sweep_a{SWEEP_ALPHAS[0]:g}", 0, 1)
    for alpha in SWEEP_ALPHAS[1:]:
        other = build_config(f"sweep_a{alpha:g}", 0, 1)
        changed = set()
        for section in ("env", "obs", "train"):
            a, b = getattr(reference, section).__dict__, getattr(other, section).__dict__
            changed |= {k for k in a if a[k] != b[k]}
        assert changed == {"alpha"}


# --- capacity and league configs -------------------------------------------
def test_capacity_configs_vary_only_the_trunk():
    from benchmark import CAPACITIES, SETTLED_ALPHA

    reference = build_config("cap_base", 0, 1)
    for size in CAPACITIES:
        cfg = build_config(f"cap_{size}", 0, 1)
        assert cfg.train.alpha == SETTLED_ALPHA
        for section in ("env", "obs", "train"):
            assert getattr(cfg, section).__dict__ == getattr(reference, section).__dict__


def test_capacity_sizes_are_strictly_increasing():
    from model.network import build_network

    counts = [
        build_network(build_config(f"cap_{size}", 0, 1)).num_parameters()
        for size in ("small", "base", "large", "xlarge")
    ]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1] / 10  # a genuine range, not a token change


def test_cap_base_matches_the_settled_sweep_configuration():
    """`cap_base` must be the incumbent, or the sweep is not a baseline."""
    base = build_config("cap_base", 0, 1)
    settled = build_config("sweep_a0.02", 0, 1)
    for section in ("env", "obs", "train"):
        assert getattr(base, section).__dict__ == getattr(settled, section).__dict__
    assert base.model.__dict__ == settled.model.__dict__


def test_long_configs_reuse_the_capacity_settings():
    for long_name, cap_name in (("long_base", "cap_base"), ("long_large", "cap_large")):
        long_cfg = build_config(long_name, 0, 1)
        cap_cfg = build_config(cap_name, 0, 1)
        assert long_cfg.model.__dict__ == cap_cfg.model.__dict__
        for section in ("env", "obs", "train"):
            assert getattr(long_cfg, section).__dict__ == getattr(cap_cfg, section).__dict__


def test_broad_league_deepens_the_league_without_adding_fixed_bots():
    base = build_config("cap_base", 0, 1)
    broad = build_config("league_broad", 0, 1)

    # Still self-snapshots only: adding a fixed bot would reintroduce the
    # contamination the alpha sweep removed.
    assert broad.train.opponent_pool == ("checkpoint",)
    assert broad.train.league_size > base.train.league_size
    assert broad.train.league_snapshot_every < base.train.league_snapshot_every
    assert broad.train.opponent_mix_prob > base.train.opponent_mix_prob


# --- population config ------------------------------------------------------
def test_population_config_enables_population_self_play():
    cfg = build_config("population", 0, 1)
    assert cfg.train.population_self_play is True
    assert cfg.train.league_weighting == "linear"
    # No fixed bot in the library -- all baselines stay held out.
    assert cfg.train.opponent_pool == ("checkpoint",) or not any(
        b in cfg.train.opponent_pool
        for b in ("random", "calling_station", "heuristic", "tight_aggressive")
    )
    assert cfg.env.all_in_ev_runout is True  # inherits the settled setup
    assert cfg.train.alpha == 0.02


def test_population_config_keeps_the_finer_abstraction():
    from environment.state import action_space_for

    assert action_space_for(build_config("population", 0, 1).env).num_actions == 14
