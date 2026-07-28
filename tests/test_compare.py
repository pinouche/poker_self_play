"""The fixed-vs-iterative comparison harness.

Everything here runs at toy sizes.  The point is not that a 12-update student
learns anything — it cannot — but that the *harness* is honest: equal budgets
actually equal, shared weights actually shared, held-out boards actually held
out, and every artifact landing where the layout says it will.  Those are the
properties a real run's conclusions rest on, and they are cheap to assert.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from holdem.compare.common.budget import CountingLeafValues, SpendRecord
from holdem.compare.common.evaluation import (
    EvaluationConfig,
    all_held_out_boards,
    evaluate_agent,
    make_held_out_situations,
)
from holdem.compare.common.situations import StreetMix, sample_mixed_situation
from holdem.compare.common.storage import RunLayout, read_json, write_json
from holdem.compare.experiment import ComparisonConfig, run_comparison
from holdem.compare.fixed.build import DatasetBuildConfig, build_layered_dataset, slice_examples
from holdem.compare.fixed.store import DatasetStore
from holdem.compare.fixed.student import FixedStudentConfig, fit_fixed_student
from holdem.compare.iterative.journal import JournalEntry, TrajectoryJournal
from holdem.compare.iterative.student import OnlineStudentConfig, fit_online_student
from holdem.compare.relabel import decode_situation, measure_label_drift
from holdem.generation import GenerationConfig
from holdem.net import HoldemValueNet, HoldemValueNetConfig
from holdem.rebel import HoldemSelfPlayConfig
from holdem.sampling import SituationConfig, conflicts_with_held_out, held_out_boards

TINY_NET = HoldemValueNetConfig(hidden_dim=64, num_residual_blocks=1, card_embedding_dim=16)


def tiny_dataset_config(**overrides) -> DatasetBuildConfig:
    config = DatasetBuildConfig(
        river_examples=24,
        turn_examples=8,
        flop_examples=4,
        self_play_examples=0,
        teacher_updates=2,
        teacher_batch_size=8,
        generation=GenerationConfig(situations_per_board=4, cfr_iterations=3),
        bootstrap_cfr_iterations=3,
        situations_per_board=2,
        self_play=HoldemSelfPlayConfig(search_iterations=3, river_iterations=3),
        situations=SituationConfig(board_cards=4),
        value_net=TINY_NET,
        workers=2,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def tiny_comparison(run_path, **overrides) -> ComparisonConfig:
    config = ComparisonConfig(
        run_path=str(run_path),
        label_budget=32,
        update_budget=8,
        dataset=tiny_dataset_config(),
        fixed=FixedStudentConfig(label_budget=0, update_budget=0, batch_size=8),
        iterative=OnlineStudentConfig(
            trajectories_per_iteration=2,
            updates_per_iteration=4,
            batch_size=8,
            self_play=HoldemSelfPlayConfig(search_iterations=3, river_iterations=3),
            situations=SituationConfig(board_cards=4),
            checkpoint_every=1,
        ),
        evaluation=EvaluationConfig(
            situations=SituationConfig(board_cards=4),
            streets=(4, 5),
            boards_per_street=1,
            search_iterations=3,
            flop_tree_depth_limit=1,
        ),
        value_net=TINY_NET,
        seed=0,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# --- held-out boards leak across streets unless exclusion is subset-aware ---
def test_a_held_out_flop_also_excludes_the_turns_that_contain_it():
    flop = (2, 7, 30)
    assert conflicts_with_held_out((2, 7, 30, 45), [flop])
    assert conflicts_with_held_out((2, 7, 30, 45, 51), [flop])
    # And the other direction: a held-out turn excludes its own flop.
    assert conflicts_with_held_out(flop, [(2, 7, 30, 45)])
    # Sorting does not make a flop a prefix of its turn, so this must be a
    # set relation rather than a prefix one.
    assert conflicts_with_held_out((2, 7, 30, 45), [(7, 30, 45)])
    assert not conflicts_with_held_out((2, 7, 31), [flop])


def test_held_out_boards_do_not_collide_with_each_other():
    rng = np.random.default_rng(0)
    boards = held_out_boards(rng, 6, 4)
    for i, board in enumerate(boards):
        assert not conflicts_with_held_out(board, boards[:i] + boards[i + 1 :])


def test_held_out_situations_span_streets_without_overlapping():
    rng = np.random.default_rng(1)
    situations = make_held_out_situations(
        rng, SituationConfig(board_cards=4), streets=(3, 4, 5), boards_per_street=2
    )
    assert set(situations) == {3, 4, 5}
    for cards, street in situations.items():
        assert all(len(s.root.board) == cards for s in street)
    boards = all_held_out_boards(situations)
    assert len(boards) == 6
    for i, board in enumerate(boards):
        assert not conflicts_with_held_out(board, boards[:i] + boards[i + 1 :])


# --- the shared budget meter -----------------------------------------------
def test_counting_leaf_values_tallies_without_changing_the_answer():
    spend = SpendRecord()

    def inner(states):
        return np.ones((len(states), 2, 3))

    counting = CountingLeafValues(inner, spend)
    out = counting([object(), object()])
    assert out.shape == (2, 2, 3)
    assert spend.solver_calls == 1
    assert spend.leaf_evaluations == 2
    counting([object()])
    assert spend.solver_calls == 2
    assert spend.leaf_evaluations == 3


# --- arm 1: the artifact ----------------------------------------------------
def test_building_the_dataset_hits_exact_counts_and_saves_its_teachers(tmp_path):
    config = tiny_dataset_config()
    store, spend = build_layered_dataset(
        tmp_path / "dataset", config, teachers_path=tmp_path / "teachers"
    )
    counts = store.counts()
    assert counts["river"] == 24
    assert counts["turn"] == 8
    assert counts["flop"] == 4
    assert len(store) == config.total_examples == spend.labels

    # Two teachers, not three: each exists to price the street above it, and
    # nothing is bootstrapped from the flop.
    assert (tmp_path / "teachers" / "teacher-after-river.pt").exists()
    assert (tmp_path / "teachers" / "teacher-after-turn.pt").exists()
    assert not (tmp_path / "teachers" / "teacher-after-flop.pt").exists()

    stages = store.manifest.teacher_stages
    assert [s["after_source"] for s in stages] == ["river", "turn"]
    # And every bootstrapped street names the teacher that priced its leaves.
    assert store.manifest.generation["turn"]["teacher"] == "teacher-after-river.pt"
    assert store.manifest.generation["flop"]["teacher"] == "teacher-after-turn.pt"


def test_the_self_play_control_source_adds_the_third_teacher(tmp_path):
    """The one case where a teacher is fitted after the top street."""
    store, _ = build_layered_dataset(
        tmp_path / "dataset",
        tiny_dataset_config(self_play_examples=4),
        teachers_path=tmp_path / "teachers",
    )
    assert (tmp_path / "teachers" / "teacher-after-flop.pt").exists()
    assert [s["after_source"] for s in store.manifest.teacher_stages] == [
        "river",
        "turn",
        "flop",
    ]
    assert store.counts()["self_play"] == 4
    assert store.manifest.generation["self_play"]["teacher"] == "teacher-after-flop.pt"


def test_the_teacher_named_is_the_last_one_actually_fitted(tmp_path):
    """With no flop stage, self-play is priced by the turn teacher, and says so."""
    store, _ = build_layered_dataset(
        tmp_path / "dataset",
        tiny_dataset_config(flop_examples=0, self_play_examples=4),
        teachers_path=tmp_path / "teachers",
    )
    assert not (tmp_path / "teachers" / "teacher-after-flop.pt").exists()
    assert store.manifest.generation["self_play"]["teacher"] == "teacher-after-turn.pt"


def test_the_artifact_excludes_the_held_out_boards(tmp_path):
    rng = np.random.default_rng(3)
    tests = make_held_out_situations(
        rng, SituationConfig(board_cards=4), streets=(4,), boards_per_street=1
    )
    excluded = all_held_out_boards(tests)
    store, _ = build_layered_dataset(
        tmp_path / "dataset",
        tiny_dataset_config(flop_examples=0),
        excluded_boards=excluded,
        teachers_path=tmp_path / "teachers",
    )
    # Recover each stored board from its encoded features and check it against
    # the held-out set at every street, not just its own.
    for source in ("river", "turn"):
        examples = store.read_all(source)
        for row in range(len(examples)):
            _, public, _ = decode_situation(examples.features[row], examples.masks[row])
            assert not conflicts_with_held_out(public.board, excluded)


def test_the_fixed_student_never_touches_a_solver(tmp_path):
    store, _ = build_layered_dataset(
        tmp_path / "dataset", tiny_dataset_config(), teachers_path=tmp_path / "teachers"
    )
    net = HoldemValueNet(TINY_NET)
    result = fit_fixed_student(
        store,
        FixedStudentConfig(label_budget=30, update_budget=6, batch_size=8, value_net=TINY_NET),
        net=net,
        rng=np.random.default_rng(0),
    )
    assert result.spend.labels == 30
    assert result.spend.updates == 6
    # The offline arm's whole premise: no search happens during training.
    assert result.spend.leaf_evaluations == 0
    assert result.spend.solver_calls == 0


def test_the_fixed_student_draws_only_its_budget_from_a_larger_artifact(tmp_path):
    """The fairness claim: a budget smaller than the artifact really binds.

    The artifact is deliberately built larger than one student's budget so it
    can be reused.  If the student sampled from all of it while reporting
    ``label_budget`` spent, "both arms consumed the same number of labels"
    would be false by the difference.
    """
    from holdem.compare.fixed.student import _draw_budget_subset

    store, _ = build_layered_dataset(
        tmp_path / "dataset", tiny_dataset_config(), teachers_path=tmp_path / "teachers"
    )
    assert len(store) == 36  # 24 river + 8 turn + 4 flop
    subset = _draw_budget_subset(store, 12, None, np.random.default_rng(0))
    assert sum(len(rows) for rows in subset.values()) == 12
    # Drawn in the artifact's own proportions, and without repeats.
    assert len(subset["river"]) == 8  # 24/36 of 12
    for rows in subset.values():
        assert len(set(rows.tolist())) == len(rows)


def test_the_budget_subset_never_asks_a_source_for_more_than_it_holds(tmp_path):
    from holdem.compare.fixed.student import _draw_budget_subset

    store, _ = build_layered_dataset(
        tmp_path / "dataset", tiny_dataset_config(), teachers_path=tmp_path / "teachers"
    )
    # Weighted entirely onto the smallest source, which holds only 4.
    subset = _draw_budget_subset(
        store, 12, {"flop": 1.0, "river": 0.0001}, np.random.default_rng(0)
    )
    assert sum(len(rows) for rows in subset.values()) == 12
    assert len(subset["flop"]) <= store.counts()["flop"]


def test_gathering_specific_rows_matches_reading_them_all(tmp_path):
    store, _ = build_layered_dataset(
        tmp_path / "dataset", tiny_dataset_config(), teachers_path=tmp_path / "teachers"
    )
    everything = store.read_all("river")
    rows = np.array([0, 5, 17, 23])
    gathered = store.gather("river", rows)
    assert np.allclose(gathered.features, everything.features[np.sort(rows)])
    assert np.allclose(gathered.targets, everything.targets[np.sort(rows)])
    with pytest.raises(IndexError):
        store.gather("river", [999])


def test_a_label_budget_larger_than_the_artifact_is_refused(tmp_path):
    store, _ = build_layered_dataset(
        tmp_path / "dataset", tiny_dataset_config(), teachers_path=tmp_path / "teachers"
    )
    with pytest.raises(ValueError, match="label budget"):
        fit_fixed_student(
            store,
            FixedStudentConfig(label_budget=10_000, update_budget=4, value_net=TINY_NET),
            rng=np.random.default_rng(0),
        )


# --- arm 2: the online loop and its journal ---------------------------------
def test_the_online_student_spends_its_budgets_exactly(tmp_path):
    net = HoldemValueNet(TINY_NET)
    result = fit_online_student(
        OnlineStudentConfig(
            trajectories_per_iteration=2,
            updates_per_iteration=3,
            batch_size=8,
            self_play=HoldemSelfPlayConfig(search_iterations=3, river_iterations=3),
            situations=SituationConfig(board_cards=4),
            value_net=TINY_NET,
        ),
        label_budget=17,  # deliberately not a multiple of a trajectory's length
        update_budget=7,
        net=net,
        rng=np.random.default_rng(0),
        journal_path=tmp_path / "journal",
    )
    assert result.spend.labels == 17
    assert result.spend.updates == 7
    # Unlike the offline arm, this one is search all the way down.
    assert result.spend.leaf_evaluations > 0


def test_the_journal_records_every_label_with_its_iteration(tmp_path):
    net = HoldemValueNet(TINY_NET)
    result = fit_online_student(
        OnlineStudentConfig(
            trajectories_per_iteration=2,
            updates_per_iteration=3,
            batch_size=8,
            self_play=HoldemSelfPlayConfig(search_iterations=3, river_iterations=3),
            situations=SituationConfig(board_cards=4),
            value_net=TINY_NET,
        ),
        label_budget=20,
        update_budget=6,
        net=net,
        rng=np.random.default_rng(0),
        journal_path=tmp_path / "journal",
    )
    journal = TrajectoryJournal.open(tmp_path / "journal")
    assert len(journal) == result.spend.labels

    examples, meta = journal.read()
    assert len(examples) == result.spend.labels
    assert meta.shape == (result.spend.labels, 4)
    # Iterations are recorded in order and start at one.
    iterations = journal.iterations()
    assert iterations[0] == 1
    assert list(iterations) == sorted(iterations)
    # Every row carries a real postflop street.
    assert set(np.unique(meta[:, 3])).issubset({3, 4, 5})
    # A trajectory's steps are numbered from zero, descending the streets.
    first = meta[meta[:, 1] == meta[0, 1]]
    assert list(first[:, 2]) == list(range(len(first)))


def test_the_journal_can_be_read_one_iteration_at_a_time(tmp_path):
    journal = TrajectoryJournal(tmp_path / "journal")
    rng = np.random.default_rng(0)
    for iteration in (1, 2):
        journal.add(
            [
                JournalEntry(
                    features=rng.standard_normal(2852).astype(np.float32),
                    mask=np.ones(1326, dtype=np.float32),
                    values=np.zeros((2, 1326), dtype=np.float32),
                    iteration=iteration,
                    trajectory=iteration,
                    step=0,
                    board_cards=4,
                )
            ]
        )
        journal.flush(iteration)
    reopened = TrajectoryJournal.open(tmp_path / "journal")
    assert reopened.iterations() == (1, 2)
    examples, meta = reopened.read(iteration=2)
    assert len(examples) == 1
    assert int(meta[0, 0]) == 2


# --- the street mixture -----------------------------------------------------
def test_the_street_mix_follows_the_artifacts_composition():
    mix = StreetMix.from_counts({"river": 100, "turn": 50, "flop": 50})
    weights = mix.normalised()
    assert weights["river"] == pytest.approx(0.5)
    assert weights["turn"] == pytest.approx(0.25)
    rng = np.random.default_rng(0)
    drawn = [
        sample_mixed_situation(rng, SituationConfig(), mix)[3] for _ in range(60)
    ]
    assert set(drawn) == {3, 4, 5}


def test_a_street_mix_needs_at_least_one_positive_weight():
    with pytest.raises(ValueError, match="positive weight"):
        StreetMix(river=0, turn=0, flop=0).normalised()


# --- the staleness probe ----------------------------------------------------
def test_decoding_recovers_the_board_pot_and_ranges(tmp_path):
    store, _ = build_layered_dataset(
        tmp_path / "dataset",
        tiny_dataset_config(turn_examples=4, flop_examples=0),
        teachers_path=tmp_path / "teachers",
    )
    examples = store.read_all("turn")
    space, public, ranges = decode_situation(examples.features[0], examples.masks[0])
    assert len(public.board) == 4
    assert public.betting.starting_pot > 0
    assert ranges.shape == (2, 1326)
    assert np.all(ranges >= 0.0)
    assert ranges.sum(axis=1) == pytest.approx(np.ones(2), abs=1e-5)


def test_exactly_solved_river_labels_survive_relabelling(tmp_path):
    """The probe's own control: river labels are exact and must reproduce.

    A river root has no leaves, so the network is never consulted and the
    re-solve should return the stored label.  If this drifts, either
    ``decode_situation`` is not round-tripping the belief state or the probe is
    solving at a different budget than generation did — and every turn/flop
    number it reports would be measuring that instead of label staleness.
    """
    store, _ = build_layered_dataset(
        tmp_path / "dataset",
        tiny_dataset_config(turn_examples=4, flop_examples=0),
        teachers_path=tmp_path / "teachers",
    )
    net = HoldemValueNet(TINY_NET)
    drift = measure_label_drift(store, net, sample_size=4)
    assert drift["sources"]["river"]["relative_drift"] < 0.02
    # The default must be the count the labels were generated at, not a
    # constant: solving the same river spot at a different budget moves the
    # values and would be misreported as drift.
    assert drift["sources"]["river"]["cfr_iterations"] == 3


def test_relabelling_at_a_different_budget_is_not_the_control(tmp_path):
    """Guards the gotcha above: the river only reproduces at matched budgets."""
    store, _ = build_layered_dataset(
        tmp_path / "dataset",
        tiny_dataset_config(turn_examples=4, flop_examples=0),
        teachers_path=tmp_path / "teachers",
    )
    net = HoldemValueNet(TINY_NET)
    matched = measure_label_drift(store, net, sample_size=4)
    mismatched = measure_label_drift(store, net, sample_size=4, cfr_iterations=30)
    assert (
        mismatched["sources"]["river"]["relative_drift"]
        > matched["sources"]["river"]["relative_drift"]
    )


# --- the runner -------------------------------------------------------------
def test_a_comparison_holds_weights_budgets_and_boards_equal(tmp_path):
    result = run_comparison(tiny_comparison(tmp_path / "run"))

    assert result.weights_identical
    assert result.fixed.spend.labels == result.iterative.spend.labels == 32
    assert result.fixed.spend.updates == result.iterative.spend.updates == 8
    # Both arms scored on the same streets, and the aggregate is over streets.
    for arm in (result.fixed, result.iterative):
        assert {"turn", "river", "aggregate"} <= set(arm.per_street)
        assert np.isfinite(arm.aggregate_exploitability)


def test_a_comparison_writes_each_arm_into_its_own_folder(tmp_path):
    run = tmp_path / "run"
    run_comparison(tiny_comparison(run))
    layout = RunLayout.at(run)

    assert layout.config.exists()
    assert layout.results.exists()
    assert layout.held_out.exists()
    # Arm 1: the frozen labels and the teachers that made them.
    assert (layout.dataset / "manifest.json").exists()
    assert layout.teacher("river").exists()
    assert layout.fixed_student.exists()
    assert layout.fixed_history.exists()
    # Arm 2: the journal and the student over time.
    assert (layout.journal / "manifest.json").exists()
    assert layout.iterative_student.exists()
    assert layout.iterative_history.exists()
    assert list(layout.iterative_checkpoints.glob("student-iter-*.pt"))
    # Nothing from one arm leaks into the other's folder.
    assert not list(layout.fixed.glob("**/iter-*.npy"))
    assert not list(layout.iterative.glob("**/teacher-after-*.pt"))


def test_a_second_run_reuses_the_artifact_rather_than_rebuilding(tmp_path):
    run = tmp_path / "run"
    first = run_comparison(tiny_comparison(run))
    assert first.dataset_preparation.generation_seconds > 0.0

    second = run_comparison(tiny_comparison(run))
    # Reused, not regenerated: no generation time, same manifest.
    assert second.dataset_preparation.generation_seconds == 0.0
    assert second.dataset_manifest["sources"] == first.dataset_manifest["sources"]


def test_the_results_file_round_trips(tmp_path):
    from holdem.compare.experiment import ComparisonResult

    run = tmp_path / "run"
    result = run_comparison(tiny_comparison(run))
    reloaded = ComparisonResult.from_dict(read_json(RunLayout.at(run).results))
    assert reloaded.seed == result.seed
    assert reloaded.fixed.spend.labels == result.fixed.spend.labels
    assert reloaded.weights_identical == result.weights_identical
    assert reloaded.iterative.per_street == pytest.approx(result.iterative.per_street)


def test_budgets_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        run_comparison(tiny_comparison(tmp_path / "run", label_budget=0))


# --- storage ----------------------------------------------------------------
def test_writes_are_atomic_and_leave_no_temporary_files(tmp_path):
    path = tmp_path / "thing.json"
    write_json({"a": 1}, path)
    write_json({"a": 2}, path)
    assert read_json(path) == {"a": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["thing.json"]
