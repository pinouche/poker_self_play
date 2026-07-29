"""The runner: one call, two arms, one comparable answer.

Three things are held identical, and a comparison that blurs any of them stops
being a comparison:

* **One seed, one set of weights.**  A single :class:`~holdem.net.HoldemValueNet`
  is built once and its ``state_dict`` cloned into both students, so "started
  from the same place" is *verifiable* — both results carry an
  ``initial_state`` snapshot — rather than assumed.
* **One held-out set.**  Flop, turn and river situations are built once and
  handed, as the same Python objects, to the dataset builder's exclusion list
  and to both arms' evaluation.
* **Equal, separately-accounted cost.**  Both arms consume exactly
  ``label_budget`` labels and take exactly ``update_budget`` gradient steps.
  The offline arm's teacher preparation is measured but reported *apart* from
  student training, because it is not a cost the online arm pays at all — and
  because the artifact is meant to be amortised over many students, which is
  the claim being tested.  ``amortisation_break_even`` says how many students
  it would take before the offline route is cheaper end to end.

What the shape of the result means, since a single aggregate number hides it:
the **river should tie** (both arms solve it exactly, so it is a free check
that the pipeline works), the **turn should show a small gap**, and the **flop
the largest**, because flop labels are bootstrapped through two layers of
possibly-stale teacher.  A flat difference across all three streets is a signal
that something other than label staleness is driving the result — most likely
the input-distribution confound, which ``self_play_examples`` exists to control.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from paradigm_b.holdem.arms_common.budget import SpendRecord
from paradigm_b.holdem.arms_common.evaluation import (
    EvaluationConfig,
    TestSituation,
    all_held_out_boards,
    evaluate_agent,
    make_held_out_situations,
)
from paradigm_b.holdem.arms_common.fitting import StudentResult
from paradigm_b.holdem.arms_common.situations import STREET_NAMES, StreetMix
from paradigm_b.holdem.arms_common.storage import PathLike, RunLayout, save_checkpoint, write_json
from paradigm_b.holdem.arm1_fixed.build import DatasetBuildConfig, build_layered_dataset
from paradigm_b.holdem.arm1_fixed.store import MANIFEST_FILENAME, STREET_SOURCES, DatasetStore
from paradigm_b.holdem.arm1_fixed.student import FixedStudentConfig, fit_fixed_student
from paradigm_b.holdem.arm2_iterative.student import OnlineStudentConfig, fit_online_student
from paradigm_b.holdem.net.value_net import HoldemValueNet, HoldemValueNetConfig


@dataclass
class ComparisonConfig:
    """Everything :func:`run_comparison` needs, and nothing it has to guess."""

    run_path: str
    label_budget: int
    update_budget: int
    dataset: DatasetBuildConfig = field(default_factory=DatasetBuildConfig)
    # An existing, valid artifact is reused; this must be set to rebuild one.
    # It never rebuilds silently just because a config looks different.
    rebuild_dataset: bool = False
    fixed: FixedStudentConfig = field(
        default_factory=lambda: FixedStudentConfig(label_budget=0, update_budget=0)
    )
    iterative: OnlineStudentConfig = field(default_factory=OnlineStudentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    value_net: HoldemValueNetConfig = field(default_factory=HoldemValueNetConfig)
    # Make the online arm draw streets in the artifact's own proportions.
    match_street_mix: bool = True
    seed: int = 0
    device: str = "cpu"


@dataclass
class ArmResult:
    """What one arm spent, how long it took, and how it scored."""

    spend: SpendRecord
    per_street: Dict[str, float]
    aggregate_exploitability: float
    history: List[Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spend": self.spend.to_dict(),
            "per_street": dict(self.per_street),
            "aggregate_exploitability": self.aggregate_exploitability,
            "history": [dict(record) for record in self.history],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArmResult":
        return cls(
            spend=SpendRecord.from_dict(data["spend"]),
            per_street={k: float(v) for k, v in data["per_street"].items()},
            aggregate_exploitability=float(data["aggregate_exploitability"]),
            history=[dict(record) for record in data["history"]],
        )


@dataclass
class ComparisonResult:
    """A JSON-serialisable answer: no tensors, only measurements and provenance."""

    fixed: ArmResult
    iterative: ArmResult
    dataset_preparation: SpendRecord
    dataset_manifest: Dict[str, Any]
    held_out_boards: Dict[str, List[Tuple[int, ...]]]
    weights_identical: bool
    run_path: str
    seed: int

    @property
    def amortisation_break_even(self) -> float:
        """Students the artifact must serve before offline is cheaper overall.

        Below this many students the online arm wins on total wall clock; above
        it, the one-time cost of building the dataset has been paid off.
        """
        saved = self.iterative.spend.total_seconds - self.fixed.spend.total_seconds
        if saved <= 0:
            return float("inf")
        return self.dataset_preparation.total_seconds / saved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fixed": self.fixed.to_dict(),
            "iterative": self.iterative.to_dict(),
            "dataset_preparation": self.dataset_preparation.to_dict(),
            "dataset_manifest": self.dataset_manifest,
            "held_out_boards": {
                k: [list(b) for b in v] for k, v in self.held_out_boards.items()
            },
            "weights_identical": self.weights_identical,
            "amortisation_break_even": self.amortisation_break_even,
            "run_path": self.run_path,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComparisonResult":
        return cls(
            fixed=ArmResult.from_dict(data["fixed"]),
            iterative=ArmResult.from_dict(data["iterative"]),
            dataset_preparation=SpendRecord.from_dict(data["dataset_preparation"]),
            dataset_manifest=dict(data["dataset_manifest"]),
            held_out_boards={
                k: [tuple(int(c) for c in b) for b in v]
                for k, v in data["held_out_boards"].items()
            },
            weights_identical=bool(data["weights_identical"]),
            run_path=str(data["run_path"]),
            seed=int(data["seed"]),
        )


def run_comparison(
    config: ComparisonConfig, verbose: bool = False
) -> ComparisonResult:
    """Run both arms once, under identical conditions, and persist everything."""
    if config.label_budget <= 0 or config.update_budget <= 0:
        raise ValueError("label_budget and update_budget must be positive")

    layout = RunLayout.at(config.run_path).prepare()
    write_json(_config_to_dict(config), layout.config)

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    # 1. The same held-out situations, built once, used by everything.
    tests = make_held_out_situations(
        rng,
        config.evaluation.situations,
        streets=config.evaluation.streets,
        boards_per_street=config.evaluation.boards_per_street,
    )
    excluded = all_held_out_boards(tests)
    held_out = {
        STREET_NAMES.get(cards, str(cards)): [s.root.board for s in situations]
        for cards, situations in tests.items()
    }
    write_json(held_out, layout.held_out)
    if verbose:
        print(f"held-out boards: {held_out}", flush=True)

    # 2. One seeded initial network, cloned byte-identically into both students.
    initial = HoldemValueNet(config.value_net)
    initial_state = {k: v.detach().clone() for k, v in initial.state_dict().items()}
    fixed_net = _clone(config.value_net, initial_state)
    iterative_net = _clone(config.value_net, initial_state)

    # 3. Arm 1's artifact, built (or reused) with the held-out boards excluded.
    store, preparation = _open_or_build(layout, config, excluded, verbose)
    if len(store) < config.label_budget:
        raise ValueError(
            f"the artifact holds {len(store)} example(s) but label_budget is "
            f"{config.label_budget}; raise the dataset stage sizes or lower the budget"
        )

    # 4. Arm 1's student: no generation, budgets spent purely on the frozen file.
    if verbose:
        print("fitting the fixed student ...", flush=True)
    fixed_config = replace(
        config.fixed,
        label_budget=config.label_budget,
        update_budget=config.update_budget,
        value_net=config.value_net,
        seed=config.seed,
        device=config.device,
    )
    fixed_student = fit_fixed_student(
        store,
        fixed_config,
        net=fixed_net,
        rng=np.random.default_rng(config.seed),
        tests=tests,
        evaluation=config.evaluation,
    )
    save_checkpoint(fixed_student.net.state_dict(), layout.fixed_student)
    write_json(fixed_student.history, layout.fixed_history)

    # 5. Arm 2's student: fresh trajectories throughout, same budgets.
    if verbose:
        print("fitting the iterative student ...", flush=True)
    iterative_config = replace(
        config.iterative,
        situations=replace(config.iterative.situations, excluded_boards=excluded),
        value_net=config.value_net,
        seed=config.seed,
        device=config.device,
    )
    if config.match_street_mix:
        counts = {k: v for k, v in store.counts().items() if k in STREET_SOURCES}
        if counts:
            iterative_config = replace(
                iterative_config, street_mix=StreetMix.from_label_counts(counts)
            )
    iterative_student = fit_online_student(
        iterative_config,
        label_budget=config.label_budget,
        update_budget=config.update_budget,
        net=iterative_net,
        rng=np.random.default_rng(config.seed),
        tests=tests,
        evaluation=config.evaluation,
        journal_path=layout.journal,
        checkpoint_path=layout.iterative_checkpoints,
    )
    save_checkpoint(iterative_student.net.state_dict(), layout.iterative_student)
    write_json(iterative_student.history, layout.iterative_history)

    # 6. Both agents, scored on the identical test objects.
    if verbose:
        print("scoring both agents ...", flush=True)
    fixed_scores = evaluate_agent(fixed_student.net, tests, config.evaluation)
    iterative_scores = evaluate_agent(iterative_student.net, tests, config.evaluation)

    result = ComparisonResult(
        fixed=_arm_result(fixed_student, fixed_scores),
        iterative=_arm_result(iterative_student, iterative_scores),
        dataset_preparation=preparation,
        dataset_manifest=store.manifest.to_dict(),
        held_out_boards=held_out,
        weights_identical=_same_weights(
            fixed_student.initial_state, iterative_student.initial_state
        ),
        run_path=str(layout.root),
        seed=config.seed,
    )
    write_json(result.to_dict(), layout.results)
    if verbose:
        print(format_report(result), flush=True)
    return result


def _arm_result(student: StudentResult, scores: Dict[str, float]) -> ArmResult:
    return ArmResult(
        spend=student.spend,
        per_street=scores,
        aggregate_exploitability=float(scores.get("aggregate", float("nan"))),
        history=student.history,
    )


def _clone(config: HoldemValueNetConfig, state: Dict[str, torch.Tensor]) -> HoldemValueNet:
    net = HoldemValueNet(config)
    net.load_state_dict(state)
    return net


def _same_weights(left: Dict[str, torch.Tensor], right: Dict[str, torch.Tensor]) -> bool:
    if set(left) != set(right):
        return False
    return all(torch.equal(left[k].cpu(), right[k].cpu()) for k in left)


def _open_or_build(
    layout: RunLayout,
    config: ComparisonConfig,
    excluded: Tuple[Tuple[int, ...], ...],
    verbose: bool,
) -> Tuple[DatasetStore, SpendRecord]:
    """Reuse a valid artifact; build one only when told to, never silently."""
    manifest = layout.dataset / MANIFEST_FILENAME
    if manifest.exists() and not config.rebuild_dataset:
        if verbose:
            print(f"reusing the artifact at {layout.dataset}", flush=True)
        store = DatasetStore.open(layout.dataset)
        return store, SpendRecord(labels=len(store))

    if layout.dataset.exists():
        shutil.rmtree(layout.dataset)
    if verbose:
        print(f"building the artifact at {layout.dataset} ...", flush=True)
    return build_layered_dataset(
        layout.dataset,
        replace(config.dataset, seed=config.seed, device=config.device),
        excluded_boards=excluded,
        teachers_path=layout.teachers,
    )


def _config_to_dict(config: ComparisonConfig) -> Dict[str, Any]:
    try:
        return asdict(config)
    except TypeError:  # a nested config holding something asdict cannot walk
        return {"run_path": config.run_path, "seed": config.seed}


def format_report(result: ComparisonResult) -> str:
    """A short text table — the thing worth pasting into a write-up."""
    lines = [
        "",
        f"run: {result.run_path}   seed: {result.seed}   "
        f"identical initial weights: {result.weights_identical}",
        "",
        f"{'':<22}{'fixed':>14}{'iterative':>14}",
        "-" * 50,
    ]
    streets = [k for k in result.fixed.per_street if not k.endswith("_worst")]
    for street in streets:
        left = result.fixed.per_street.get(street, float("nan"))
        right = result.iterative.per_street.get(street, float("nan"))
        lines.append(f"{'exploitability ' + street:<22}{left:>14.5g}{right:>14.5g}")
    lines.append("-" * 50)
    for label, attribute in (
        ("labels", "labels"),
        ("updates", "updates"),
        ("leaf evaluations", "leaf_evaluations"),
        ("solver calls", "solver_calls"),
        ("seconds", "total_seconds"),
    ):
        left = getattr(result.fixed.spend, attribute)
        right = getattr(result.iterative.spend, attribute)
        lines.append(f"{label:<22}{left:>14.6g}{right:>14.6g}")
    lines.append("")
    lines.append(
        f"dataset preparation: {result.dataset_preparation.total_seconds:.1f}s "
        f"(excluded from the arms above; break-even at "
        f"{result.amortisation_break_even:.1f} students)"
    )
    return "\n".join(lines)
