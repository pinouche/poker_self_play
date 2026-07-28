"""The shared runner: one call, two regimes, one comparable answer.

``holdem/curriculum.py`` can build a reusable layered dataset and fit a fixed
student from it without generating anything; ``holdem/training.py`` can fit an
online ReBeL student from supplied initial weights, held-out test situations
and explicit budgets.  Nothing yet ties them to the *same* initial weights,
the *same* held-out boards, and the *same* budgets, then reports the *same*
kind of measurement back for both.  That is what this module does.

The design deliberately keeps three things separate and explicit, because a
comparison that blurs any of them stops being a comparison:

* **One seed, one set of weights.**  A single :class:`~holdem.net.HoldemValueNet`
  is constructed once and its ``state_dict`` cloned into both students, so
  "started from the same place" is verifiable rather than assumed.
* **One held-out set.**  Flop, turn and river :class:`~holdem.training.TestSituation`
  objects are built once and handed, as the same Python objects, to both the
  dataset-exclusion machinery and both regimes' evaluation.
* **Equal, separately-accounted cost.**  Both trainers take exactly
  ``label_budget`` labelled examples and ``update_budget`` gradient updates;
  the offline regime's teacher-preparation time is measured but reported
  apart from student training time, because it is not a cost the online
  regime pays at all.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from holdem.curriculum import DatasetBuildConfig, FixedStudentConfig, build_layered_dataset, fit_fixed_student
from holdem.dataset import MANIFEST_FILENAME, DatasetStore
from holdem.fitting import StudentResult
from holdem.net import HoldemValueNet, HoldemValueNetConfig
from holdem.sampling import SituationConfig
from holdem.training import (
    RandomisedReBeLConfig,
    TestSituation,
    exploitability_on,
    fit_online_student,
    make_held_out_situations,
)

PathLike = Union[str, Path]
STREET_NAMES: Dict[int, str] = {3: "flop", 4: "turn", 5: "river"}


@dataclass
class DatasetBuildOptions:
    """Whether, and how, to (re)build the offline artifact at ``dataset_path``."""

    config: DatasetBuildConfig = field(default_factory=DatasetBuildConfig)
    # An existing, valid artifact is always reopened and reused; this must be
    # set explicitly to rebuild one, and it never rebuilds silently just
    # because something -- a config, a seed -- looks different from before.
    rebuild: bool = False


@dataclass
class OfflineConfig:
    """The fixed regime: build/reuse the artifact, then fit only from it."""

    dataset: DatasetBuildOptions = field(default_factory=DatasetBuildOptions)
    batch_size: int = 128
    learning_rate: float = 1e-3
    source_weights: Optional[Dict[str, float]] = None


@dataclass
class EvaluationConfig:
    """The held-out situations, and how strictly each street is scored."""

    situations: SituationConfig = field(default_factory=SituationConfig)
    streets: Tuple[int, ...] = (3, 4, 5)
    search_iterations: int = 40
    safe_resolving: bool = False
    # A flop-rooted exact best response walks two full undealt streets and is
    # prohibitively expensive; bounding it turns river-starting nodes into
    # leaves valued by the net itself.  ``None`` here (or for any street) asks
    # for the exact, unbounded tree instead.
    flop_tree_depth_limit: Optional[int] = 2
    device: str = "cpu"

    def tree_depth_limit(self, board_cards: int) -> Optional[int]:
        return self.flop_tree_depth_limit if board_cards == 3 else None


@dataclass
class ComparisonConfig:
    """Everything :func:`run_comparison` needs, and nothing it has to guess.

    ``label_budget`` and ``update_budget`` are spent identically by both
    regimes.  The fixed regime's artifact must contain exactly
    ``label_budget`` examples in total (river + turn + flop combined) --
    :func:`run_comparison` checks this before doing any training so a
    mis-sized ``offline.dataset.config`` fails fast rather than after the
    online regime has already spent its budget.
    """

    dataset_path: str
    label_budget: int
    update_budget: int
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    iterative: RandomisedReBeLConfig = field(default_factory=RandomisedReBeLConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    value_net: HoldemValueNetConfig = field(default_factory=HoldemValueNetConfig)
    seed: int = 0
    device: str = "cpu"


@dataclass
class RegimeResult:
    """What one regime spent, how long it took, and how it scored."""

    labels: int
    updates: int
    train_seconds: float
    per_street: Dict[str, float]
    aggregate_exploitability: float
    history: List[Dict[str, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "labels": self.labels,
            "updates": self.updates,
            "train_seconds": self.train_seconds,
            "per_street": dict(self.per_street),
            "aggregate_exploitability": self.aggregate_exploitability,
            "history": [dict(record) for record in self.history],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegimeResult":
        return cls(
            labels=int(data["labels"]),
            updates=int(data["updates"]),
            train_seconds=float(data["train_seconds"]),
            per_street={k: float(v) for k, v in data["per_street"].items()},
            aggregate_exploitability=float(data["aggregate_exploitability"]),
            history=[dict(record) for record in data["history"]],
        )


@dataclass
class ComparisonResult:
    """A JSON-serializable answer: no tensors, only measurements and provenance."""

    fixed: RegimeResult
    iterative: RegimeResult
    offline_preparation_seconds: float
    dataset_path: str
    dataset_manifest: Dict[str, Any]
    held_out_boards: Dict[str, Tuple[int, ...]]
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fixed": self.fixed.to_dict(),
            "iterative": self.iterative.to_dict(),
            "offline_preparation_seconds": self.offline_preparation_seconds,
            "dataset_path": self.dataset_path,
            "dataset_manifest": self.dataset_manifest,
            "held_out_boards": {k: list(v) for k, v in self.held_out_boards.items()},
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComparisonResult":
        return cls(
            fixed=RegimeResult.from_dict(data["fixed"]),
            iterative=RegimeResult.from_dict(data["iterative"]),
            offline_preparation_seconds=float(data["offline_preparation_seconds"]),
            dataset_path=str(data["dataset_path"]),
            dataset_manifest=dict(data["dataset_manifest"]),
            held_out_boards={
                k: tuple(int(c) for c in v) for k, v in data["held_out_boards"].items()
            },
            seed=int(data["seed"]),
        )


def _open_or_build_dataset(
    path: Path, options: DatasetBuildOptions, excluded_boards: Tuple[Tuple[int, ...], ...]
) -> Tuple[DatasetStore, float]:
    """Reuse a valid artifact at ``path``; only build one when told to.

    Rebuilding is never silent: it happens only when ``options.rebuild`` is
    set, and even then only after clearing whatever was at ``path``, since
    :meth:`~holdem.dataset.DatasetStore.create` refuses to write into a
    non-empty directory.  ``excluded_boards`` -- the held-out set -- is
    threaded into the build's generation config so a freshly built artifact
    can never contain the boards the comparison tests on; it has no effect
    when the artifact is merely reopened, since that data already exists.
    """
    manifest_path = path / MANIFEST_FILENAME
    start = time.perf_counter()
    if options.rebuild or not manifest_path.exists():
        if path.exists():
            shutil.rmtree(path)
        build_config = replace(
            options.config,
            generation=replace(options.config.generation, excluded_boards=excluded_boards),
        )
        store = build_layered_dataset(path, build_config)
    else:
        store = DatasetStore.open(path)
    elapsed = time.perf_counter() - start
    return store, elapsed


def evaluate_agent(
    net: HoldemValueNet,
    tests: Dict[int, TestSituation],
    evaluation: EvaluationConfig,
) -> Dict[str, float]:
    """Per-street exploitability of ``net`` as a continual-resolving agent."""
    net.eval()
    scores = {}
    for board_cards, situation in tests.items():
        name = STREET_NAMES.get(board_cards, str(board_cards))
        scores[name] = exploitability_on(
            net,
            situation,
            evaluation.search_iterations,
            evaluation.safe_resolving,
            evaluation.device,
            full_tree_depth_limit=evaluation.tree_depth_limit(board_cards),
        )
    return scores


def _clone_initial_net(config: HoldemValueNetConfig, state: Dict[str, torch.Tensor]) -> HoldemValueNet:
    net = HoldemValueNet(config)
    net.load_state_dict(state)
    return net


def run_comparison(
    config: ComparisonConfig,
    results_path: Optional[PathLike] = None,
    overwrite_results: bool = False,
) -> ComparisonResult:
    """Run the fixed and iterative regimes once, under identical conditions.

    When ``results_path`` is given, the returned :class:`ComparisonResult` is
    also persisted there as JSON (see :func:`save_results`); a caller that
    only wants the in-memory result can leave it ``None``.
    """
    if config.label_budget <= 0 or config.update_budget <= 0:
        raise ValueError("label_budget and update_budget must be positive")

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    # 1. The same held-out flop, turn and river situations, built once.
    tests = make_held_out_situations(
        rng, config.evaluation.situations, streets=config.evaluation.streets
    )
    held_out_boards = {
        STREET_NAMES.get(cards, str(cards)): situation.root.board
        for cards, situation in tests.items()
    }
    excluded_boards = tuple(situation.root.board for situation in tests.values())

    # 2. One seeded initial network, cloned byte-identically into both students.
    initial_net = HoldemValueNet(config.value_net)
    initial_state = {k: v.detach().clone() for k, v in initial_net.state_dict().items()}
    fixed_net = _clone_initial_net(config.value_net, initial_state)
    iterative_net = _clone_initial_net(config.value_net, initial_state)

    # 3. Build or reopen the persistent artifact, excluding the held-out set.
    dataset_path = Path(config.dataset_path)
    store, offline_preparation_seconds = _open_or_build_dataset(
        dataset_path, config.offline.dataset, excluded_boards
    )
    if len(store) != config.label_budget:
        raise ValueError(
            "the dataset artifact must contain exactly label_budget examples "
            f"(expected {config.label_budget}, found {len(store)}); configure "
            "offline.dataset.config's stage sizes to match, or rebuild"
        )

    # 4. Fixed offline student: no generation, budgets spent purely on the artifact.
    fixed_config = FixedStudentConfig(
        label_budget=config.label_budget,
        update_budget=config.update_budget,
        batch_size=config.offline.batch_size,
        learning_rate=config.offline.learning_rate,
        source_weights=config.offline.source_weights,
        value_net=config.value_net,
        seed=config.seed,
        device=config.device,
    )
    fixed_start = time.perf_counter()
    fixed_student: StudentResult = fit_fixed_student(
        store, fixed_config, net=fixed_net, rng=np.random.default_rng(config.seed)
    )
    fixed_seconds = time.perf_counter() - fixed_start

    # 5. Iterative online student: fresh trajectories every iteration, same budgets.
    iterative_config = replace(
        config.iterative,
        situations=replace(config.iterative.situations, excluded_boards=excluded_boards),
    )
    iterative_start = time.perf_counter()
    iterative_student: StudentResult = fit_online_student(
        tests=list(tests.values()),
        label_budget=config.label_budget,
        update_budget=config.update_budget,
        config=iterative_config,
        net=iterative_net,
        rng=np.random.default_rng(config.seed),
    )
    iterative_seconds = time.perf_counter() - iterative_start

    # 6. Both agents, scored on the identical test objects.
    fixed_scores = evaluate_agent(fixed_student.net, tests, config.evaluation)
    iterative_scores = evaluate_agent(iterative_student.net, tests, config.evaluation)

    result = ComparisonResult(
        fixed=RegimeResult(
            labels=fixed_student.labels,
            updates=fixed_student.updates,
            train_seconds=fixed_seconds,
            per_street=fixed_scores,
            aggregate_exploitability=float(np.mean(list(fixed_scores.values()))),
            history=fixed_student.history,
        ),
        iterative=RegimeResult(
            labels=iterative_student.labels,
            updates=iterative_student.updates,
            train_seconds=iterative_seconds,
            per_street=iterative_scores,
            aggregate_exploitability=float(np.mean(list(iterative_scores.values()))),
            history=iterative_student.history,
        ),
        offline_preparation_seconds=offline_preparation_seconds,
        dataset_path=str(dataset_path),
        dataset_manifest=store.manifest.to_dict(),
        held_out_boards=held_out_boards,
        seed=config.seed,
    )

    if results_path is not None:
        save_results(result, results_path, overwrite=overwrite_results)
    return result


def save_results(result: ComparisonResult, path: PathLike, overwrite: bool = False) -> Path:
    """Persist ``result`` as JSON, atomically, refusing to clobber by accident.

    The file is written to a temporary sibling and moved into place with
    :func:`os.replace`, so a reader never sees a partially-written file and a
    crash mid-write leaves the original (if any) untouched.  An existing file
    at ``path`` is left alone unless ``overwrite`` is explicitly ``True``.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"results file already exists: {path} (pass overwrite=True to replace it)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise
    return path


def load_results(path: PathLike) -> ComparisonResult:
    """Reload a :class:`ComparisonResult` persisted by :func:`save_results`."""
    return ComparisonResult.from_dict(json.loads(Path(path).read_text()))
