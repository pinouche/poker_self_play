"""The staleness probe: how far the frozen labels drifted from the truth.

The training comparison is indirect.  It tells you which student ended up
better, not *why*.  This asks the question directly, and in the form it was
originally posed:

    How close are the labels in that dataset to the labels I would generate
    later with a much stronger evaluator?

The procedure is cheap and completely mechanical.  Take the artifact's stored
**inputs** — the same belief states, untouched — and re-solve them with a
**stronger** network at the leaves.  Compare the new labels against the frozen
ones.  Whatever the turn and flop labels moved by is the answer, and it costs a
fraction of a paired training run.

**The river is the control.**  A river root has no leaves — the tree runs
straight to real showdowns — so the network is not consulted at all and
re-solving must reproduce the stored label.  A river drift that is not ~0 means
something upstream is broken and none of the other numbers can be trusted,
which makes this probe a free validation of the whole pipeline as well as a
measurement.

**Re-solve at the iteration count the labels were generated with.**  "Exact"
means exact terminal values, not a converged strategy: DCFR at 3 iterations and
at 6 produces measurably different root values on the same river spot, and
relabelling at a different count reports that gap as drift.  The iteration count
is therefore taken from the manifest per source by default, so the control
measures the labels rather than the solver budget.  Overriding
``cfr_iterations`` compares against a *better-solved* version instead, which is
a different and also useful question — just not the control.

Read the result like this:

* **drift ~ 0 on every street** — the labels were already what a stronger
  evaluator would produce.  Freezing them costs nothing; the dataset is a
  genuinely reusable asset and the offline arm should match the online one.
* **drift growing river -> turn -> flop** — bootstrapping compounded the
  teacher's error upward, exactly as the theory predicts, and regenerating the
  upper streets is worth paying for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from paradigm_b.holdem.engine.betting import Betting
from paradigm_b.holdem.engine.combos import NUM_COMBOS, board_mask
from paradigm_b.holdem.arms_common.budget import CountingLeafValues, SpendRecord
from paradigm_b.holdem.arms_common.storage import PathLike, write_json
from paradigm_b.holdem.arm1_fixed.store import STREET_SOURCES, DatasetStore
from paradigm_b.holdem.net.features import (
    BOARD_OFFSET,
    CARD_SET_DIM,
    GLOBAL_CHIP_SCALE,
    POT_CHIPS_INDEX,
    RANGE_DIM,
    SCALAR_OFFSET,
)
from paradigm_b.holdem.net.value_net import HoldemValueNet
from paradigm_b.holdem.engine.public_tree import PublicState, build_endgame_tree
from paradigm_b.holdem.engine.space import EndgameSpace
from paradigm_b.holdem.net.leaf_values import NetLeafValues
from paradigm_b.core.search.subgame import SubgameSolver
from paradigm_b.core.cfr.tabular_cfr import CFRConfig

NUM_PLAYERS = 2


@dataclass
class DriftResult:
    """How far one source's stored labels sit from freshly-solved ones."""

    source: str
    examples: int
    mean_absolute_drift: float
    median_absolute_drift: float
    max_absolute_drift: float
    relative_drift: float  # drift as a fraction of the stored labels' own scale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "examples": self.examples,
            "mean_absolute_drift": self.mean_absolute_drift,
            "median_absolute_drift": self.median_absolute_drift,
            "max_absolute_drift": self.max_absolute_drift,
            "relative_drift": self.relative_drift,
        }


def decode_situation(
    features: np.ndarray, mask: np.ndarray
) -> Tuple[EndgameSpace, PublicState, np.ndarray]:
    """Recover the belief state an encoded row came from.

    The encoder is lossless for everything a re-solve needs: both ranges sit in
    the first ``RANGE_DIM`` entries, the board is three one-hot card sets, and
    the pot and stack are stored in absolute chips on a global scale precisely
    so they survive this round trip.
    """
    ranges = np.asarray(features[:RANGE_DIM], dtype=np.float64).reshape(
        NUM_PLAYERS, NUM_COMBOS
    )
    board: List[int] = []
    for set_index in range(3):
        offset = BOARD_OFFSET + set_index * CARD_SET_DIM
        board.extend(int(c) for c in np.flatnonzero(features[offset : offset + CARD_SET_DIM]))
    board = tuple(sorted(board))

    pot = int(round(float(features[POT_CHIPS_INDEX]) * GLOBAL_CHIP_SCALE))
    stack = int(round(float(features[POT_CHIPS_INDEX + 1]) * GLOBAL_CHIP_SCALE))
    betting = Betting(
        starting_pot=max(pot, 1),
        stack=max(stack, 2),
        max_raises=1,
        num_rounds=max(6 - len(board), 1),
    )
    return EndgameSpace(board), PublicState(betting=betting, board=board), ranges


def _normalise(values: np.ndarray, reach: np.ndarray, correction: float):
    masses = reach.sum(axis=1)
    if masses.min() <= 0.0:
        return None
    out = np.empty_like(values)
    out[0] = values[0] / (correction * masses[1])
    out[1] = values[1] / (correction * masses[0])
    return out


def generated_cfr_iterations(store: DatasetStore, source: str, default: int = 40) -> int:
    """The iteration count ``source`` was originally solved at, per the manifest."""
    recorded = store.manifest.generation.get(source, {}).get("cfr_iterations")
    return int(recorded) if recorded else default


def relabel_source(
    store: DatasetStore,
    source: str,
    evaluator,
    sample_size: int,
    rng: np.random.Generator,
    cfr_iterations: Optional[int] = None,
    cfr: Optional[CFRConfig] = None,
) -> Optional[DriftResult]:
    """Re-solve a random sample of ``source`` and measure how far its labels moved.

    ``cfr_iterations`` defaults to whatever the manifest says this source was
    generated with; see the module docstring for why that matters.
    """
    counts = store.counts()
    if not counts.get(source):
        return None
    if cfr_iterations is None:
        cfr_iterations = generated_cfr_iterations(store, source)
    stored = store.read_all(source)
    take = min(sample_size, len(stored))
    index = rng.choice(len(stored), size=take, replace=False)

    drifts: List[np.ndarray] = []
    scales: List[float] = []
    for row in index:
        features = stored.features[row]
        mask = stored.masks[row]
        space, public, reach = decode_situation(features, mask)
        tree = build_endgame_tree(public, depth_limit=1)
        solver = SubgameSolver(
            tree,
            leaf_value_fn=evaluator if tree.leaves() else None,
            config=cfr or CFRConfig.dcfr(),
            space=space,
        )
        solver.solve(reach=reach, iterations=cfr_iterations)
        fresh = _normalise(solver.root_values(), reach, space.pair_correction)
        if fresh is None:
            continue
        fresh = fresh * mask
        drifts.append(np.abs(fresh - stored.targets[row]))
        scales.append(float(np.abs(stored.targets[row]).mean()))

    if not drifts:
        return None
    stacked = np.concatenate([d.reshape(-1) for d in drifts])
    scale = float(np.mean(scales)) or 1.0
    return DriftResult(
        source=source,
        examples=len(drifts),
        mean_absolute_drift=float(stacked.mean()),
        median_absolute_drift=float(np.median(stacked)),
        max_absolute_drift=float(stacked.max()),
        relative_drift=float(stacked.mean() / scale),
    )


def measure_label_drift(
    store: DatasetStore,
    net: HoldemValueNet,
    sample_size: int = 64,
    seed: int = 0,
    cfr_iterations: Optional[int] = None,
    device: str = "cpu",
    results_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Drift of every street's stored labels against ``net`` as the evaluator.

    ``net`` should be the *strongest* network available — normally the winning
    arm's finished student, which is by construction a better evaluator than
    the teacher that wrote the frozen labels.  ``cfr_iterations`` defaults to
    each source's own recorded count, which is what makes the river a control.
    """
    rng = np.random.default_rng(seed)
    spend = SpendRecord()
    evaluator = CountingLeafValues(NetLeafValues(net, device=device), spend)

    results: Dict[str, Any] = {"sample_size": sample_size, "sources": {}}
    for source in STREET_SOURCES:
        iterations = cfr_iterations or generated_cfr_iterations(store, source)
        drift = relabel_source(
            store, source, evaluator, sample_size, rng, iterations
        )
        if drift is not None:
            entry = drift.to_dict()
            entry["cfr_iterations"] = iterations
            results["sources"][source] = entry
    results["spend"] = spend.to_dict()
    results["verdict"] = _verdict(results["sources"])
    if results_path is not None:
        write_json(results, results_path)
    return results


def _verdict(sources: Dict[str, Any]) -> str:
    """Plain-language reading of the drift profile."""
    river = sources.get("river", {}).get("relative_drift")
    if river is not None and river > 0.05:
        return (
            "river labels moved, but they were solved exactly and must not — "
            "the pipeline is inconsistent and the other numbers are unreliable"
        )
    upper = [
        sources[s]["relative_drift"] for s in ("turn", "flop") if s in sources
    ]
    if not upper:
        return "no bootstrapped sources to measure"
    if max(upper) < 0.05:
        return (
            "bootstrapped labels are within 5% of freshly-solved ones: freezing "
            "them costs little and the dataset is a reusable asset"
        )
    return (
        f"bootstrapped labels drift by up to {max(upper):.1%} against a stronger "
        "evaluator: regenerating the upper streets is worth paying for"
    )
