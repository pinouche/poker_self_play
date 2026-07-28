"""Arm 1, step two: train a fresh student on the frozen file, and nothing else.

No solver runs here.  No teacher is consulted.  The student sees only the
labels that were written to disk once, however stale they may have become —
that is the regime, and it is what makes this a test of the claim *"a labelled
dataset is a reusable asset"* rather than a test of curriculum training.

The student's ``label_budget`` is the number of *distinct* labels it is allowed
to draw on, not the number of batches it sees.  With a budget equal to the
artifact's size it uses all of it, and the ``update_budget`` gradient steps
revisit those labels many times over — which is exactly the point.  The online
arm spends the same label budget on labels it makes fresh and then throws away.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from holdem.compare.common.budget import SpendRecord
from holdem.compare.common.evaluation import EvaluationConfig, TestSituation, evaluate_agent
from holdem.compare.common.fitting import StudentResult, fit_value_net
from holdem.compare.fixed.store import DatasetStore
from holdem.generation import Examples
from holdem.net import HoldemValueNet, HoldemValueNetConfig


@dataclass
class FixedStudentConfig:
    """Budgets and optimiser settings for the offline student."""

    label_budget: int
    update_budget: int
    batch_size: int = 128
    learning_rate: float = 1e-3
    # How to weight the artifact's sources when drawing a batch.  ``None``
    # weights every non-empty source equally, which is rarely what you want:
    # the river is usually far larger than the streets above it.
    source_weights: Optional[Dict[str, float]] = None
    # Gradient steps between evaluations; ``None`` scores only at the end.
    eval_every: Optional[int] = None
    value_net: HoldemValueNetConfig = field(default_factory=HoldemValueNetConfig)
    seed: int = 0
    device: str = "cpu"


def fit_fixed_student(
    store: DatasetStore,
    config: FixedStudentConfig,
    net: Optional[HoldemValueNet] = None,
    rng: Optional[np.random.Generator] = None,
    tests: Optional[Dict[int, tuple]] = None,
    evaluation: Optional[EvaluationConfig] = None,
) -> StudentResult:
    """Fit ``net`` from ``store`` alone, spending exactly the configured budgets."""
    if config.label_budget <= 0 or config.update_budget <= 0:
        raise ValueError("label_budget and update_budget must be positive")
    available = len(store)
    if available < config.label_budget:
        raise ValueError(
            f"artifact holds {available} example(s) but the label budget is "
            f"{config.label_budget}; build a larger dataset or lower the budget"
        )

    rng = rng if rng is not None else np.random.default_rng(config.seed)
    device = torch.device(config.device)
    net = net if net is not None else HoldemValueNet(config.value_net)
    net.to(device)
    initial_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

    optimiser = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    loss_fn = nn.HuberLoss(reduction="mean")
    subset = _draw_budget_subset(store, config.label_budget, config.source_weights, rng)
    sample_fn = _sampler(store, subset, config.source_weights)

    spend = SpendRecord(labels=config.label_budget)
    history: List[Dict[str, float]] = []
    started = time.perf_counter()

    step_size = config.eval_every or config.update_budget
    taken = 0
    while taken < config.update_budget:
        steps = min(step_size, config.update_budget - taken)
        total, done = fit_value_net(
            net, optimiser, loss_fn, sample_fn, steps, config.batch_size, device, rng
        )
        taken += done
        spend.updates += done
        record: Dict[str, float] = {
            "updates": float(taken),
            "loss": total / max(done, 1),
        }
        # Only on the requested cadence.  The *final* score is the caller's job
        # (:func:`~holdem.compare.experiment.run_comparison` step 6), and
        # scoring here as well would pay for the expensive flop and turn
        # evaluations twice for the same weights.
        if config.eval_every is not None and tests is not None and evaluation is not None:
            record.update(evaluate_agent(net, tests, evaluation))
        history.append(record)

    spend.training_seconds = time.perf_counter() - started
    return StudentResult(
        net=net, history=history, initial_state=initial_state, spend=spend
    )


def _draw_budget_subset(
    store: DatasetStore,
    label_budget: int,
    source_weights: Optional[Dict[str, float]],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Pick exactly ``label_budget`` rows of the artifact, once, up front.

    An artifact is normally built larger than any single student's budget so it
    can be reused.  Without this the student would draw batches from *all* of
    it while reporting only ``label_budget`` spent, and the experiment's central
    fairness claim — both arms consumed the same number of labels — would be
    false by however much the artifact exceeded the budget.

    The subset is drawn in the artifact's own source proportions (or in
    ``source_weights`` if given), so restricting the budget changes how *many*
    labels the student sees, not the mixture.
    """
    counts = {source: n for source, n in store.counts().items() if n > 0}
    if source_weights is not None:
        counts = {s: n for s, n in counts.items() if source_weights.get(s, 0.0) > 0}
        if not counts:
            raise ValueError("source_weights has no positive weight on an available source")
        weights = np.array([source_weights[s] for s in counts], dtype=np.float64)
    else:
        weights = np.array([counts[s] for s in counts], dtype=np.float64)
    weights = weights / weights.sum()

    sources = list(counts)
    exact = weights * label_budget
    wanted = np.floor(exact).astype(np.int64)
    # Largest-remainder apportionment: the seats left over by flooring go to
    # the sources that were rounded down hardest, so the realised mixture is
    # the closest achievable one rather than one biased toward the big sources.
    for index in np.argsort(-(exact - wanted))[: int(label_budget - wanted.sum())]:
        wanted[index] += 1
    # A source cannot give more rows than it holds; spill the excess elsewhere.
    available = np.array([counts[s] for s in sources], dtype=np.int64)
    overflow = int(np.maximum(wanted - available, 0).sum())
    wanted = np.minimum(wanted, available)
    while overflow > 0:
        room = available - wanted
        if not room.any():
            raise ValueError(
                f"artifact holds {int(available.sum())} example(s); cannot draw "
                f"a subset of {label_budget}"
            )
        take = min(overflow, int(room.max()))
        wanted[int(np.argmax(room))] += take
        overflow -= take

    return {
        source: rng.choice(counts[source], size=int(n), replace=False)
        for source, n in zip(sources, wanted)
        if n > 0
    }


def _sampler(
    store: DatasetStore,
    subset: Dict[str, np.ndarray],
    source_weights: Optional[Dict[str, float]],
):
    """Batches drawn only from the pinned subset, in its own proportions."""
    sources = list(subset)
    sizes = np.array([len(subset[s]) for s in sources], dtype=np.float64)
    if source_weights is None:
        weights = sizes / sizes.sum()
    else:
        raw = np.array([source_weights.get(s, 0.0) for s in sources], dtype=np.float64)
        weights = raw / raw.sum()

    def sample(batch_size: int, rng: np.random.Generator):
        drawn = rng.multinomial(batch_size, weights)
        parts = []
        for source, count in zip(sources, drawn):
            if not count:
                continue
            rows = rng.choice(subset[source], size=int(count), replace=True)
            parts.append(store.gather(source, rows))
        batch = Examples.concatenate(parts)
        return batch.features, batch.masks, batch.targets

    return sample
