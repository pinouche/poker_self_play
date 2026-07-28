"""Arm 2: ReBeL Algorithm 1 — generate labels with the current net, forever.

The loop is the paper's: sample a situation, run depth-limited search using the
**current** network at the leaves, record what search concluded, take gradient
steps, repeat.  Labels improve as the network improves, and stale ones age out
of the replay buffer.  Nothing is ever frozen.

Two things this has to get right for the comparison to mean anything:

**Exact budgets.**  ``collect_trajectory`` returns a variable number of examples
— one per street it passes through — so a loop that stopped at "the first
iteration past the budget" would let this arm overspend by up to a trajectory
every run.  The last trajectory is truncated so the arm consumes *precisely*
``label_budget`` labels and takes *precisely* ``update_budget`` steps.

**The same street distribution as the artifact.**  A ``SituationConfig`` pins
one street; left alone, this arm would train on turn endgames and be scored on
flops.  ``street_mix`` is set from the offline artifact's own composition, so
both arms see the same spread of streets and differ only in when their labels
were made.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from paradigm_b.holdem.arms_common.budget import CountingLeafValues, SpendRecord
from paradigm_b.holdem.arms_common.evaluation import EvaluationConfig, TestSituation, evaluate_agent
from paradigm_b.holdem.arms_common.fitting import StudentResult, fit_value_net
from paradigm_b.holdem.arms_common.situations import StreetMix, sample_mixed_situation
from paradigm_b.holdem.arms_common.storage import PathLike, save_checkpoint
from paradigm_b.holdem.arm2_iterative.journal import JournalEntry, TrajectoryJournal
from paradigm_b.holdem.net.value_net import HoldemValueNet, HoldemValueNetConfig
from paradigm_b.holdem.selfplay import Buffer, HoldemSelfPlayConfig, collect_trajectory
from paradigm_b.holdem.data.sampling import SituationConfig
from paradigm_b.holdem.net.leaf_values import NetLeafValues


@dataclass
class OnlineStudentConfig:
    """The online arm's knobs.  Budgets come from the comparison, not from here."""

    trajectories_per_iteration: int = 16
    updates_per_iteration: int = 40
    buffer_size: int = 60_000
    batch_size: int = 128
    learning_rate: float = 1e-3
    self_play: HoldemSelfPlayConfig = field(default_factory=HoldemSelfPlayConfig)
    situations: SituationConfig = field(default_factory=SituationConfig)
    street_mix: StreetMix = field(default_factory=StreetMix)
    # Score the agent every N iterations; ``None`` scores only at the end.
    eval_every: Optional[int] = None
    # Keep a copy of the student every N iterations, so the run can be replayed.
    checkpoint_every: Optional[int] = None
    value_net: HoldemValueNetConfig = field(default_factory=HoldemValueNetConfig)
    seed: int = 0
    device: str = "cpu"


def fit_online_student(
    config: OnlineStudentConfig,
    label_budget: int,
    update_budget: int,
    net: Optional[HoldemValueNet] = None,
    rng: Optional[np.random.Generator] = None,
    tests: Optional[Dict[int, Tuple[TestSituation, ...]]] = None,
    evaluation: Optional[EvaluationConfig] = None,
    journal_path: Optional[PathLike] = None,
    checkpoint_path: Optional[PathLike] = None,
) -> StudentResult:
    """Run Algorithm 1 until both budgets are exactly spent."""
    if label_budget <= 0 or update_budget <= 0:
        raise ValueError("label_budget and update_budget must be positive")

    rng = rng if rng is not None else np.random.default_rng(config.seed)
    device = torch.device(config.device)
    net = net if net is not None else HoldemValueNet(config.value_net)
    net.to(device)
    initial_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

    optimiser = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    loss_fn = nn.HuberLoss(reduction="mean")
    buffer = Buffer(config.buffer_size)
    journal = TrajectoryJournal(journal_path) if journal_path is not None else None
    spend = SpendRecord()
    history: List[Dict[str, float]] = []

    situations = config.situations
    trajectory_id = 0
    iteration = 0

    while spend.labels < label_budget or spend.updates < update_budget:
        iteration += 1

        # -- generate ------------------------------------------------------
        generation_started = time.perf_counter()
        leaf_values = CountingLeafValues(
            NetLeafValues(net, device=config.device), spend
        )
        produced = 0
        for _ in range(config.trajectories_per_iteration):
            if spend.labels >= label_budget:
                break
            space, root, reach, board_cards = sample_mixed_situation(
                rng, situations, config.street_mix
            )
            examples = collect_trajectory(
                leaf_values, space, root, config.self_play, rng, reach=reach
            )
            # Truncate so the arm lands exactly on its budget.
            examples = examples[: label_budget - spend.labels]
            if not examples:
                continue
            buffer.add(examples)
            spend.labels += len(examples)
            produced += len(examples)
            if journal is not None:
                journal.add(
                    [
                        JournalEntry(
                            features=example.features,
                            mask=example.mask,
                            values=example.values,
                            iteration=iteration,
                            trajectory=trajectory_id,
                            step=step,
                            board_cards=len(example.board) or board_cards,
                        )
                        for step, example in enumerate(examples)
                    ]
                )
            trajectory_id += 1
        spend.generation_seconds += time.perf_counter() - generation_started
        if journal is not None:
            journal.flush(iteration)

        # -- train ---------------------------------------------------------
        training_started = time.perf_counter()
        steps = min(config.updates_per_iteration, update_budget - spend.updates)
        total, done = fit_value_net(
            net, optimiser, loss_fn, _sampler(buffer), steps, config.batch_size, device, rng
        )
        spend.updates += done
        spend.training_seconds += time.perf_counter() - training_started

        record: Dict[str, float] = {
            "iteration": float(iteration),
            "labels": float(spend.labels),
            "updates": float(spend.updates),
            "generated": float(produced),
            "buffer": float(len(buffer)),
            "loss": total / max(done, 1),
        }
        finished = spend.labels >= label_budget and spend.updates >= update_budget
        # Cadence only — never forced at the finish.  The final score belongs to
        # :func:`~holdem.compare.experiment.run_comparison`, and doing it here
        # too would pay for the expensive flop and turn evaluations twice.
        if (
            tests is not None
            and evaluation is not None
            and config.eval_every is not None
            and iteration % config.eval_every == 0
        ):
            record.update(evaluate_agent(net, tests, evaluation))
        history.append(record)

        if checkpoint_path is not None and _due(iteration, config.checkpoint_every, finished):
            save_checkpoint(
                net.state_dict(),
                Path(checkpoint_path) / f"student-iter-{iteration:05d}.pt",
            )

        if produced == 0 and done == 0:
            raise RuntimeError(
                "online student made no progress in a full iteration; "
                "check trajectories_per_iteration and updates_per_iteration"
            )

    if journal is not None:
        journal.flush(iteration)
    return StudentResult(
        net=net, history=history, initial_state=initial_state, spend=spend
    )


def _due(iteration: int, every: Optional[int], finished: bool) -> bool:
    if finished:
        return True
    return every is not None and iteration % every == 0


def _sampler(buffer: Buffer):
    def sample(batch_size: int, rng: np.random.Generator):
        return buffer.sample(batch_size, rng)

    return sample
