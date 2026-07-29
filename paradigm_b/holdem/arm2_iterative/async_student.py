"""The learner half of the actor/learner loop.

Actors run Algorithm 1 in parallel processes (see :mod:`.actors`); this is the
single learner that consumes what they produce.  Its loop is deliberately
simple, and the ordering inside it is what makes the budgets exact:

1. drain whatever the actors have finished into the replay buffer, **clipping
   the batch that crosses the label budget** so the arm lands on it precisely;
2. take gradient steps, capped by what is left of the update budget;
3. publish weights every ``weight_sync_every`` steps, so actors keep working
   from a network that is close to current;
4. stop when both budgets are spent, then shut the actors down.

Two things worth knowing before reading the numbers this produces.

**It is not reproducible.**  How many trajectories arrive between two gradient
steps depends on process scheduling. The synchronous loop in ``student.py``
remains the one the fixed-vs-iterative comparison uses, because that comparison
rests on both arms being deterministic given a seed.

**Labels count when accepted, not when generated.**  Actors are stopped on the
label that hits the cap, and anything already in flight is dropped rather than
trained on. So the arm is charged for exactly ``label_budget`` labels — but a
little more generation work than that was really performed, and the wall clock
reflects it. The alternative (counting everything the actors made) would let
the budget drift with the worker count, which would be worse.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from paradigm_b.holdem.arm2_iterative.actors import ActorBatch, ActorPool
from paradigm_b.holdem.arm2_iterative.journal import JournalEntry, TrajectoryJournal
from paradigm_b.holdem.arms_common.budget import SpendRecord
from paradigm_b.holdem.arms_common.evaluation import EvaluationConfig, TestSituation, evaluate_agent
from paradigm_b.holdem.arms_common.fitting import StudentResult, fit_value_net
from paradigm_b.holdem.arms_common.storage import PathLike, save_checkpoint
from paradigm_b.holdem.net.value_net import HoldemValueNet
from paradigm_b.holdem.selfplay import Buffer, Example


def _clip(batch: ActorBatch, room: int) -> ActorBatch:
    """The first ``room`` labels of ``batch``."""
    if room >= len(batch):
        return batch
    return ActorBatch(
        features=batch.features[:room],
        masks=batch.masks[:room],
        targets=batch.targets[:room],
        boards=batch.boards[:room],
        leaf_evaluations=batch.leaf_evaluations,
        solver_calls=batch.solver_calls,
        weight_version=batch.weight_version,
    )


def fit_online_student_async(
    config,
    label_budget: int,
    update_budget: int,
    net: Optional[HoldemValueNet] = None,
    rng: Optional[np.random.Generator] = None,
    tests: Optional[Dict[int, Tuple[TestSituation, ...]]] = None,
    evaluation: Optional[EvaluationConfig] = None,
    journal_path: Optional[PathLike] = None,
    checkpoint_path: Optional[PathLike] = None,
) -> StudentResult:
    """Algorithm 1 across ``config.actors`` processes, one learner here."""
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

    pool = ActorPool(
        workers=config.actors,
        net=net,
        net_config=config.value_net,
        self_play=config.self_play,
        situations=config.situations,
        street_mix=config.street_mix,
        seed=config.seed,
        device=config.device,
    )
    trajectory_id = 0
    iteration = 0
    started = time.perf_counter()

    try:
        with pool:
            while spend.labels < label_budget or spend.updates < update_budget:
                iteration += 1

                # 1. Collect whatever finished while we were training.
                waiting = time.perf_counter()
                produced = 0
                if spend.labels < label_budget:
                    for batch in pool.drain():
                        room = label_budget - spend.labels
                        if room <= 0:
                            break
                        batch = _clip(batch, room)
                        buffer.add(
                            [
                                Example(
                                    features=batch.features[i],
                                    mask=batch.masks[i],
                                    values=batch.targets[i],
                                )
                                for i in range(len(batch))
                            ]
                        )
                        spend.labels += len(batch)
                        spend.leaf_evaluations += batch.leaf_evaluations
                        spend.solver_calls += batch.solver_calls
                        produced += len(batch)
                        if journal is not None:
                            journal.add(
                                [
                                    JournalEntry(
                                        features=batch.features[i],
                                        mask=batch.masks[i],
                                        values=batch.targets[i],
                                        iteration=iteration,
                                        trajectory=trajectory_id,
                                        step=i,
                                        board_cards=batch.boards[i],
                                    )
                                    for i in range(len(batch))
                                ]
                            )
                        trajectory_id += 1
                spend.generation_seconds += time.perf_counter() - waiting
                if journal is not None:
                    journal.flush(iteration)

                purged = 0
                if (
                    config.purge_after_iterations is not None
                    and iteration == config.purge_after_iterations
                ):
                    purged = buffer.purge_oldest(config.purge_fraction)

                # 2. Train on whatever is in the buffer.  An empty buffer only
                #    happens on the first pass, before any actor has finished.
                training = time.perf_counter()
                steps = min(config.updates_per_iteration, update_budget - spend.updates)
                total, done = 0.0, 0
                if len(buffer):
                    total, done = fit_value_net(
                        net,
                        optimiser,
                        loss_fn,
                        lambda size, generator: buffer.sample(size, generator),
                        steps,
                        config.batch_size,
                        device,
                        rng,
                    )
                    spend.updates += done
                spend.training_seconds += time.perf_counter() - training

                # 3. Let the actors catch up to the learner.
                if done and spend.updates % config.weight_sync_every < done:
                    pool.publish(net)

                finished = spend.labels >= label_budget and spend.updates >= update_budget
                record: Dict[str, float] = {
                    "iteration": float(iteration),
                    "labels": float(spend.labels),
                    "updates": float(spend.updates),
                    "generated": float(produced),
                    "buffer": float(len(buffer)),
                    "purged": float(purged),
                    "loss": total / max(done, 1),
                }
                if (
                    tests is not None
                    and evaluation is not None
                    and config.eval_every is not None
                    and iteration % config.eval_every == 0
                ):
                    record.update(evaluate_agent(net, tests, evaluation))
                history.append(record)

                if checkpoint_path is not None and (
                    finished
                    or (
                        config.checkpoint_every is not None
                        and iteration % config.checkpoint_every == 0
                    )
                ):
                    save_checkpoint(
                        net.state_dict(),
                        Path(checkpoint_path) / f"student-iter-{iteration:05d}.pt",
                    )
    finally:
        if journal is not None:
            journal.flush(iteration)

    spend.training_seconds = min(spend.training_seconds, time.perf_counter() - started)
    return StudentResult(
        net=net, history=history, initial_state=initial_state, spend=spend
    )
