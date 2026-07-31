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

**The learner is held to a share of the cores.**  Actors pin themselves to one
thread each, so a learner left on torch's default pool would run its backward
pass across every core while all of them were already busy.  See
:func:`learner_thread_count`.

**Labels count when accepted, not when generated.**  Actors are stopped on the
label that hits the cap, and anything already in flight is dropped rather than
trained on. So the arm is charged for exactly ``label_budget`` labels — but a
little more generation work than that was really performed, and the wall clock
reflects it. The alternative (counting everything the actors made) would let
the budget drift with the worker count, which would be worse.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from paradigm_b.holdem.arm2_iterative.actors import ActorBatch, ActorPool
from paradigm_b.holdem.arm2_iterative.journal import JournalEntry, TrajectoryJournal
from paradigm_b.holdem.arms_common.budget import SpendRecord
from paradigm_b.holdem.arms_common.evaluation import EvaluationConfig, TestSituation, evaluate_agent
from paradigm_b.holdem.arms_common.fitting import StudentResult, fit_value_net
from paradigm_b.holdem.arms_common.storage import (
    PathLike,
    load_run_state,
    save_checkpoint,
    save_run_state,
)
from paradigm_b.holdem.arm2_iterative.student import _state_directory
from paradigm_b.holdem.net.policy import (
    HoldemPolicyNet,
    PolicyReplayBuffer,
    train_policy_net,
)
from paradigm_b.holdem.net.value_net import HoldemValueNet
from paradigm_b.holdem.selfplay import Buffer, Example


def learner_thread_count(actors: int, requested: Optional[int] = None) -> int:
    """How many threads the learner should take when ``actors`` are running.

    Actors call ``torch.set_num_threads(1)``; the learner never did, so torch
    sized its intra-op pool from the whole machine and every gradient step ran
    a backward pass across all cores while eight actor processes were already
    on them.  The oversubscription costs both sides — the learner's threads
    spend their slice descheduling, and the actors lose the cores underneath
    them mid-solve.

    Leaving the learner the cores the actors are not using is the conservative
    fix: it never takes a core an actor was counting on, and on the usual
    ``actors < cores`` setting it still gets several.

    Measured, 8 actors / xlarge net / 250 labels / 400 updates, medians of
    three runs on a 16-logical-core M4 Max — the wall clock moves little
    because these runs are generation-bound, but the learner's own time is
    where the contention was, and it is unambiguous:

    ====== ========== ============== ========
    threads  wall      learner time   speedup
    ====== ========== ============== ========
    16      99.4s      87.9s          1.00x
    8       92.1s      67.8s          1.08x
    1       88.6s      39.6s          1.12x
    ====== ========== ============== ========

    So ``cpu_count - actors`` is the adaptive default, and it is most of the
    win.  One thread measured slightly better still, because this machine's
    16 logical cores are 12 performance plus 4 efficiency and eight actors
    have already taken the cores worth having — a distinction no portable
    formula can see.  Pass ``learner_threads=1`` for generation-bound runs;
    raise it when the update budget is large enough that the learner, not the
    actors, is the thing waiting.
    """
    if requested is not None:
        return max(1, int(requested))
    return max(1, (os.cpu_count() or 1) - max(actors, 0))


@contextmanager
def _learner_threads(count: int) -> Iterator[int]:
    """Hold the learner to ``count`` threads, restoring the previous setting."""
    previous = torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        yield count
    finally:
        torch.set_num_threads(previous)


def _clip(batch: ActorBatch, room: int) -> ActorBatch:
    """The first ``room`` labels of ``batch``, and their policy targets."""
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
        policies=batch.policies[:room],
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
    state_path: Optional[PathLike] = None,
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

    policy_net: Optional[HoldemPolicyNet] = None
    policy_optimiser = None
    policy_buffer = None
    if config.uses_policy_net:
        policy_net = HoldemPolicyNet(config.policy_net).to(device)
        policy_optimiser = torch.optim.Adam(
            policy_net.parameters(), lr=config.learning_rate
        )
        policy_buffer = PolicyReplayBuffer(config.policy_buffer_size)

    state_directory = _state_directory(config, state_path, checkpoint_path)
    trajectory_id = 0
    iteration = 0
    if config.resume_from is not None:
        restored = load_run_state(
            config.resume_from,
            config=config,
            net=net,
            optimiser=optimiser,
            buffer=buffer,
            rng=rng,
            policy_net=policy_net,
            policy_optimiser=policy_optimiser,
            policy_buffer=policy_buffer,
        )
        spend = restored["spend"]
        iteration = restored["iteration"]
        trajectory_id = restored["trajectory_id"]
        initial_state = restored["initial_state"]
        net.to(device)
        if policy_net is not None:
            policy_net.to(device)

    # Built after the restore on purpose: ``ActorPool`` snapshots the network
    # into shared memory at construction, so spawning it first would start
    # every actor generating from freshly initialised weights and quietly
    # poison the buffer for the first sync interval.
    pool = ActorPool(
        workers=config.actors,
        net=net,
        net_config=config.value_net,
        self_play=config.generation_config(),
        situations=config.situations,
        street_mix=config.street_mix,
        preflop_start=config.preflop_start,
        seed=config.seed,
        device=config.device,
        policy_net=policy_net,
        policy_config=config.policy_net if policy_net is not None else None,
    )
    started = time.perf_counter()

    threads = learner_thread_count(
        config.actors, getattr(config, "learner_threads", None)
    )

    try:
        with _learner_threads(threads), pool:
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
                        if policy_buffer is not None:
                            policy_buffer.add(
                                [
                                    policy
                                    for group in batch.policies
                                    for policy in group
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

                # 2b. theta_pi, outside ``update_budget`` — see
                #     :class:`OnlineStudentConfig`.
                policy_loss = None
                if policy_net is not None and config.policy_updates_per_iteration > 0:
                    policy_loss = train_policy_net(
                        policy_net,
                        policy_buffer,
                        config.policy_updates_per_iteration,
                        config.batch_size,
                        config.learning_rate,
                        rng,
                        device,
                        policy_optimiser,
                    )

                # 3. Let the actors catch up to the learner.
                if done and spend.updates % config.weight_sync_every < done:
                    pool.publish(net, policy_net)

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
                if policy_loss is not None:
                    record["policy_loss"] = policy_loss
                if (
                    tests is not None
                    and evaluation is not None
                    and config.eval_every is not None
                    and iteration % config.eval_every == 0
                ):
                    record.update(evaluate_agent(net, tests, evaluation))
                history.append(record)

                if state_directory is not None and (
                    finished
                    or (
                        config.state_every is not None
                        and iteration % config.state_every == 0
                    )
                ):
                    # The actors keep generating through this; the learner is
                    # the only writer of everything being saved, so a snapshot
                    # taken here is self-consistent even though trajectories
                    # are in flight.  Those in-flight labels are simply lost on
                    # a resume, which costs at most one drain's worth.
                    save_run_state(
                        state_directory,
                        config=config,
                        spend=spend,
                        iteration=iteration,
                        trajectory_id=trajectory_id,
                        rng=rng,
                        net=net,
                        optimiser=optimiser,
                        buffer=buffer,
                        initial_state=initial_state,
                        policy_net=policy_net,
                        policy_optimiser=policy_optimiser,
                        policy_buffer=policy_buffer,
                    )

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
        net=net,
        history=history,
        initial_state=initial_state,
        spend=spend,
        policy_net=policy_net,
    )
