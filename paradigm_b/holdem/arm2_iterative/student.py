"""Arm 2: ReBeL Algorithm 2 — generate labels with the current net, forever.

The loop is the paper's: sample a situation, run Linear CFR-D depth-limited
search using the **current** network at the leaves (see
:func:`~paradigm_b.holdem.selfplay.collect_trajectory`, which is the pseudocode
line by line), record what search concluded, take gradient steps, repeat.
Labels improve as the network improves, and stale ones age out of the replay
buffer.  Nothing is ever frozen.

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
import warnings
from dataclasses import dataclass, field, replace
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
from paradigm_b.holdem.net.policy import (
    HoldemPolicyNet,
    HoldemPolicyNetConfig,
    PolicyReplayBuffer,
    train_policy_net,
)
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
    # ReBeL appendix E removes half the replay buffer after 20 of its 1,750
    # epochs, because the earliest labels were written by a random network.
    # ``None`` disables it; the labels still count against the label budget,
    # because generating them is a cost that was really paid.
    # Actor processes generating trajectories in parallel.  0 keeps the
    # synchronous generate-then-train loop, which is exactly reproducible and
    # is what the fixed-vs-iterative comparison uses.  Anything higher runs the
    # Ape-X shape from ``actors.py``: much faster, not bit-reproducible.
    actors: int = 0
    # Threads the learner may use for its own forward/backward passes.  Actors
    # pin themselves to one thread each (``actors.py``); the learner does not,
    # so by default torch hands it every core and its backward pass competes
    # with every actor for the same cores.  ``None`` leaves it the cores the
    # actors are not already occupying; an explicit value overrides that, and
    # is ignored entirely on the synchronous path, where nothing competes.
    learner_threads: Optional[int] = None
    # How many gradient steps between publishing weights to the actors.  Lower
    # means fresher labels and more copying; ReBeL's actors run a network some
    # way behind the learner's and it costs little.
    weight_sync_every: int = 50
    purge_after_iterations: Optional[int] = None
    purge_fraction: float = 0.5
    # Score the agent every N iterations; ``None`` scores only at the end.
    eval_every: Optional[int] = None
    # Keep a copy of the student every N iterations, so the run can be replayed.
    checkpoint_every: Optional[int] = None
    value_net: HoldemValueNetConfig = field(default_factory=HoldemValueNetConfig)
    # theta_pi and D_pi.  Algorithm 2 carries a policy network alongside the
    # value network; it exists to warm-start search
    # (``self_play.warm_start_iterations``) and it is the one line of the
    # pseudocode marked "optional".
    #
    # Both are off by default, for a reason the comparison depends on: gradient
    # steps spent on theta_pi are compute arm 1 does not spend, and
    # ``update_budget`` counts only value-net steps, so a run with the policy
    # net on is no longer the equal-budget comparison
    # :mod:`~paradigm_b.holdem.compare.experiment` is measuring.  Turn them on
    # to run the full algorithm; leave them off to run the comparison.
    policy_net: HoldemPolicyNetConfig = field(default_factory=HoldemPolicyNetConfig)
    policy_buffer_size: int = 20_000
    policy_updates_per_iteration: int = 0
    seed: int = 0
    device: str = "cpu"

    @property
    def uses_policy_net(self) -> bool:
        """Whether theta_pi is trained, warm-starts search, or both."""
        return (
            self.policy_updates_per_iteration > 0
            or self.self_play.warm_start_iterations > 0
        )

    def generation_config(self) -> HoldemSelfPlayConfig:
        """The self-play config to generate with, with D_pi off when unused."""
        if self.uses_policy_net:
            return self.self_play
        return replace(self.self_play, policy_targets=False)


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
    """Run Algorithm 1 until both budgets are exactly spent.

    Dispatches to the parallel actor/learner loop when ``config.actors > 0``;
    see :mod:`paradigm_b.holdem.arm2_iterative.async_student`.
    """
    if label_budget <= 0 or update_budget <= 0:
        raise ValueError("label_budget and update_budget must be positive")

    if config.actors > 0:
        from paradigm_b.holdem.arm2_iterative.async_student import (
            fit_online_student_async,
        )

        return fit_online_student_async(
            config,
            label_budget,
            update_budget,
            net=net,
            rng=rng,
            tests=tests,
            evaluation=evaluation,
            journal_path=journal_path,
            checkpoint_path=checkpoint_path,
        )

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

    self_play = config.generation_config()
    situations = config.situations
    trajectory_id = 0
    iteration = 0
    _warn_if_budgets_are_mismatched(config, label_budget, update_budget)

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
                leaf_values,
                space,
                root,
                self_play,
                rng,
                reach=reach,
                policy_net=policy_net,
                device=config.device,
            )
            # Truncate so the arm lands exactly on its budget.
            examples = examples[: label_budget - spend.labels]
            if not examples:
                continue
            buffer.add(examples)
            if policy_buffer is not None:
                policy_buffer.add(
                    [policy for example in examples for policy in example.policies]
                )
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

        # Flush the random-network era out of the buffer, once.
        purged = 0
        if (
            config.purge_after_iterations is not None
            and iteration == config.purge_after_iterations
        ):
            purged = buffer.purge_oldest(config.purge_fraction)

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
            "purged": float(purged),
            "loss": total / max(done, 1),
        }
        # theta_pi's steps are deliberately outside ``update_budget``; see
        # :class:`OnlineStudentConfig`.
        if policy_net is not None and config.policy_updates_per_iteration > 0:
            record["policy_loss"] = train_policy_net(
                policy_net,
                policy_buffer,
                config.policy_updates_per_iteration,
                config.batch_size,
                config.learning_rate,
                rng,
                device,
                policy_optimiser,
            )
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
        net=net,
        history=history,
        initial_state=initial_state,
        spend=spend,
        policy_net=policy_net,
    )


def _warn_if_budgets_are_mismatched(
    config: OnlineStudentConfig, label_budget: int, update_budget: int
) -> None:
    """Shout when the per-iteration rates cannot spend both budgets together.

    The loop runs until *both* budgets are exhausted, so a mismatched ratio
    does not fail — it silently wastes.  Exhaust updates first and the tail of
    the run generates labels it never trains on; exhaust labels first and the
    tail trains on a frozen buffer.  A run that burns 46% of its label budget
    after training has stopped looks like a fair comparison in the results file
    and is not one, so this is worth a loud warning rather than a docstring.
    """
    per_iteration = (
        config.trajectories_per_iteration * config.street_mix.labels_per_trajectory
    )
    if per_iteration <= 0 or config.updates_per_iteration <= 0:
        return

    label_iterations = label_budget / per_iteration
    update_iterations = update_budget / config.updates_per_iteration
    # Judge the waste itself rather than the ratio.  A ratio that looks only
    # mildly off — 0.27 labels per update against a requested 0.5 — still burns
    # 46% of the run, which is what happened the first time this was run for
    # real, so the threshold has to be on the consequence.
    longer, shorter = max(label_iterations, update_iterations), min(
        label_iterations, update_iterations
    )
    waste = 1.0 - shorter / longer
    if waste <= 0.15:
        return

    updates_first = update_iterations < label_iterations
    tail = (
        "generating labels it never trains on"
        if updates_first
        else "training on a frozen buffer"
    )
    suggestion = max(
        round(
            (label_budget / update_budget)
            * config.updates_per_iteration
            / config.street_mix.labels_per_trajectory
        ),
        1,
    )
    warnings.warn(
        f"budget ratio mismatch: the run produces "
        f"{per_iteration / config.updates_per_iteration:.2f} labels per update "
        f"but the budgets ask for {label_budget / update_budget:.2f}.  "
        f"{'Updates' if updates_first else 'Labels'} will run out first, leaving "
        f"roughly {waste:.0%} of the run {tail}.  Set "
        f"trajectories_per_iteration to about {suggestion}.",
        RuntimeWarning,
        stacklevel=3,
    )


def _due(iteration: int, every: Optional[int], finished: bool) -> bool:
    if finished:
        return True
    return every is not None and iteration % every == 0


def _sampler(buffer: Buffer):
    def sample(batch_size: int, rng: np.random.Generator):
        return buffer.sample(batch_size, rng)

    return sample
