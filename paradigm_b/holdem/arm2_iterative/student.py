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
from paradigm_b.holdem.arms_common.situations import (
    StreetMix,
    labels_from_street,
    sample_trajectory_start,
)
from paradigm_b.holdem.arms_common.storage import (
    PathLike,
    load_run_state,
    save_checkpoint,
    save_run_state,
)
from paradigm_b.holdem.arm2_iterative.journal import JournalEntry, TrajectoryJournal
from paradigm_b.holdem.net.policy import (
    HoldemPolicyNet,
    HoldemPolicyNetConfig,
    PolicyReplayBuffer,
    train_policy_net,
)
from paradigm_b.holdem.net.value_net import HoldemValueNet, HoldemValueNetConfig
from paradigm_b.holdem.data.store import build_buffer
from paradigm_b.holdem.selfplay import HoldemSelfPlayConfig, collect_trajectory
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
    # Start every trajectory at the blinds, which is ReBeL's own regime: the
    # initial belief state is common knowledge, so no PBS distribution has to
    # be handcrafted and ``street_mix`` is ignored.  Set it False to keep the
    # DeepStack-style postflop sampler, which the fixed-vs-iterative comparison
    # needs because it scores each street on its own held-out boards.
    preflop_start: bool = True
    # Both of ReBeL's augmentation clauses, applied when a row is read rather
    # than when it is written — see
    # :func:`~paradigm_b.holdem.data.augmentation.transform_batch`.
    #
    # ``K`` used to be a storage multiplier in the paper: K transformed copies
    # of each solve in the buffer, each pinned to one point of its orbit, plus a
    # periodic in-place pass to re-transform them.  Drawing the transformation
    # at sample time instead makes both unnecessary and is strictly stronger —
    # one canonical row per solve, a fresh orbit point on every draw — so this
    # is a switch rather than a count.  Any value >= 1 means "transform on
    # read", which is the default and what the paper's K=2 becomes here.  ``0``
    # is the ablation: canonical rows served exactly as stored.
    suit_augmentations: int = 2
    # Where the replay rows live.  ``None`` keeps them in memory, which is right
    # up to a few million.  A path puts them in append-only shards on disk
    # (:class:`~paradigm_b.holdem.data.store.ShardedReplayBuffer`) and is what
    # makes a paper-sized capacity reachable; it is expected to be NVMe and may
    # sit outside the run directory.
    buffer_dir: Optional[str] = None
    buffer_shard_rows: int = 16_384
    # Rows of the newest data mirrored in memory when the buffer is sharded.
    # A cache, not a tier — sampling stays uniform over every live row.
    # 262,144 rows is ~2.9 GB at the fp16 row size.
    buffer_hot_rows: int = 262_144
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
    # Write the *resumable* state — both networks, both optimisers, both
    # buffers, the spend, the counters and the RNG — every N iterations.  A
    # checkpoint is weights only and cannot continue a run: restarting from one
    # rebuilds Adam's moments from scratch and trains the first few hundred
    # steps against an empty replay buffer, which is a worse starting point
    # than it looks.  ``None`` disables it.
    #
    # The replay arrays grow lazily toward ``buffer_size``, and a state file
    # contains only populated rows — in canonical fp16, so 2.5x smaller than the
    # encoding-plus-mask form it replaces (60,000 examples is ~0.66GB, not
    # ~1.6GB).  With ``buffer_dir`` set the rows are not copied here at all: the
    # shards are already durable and the state records a manifest into them.
    state_every: Optional[int] = None
    # Continue from a directory written by ``state_every``.  Budgets are read as
    # *totals*, so a run resumed at 400k of a 1M label budget generates the
    # remaining 600k and then stops.
    resume_from: Optional[str] = None
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
    # ReBeL appendix D, the full-game recipe: "Adam optimizer with learning
    # rate 3 x 10^-4 and halved the learning rate every 800 epochs.  One epoch
    # is 2,560,000 examples and the batch size 1024."
    #
    # The halving is expressed in *examples seen* rather than held on a
    # scheduler object, so it survives a resume for free: the schedule is a
    # pure function of ``spend.updates * batch_size``, which the run state
    # already carries.  A stateful ``lr_scheduler`` would have to be saved and
    # restored, and silently resets to full learning rate if it is not.
    #
    # At this project's scale it never fires.  800 epochs is 2,000,000 steps at
    # batch 1024; a 12-hour run here is ~45,000.  It is here so the recipe is
    # the paper's rather than nearly the paper's, and so a longer run behaves.
    examples_per_epoch: int = 2_560_000
    lr_halve_every_epochs: Optional[int] = 800
    # Stop cleanly after this many seconds, whatever the budgets say.  Sizing a
    # run by label budget requires knowing the generation rate in advance, and
    # that rate moves with every config change -- ``updates_per_iteration``
    # alone swings it several-fold, because the learner and the actors share
    # one GPU.  A wall-clock bound makes "a twelve-hour run" mean twelve hours
    # rather than a guess, and the per-iteration ratio (112 labels to 5
    # updates) is preserved whatever it gets through.  State is written on the
    # way out, so the run is resumable and the budgets stay totals.
    max_seconds: Optional[float] = None
    # Labels that must be *accepted* per gradient step.  This is the knob that
    # sets reuse (presentations per label = batch_size / labels_per_update), and
    # on the actor path it is the only thing that can.
    #
    # ``trajectories_per_iteration`` governs the ratio on the synchronous path
    # only: there, one iteration generates exactly that many trajectories and
    # then trains.  An actor-driven learner instead drains whatever has arrived
    # and takes ``updates_per_iteration`` steps regardless, so the ratio falls
    # out of how fast the actors happen to be relative to the learner -- 4.3
    # labels per update when 22.4 was intended, i.e. 236x reuse instead of 46x.
    # Setting this throttles the learner to the intended ratio whatever the
    # relative speeds turn out to be.
    labels_per_update: Optional[float] = None
    seed: int = 0
    device: str = "cpu"

    def learning_rate_at(self, examples_seen: int) -> float:
        """ReBeL's halving schedule, as a function of examples presented."""
        if not self.lr_halve_every_epochs:
            return self.learning_rate
        epochs = examples_seen / max(self.examples_per_epoch, 1)
        halvings = int(epochs // self.lr_halve_every_epochs)
        return self.learning_rate * (0.5 ** halvings)

    @property
    def augment_on_read(self) -> bool:
        """Whether sampled rows are re-transformed; see ``suit_augmentations``."""
        return self.suit_augmentations >= 1

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
    state_path: Optional[PathLike] = None,
    progress=None,
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
            state_path=state_path,
            progress=progress,
        )
    if progress is not None:
        raise ValueError(
            "progress evaluation needs the async loop; it exists to keep the "
            "learner working while a measurement runs, and the synchronous "
            "path has no other thread to keep working"
        )

    rng = rng if rng is not None else np.random.default_rng(config.seed)
    # Augmentation draws from a stream of its own.  It is handed to the buffer,
    # which applies the isomorphisms at sample time; sharing the generation rng
    # would let each transform advance the stream that decides which situations
    # get sampled next, so a run with ``suit_augmentations`` set would search
    # different subgames than one without — destroying the only comparison the
    # setting is supposed to allow, and doing it invisibly.
    augment_rng = np.random.default_rng(np.random.SeedSequence(config.seed).spawn(2)[1])
    device = torch.device(config.device)
    net = net if net is not None else HoldemValueNet(config.value_net)
    net.to(device)
    initial_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

    optimiser = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    loss_fn = nn.HuberLoss(reduction="mean")
    buffer = build_buffer(
        config.buffer_size,
        directory=config.buffer_dir,
        shard_rows=config.buffer_shard_rows,
        hot_rows=config.buffer_hot_rows,
        augment=config.augment_on_read,
        augment_rng=augment_rng,
    )
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

    state_directory = _state_directory(config, state_path, checkpoint_path)
    if config.resume_from is not None:
        restored = load_run_state(
            config.resume_from,
            config=config,
            net=net,
            optimiser=optimiser,
            buffer=buffer,
            rng=rng,
            augment_rng=augment_rng,
            policy_net=policy_net,
            policy_optimiser=policy_optimiser,
            policy_buffer=policy_buffer,
        )
        spend = restored["spend"]
        iteration = restored["iteration"]
        trajectory_id = restored["trajectory_id"]
        initial_state = restored["initial_state"]
        net.to(device)

    _warn_if_budgets_are_mismatched(config, label_budget, update_budget)

    deadline = None if config.max_seconds is None else time.perf_counter() + config.max_seconds
    while spend.labels < label_budget or spend.updates < update_budget:
        if deadline is not None and time.perf_counter() >= deadline:
            break
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
            space, root, reach, board_cards = sample_trajectory_start(
                rng, situations, config.street_mix, config.preflop_start
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
            journalling = time.perf_counter()
            journal.flush(iteration)
            spend.journal_seconds += time.perf_counter() - journalling

        # Flush the random-network era out of the buffer, once.
        purged = 0
        if (
            config.purge_after_iterations is not None
            and iteration == config.purge_after_iterations
        ):
            purged = buffer.purge_oldest(config.purge_fraction)

        # -- train ---------------------------------------------------------
        training_started = time.perf_counter()
        _apply_learning_rate(optimiser, config, spend.updates * config.batch_size)
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

        if state_directory is not None and _due(iteration, config.state_every, finished):
            save_run_state(
                state_directory,
                config=config,
                spend=spend,
                iteration=iteration,
                trajectory_id=trajectory_id,
                rng=rng,
                augment_rng=augment_rng,
                net=net,
                optimiser=optimiser,
                buffer=buffer,
                initial_state=initial_state,
                policy_net=policy_net,
                policy_optimiser=policy_optimiser,
                policy_buffer=policy_buffer,
            )

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
        journal.close()
    # Always on the way out, not only on the cadence: a run stopped by
    # ``max_seconds`` breaks at the top of an iteration, so the last scheduled
    # write could otherwise be up to ``state_every`` iterations stale and that
    # much generation would have to be redone on resume.
    if state_directory is not None:
        save_run_state(
            state_directory,
            config=config,
            spend=spend,
            iteration=iteration,
            trajectory_id=trajectory_id,
            rng=rng,
            augment_rng=augment_rng,
            net=net,
            optimiser=optimiser,
            buffer=buffer,
            initial_state=initial_state,
            policy_net=policy_net,
            policy_optimiser=policy_optimiser,
            policy_buffer=policy_buffer,
        )
    return StudentResult(
        net=net,
        history=history,
        initial_state=initial_state,
        spend=spend,
        policy_net=policy_net,
    )


def _labels_per_trajectory(config: "OnlineStudentConfig") -> float:
    """How many labels one trajectory is expected to yield, under either regime.

    A preflop-rooted hand passes through four betting rounds and emits a label
    at each root plus one at each of the three pre-deal belief states it stops
    in front of.  A postflop start emits fewer, and how many depends on the
    street mixture.  Budget sizing needs the right one or a run silently spends
    one of its two budgets long before the other.
    """
    if config.preflop_start:
        return float(labels_from_street(0))
    return config.street_mix.labels_per_trajectory


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
    per_iteration = config.trajectories_per_iteration * _labels_per_trajectory(config)
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
            / _labels_per_trajectory(config)
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


def _apply_learning_rate(optimiser, config, examples_seen: int) -> float:
    """Set the optimiser's learning rate from the schedule; returns it."""
    lr = config.learning_rate_at(examples_seen)
    for group in optimiser.param_groups:
        group["lr"] = lr
    return lr


def _state_directory(
    config, state_path: Optional[PathLike], checkpoint_path: Optional[PathLike]
) -> Optional[Path]:
    """Where the resumable state goes, if the run wants one.

    Defaults to a sibling of the checkpoint directory rather than inside it:
    checkpoints are a series and get globbed, the run state is a single
    directory that is swapped in place, and mixing them makes both harder to
    reason about.
    """
    if config.state_every is None and state_path is None:
        return None
    if state_path is not None:
        return Path(state_path)
    if checkpoint_path is None:
        raise ValueError(
            "state_every needs somewhere to write: pass state_path, or a "
            "checkpoint_path to derive it from"
        )
    return Path(checkpoint_path).parent / "run_state"


def _due(iteration: int, every: Optional[int], finished: bool) -> bool:
    if finished:
        return True
    return every is not None and iteration % every == 0


def _sampler(buffer: Buffer):
    def sample(batch_size: int, rng: np.random.Generator):
        return buffer.sample(batch_size, rng)

    return sample
