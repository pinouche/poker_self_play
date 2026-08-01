"""Is it actually getting better?  Measured during the run, not after it.

A training loop reports loss, and loss is not the question.  The question is
whether the agent *plays* better than it did an hour ago, and the only answers
worth having are the two this project can produce: exploitability against a
best response on held-out boards, and chips against Slumbot.  Both are far too
slow to run every iteration, so they run on a cadence and their results land in
``progress.jsonl`` next to the run, one line per evaluation, ready to plot.

**Everything here happens on a background thread, and that is not an
optimisation.**  A Slumbot session is network-bound at roughly one hand per
second; 200 hands is three minutes during which a foreground evaluator would
have the learner sitting idle, and over a twelve-hour run at a sane cadence
that is hours of training thrown away to measure training.  So an evaluation
takes a *snapshot* of the weights and runs against that, while the learner
carries on with the live ones.  The cost is that a result describes the network
as it was at the iteration named in the record rather than the one current when
the line is written; the alternative is not measuring at all.

Only one evaluation runs at a time.  If the cadence comes round again while the
previous one is still going, the new one is skipped and counted — a run whose
``skipped`` climbs is asking for a longer ``every`` or fewer hands, and seeing
that in the file beats wondering why the lines are irregular.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from paradigm_b.holdem.arms_common.evaluation import EvaluationConfig, evaluate_agent
from paradigm_b.holdem.arms_common.lbr_match import LbrMatchConfig
from paradigm_b.holdem.arms_common.play import PlayConfig
from paradigm_b.holdem.net.value_net import HoldemValueNet, HoldemValueNetConfig


@dataclass
class ProgressConfig:
    """When to measure, and how much of each measurement to buy."""

    every: Optional[int] = None  # iterations between evaluations; None = off
    # Wall clock between evaluations, which is usually the cadence actually
    # wanted.  An iteration is not a fixed amount of work: on the actor path it
    # is one drain plus whatever gradient steps the label throttle has earned,
    # so its rate moves several-fold with ``labels_per_update`` alone and
    # "every 90,000 iterations" is a guess at "every five hours" that can be out
    # by 2-3x.  Set this and the cadence is the clock.  Both bounds may be set,
    # in which case whichever comes round first fires.
    every_seconds: Optional[float] = None
    # Held-out boards per street for the exploitability / LBR half.  This is
    # the same knob ``run_arm2`` uses for its final score, kept small here
    # because the whole point is that it runs many times.
    boards: int = 1
    search_iterations: int = 40
    local_best_response: bool = True
    # Hands per Slumbot session.  Zero turns the Slumbot half off, which is the
    # default: it talks to a third-party server over the network, and a
    # training run should not start doing that unless it was asked to.
    slumbot_hands: int = 0
    slumbot_play: PlayConfig = field(default_factory=PlayConfig)
    # Hands of *LBR as an opponent* — ReBeL Table 1's form of the number, where
    # LBR is dealt cards and the ± is sampling error over hands.  Zero (the
    # default) leaves only the exact tree LBR above, which is the stronger
    # measurement; this one exists to be quotable next to a published figure.
    # See :mod:`.lbr_match` for why they are not the same quantity.
    lbr_match_hands: int = 0
    lbr_match_workers: int = 1
    lbr_match: LbrMatchConfig = field(default_factory=LbrMatchConfig)
    # Concurrent Slumbot sessions.  One is the safe default *during training*:
    # this evaluation already shares a machine with the learner, and buying a
    # faster measurement with the learner's cores is usually the wrong trade.
    # Raise it for a standalone evaluation run, where nothing else is competing.
    slumbot_workers: int = 1
    device: str = "cpu"

    @property
    def enabled(self) -> bool:
        return (self.every is not None and self.every > 0) or (
            self.every_seconds is not None and self.every_seconds > 0
        )


def _snapshot(net: HoldemValueNet, config: HoldemValueNetConfig) -> HoldemValueNet:
    """A frozen CPU copy of the live network.

    The learner keeps training the original while the evaluation runs, so the
    weights must be copied rather than referenced — reading a tensor mid-update
    would measure a network that never existed.
    """
    clone = HoldemValueNet(config)
    clone.load_state_dict({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    clone.eval()
    return clone


def evaluate_against_slumbot(
    net: HoldemValueNet,
    hands: int,
    play: Optional[PlayConfig] = None,
    client_factory=None,
    seed: int = 0,
    workers: int = 1,
) -> Dict[str, float]:
    """Play ``hands`` against Slumbot and report mbb/g.

    Imported lazily so that a run with the Slumbot half switched off never
    reaches for the network stack at all.

    ``workers`` plays the session as that many concurrent Slumbot sessions,
    each with its own client, policy and seed — see
    :func:`~.slumbot.play_session_parallel` for why that is both faster and
    fair.  The network is shared across them read-only, which is safe because
    this is a frozen snapshot in ``eval`` mode; the *policies* cannot be shared,
    because each one carries a hand's belief state.
    """
    from paradigm_b.holdem.arms_common.slumbot import play_session_parallel
    from paradigm_b.holdem.arms_common.slumbot_agent import (
        SlumbotAgentConfig,
        SlumbotAgentPolicy,
    )

    config = SlumbotAgentConfig(play=play or PlayConfig())
    policies = []

    def policy_factory(worker: int):
        policy = SlumbotAgentPolicy(net, config, rng=np.random.default_rng(seed + worker))
        policies.append(policy)
        return policy

    summary = play_session_parallel(
        hands,
        policy_factory=policy_factory,
        workers=workers,
        client_factory=client_factory,
    )
    return {
        "slumbot_mbb_per_game": float(summary.mbb_per_game),
        "slumbot_stderr_mbb": float(summary.stderr_mbb),
        "slumbot_hands": float(summary.hands),
        "slumbot_seconds": float(summary.seconds),
        # A session full of these is measuring the action abstraction rather
        # than the network; see :mod:`.slumbot_agent`.
        "slumbot_translation_fallbacks": float(
            sum(p.translation_fallbacks for p in policies)
        ),
    }


def _pm(stderr: Optional[float]) -> str:
    """``+/-`` the standard error, or nothing when there is not one.

    A single held-out board per street gives a mean and no spread, and printing
    ``+/-0.0`` there would read as "measured precisely" when it means "measured
    once".  See :func:`~.evaluation._stderr` for what this dispersion is over —
    situations, not hands, which is not the same quantity as the ± in ReBeL's
    Table 1.
    """
    return f"+/-{stderr:.1f}" if stderr else ""


def summarise(record: Dict[str, float]) -> str:
    """One line per evaluation, for the run's log.

    ``progress.jsonl`` has everything and is what gets plotted, but a run whose
    only visible output is a file nobody is tailing looks identical to a run
    that has quietly stopped measuring.  This is the line that says otherwise,
    and it carries the two numbers the run exists to move: exploitability on
    held-out boards, and chips against Slumbot.
    """
    parts = [
        f"progress iter {int(record.get('iteration', 0)):,}",
        f"labels {int(record.get('labels', 0)):,}",
        f"updates {int(record.get('updates', 0)):,}",
    ]
    if "aggregate" in record:
        streets = " ".join(
            f"{name[0]}{record[name]:.1f}"
            for name in ("flop", "turn", "river")
            if name in record
        )
        parts.append(
            f"expl {record['aggregate']:.2f}{_pm(record.get('aggregate_stderr'))}"
            f" ({streets})"
        )
    if "aggregate_lbr_full" in record:
        parts.append(
            f"lbr {record['aggregate_lbr_full']:.1f}"
            f"{_pm(record.get('aggregate_lbr_full_stderr'))}"
        )
    if record.get("flop_boards", 0) > 1:
        # Only worth saying when there is a spread to have: at one board a
        # street has a mean and no dispersion, and the +/- above is absent
        # rather than zero-width by luck.
        parts.append(f"n={int(record['flop_boards'])}/street")
    if "slumbot_hands" in record:
        # Never the point estimate alone: at a few hundred hands the interval
        # is wider than any edge this project can produce.
        parts.append(
            f"slumbot {record.get('slumbot_mbb_per_game', 0.0):+.0f}"
            f"+/-{record.get('slumbot_stderr_mbb', 0.0):.0f} mbb/g"
            f" over {int(record['slumbot_hands'])} hands"
        )
    if "error" in record:
        parts.append(f"ERROR {record['error']}")
    parts.append(f"[{record.get('eval_seconds', 0.0)/60:.1f} min]")
    return "  ".join(parts)


class ProgressEvaluator:
    """Runs the periodic evaluation off the learner's thread.

    ``slumbot_client_factory`` takes the worker index and returns a client, one
    per concurrent session; ``None`` means real ones.  It is a factory rather
    than a client because two sessions must never share a token — the tests
    pass a factory of fakes, a real run passes nothing.
    """

    def __init__(
        self,
        config: ProgressConfig,
        net_config: HoldemValueNetConfig,
        tests: Optional[Dict[int, Tuple[Any, ...]]] = None,
        path: Optional[Path] = None,
        slumbot_client_factory=None,
        echo: bool = True,
    ) -> None:
        self.config = config
        self.net_config = net_config
        self.tests = tests
        self.path = Path(path) if path is not None else None
        self.slumbot_client_factory = slumbot_client_factory
        # Printed from the evaluation's own thread, which is safe here: one
        # evaluation runs at a time and a single ``print`` of one line is
        # atomic enough against a learner that prints only at the ends of a run.
        self.echo = echo
        self.skipped = 0
        self.failures = 0
        self.completed = 0
        self._thread: Optional[threading.Thread] = None
        self._done: list = []
        self._lock = threading.Lock()
        # The clock for ``every_seconds`` starts now rather than at the first
        # evaluation, so the first one lands one interval into the run instead
        # of immediately — a network that has seen nothing is not worth 1000
        # hands against Slumbot.
        self._last_launched = time.perf_counter()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def due(self, iteration: int) -> bool:
        if not self.config.enabled or iteration <= 0:
            return False
        if self.config.every and iteration % int(self.config.every) == 0:
            return True
        return bool(self.config.every_seconds) and (
            time.perf_counter() - self._last_launched >= float(self.config.every_seconds)
        )

    def maybe_start(self, iteration: int, net: HoldemValueNet, spend=None) -> bool:
        """Launch an evaluation if one is due and none is in flight."""
        if not self.due(iteration):
            return False
        # Reset the wall-clock timer on *any* due tick, launched or skipped.
        # Otherwise a skip leaves the interval permanently elapsed and every
        # subsequent iteration — five a second — counts another skip, which
        # buries the one number that says the cadence is too tight.
        self._last_launched = time.perf_counter()
        if self.running:
            self.skipped += 1
            return False
        # The snapshot is taken here, on the learner's thread, because that is
        # the only place the weights are guaranteed not to be mid-update.
        clone = _snapshot(net, self.net_config)
        labels = float(getattr(spend, "labels", 0.0)) if spend is not None else 0.0
        updates = float(getattr(spend, "updates", 0.0)) if spend is not None else 0.0
        self._thread = threading.Thread(
            target=self._run,
            args=(iteration, clone, labels, updates),
            daemon=True,
            name=f"progress-{iteration}",
        )
        self._thread.start()
        return True

    def _run(self, iteration: int, net: HoldemValueNet, labels: float, updates: float) -> None:
        started = time.perf_counter()
        record: Dict[str, float] = {
            "iteration": float(iteration),
            "labels": labels,
            "updates": updates,
        }
        try:
            if self.tests:
                record.update(
                    evaluate_agent(
                        net,
                        self.tests,
                        EvaluationConfig(
                            boards_per_street=self.config.boards,
                            search_iterations=self.config.search_iterations,
                            device=self.config.device,
                            local_best_response=self.config.local_best_response,
                        ),
                    )
                )
            if self.config.lbr_match_hands > 0:
                record.update(
                    evaluate_against_lbr(
                        net,
                        self.config.lbr_match_hands,
                        self.config.lbr_match,
                        seed=iteration,
                        workers=self.config.lbr_match_workers,
                    )
                )
            if self.config.slumbot_hands > 0:
                record.update(
                    evaluate_against_slumbot(
                        net,
                        self.config.slumbot_hands,
                        self.config.slumbot_play,
                        client_factory=self.slumbot_client_factory,
                        seed=iteration,
                        workers=self.config.slumbot_workers,
                    )
                )
        except Exception as error:  # noqa: BLE001 - a bad eval must not kill a run
            # Twelve hours of training is worth more than one measurement, and
            # a Slumbot session in particular fails for reasons that have
            # nothing to do with this process — so the failure is recorded and
            # the run continues.
            record["error"] = f"{type(error).__name__}: {error}"
            self.failures += 1
        record["eval_seconds"] = time.perf_counter() - started
        with self._lock:
            self._done.append(record)
            self.completed += 1
        self._write(record)
        if self.echo:
            print(summarise(record), flush=True)

    def _write(self, record: Dict[str, float]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def drain(self) -> list:
        """Results finished since the last call, for the caller's history."""
        with self._lock:
            done, self._done = self._done, []
        return done

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
