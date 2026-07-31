#!/usr/bin/env python
"""Launch (or resume) a long arm-2 run: ReBeL Algorithm 2, generation-bound.

``solve.py compare`` runs *both* arms at equal budgets, which is the experiment;
this runs arm 2 on its own, which is the training run.

The defaults are the 12-hour configuration derived in the session that added
this file, and every one of them is either measured or taken from ReBeL's
appendix D:

* ``--batch-size 1024`` and ``--learning-rate 3e-4`` are the paper's, as is the
  halving schedule (inert here -- its first halving is at 2,000,000 steps and
  this run takes ~45,000).
* The 112-labels-to-5-updates ratio holds **45.7 presentations per label**.
  The paper is single-pass (4.48e9 presentations against ~4.5e9 samples), but
  that ratio is a consequence of having 4,500x more data than this run: copied
  here it would be 978 gradient steps and 0.05 presentations per parameter,
  which does not train an 18.6M-parameter network.
* **The run is bounded by ``--hours``, not by the budgets.**  Sizing by label
  count needs the generation rate known in advance, and that rate moves several
  fold with ``--updates-per-iteration`` alone, because the learner and the
  actors share one GPU.  The budgets are set generously and the clock decides;
  whatever the run gets through keeps the ratio.  Budgets are still *totals*,
  so ``--resume`` continues rather than restarts.

**Guarded on purpose.**  Actors spawn, so every child re-imports ``__main__``;
an unguarded call at module level spawns recursively until the machine dies.
See the warning in ``arm2_iterative/actors.py``.

Typical use::

    nohup python run_arm2.py --run runs/arm2-12h > runs/arm2-12h/log.txt 2>&1 &
    python run_arm2.py --run runs/arm2-12h --resume     # after a crash
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from paradigm_b.holdem.arm2_iterative.student import (
    OnlineStudentConfig,
    fit_online_student,
)
from paradigm_b.holdem.arms_common.evaluation import (
    EvaluationConfig,
    all_held_out_boards,
    evaluate_agent,
    make_held_out_situations,
)
from paradigm_b.holdem.arms_common.lbr import LBRConfig
from paradigm_b.holdem.arms_common.storage import save_checkpoint, write_json
from paradigm_b.holdem.data.sampling import SituationConfig
from paradigm_b.holdem.net.value_net import HoldemValueNetConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="directory for this run")
    # Generous, because the run is bounded by --hours, not by these.  The
    # per-iteration ratio (112 labels to 5 updates) holds regardless, so
    # whatever it gets through keeps 45.7 presentations per label.
    parser.add_argument("--label-budget", type=int, default=4_000_000)
    parser.add_argument("--update-budget", type=int, default=178_570)
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--trajectories-per-iteration", type=int, default=16)
    parser.add_argument("--updates-per-iteration", type=int, default=5)
    # batch_size / labels_per_update = presentations per label.
    # 1024 / 22.4 = 45.7x reuse.
    parser.add_argument("--labels-per-update", type=float, default=22.4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=60_000)
    parser.add_argument("--purge-after-iterations", type=int, default=100)
    parser.add_argument("--state-every", type=int, default=400)
    parser.add_argument("--actors", type=int, default=8)
    parser.add_argument("--learner-threads", type=int, default=1)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--hidden-dim", type=int, default=1536)
    parser.add_argument("--residual-blocks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--journal",
        action="store_true",
        default=True,
        help="log every label to disk (~27GB at 1M) so the reuse ratio can be "
        "swept later without regenerating anything",
    )
    parser.add_argument("--no-journal", dest="journal", action="store_false")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from <run>/run_state rather than starting fresh",
    )
    parser.add_argument("--eval-boards", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="print and exit")
    return parser.parse_args()


def build(args: argparse.Namespace, excluded) -> OnlineStudentConfig:
    state = Path(args.run) / "run_state"
    return OnlineStudentConfig(
        trajectories_per_iteration=args.trajectories_per_iteration,
        updates_per_iteration=args.updates_per_iteration,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        situations=SituationConfig(excluded_boards=excluded),
        purge_after_iterations=args.purge_after_iterations,
        state_every=args.state_every,
        max_seconds=args.hours * 3600.0,
        labels_per_update=args.labels_per_update,
        resume_from=str(state) if args.resume else None,
        actors=args.actors,
        learner_threads=args.learner_threads,
        device=args.device,
        seed=args.seed,
        value_net=HoldemValueNetConfig(
            hidden_dim=args.hidden_dim, num_residual_blocks=args.residual_blocks
        ),
    )


def describe(args: argparse.Namespace, config: OnlineStudentConfig) -> dict:
    labels_per_iteration = config.trajectories_per_iteration * 7
    iterations = args.label_budget / labels_per_iteration
    presentations = args.update_budget * config.batch_size
    return {
        "label_budget": args.label_budget,
        "update_budget": args.update_budget,
        "labels_per_iteration": labels_per_iteration,
        "iterations": round(iterations),
        "labels_per_update": args.label_budget / max(args.update_budget, 1),
        "presentations": presentations,
        "reuse_per_label": config.batch_size / args.labels_per_update,
        "labels_per_update_enforced": args.labels_per_update,
        "rebel_epochs": presentations / config.examples_per_epoch,
        "wall_clock_bound_hours": args.hours,
        "labels_if_25_per_second": round(args.hours * 3600 * 25),
        "labels_if_80_per_second": round(args.hours * 3600 * 80),
        "state_writes": round(iterations / max(config.state_every, 1)),
    }


def main() -> None:
    args = parse_args()
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)

    # Held-out boards, fixed by seed so a resumed run scores on the same ones.
    rng = np.random.default_rng(args.seed)
    tests = make_held_out_situations(
        rng, SituationConfig(), streets=(3, 4, 5), boards_per_street=args.eval_boards
    )
    excluded = all_held_out_boards(tests)

    config = build(args, excluded)
    plan = describe(args, config)
    write_json({"args": vars(args), "plan": plan}, run / "config.json")
    print("plan:", flush=True)
    for key, value in plan.items():
        print(f"  {key:34s} {value}", flush=True)
    if args.dry_run:
        return

    torch.manual_seed(args.seed)
    started = time.perf_counter()
    result = fit_online_student(
        config,
        label_budget=args.label_budget,
        update_budget=args.update_budget,
        journal_path=(run / "journal") if args.journal else None,
        checkpoint_path=run / "checkpoints",
        state_path=run / "run_state",
    )
    wall = time.perf_counter() - started

    save_checkpoint(result.net.state_dict(), run / "student.pt")
    write_json(result.history, run / "history.json")
    print(
        f"generation done: {result.spend.labels} labels, {result.spend.updates} "
        f"updates, {wall/3600:.2f} h ({result.spend.labels/wall:.2f} labels/s)",
        flush=True,
    )

    # Score it, including the off-abstraction responder.
    evaluation = EvaluationConfig(
        streets=(3, 4, 5),
        boards_per_street=args.eval_boards,
        search_iterations=40,
        device=args.device,
        local_best_response=True,
        lbr=LBRConfig(),
    )
    scores = evaluate_agent(result.net, tests, evaluation)
    write_json(
        {"scores": scores, "spend": result.spend.to_dict(), "wall_seconds": wall},
        run / "results.json",
    )
    print("scores:", flush=True)
    for key in sorted(scores):
        print(f"  {key:34s} {scores[key]:12.4f}", flush=True)


if __name__ == "__main__":
    main()
