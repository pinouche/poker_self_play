"""Arm 1 vs arm 2: are labels a reusable asset, or do they go stale?

``compare``      Run both arms at equal label and update budgets, from one
                 seeded set of starting weights, and score them by held-out
                 exploitability per street.
``label-drift``  The follow-up probe: re-solve a frozen dataset's own inputs
                 with a stronger evaluator and measure how far its labels had
                 drifted.  The river is the control — it has no leaves, so a
                 non-zero river drift means the pipeline is inconsistent and no
                 other number can be trusted.

Examples::

    python -m paradigm_b.cli.compare compare --run runs/labels-01 --label-budget 20000
    python -m paradigm_b.cli.compare label-drift --run runs/labels-01

Reachable as ``python solve.py compare`` and ``python solve.py label-drift``.
Full write-up, including what is held equal and how to read the result:
``docs/arms.md``.
"""

from __future__ import annotations

import argparse
import json

import torch

from paradigm_b.holdem.arm1_fixed.build import DatasetBuildConfig
from paradigm_b.holdem.arm1_fixed.store import DatasetStore
from paradigm_b.holdem.arm1_fixed.student import FixedStudentConfig
from paradigm_b.holdem.arm2_iterative.student import OnlineStudentConfig
from paradigm_b.holdem.arms_common.evaluation import EvaluationConfig
from paradigm_b.holdem.arms_common.storage import RunLayout
from paradigm_b.holdem.compare.experiment import ComparisonConfig, run_comparison
from paradigm_b.holdem.compare.relabel import measure_label_drift
from paradigm_b.holdem.data.sampling import SituationConfig
from paradigm_b.holdem.net.value_net import HoldemValueNet, HoldemValueNetConfig


# --- argument wiring -------------------------------------------------------
def _add_compare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=str, required=True, help="run directory")
    parser.add_argument("--label-budget", type=int, default=20_000)
    parser.add_argument("--update-budget", type=int, default=4_000)
    parser.add_argument("--river-examples", type=int, default=20_000)
    parser.add_argument("--turn-examples", type=int, default=6_000)
    parser.add_argument("--flop-examples", type=int, default=2_000)
    parser.add_argument(
        "--self-play-examples",
        type=int,
        default=0,
        help="frozen on-policy control source; separates stale labels from "
        "a different input distribution",
    )
    parser.add_argument("--teacher-updates", type=int, default=4_000)
    parser.add_argument(
        "--trajectories-per-iteration",
        type=int,
        default=16,
        help="arm 2 only; with updates-per-iteration this sets the labels-per-"
        "update ratio, which should roughly match label-budget/update-budget",
    )
    parser.add_argument("--updates-per-iteration", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=1536)
    parser.add_argument("--residual-blocks", type=int, default=6)
    parser.add_argument("--card-embedding-dim", type=int, default=128)
    parser.add_argument("--eval-boards", type=int, default=2)
    parser.add_argument("--eval-iterations", type=int, default=40)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="score mid-run every N iterations (arm 2) / N updates (arm 1); "
        "off by default because a flop+turn score is not cheap",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--flop-depth-limit",
        type=int,
        default=1,
        help="betting rounds of flop lookahead when scoring; 2 is ~170x the work",
    )
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")


def _add_drift_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", type=str, required=True, help="an existing run directory")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="evaluator to re-solve with; defaults to the iterative student",
    )
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument(
        "--cfr-iterations",
        type=int,
        default=None,
        help="override; by default each source is re-solved at the count it "
        "was generated with, which is what makes the river a control",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")


def register(sub) -> None:
    """Add this module's subcommands to a shared subparser object."""
    parser = sub.add_parser(
        "compare", help="frozen labels vs ReBeL-refreshed labels, at equal budgets"
    )
    _add_compare_args(parser)
    parser.set_defaults(func=run_compare)

    parser = sub.add_parser(
        "label-drift",
        help="how far a frozen dataset's labels sit from freshly-solved ones",
    )
    _add_drift_args(parser)
    parser.set_defaults(func=run_label_drift)


# --- commands --------------------------------------------------------------
def run_compare(args: argparse.Namespace) -> None:
    """Both arms, equal budgets, one report."""
    value_net = HoldemValueNetConfig(
        hidden_dim=args.hidden_dim,
        num_residual_blocks=args.residual_blocks,
        card_embedding_dim=args.card_embedding_dim,
    )
    config = ComparisonConfig(
        run_path=args.run,
        label_budget=args.label_budget,
        update_budget=args.update_budget,
        dataset=DatasetBuildConfig(
            river_examples=args.river_examples,
            turn_examples=args.turn_examples,
            flop_examples=args.flop_examples,
            self_play_examples=args.self_play_examples,
            teacher_updates=args.teacher_updates,
            teacher_batch_size=args.batch_size,
            teacher_learning_rate=args.learning_rate,
            value_net=value_net,
            workers=args.workers,
        ),
        rebuild_dataset=args.rebuild_dataset,
        fixed=FixedStudentConfig(
            label_budget=0,
            update_budget=0,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            eval_every=args.eval_every,
        ),
        iterative=OnlineStudentConfig(
            trajectories_per_iteration=args.trajectories_per_iteration,
            updates_per_iteration=args.updates_per_iteration,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            eval_every=args.eval_every,
            checkpoint_every=args.checkpoint_every,
        ),
        evaluation=EvaluationConfig(
            situations=SituationConfig(board_cards=4),
            boards_per_street=args.eval_boards,
            search_iterations=args.eval_iterations,
            flop_tree_depth_limit=args.flop_depth_limit,
            device=args.device,
        ),
        value_net=value_net,
        seed=args.seed,
        device=args.device,
    )
    run_comparison(config, verbose=True)


def _net_config_from_state(state) -> HoldemValueNetConfig:
    """Recover a value net's shape from its own weights.

    Saves the caller from having to repeat ``--hidden-dim`` and friends when
    probing a run that was trained with non-default sizes — and from the silent
    ``load_state_dict`` failure that follows when they forget.
    """
    linear = [
        k
        for k, v in state.items()
        if k.startswith("trunk.layers") and k.endswith(".weight") and v.dim() == 2
    ]
    return HoldemValueNetConfig(
        hidden_dim=int(state["trunk.layers.0.weight"].shape[0]),
        num_residual_blocks=len(linear),
        card_embedding_dim=int(state["board_embedding.card_embedding.weight"].shape[1]),
    )


def run_label_drift(args: argparse.Namespace) -> None:
    """Re-solve a frozen dataset's own inputs with a stronger evaluator."""
    layout = RunLayout.at(args.run)
    store = DatasetStore.open(layout.dataset)
    checkpoint = args.checkpoint or layout.iterative_student
    state = torch.load(checkpoint, map_location=args.device)
    # Size the network from the checkpoint rather than the defaults, so a run
    # made with --hidden-dim can be probed without repeating the flag.
    net = HoldemValueNet(_net_config_from_state(state))
    net.load_state_dict(state)

    results = measure_label_drift(
        store,
        net,
        sample_size=args.sample_size,
        seed=args.seed,
        cfr_iterations=args.cfr_iterations,
        device=args.device,
        results_path=layout.relabel,
    )
    print(json.dumps(results, indent=2))
    print(f"\n{results['verdict']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
