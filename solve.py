"""Paradigm B entry point: solve Leduc, or train a ReBeL agent for it.

Three subcommands, one per thing worth running on its own:

``cfr``       Tabular CFR on the whole game tree.  The reference point — this
              is what "solved" means, and every other number is measured
              against it.
``rebel``     ReBeL self play: search with a value network at the depth limit,
              train the network on what search concluded, repeat.
``benchmark`` The ladder.  Full-depth CFR, depth-limited search with exactly
              solved leaves, depth-limited search with a trained network, and
              search with no value function at all — all scored by the same
              exact best response.  Its timings cover behaviour at *every*
              reachable belief state, because that is what scoring needs.
``holdem``    Solve a real hold'em turn endgame — 1,326-combination ranges, no
              card abstraction — and report exact exploitability.
``decision-time``
              What playing actually costs: the wall clock for the two solves a
              single hand requires.

Examples::

    python solve.py cfr --iterations 1000
    python solve.py rebel --iterations 60 --checkpoint checkpoints/rebel.pt
    python solve.py benchmark --checkpoint checkpoints/rebel.pt
    python solve.py decision-time --checkpoint checkpoints/rebel.pt
    python solve.py holdem --iterations 400
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import torch

from belief import PublicState, build_public_tree, initial_reach
from cfr import CFRConfig, CFRSolver, expected_values, exploitability
from game import KuhnPoker, LeducHoldem, build_tree
from rebel import ReBeLConfig, SelfPlayConfig, train_rebel
from search import (
    ContinualResolver,
    ResolveConfig,
    SubgameSolver,
    extract_tabular_policy,
)
from search.evaluate import leaf_counterfactual_values, leaf_reaches
from search.subgame import Gadget
from value_net import ExactLeafValues, PBSValueNet, ZeroLeafValues
from value_net.values import NetLeafValues

LEDUC_VALUE = -0.085606424078
GAMES = {"leduc": LeducHoldem, "kuhn": KuhnPoker}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser("cfr", help="tabular CFR on the full game tree")
    solve.add_argument("--game", choices=sorted(GAMES), default="leduc")
    solve.add_argument("--iterations", type=int, default=1000)
    solve.add_argument(
        "--variant", choices=("vanilla", "cfr_plus", "linear", "dcfr"), default="dcfr"
    )
    solve.add_argument("--report-every", type=int, default=100)

    rebel = sub.add_parser("rebel", help="ReBeL self play with a value network")
    rebel.add_argument("--iterations", type=int, default=60)
    rebel.add_argument("--trajectories", type=int, default=96)
    rebel.add_argument("--search-iterations", type=int, default=100)
    rebel.add_argument("--updates", type=int, default=200)
    rebel.add_argument("--batch-size", type=int, default=512)
    rebel.add_argument("--learning-rate", type=float, default=1e-3)
    rebel.add_argument("--exploration", type=float, default=0.25)
    rebel.add_argument("--eval-every", type=int, default=5)
    rebel.add_argument("--eval-iterations", type=int, default=1000)
    rebel.add_argument("--seed", type=int, default=0)
    rebel.add_argument("--device", type=str, default="cpu")
    rebel.add_argument("--checkpoint", type=str, default=None)

    bench = sub.add_parser("benchmark", help="score every method by exploitability")
    bench.add_argument("--checkpoint", type=str, default=None)
    bench.add_argument("--cfr-iterations", type=int, default=1000)
    bench.add_argument("--search-iterations", type=int, default=1000)
    bench.add_argument("--leaf-iterations", type=int, default=100)
    bench.add_argument("--skip-exact", action="store_true", help="the slow one")

    endgame = sub.add_parser(
        "holdem", help="solve a real hold'em turn endgame (1,326-combo ranges)"
    )
    endgame.add_argument("--iterations", type=int, default=400)
    endgame.add_argument("--report-every", type=int, default=50)
    endgame.add_argument("--pot", type=int, default=20)
    endgame.add_argument("--stack", type=int, default=100)
    endgame.add_argument("--max-raises", type=int, default=1)
    endgame.add_argument(
        "--board", type=str, default="As Ks 7h 3h", help="four turn cards"
    )

    timing = sub.add_parser(
        "decision-time", help="wall-clock cost of playing a single hand"
    )
    timing.add_argument("--checkpoint", type=str, default=None)
    timing.add_argument("--search-iterations", type=int, default=300)
    timing.add_argument("--leaf-iterations", type=int, default=100)
    timing.add_argument("--skip-exact", action="store_true")
    return parser.parse_args()


def run_cfr(args: argparse.Namespace) -> None:
    tree = build_tree(GAMES[args.game]())
    solver = CFRSolver(tree, getattr(CFRConfig, args.variant)())
    print(f"{args.game}: {tree.num_infosets} information sets, {tree.num_nodes} nodes")
    started = time.time()
    while solver.iteration < args.iterations:
        solver.iterate(min(args.report_every, args.iterations - solver.iteration))
        policy = solver.average_policy()
        print(
            f"  iter {solver.iteration:>6}  exploitability {exploitability(tree, policy):.6f}"
            f"  value {expected_values(tree, policy)[0]:+.7f}"
            f"  ({time.time() - started:.0f}s)",
            flush=True,
        )
    if args.game == "leduc":
        print(f"  exact Leduc value for comparison: {LEDUC_VALUE:+.7f}")


def run_rebel(args: argparse.Namespace) -> None:
    config = ReBeLConfig(
        iterations=args.iterations,
        updates_per_iteration=args.updates,
        batch_size=args.batch_size,
        evaluate_every=args.eval_every,
        seed=args.seed,
        device=args.device,
        self_play=SelfPlayConfig(
            trajectories_per_iteration=args.trajectories,
            search_iterations=args.search_iterations,
            exploration=args.exploration,
        ),
        evaluation=ResolveConfig(iterations=args.eval_iterations, depth_limit=1),
    )
    config.training.learning_rate = args.learning_rate
    started = time.time()
    run = train_rebel(config, verbose=True)
    print(f"trained in {time.time() - started:.0f}s")
    if args.checkpoint:
        torch.save(
            {"state_dict": run.net.state_dict(), "config": run.net.config},
            args.checkpoint,
        )
        print(f"wrote {args.checkpoint}")


def load_net(path: Optional[str]) -> Optional[PBSValueNet]:
    if not path:
        return None
    checkpoint = torch.load(path, weights_only=False)
    net = PBSValueNet(checkpoint["config"])
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net


def run_benchmark(args: argparse.Namespace) -> None:
    world = build_tree(LeducHoldem())
    rows = []

    started = time.time()
    solver = CFRSolver(world, CFRConfig.dcfr())
    solver.iterate(args.cfr_iterations)
    rows.append(
        (
            f"full-depth CFR, {args.cfr_iterations} iterations",
            solver.average_policy(),
            time.time() - started,
        )
    )

    started = time.time()
    range_solver = SubgameSolver(build_public_tree(), config=CFRConfig.dcfr())
    range_solver.solve(iterations=args.search_iterations)
    rows.append(
        (
            f"full-depth range CFR, {args.search_iterations} iterations",
            extract_tabular_policy(range_solver, world),
            time.time() - started,
        )
    )

    # The zero baseline is scored without the re-solving gadget: a value
    # function that predicts nothing has nothing to promise, and holding a
    # re-solve to a promise of zero is worse than not promising at all.
    plans = [
        (
            "full lookahead + safe re-solving",
            None,
            ResolveConfig(iterations=args.search_iterations, depth_limit=2),
        ),
        (
            "depth-limited, no value function",
            ZeroLeafValues(),
            ResolveConfig(iterations=args.search_iterations, safe_resolving=False),
        ),
        (
            "depth-limited, exact leaf values",
            None if args.skip_exact else ExactLeafValues(iterations=args.leaf_iterations),
            ResolveConfig(iterations=args.search_iterations),
        ),
        (
            "depth-limited, ReBeL value network",
            _net_values(args),
            ResolveConfig(iterations=args.search_iterations),
        ),
    ]
    for name, leaf_values, resolve_config in plans:
        if leaf_values is None and resolve_config.depth_limit == 1:
            continue  # a depth-limited row with nothing to evaluate its leaves
        started = time.time()
        resolver = ContinualResolver(leaf_values, resolve_config)
        rows.append((name, resolver.policy(world), time.time() - started))

    print(f"{'method':<42} {'exploitability':>14} {'value':>11} {'seconds':>8}")
    for name, policy, seconds in rows:
        print(
            f"{name:<42} {exploitability(world, policy):>14.5f} "
            f"{expected_values(world, policy)[0]:>+11.5f} {seconds:>8.0f}"
        )
    print(f"{'exact equilibrium (literature)':<42} {0.0:>14.5f} {LEDUC_VALUE:>+11.5f}")
    print(
        "\nexploitability: chips/hand a perfect counter-strategy wins; 0 is a Nash"
        "\n                equilibrium.  value: chips/hand to player 0 with this"
        "\n                strategy in both seats -- a consistency check against the"
        "\n                known game value, not a measure of quality."
        "\nseconds:        time to produce behaviour at *every* reachable belief"
        "\n                state, which is what scoring requires.  Playing one hand"
        "\n                needs two solves, not sixty -- see `decision-time`."
    )


def _net_values(args: argparse.Namespace):
    net = load_net(args.checkpoint)
    return NetLeafValues(net) if net is not None else None


def run_holdem(args: argparse.Namespace) -> None:
    """Solve a hold'em turn endgame and report exact exploitability.

    Real cards, both players' ranges over every one of the 1,326 two-card
    combinations, exact card removal, no bucketing.  The betting is abstracted
    (half pot / pot / all-in, capped) — that is the one approximation, and it is
    the one that would need action translation before facing a real opponent.
    """
    from environment.cards import Card
    from holdem import PublicState, TurnEndgameSpace, build_turn_tree
    from holdem.betting import Betting
    from search import SubgameSolver, strategy_map
    from search.best_response import subgame_exploitability

    board = tuple(Card.from_str(c).id for c in args.board.split())
    space = TurnEndgameSpace(board)
    root = PublicState(
        betting=Betting(
            starting_pot=args.pot, stack=args.stack, max_raises=args.max_raises
        ),
        board=board,
    )
    tree = build_turn_tree(root, depth_limit=None)
    print(
        f"turn endgame on {args.board}: {len(tree.decision_nodes())} public decision "
        f"nodes, {int(space.root_mask().sum())} hands each, pot {args.pot}, "
        f"stack {args.stack}"
    )
    solver = SubgameSolver(tree, config=CFRConfig.dcfr(), space=space)
    reach = space.initial_reach()
    started = time.time()
    while solver.iteration < args.iterations:
        solver.solve(
            reach=reach,
            iterations=min(args.report_every, args.iterations - solver.iteration),
        )
        total, per = subgame_exploitability(
            tree, strategy_map(solver), reach, space=space
        )
        print(
            f"  iter {solver.iteration:>5}  exploitability {total:8.4f} chips"
            f"  best responses {per[0]:+.4f} / {per[1]:+.4f}"
            f"  ({time.time() - started:.0f}s)",
            flush=True,
        )


def run_decision_time(args: argparse.Namespace) -> None:
    """What it costs to actually play, as opposed to what it costs to score.

    The benchmark's timings cover every reachable belief state, because exact
    exploitability demands behaviour everywhere.  A hand only ever visits two of
    them: one solve when the first round starts, one when the board lands.
    """
    candidates = [
        ("ReBeL value network", _net_values(args)),
        (
            "exact leaf values",
            None if args.skip_exact else ExactLeafValues(iterations=args.leaf_iterations),
        ),
    ]
    print(
        f"{'leaf values':<24} {'round 1':>10} {'round 2':>10} {'per hand':>10}"
        f"   ({args.search_iterations} CFR iterations per decision)"
    )
    for name, leaf_values in candidates:
        if leaf_values is None:
            continue
        first, second = _time_one_hand(leaf_values, args.search_iterations)
        print(
            f"{name:<24} {first:>9.2f}s {second:>9.2f}s {first + second:>9.2f}s"
        )


def _time_one_hand(leaf_value_fn, iterations: int):
    tree = build_public_tree(PublicState(), depth_limit=1)
    started = time.time()
    solver = SubgameSolver(tree, leaf_value_fn=leaf_value_fn, config=CFRConfig.dcfr())
    solver.solve(reach=initial_reach(), iterations=iterations)
    first = time.time() - started

    strategies = {n.public: solver.average_strategy(n) for n in tree.decision_nodes()}
    reachable = [
        (leaf, reach)
        for leaf, reach in leaf_reaches(tree, strategies, initial_reach())
        if reach.sum(axis=1).min() > 0.0
    ]
    leaf, reach = reachable[0]
    promised = leaf_counterfactual_values(leaf.public, reach, leaf_value_fn)
    started = time.time()
    for player in range(2):
        guarded = SubgameSolver(
            build_public_tree(leaf.public, depth_limit=1), config=CFRConfig.dcfr()
        )
        guarded.solve(
            reach=reach,
            iterations=iterations,
            gadget=Gadget(player=player, promised=promised[1 - player]),
        )
    return first, time.time() - started


def main() -> None:
    args = parse_args()
    commands = {
        "cfr": run_cfr,
        "rebel": run_rebel,
        "benchmark": run_benchmark,
        "holdem": run_holdem,
        "decision-time": run_decision_time,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
