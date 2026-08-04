"""Stage 4: solve a real hold'em endgame and report exact exploitability.

Real cards, both players' ranges over every one of the 1,326 two-card
combinations, exact card removal, no bucketing.  The betting is abstracted
(half pot / pot / all-in, capped) — that is the one approximation, and it is the
one that would need action translation before facing a real opponent.

Examples::

    python -m paradigm_b.cli.holdem --iterations 400
    python -m paradigm_b.cli.holdem --board "As Ks 7h 3h" --pot 20 --stack 100

Reachable as ``python solve.py holdem`` too.
"""

from __future__ import annotations

import argparse
import time

from common.cards import Card
from paradigm_b.core.cfr import CFRConfig
from paradigm_b.core.search import SubgameSolver, strategy_map
from paradigm_b.core.search.best_response import subgame_exploitability
from paradigm_b.holdem import PublicState, TurnEndgameSpace, build_turn_tree
from paradigm_b.holdem.engine.betting import Betting


def _add_endgame_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--pot", type=int, default=20)
    parser.add_argument("--stack", type=int, default=100)
    parser.add_argument("--max-raises", type=int, default=1)
    parser.add_argument(
        "--board", type=str, default="As Ks 7h 3h", help="four turn cards"
    )


def register(sub) -> None:
    """Add this module's subcommand to a shared subparser object."""
    parser = sub.add_parser(
        "holdem", help="solve a real hold'em turn endgame (1,326-combo ranges)"
    )
    _add_endgame_args(parser)
    parser.set_defaults(func=run_endgame)


def run_endgame(args: argparse.Namespace) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _add_endgame_args(parser)
    run_endgame(parser.parse_args())


if __name__ == "__main__":
    main()
