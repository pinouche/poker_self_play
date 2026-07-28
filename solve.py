#!/usr/bin/env python
"""Paradigm B entry point: one door onto every solver command.

The commands live in :mod:`paradigm_b.cli`, three modules split by what they
drive; this file only gathers their subparsers so you do not have to remember
which module a command belongs to.

Stages 0-3, Leduc (:mod:`paradigm_b.cli.leduc`):

``cfr``       Tabular CFR on the whole game tree.  The reference point — this
              is what "solved" means, and every other number is measured
              against it.
``rebel``     ReBeL self play: search with a value network at the depth limit,
              train the network on what search concluded, repeat.
``benchmark`` The ladder: full-depth CFR, depth-limited search with exactly
              solved leaves, with a trained network, and with no value function
              at all — all scored by the same exact best response.
``decision-time``
              What playing actually costs: the wall clock for the two solves a
              single hand requires.

Stage 4, real hold'em (:mod:`paradigm_b.cli.holdem`):

``holdem``    Solve a real hold'em turn endgame — 1,326-combination ranges, no
              card abstraction — and report exact exploitability.

The label-regime experiment (:mod:`paradigm_b.cli.compare`):

``compare``   Arm 1 (frozen labels) vs arm 2 (ReBeL-refreshed labels), at equal
              label and update budgets, scored by held-out exploitability per
              street.
``label-drift``
              The follow-up: re-solve a frozen dataset's own inputs with a
              stronger evaluator and measure how far its labels had drifted.

Examples::

    python solve.py cfr --iterations 1000
    python solve.py rebel --iterations 60 --checkpoint checkpoints/rebel.pt
    python solve.py benchmark --checkpoint checkpoints/rebel.pt
    python solve.py decision-time --checkpoint checkpoints/rebel.pt
    python solve.py holdem --iterations 400
    python solve.py compare --run runs/labels-01 --label-budget 20000
    python solve.py label-drift --run runs/labels-01
"""

from __future__ import annotations

import argparse

from paradigm_b.cli import compare, holdem, leduc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for module in (leduc, holdem, compare):
        module.register(sub)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
