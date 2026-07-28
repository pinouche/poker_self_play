"""Paradigm B entry points, one module per thing you would run.

``leduc``    stages 0-3 on the toy game: tabular CFR, ReBeL self play, and the
             benchmark ladder that scores every method by exact exploitability.
``holdem``   stage 4: solve a real hold'em endgame and report exploitability.
``compare``  the arm 1 vs arm 2 label-regime experiment, and its drift probe.

Each runs standalone (``python -m paradigm_b.cli.leduc cfr``) and each
registers its subcommands with the top-level ``solve.py``, so both spellings
work and neither is the "real" one.
"""
