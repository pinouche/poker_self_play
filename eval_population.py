#!/usr/bin/env python3
"""Evaluate every network of a co-evolution run against the baseline suite.

Each ``latest_net<k>.pt`` (or ``iter_<it>_net<k>.pt``) is scored in the standard
battery from :mod:`evaluation.evaluate`: self-play, then heads-up-ish matchups
against the random, calling-station and tight-aggressive heuristic baselines
(the network in one seat, the baseline in the other two).

    python eval_population.py                                   # checkpoints/latest_net*.pt
    python eval_population.py --checkpoint-dir checkpoints/pop5 --hands 2000
    python eval_population.py --which iter_000200               # a specific snapshot
"""

from __future__ import annotations

import argparse
import glob
import os
import re

from config import resolve_device
from evaluation.evaluate import evaluate_suite, format_results, summarize_headline
from model.network import load_checkpoint


def _net_index(path: str) -> int:
    match = re.search(r"net(\d+)\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument(
        "--which", default="latest",
        help="checkpoint prefix: 'latest' or a snapshot like 'iter_000200'",
    )
    parser.add_argument("--hands", type=int, default=1000, help="hands per matchup")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    pattern = os.path.join(args.checkpoint_dir, f"{args.which}_net*.pt")
    paths = sorted(glob.glob(pattern), key=_net_index)
    if not paths:
        raise SystemExit(f"no checkpoints matched {pattern}")

    device = resolve_device(args.device)
    print(f"device={device}  evaluating {len(paths)} networks, {args.hands} hands each")

    headlines = {}
    for path in paths:
        k = _net_index(path)
        network, cfg, extra = load_checkpoint(path, device=device)
        results = evaluate_suite(
            network, cfg, num_hands=args.hands, seed=args.seed, device=device
        )
        print(
            f"\n================ net{k}  ({os.path.basename(path)}, "
            f"iter {extra.get('iteration', '?')}) ================"
        )
        print(format_results(results))
        headlines[k] = summarize_headline(results)

    # Compact cross-network comparison on the three baseline matchups.
    matchups = ["vs_random", "vs_calling_station", "vs_heuristic"]
    print("\n\n=== bb/100 vs baselines (higher is better) ===")
    print("  net " + "".join(f"{label:>22}" for label in matchups))
    for k in sorted(headlines):
        cells = "".join(
            f"{headlines[k].get(f'{label}/bb_per_100', float('nan')):>22.1f}"
            for label in matchups
        )
        print(f"  {k:<4}{cells}")


if __name__ == "__main__":
    main()
