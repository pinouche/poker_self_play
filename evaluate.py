#!/usr/bin/env python3
"""Evaluation entry point.

    python evaluate.py --checkpoint checkpoints/latest.pt --hands 600
    python evaluate.py --checkpoint checkpoints/latest.pt \
                       --against checkpoints/iter_000100.pt
"""

from __future__ import annotations

import argparse
import json

from config import Config, resolve_device
from evaluation.evaluate import (
    evaluate_against_checkpoint,
    evaluate_suite,
    format_results,
)
from model.network import build_network, load_checkpoint
from representation.observation_encoder import ObservationEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained poker network")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--against", type=str, default=None, help="a second checkpoint to play against"
    )
    parser.add_argument("--hands", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--json", action="store_true", help="emit raw JSON results")
    parser.add_argument(
        "--no-heuristic",
        action="store_true",
        help="skip the Monte-Carlo heuristic bot (much faster)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.checkpoint:
        network, cfg, extra = load_checkpoint(args.checkpoint, device=device)
        print(f"loaded {args.checkpoint} (iteration {extra.get('iteration', '?')})")
    else:
        cfg = Config()
        network = build_network(cfg).to(device)
        print("no checkpoint supplied: evaluating a randomly initialised network")

    encoder = ObservationEncoder(cfg.obs)
    print(encoder.describe())

    results = evaluate_suite(
        network,
        cfg,
        num_hands=args.hands,
        seed=args.seed,
        device=device,
        include_heuristic=not args.no_heuristic,
    )

    if args.against:
        opponent, opponent_cfg, opponent_extra = load_checkpoint(args.against, device=device)
        if opponent_cfg.obs != cfg.obs:
            raise SystemExit("checkpoints have incompatible observation configs")
        results["vs_checkpoint"] = evaluate_against_checkpoint(
            network, opponent, cfg, num_hands=args.hands, seed=args.seed, device=device
        )
        print(f"opponent checkpoint: {args.against} (iteration {opponent_extra.get('iteration', '?')})")

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_results(results))


if __name__ == "__main__":
    main()
