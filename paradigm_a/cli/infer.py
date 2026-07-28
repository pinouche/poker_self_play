#!/usr/bin/env python3
"""Inference entry point.

    python infer.py                                   # built-in example state
    python infer.py --table-state hand.json
    cat hand.json | python infer.py --table-state -
    python infer.py --checkpoint checkpoints/latest.pt --mode sample --temperature 0.8
"""

from __future__ import annotations

import argparse
import json
import sys

from paradigm_a.config import Config, resolve_device
from paradigm_a.inference.suggest_action import SuggestionEngine, load_table_state
from paradigm_a.representation.observation_encoder import ObservationEncoder

EXAMPLE_TABLE_STATE = {
    "hero": {
        "cards": [{"rank": "4", "suit": "h"}, {"rank": "2", "suit": "s"}],
        "stack": 280,
        "bet": 0,
        "equity_pct": None,
        "active": True,
    },
    "villain_left": {
        "cards": [],
        "stack": 280,
        "bet": 0,
        "equity_pct": None,
        "active": True,
    },
    "villain_right": {
        "cards": [],
        "stack": 280,
        "bet": 0,
        "equity_pct": None,
        "active": True,
    },
    "board": [
        {"rank": "8", "suit": "s"},
        {"rank": "9", "suit": "h"},
        {"rank": "K", "suit": "h"},
    ],
    "pot": 60,
    "small_blind": 10,
    "big_blind": 20,
    "dealer": "hero",
    "street": "flop",
    "showdown": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suggest a play for a table state")
    parser.add_argument(
        "--table-state",
        type=str,
        default=None,
        help="path to a table-state JSON file, or '-' for stdin",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="shapes the reported distribution and sampling; 0 collapses onto the best action",
    )
    parser.add_argument("--mode", choices=["argmax", "sample"], default="argmax")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--show-spec", action="store_true", help="print the observation layout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.table_state == "-":
        table_state = json.load(sys.stdin)
    elif args.table_state:
        table_state = load_table_state(args.table_state)
    else:
        table_state = EXAMPLE_TABLE_STATE
        print("# no --table-state given; using the built-in example", file=sys.stderr)

    if args.checkpoint:
        engine = SuggestionEngine.from_checkpoint(args.checkpoint, device=device)
    else:
        print(
            "# no --checkpoint given; using a randomly initialised network",
            file=sys.stderr,
        )
        engine = SuggestionEngine.untrained(Config(), device=device)

    if args.show_spec:
        print(ObservationEncoder(engine.cfg.obs).describe(), file=sys.stderr)

    suggestion = engine.suggest(
        table_state, temperature=args.temperature, mode=args.mode
    )
    print(json.dumps(suggestion, indent=2))


if __name__ == "__main__":
    main()
