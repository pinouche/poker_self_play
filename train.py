#!/usr/bin/env python3
"""Self-play training entry point.

    python train.py                       # defaults
    python train.py --iterations 500 --hands-per-iteration 128
    python train.py --reward-mode normalized_chip_return
    python train.py --resume checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict

import numpy as np
import torch

from config import Config, reward_bound, resolve_device
from evaluation.evaluate import evaluate_suite, format_results, summarize_headline
from model.network import build_network, load_checkpoint, save_checkpoint
from representation.observation_encoder import ObservationEncoder
from training.replay_buffer import ReplayBuffer
from training.self_play import (
    BatchedSelfPlayWorker,
    SelfPlayWorker,
    build_opponent_pool,
    snapshot_agent,
)
from training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3-player poker agent by self-play")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--hands-per-iteration", type=int, default=None)
    parser.add_argument("--updates-per-iteration", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--min-buffer", dest="min_buffer_before_training", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None, help="entropy coefficient")
    parser.add_argument("--beta", type=float, default=None, help="reverse-KL coefficient")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument(
        "--reward-mode",
        choices=["normalized_chip_return", "bb_normalized", "chip_return", "binary"],
        default=None,
    )
    parser.add_argument(
        "--reward-clip",
        type=float,
        default=None,
        help="symmetric clip for normalized_chip_return; 0 disables it",
    )
    parser.add_argument(
        "--unbounded-q",
        action="store_true",
        help="use a linear Q head even when the reward mode would allow tanh",
    )
    parser.add_argument("--starting-stack", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-hands", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="load a saved config JSON")
    parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument(
        "--opponent-pool",
        type=str,
        default=None,
        help="comma-separated opponents to mix into self-play, e.g. "
        "'tight_aggressive,checkpoint,calling_station'",
    )
    parser.add_argument("--opponent-mix-prob", type=float, default=None)
    parser.add_argument(
        "--self-play-envs",
        type=int,
        default=None,
        help="hands advanced in lockstep to batch the network (1 = sequential)",
    )
    parser.add_argument("--league-snapshot-every", type=int, default=None)
    parser.add_argument("--league-size", type=int, default=None)
    parser.add_argument("--league-recency-decay", type=float, default=None)
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config) if args.config else Config()

    for name, target in [
        ("iterations", "train"),
        ("hands_per_iteration", "train"),
        ("updates_per_iteration", "train"),
        ("batch_size", "train"),
        ("learning_rate", "train"),
        ("replay_capacity", "train"),
        ("min_buffer_before_training", "train"),
        ("self_play_envs", "train"),
        ("alpha", "train"),
        ("beta", "train"),
        ("gamma", "train"),
        ("lam", "train"),
        ("seed", "train"),
        ("device", "train"),
        ("eval_every", "train"),
        ("eval_hands", "train"),
        ("checkpoint_every", "train"),
        ("checkpoint_dir", "train"),
        ("opponent_mix_prob", "train"),
        ("league_snapshot_every", "train"),
        ("league_size", "train"),
        ("league_recency_decay", "train"),
        ("reward_mode", "env"),
        ("reward_clip", "env"),
        ("starting_stack", "env"),
    ]:
        value = getattr(args, name, None)
        if value is not None:
            setattr(getattr(cfg, target), name, value)

    if args.opponent_pool:
        cfg.train.opponent_pool = tuple(
            name.strip() for name in args.opponent_pool.split(",") if name.strip()
        )
        if cfg.train.opponent_mix_prob <= 0:
            cfg.train.opponent_mix_prob = 0.5

    # The Q head bounding is derived from the reward mode by `build_network`;
    # only an explicit --unbounded-q overrides it (experiment B).
    if args.unbounded_q:
        cfg.model.bounded_q = False
    return cfg


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_metrics(metrics: Dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    device = resolve_device(cfg.train.device)
    set_seeds(cfg.train.seed)

    encoder = ObservationEncoder.from_config(cfg)
    print(encoder.describe())

    if args.resume:
        network, loaded_cfg, extra = load_checkpoint(args.resume, device=device)
        if loaded_cfg.obs != cfg.obs:
            raise SystemExit(
                "resumed checkpoint has a different observation config; "
                "pass --config with the checkpoint's config to continue"
            )
        start_iteration = int(extra.get("iteration", 0))
        print(f"resumed from {args.resume} at iteration {start_iteration}")
    else:
        network = build_network(cfg)
        start_iteration = 0
    network.to(device)

    print(
        f"device={device}  parameters={network.num_parameters():,}  "
        f"reward_mode={cfg.env.reward_mode}  reward_bound={reward_bound(cfg.env)}  "
        f"bounded_q={network.q_head.bounded}  q_scale={network.q_head.scale}"
    )

    buffer = ReplayBuffer(
        capacity=cfg.train.replay_capacity,
        observation_dim=encoder.observation_dim,
        num_actions=encoder.spec.num_actions,
        store_next_obs=cfg.train.store_next_obs,
    )
    print(f"replay capacity={buffer.capacity:,}  ({buffer.memory_bytes() / 1e6:.0f} MB reserved)\n")

    pool = build_opponent_pool(cfg, network, device=device)
    use_league = "checkpoint" in cfg.train.opponent_pool
    worker_class = BatchedSelfPlayWorker if cfg.train.self_play_envs > 1 else SelfPlayWorker
    worker_kwargs = dict(device=device, seed=cfg.train.seed, opponent_pool=pool)
    if worker_class is BatchedSelfPlayWorker:
        worker_kwargs["num_envs"] = cfg.train.self_play_envs
    worker = worker_class(cfg, network, encoder, **worker_kwargs)
    print(
        f"self-play: {worker_class.__name__}"
        + (f" ({cfg.train.self_play_envs} envs in lockstep)" if cfg.train.self_play_envs > 1 else "")
    )
    if cfg.train.opponent_pool:
        print(
            f"opponent pool: {', '.join(cfg.train.opponent_pool)}  "
            f"(mixed into {cfg.train.opponent_mix_prob:.0%} of hands)"
        )
    trainer = Trainer(cfg, network, device=device)
    sample_rng = np.random.default_rng(cfg.train.seed)

    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.train.checkpoint_dir, "config.json"))

    history = []
    started = time.time()
    iteration = start_iteration

    try:
        for iteration in range(start_iteration + 1, cfg.train.iterations + 1):
            tic = time.time()
            if use_league and iteration % max(1, cfg.train.league_snapshot_every) == 0:
                worker.add_snapshot(snapshot_agent(cfg, network, device=device))

            transitions, sp_stats = worker.generate(cfg.train.hands_per_iteration)
            buffer.extend(transitions)
            metrics = trainer.train_iteration(buffer, rng=sample_rng)
            elapsed = time.time() - tic

            if iteration % cfg.train.log_every == 0:
                status = (
                    f"[{iteration:>5}/{cfg.train.iterations}] "
                    f"buffer={len(buffer):>7,}  "
                    f"hands/s={cfg.train.hands_per_iteration / max(elapsed, 1e-6):5.1f}  "
                    f"showdown={sp_stats['showdown_rate']:.2f}  "
                    f"dec/hand={sp_stats['decisions_per_hand']:.1f}"
                )
                if metrics:
                    status += "  " + format_metrics(metrics)
                else:
                    status += "  (filling buffer)"
                print(status, flush=True)
                history.append({"iteration": iteration, **sp_stats, **metrics})

            if (
                not args.no_eval
                and cfg.train.eval_every > 0
                and iteration % cfg.train.eval_every == 0
            ):
                results = evaluate_suite(
                    network, cfg, num_hands=cfg.train.eval_hands, seed=iteration, device=device
                )
                print(format_results(results))
                print("  headline:", format_metrics(summarize_headline(results)), flush=True)

            if cfg.train.checkpoint_every > 0 and iteration % cfg.train.checkpoint_every == 0:
                path = os.path.join(cfg.train.checkpoint_dir, f"iter_{iteration:06d}.pt")
                save_checkpoint(path, network, cfg, extra={"iteration": iteration})
                save_checkpoint(
                    os.path.join(cfg.train.checkpoint_dir, "latest.pt"),
                    network,
                    cfg,
                    extra={"iteration": iteration},
                )
                print(f"  saved {path}", flush=True)

    except KeyboardInterrupt:
        print("\ninterrupted; saving final checkpoint")

    final = os.path.join(cfg.train.checkpoint_dir, "latest.pt")
    save_checkpoint(final, network, cfg, extra={"iteration": iteration})
    with open(os.path.join(cfg.train.checkpoint_dir, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    print(
        f"done: {iteration} iterations in {time.time() - started:.1f}s; "
        f"final checkpoint {final}"
    )


if __name__ == "__main__":
    main()
