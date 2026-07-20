"""Evaluation-vs-opponent over the course of training.

The endpoint comparison (250 vs 1000 iterations) suggested more training makes
the agent worse against the heuristic, but two endpoints cannot tell monotonic
degradation from a noisy tail.  This trains the base config to 1000 iterations,
saves a checkpoint every `--every` iterations, and scores each against both the
heuristic (does it degrade?) and random (is the degradation style-specific or
global?).

    python trajectory.py --seed 0 --root traj/ --iterations 1000 --every 100
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from benchmark import build_config
from config import Config
from evaluation.evaluate import (
    bb_per_100_interval,
    duplicate_deal_scores,
    make_network_agent,
)
from evaluation.heuristic_agent import tight_aggressive
from evaluation.random_agent import CallingStationAgent, RandomAgent
from environment.state import action_space_for
from model.network import build_network, save_checkpoint
from representation.observation_encoder import ObservationEncoder
from training.replay_buffer import ReplayBuffer
from training.self_play import BatchedSelfPlayWorker, snapshot_agent
from training.trainer import Trainer


def train_with_checkpoints(config_name, seed, root, iterations, every):
    cfg = build_config(config_name, seed, iterations)
    out = os.path.join(root, f"{config_name}__seed{seed}")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder = ObservationEncoder.from_config(cfg)
    network = build_network(cfg)
    worker = BatchedSelfPlayWorker(
        cfg, network, encoder, seed=seed, num_envs=cfg.train.self_play_envs
    )
    buffer = ReplayBuffer(cfg.train.replay_capacity, encoder.observation_dim, network.num_actions)
    trainer = Trainer(cfg, network, device="cpu")
    rng = np.random.default_rng(seed)

    started = time.time()
    for iteration in range(1, iterations + 1):
        if iteration % cfg.train.league_snapshot_every == 0:
            worker.add_snapshot(snapshot_agent(cfg, network))
        transitions, _ = worker.generate(cfg.train.hands_per_iteration)
        buffer.extend(transitions)
        trainer.train_iteration(buffer, rng=rng)
        if iteration % every == 0:
            save_checkpoint(
                os.path.join(out, f"iter_{iteration:04d}.pt"), network, cfg,
                extra={"iteration": iteration, "buffer_size": len(buffer)},
            )
            print(f"[seed{seed}] saved iter {iteration} ({time.time() - started:.0f}s)", flush=True)
    return out


def evaluate_trajectory(out, deals):
    from config import EnvConfig
    from model.network import load_checkpoint

    seeds = list(range(1, deals + 1))
    curve = []
    checkpoints = sorted(f for f in os.listdir(out) if f.startswith("iter_"))
    for name in checkpoints:
        network, cfg, extra = load_checkpoint(os.path.join(out, name))
        # Score in the game the network was trained to observe (so the obs dim
        # and action space match), holding the money-determining rules and deal
        # seeds constant.  EV runouts are forced on to settle every hand the
        # same low-variance way.
        scoring = Config.from_dict(cfg.to_dict())
        scoring.env = EnvConfig(**{**scoring.env.__dict__, "all_in_ev_runout": True})
        encoder = ObservationEncoder.from_config(scoring)
        space = action_space_for(scoring.env)
        opponents = {
            "heuristic": tight_aggressive(seed=1, samples=40, action_space=space),
            "random": RandomAgent(),
            "calling_station": CallingStationAgent(),
        }
        hero = make_network_agent(network, cfg, temperature=0.0, name="hero")
        point = {"iteration": extra["iteration"], "buffer_size": extra.get("buffer_size")}
        for label, opponent in opponents.items():
            scores = duplicate_deal_scores(hero, [opponent, opponent], scoring, seeds, encoder)
            point[label] = bb_per_100_interval(scores, scoring.env.big_blind)
        curve.append(point)
        print(
            f"[{out.split('/')[-1]}] iter {point['iteration']:>4}: "
            f"heuristic {point['heuristic']['bb_per_100']:+7.0f}  "
            f"random {point['random']['bb_per_100']:+7.0f}  "
            f"station {point['calling_station']['bb_per_100']:+7.0f}",
            flush=True,
        )

    with open(os.path.join(out, "trajectory.json"), "w") as fh:
        json.dump(curve, fh, indent=2)
    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="long_base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", default="traj")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--every", type=int, default=100)
    parser.add_argument("--deals", type=int, default=500)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    out = os.path.join(args.root, f"{args.config}__seed{args.seed}")
    if not args.eval_only:
        out = train_with_checkpoints(
            args.config, args.seed, args.root, args.iterations, args.every
        )
    evaluate_trajectory(out, args.deals)


if __name__ == "__main__":
    main()
