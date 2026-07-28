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

from paradigm_a.config import (
    MODEL_PRESETS,
    Config,
    model_config,
    reward_bound,
    resolve_device,
    updates_for_transitions,
)
from paradigm_a.evaluation.evaluate import evaluate_suite, format_results, summarize_headline
from paradigm_a.model.network import build_network, load_checkpoint, save_checkpoint
from paradigm_a.representation.observation_encoder import ObservationEncoder
from paradigm_a.training.replay_buffer import ReplayBuffer
from paradigm_a.training.self_play import (
    BatchedSelfPlayWorker,
    CoevolutionSelfPlayWorker,
    PopulationSelfPlayWorker,
    SelfPlayWorker,
    build_opponent_pool,
    snapshot_agent,
)
from paradigm_a.training.trainer import Trainer


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
    parser.add_argument(
        "--network-size",
        choices=list(MODEL_PRESETS),
        default=None,
        help="trunk/embedding size preset for every network built this run "
        "(league slots included).  Default is the `medium` baseline",
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
    parser.add_argument("--league-weighting", choices=["linear", "geometric"], default=None)
    parser.add_argument(
        "--population", dest="population_self_play", action="store_true", default=None,
        help="population self-play: villains sampled from a library of checkpoints, "
        "randomised full games, later-street scenario mixture",
    )
    parser.add_argument(
        "--num-policies", dest="num_policies", type=int, default=None,
        help="co-evolving population size: number of live networks, all optimised "
        "at once, with one sampled per seat per hand (default 1 = shared-network "
        "self-play)",
    )
    parser.add_argument(
        "--league", dest="league_population", action="store_true", default=None,
        help="role-structured league: learners + champions + explorers, with "
        "behaviour-aware promotion (score = 0.6 strength + 0.4 diversity).  "
        "Champions are frozen unless --champion-mode anchored",
    )
    parser.add_argument("--league-learners", type=int, default=None)
    parser.add_argument("--league-champions", type=int, default=None)
    parser.add_argument("--league-explorers", type=int, default=None)
    parser.add_argument("--league-manage-every", type=int, default=None)
    parser.add_argument("--league-promotion-hands", type=int, default=None)
    parser.add_argument(
        "--champion-mode", choices=["frozen", "anchored"], default=None,
        help="frozen (default): every champion is a hard snapshot.  anchored: "
        "--league-champion-anchors slots stay frozen as the gauntlet spine and "
        "the rest become veterans that keep training, anchored to their own "
        "snapshot with a measured, enforced drift budget",
    )
    parser.add_argument("--league-champion-anchors", type=int, default=None)
    parser.add_argument("--league-veteran-max-kl", type=float, default=None)
    parser.add_argument("--league-veteran-lr-scale", type=float, default=None)
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config) if args.config else Config()

    # A size preset replaces the whole trunk/embedding block, so keep the head
    # knobs that are unrelated to capacity (and may come from --config).
    if args.network_size:
        cfg.model = model_config(
            args.network_size,
            dropout=cfg.model.dropout,
            bounded_q=cfg.model.bounded_q,
            q_scale=cfg.model.q_scale,
        )

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
        ("league_weighting", "train"),
        ("population_self_play", "train"),
        ("num_policies", "train"),
        ("league_population", "train"),
        ("league_learners", "train"),
        ("league_champions", "train"),
        ("league_explorers", "train"),
        ("league_manage_every", "train"),
        ("league_promotion_hands", "train"),
        ("champion_mode", "train"),
        ("league_champion_anchors", "train"),
        ("league_veteran_max_kl", "train"),
        ("league_veteran_lr_scale", "train"),
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


def train_coevolution(cfg: Config, args: argparse.Namespace, encoder, device: str) -> None:
    """Training loop for a co-evolving population of ``num_policies`` networks.

    Every network is optimised at once.  Each hand samples one network per seat
    (with replacement); each seat's transitions train the network that produced
    them.  Networks are independent -- separate random initialisations, separate
    optimisers, separate replay buffers -- and they only influence one another
    through the games they play together.

    Kept separate from :func:`main`'s single-network loop on purpose: the two
    share no state, and folding them together would only obscure both.
    """
    n = int(cfg.train.num_policies)

    # Distinct random initialisations: each ``build_network`` call advances the
    # global torch RNG, so the population starts diverse rather than identical.
    networks = [build_network(cfg).to(device) for _ in range(n)]

    # Split the replay budget evenly so total memory matches a single-network
    # run.  Each network's replay ratio still matches the single-network case:
    # it sees ~1/n of the transitions and does ~1/n of the updates.
    per_net_capacity = max(1_000, cfg.train.replay_capacity // n)
    buffers = [
        ReplayBuffer(
            capacity=per_net_capacity,
            observation_dim=encoder.observation_dim,
            num_actions=encoder.spec.num_actions,
            store_next_obs=cfg.train.store_next_obs,
        )
        for _ in range(n)
    ]
    trainers = [Trainer(cfg, net, device=device) for net in networks]
    worker = CoevolutionSelfPlayWorker(
        cfg, networks, encoder, device=device, seed=cfg.train.seed,
        num_envs=max(1, cfg.train.self_play_envs),
    )

    print(
        f"device={device}  num_policies={n}  parameters/net={networks[0].num_parameters():,}  "
        f"reward_mode={cfg.env.reward_mode}  reward_bound={reward_bound(cfg.env)}"
    )
    print(
        f"self-play: co-evolving population ({n} live networks, all optimised; "
        f"{worker.num_envs} envs in lockstep; per-net replay {per_net_capacity:,})\n"
    )

    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.train.checkpoint_dir, "config.json"))

    sample_rng = np.random.default_rng(cfg.train.seed)
    history = []
    total_transitions = total_hands = total_updates = 0
    started = time.time()
    iteration = 0

    def save_all(it: int) -> None:
        for k in range(n):
            save_checkpoint(
                os.path.join(cfg.train.checkpoint_dir, f"latest_net{k}.pt"),
                networks[k], cfg, extra={"iteration": it, "network": k},
            )

    try:
        for iteration in range(1, cfg.train.iterations + 1):
            tic = time.time()
            per_net_transitions, sp_stats = worker.generate(cfg.train.hands_per_iteration)

            # Each network trains only on the data it generated, so its replay
            # ratio is governed by its own transition count.
            iter_updates = trained = 0
            accumulated: Dict[str, float] = {}
            for k in range(n):
                buffers[k].extend(per_net_transitions[k])
                num_updates = updates_for_transitions(cfg.train, len(per_net_transitions[k]))
                metrics = trainers[k].train_iteration(
                    buffers[k], num_updates=num_updates, rng=sample_rng
                )
                if metrics:
                    trained += 1
                    iter_updates += num_updates
                    for key, value in metrics.items():
                        accumulated[key] = accumulated.get(key, 0.0) + value
            # Report the population average over the networks that trained.
            metrics = (
                {key: value / trained for key, value in accumulated.items()} if trained else {}
            )

            total_transitions += sp_stats["transitions"]
            total_hands += cfg.train.hands_per_iteration
            total_updates += iter_updates
            elapsed = time.time() - tic
            replay_ratio = (total_updates * cfg.train.batch_size) / max(1, total_transitions)

            if iteration % cfg.train.log_every == 0:
                status = (
                    f"[{iteration:>5}/{cfg.train.iterations}] "
                    f"hands={total_hands/1e3:6.0f}k  "
                    f"buf={'/'.join(f'{len(b)/1e3:.0f}k' for b in buffers)}  "
                    f"hands/s={cfg.train.hands_per_iteration / max(elapsed, 1e-6):5.0f}  "
                    f"upd/it={iter_updates:>3}  rr={replay_ratio:4.1f}  "
                    f"showdown={sp_stats['showdown_rate']:.2f}"
                )
                status += ("  " + format_metrics(metrics)) if metrics else "  (filling buffers)"
                print(status, flush=True)
                history.append({
                    "iteration": iteration,
                    "total_hands": total_hands,
                    "total_transitions": total_transitions,
                    "total_updates": total_updates,
                    "replay_ratio": replay_ratio,
                    **sp_stats,
                    **metrics,
                })

            if (
                not args.no_eval
                and cfg.train.eval_every > 0
                and iteration % cfg.train.eval_every == 0
            ):
                for k in range(n):
                    results = evaluate_suite(
                        networks[k], cfg, num_hands=cfg.train.eval_hands,
                        seed=iteration, device=device,
                    )
                    print(f"  net{k} headline:", format_metrics(summarize_headline(results)),
                          flush=True)

            if cfg.train.checkpoint_every > 0 and iteration % cfg.train.checkpoint_every == 0:
                for k in range(n):
                    save_checkpoint(
                        os.path.join(cfg.train.checkpoint_dir, f"iter_{iteration:06d}_net{k}.pt"),
                        networks[k], cfg, extra={"iteration": iteration, "network": k},
                    )
                save_all(iteration)
                print(f"  saved {n} checkpoints at iter {iteration}", flush=True)

    except KeyboardInterrupt:
        print("\ninterrupted; saving final checkpoints")

    save_all(iteration)
    with open(os.path.join(cfg.train.checkpoint_dir, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    print(
        f"done: {iteration} iterations in {time.time() - started:.1f}s; "
        f"{n} networks saved to {cfg.train.checkpoint_dir}/latest_net*.pt"
    )


def train_league(cfg: Config, args: argparse.Namespace, encoder, device: str) -> None:
    """Training loop for the role-structured league.

    Learners and explorers are optimised (one optimiser and replay buffer per
    slot); anchored champions only play.  Under ``champion_mode="anchored"``
    veteran champions are optimised too, at a reduced learning rate and with
    their improvement target anchored to their own frozen snapshot.  A management
    pass every ``league_manage_every`` iterations recomputes strength /
    diversity / coverage, promotes earned learners into the champion ladder,
    retires stale champions, culls the weakest learner and resets any veteran
    that has drifted past its budget -- and any slot whose network was replaced
    has its optimiser and buffer rebuilt here, since this loop owns them.
    """
    from paradigm_a.training.league import LeaguePopulation, Role
    from paradigm_a.training.league_play import LeagueSelfPlayWorker

    population = LeaguePopulation(cfg, device=device, seed=cfg.train.seed)
    worker = LeagueSelfPlayWorker(
        cfg, population, encoder, device=device, seed=cfg.train.seed,
        num_envs=max(1, cfg.train.self_play_envs),
    )
    trainable = population.trainable_indices()
    per_slot_capacity = max(1_000, cfg.train.replay_capacity // max(1, len(trainable)))

    def new_buffer() -> ReplayBuffer:
        return ReplayBuffer(
            capacity=per_slot_capacity,
            observation_dim=encoder.observation_dim,
            num_actions=encoder.spec.num_actions,
            store_next_obs=cfg.train.store_next_obs,
        )

    def new_trainer(slot: int) -> Trainer:
        """Optimiser for one slot; veterans get the anchored, slowed-down variant."""
        member = population.members[slot]
        if member.snapshot is None:
            return Trainer(cfg, member.network, device=device)
        return Trainer(
            cfg, member.network, device=device,
            reference_mode="snapshot", reference_network=member.snapshot,
            lr_scale=float(cfg.train.league_veteran_lr_scale),
        )

    buffers = {slot: new_buffer() for slot in trainable}
    trainers = {slot: new_trainer(slot) for slot in trainable}

    counts = population.role_counts()
    champion_detail = (
        f"{counts['champion']} champions"
        if population.champion_mode == "frozen"
        else f"{counts['anchor']} anchors + {counts['veteran']} veterans"
    )
    print(
        f"device={device}  league={len(population)} networks "
        f"({counts['learner']} learners, {champion_detail}, "
        f"{counts['explorer']} explorers)  parameters/net="
        f"{population.networks[0].num_parameters():,}"
    )
    if population.champion_mode == "anchored":
        print(
            f"  champions: anchored mode -- {counts['veteran']} veterans train at "
            f"{cfg.train.league_veteran_lr_scale:g}x lr against their own snapshot, "
            f"reset past KL {cfg.train.league_veteran_max_kl:g}; the gauntlet reads "
            f"the {counts['anchor']} frozen anchors only"
        )
        if counts["veteran"] == 0:
            print(
                "  WARNING: league_champion_anchors >= league_champions, so there are "
                "no veterans and anchored mode is identical to frozen."
            )
        if counts["anchor"] < 2:
            print(
                f"  WARNING: only {counts['anchor']} frozen anchor(s); the "
                "over-specialisation gate compares the oldest and newest generation "
                "and is inert below 2.  Raise --league-champion-anchors."
            )
    print(
        f"  matchmaking: {cfg.train.league_match_learner_vs_learner:.0%} L-v-L / "
        f"{cfg.train.league_match_learner_vs_champion:.0%} L-v-C / "
        f"{cfg.train.league_match_mixed:.0%} mixed"
    )
    print(
        f"  promotion: >= {cfg.train.league_promotion_mbb_per_100:.0f} mbb/100 over "
        f"{cfg.train.league_promotion_hands:,} hands, beating the median champion "
        f"(>= {cfg.train.league_promotion_min_generations:.0%} of generations faced)"
    )

    # Both promotion paths gate on a *per-member* hand count.  Set above what the
    # run can generate and the ladder freezes silently -- champions stay at their
    # random initialisation and every champion metric reads flat for the whole
    # run.  Say so up front rather than leaving it to be inferred from a dead plot.
    per_member_per_iter = (
        cfg.train.hands_per_iteration * cfg.env.num_players / max(1, len(population))
    )
    gate = int(cfg.train.league_promotion_hands)
    iters_to_gate = gate / max(per_member_per_iter, 1e-9)
    if iters_to_gate > cfg.train.iterations:
        print(
            f"  WARNING: no learner can reach the {gate:,}-hand promotion gate in "
            f"{cfg.train.iterations:,} iterations (a member accrues "
            f"~{per_member_per_iter:.0f} hands/iteration, so the gate needs "
            f"~{iters_to_gate:,.0f}).  The champion pool will never change and "
            f"every champion metric will read flat.  Lower "
            f"--league-promotion-hands (<= {int(per_member_per_iter * cfg.train.iterations):,})."
        )
    else:
        print(
            f"  a member accrues ~{per_member_per_iter:.0f} hands/iteration "
            f"-> promotion gate reachable from iteration ~{iters_to_gate:.0f}"
        )
    total_hands_planned = cfg.train.hands_per_iteration * cfg.train.iterations
    if int(cfg.train.league_explorer_reset_hands) > total_hands_planned:
        print(
            f"  WARNING: explorers never reset -- "
            f"league_explorer_reset_hands={cfg.train.league_explorer_reset_hands:,} "
            f"exceeds the {total_hands_planned:,} league hands this run generates."
        )
    print(
        f"  per-slot replay {per_slot_capacity:,}  ({len(trainable)} optimised slots)\n",
        flush=True,
    )

    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.train.checkpoint_dir, "config.json"))

    sample_rng = np.random.default_rng(cfg.train.seed)
    history = []
    total_transitions = total_hands = total_updates = 0
    started = time.time()
    iteration = 0

    try:
        for iteration in range(1, cfg.train.iterations + 1):
            tic = time.time()
            per_member, sp_stats = worker.generate(cfg.train.hands_per_iteration)

            iter_updates = trained = 0
            accumulated: Dict[str, float] = {}
            for slot in list(buffers):
                transitions = per_member[slot]
                buffers[slot].extend(transitions)
                num_updates = updates_for_transitions(cfg.train, len(transitions))
                metrics = trainers[slot].train_iteration(
                    buffers[slot], num_updates=num_updates, rng=sample_rng
                )
                if metrics:
                    trained += 1
                    iter_updates += num_updates
                    for key, value in metrics.items():
                        accumulated[key] = accumulated.get(key, 0.0) + value
            metrics = (
                {key: value / trained for key, value in accumulated.items()} if trained else {}
            )

            total_transitions += sp_stats["transitions"]
            total_hands += sp_stats["hands"]
            total_updates += iter_updates
            elapsed = time.time() - tic

            # Explorers are wiped on their own hand schedule, independently of
            # the management cadence.
            reset = population.maybe_reset_explorers()
            for slot in reset:
                buffers[slot] = new_buffer()
                trainers[slot] = new_trainer(slot)
            if reset:
                print(f"  reinitialised explorers {reset} from scratch", flush=True)

            if iteration % cfg.train.log_every == 0:
                status = (
                    f"[{iteration:>5}/{cfg.train.iterations}] "
                    f"hands={total_hands/1e3:7.0f}k  "
                    f"hands/s={sp_stats['hands'] / max(elapsed, 1e-6):5.0f}  "
                    f"upd/it={iter_updates:>3}  "
                    f"collected={sp_stats['collected_fraction']:.2f}  "
                    f"showdown={sp_stats['showdown_rate']:.2f}"
                )
                status += ("  " + format_metrics(metrics)) if metrics else "  (filling buffers)"
                print(status, flush=True)
                history.append({
                    "iteration": iteration, "total_hands": total_hands,
                    "total_transitions": total_transitions, "total_updates": total_updates,
                    **{k: v for k, v in sp_stats.items() if k != "transitions_per_member"},
                    **metrics,
                })

            if cfg.train.league_manage_every > 0 and iteration % cfg.train.league_manage_every == 0:
                # Fixed seed: paired trends (same validation states, same deals),
                # so a frozen ladder reads flat instead of swinging on card luck.
                eval_seed = int(cfg.train.league_eval_seed)
                league_metrics = population.evaluate(encoder, seed=eval_seed)
                gauntlet = population.champion_gauntlet(
                    min_hands=int(cfg.train.league_gauntlet_min_hands)
                )

                # Held-out heuristic, measured *before* managing so it can retire
                # champions that are catastrophic against an external opponent.
                external_scores = np.full(len(population), np.nan)
                heuristic_line = None
                if not args.no_eval and cfg.train.eval_hands > 0:
                    from paradigm_a.training.league_metrics import evaluate_vs_heuristic

                    champions = population.champion_generations()
                    champ_scores = evaluate_vs_heuristic(
                        [population.members[c].network for c in champions],
                        cfg, cfg.train.eval_hands, encoder, device=device, seed=eval_seed,
                    )
                    for slot, score in zip(champions, champ_scores):
                        external_scores[slot] = score
                    best_learner = max(
                        population.indices(Role.LEARNER),
                        key=lambda i: league_metrics["score"][i],
                    )
                    learner_score = evaluate_vs_heuristic(
                        [population.members[best_learner].network],
                        cfg, cfg.train.eval_hands, encoder, device=device, seed=eval_seed,
                    )[0]
                    heuristic_line = (
                        "  vs tight_aggressive (bb/100): champions oldest->newest "
                        + " ".join(f"{s:+.0f}" for s in champ_scores)
                        + f"  |  mean {champ_scores.mean():+.0f}  best {champ_scores.max():+.0f}"
                        + f"  |  top learner {learner_score:+.0f}"
                    )
                    history.append({
                        "iteration": iteration,
                        "champions_vs_heuristic": champ_scores.tolist(),
                        "champions_vs_heuristic_mean": float(champ_scores.mean()),
                        "champions_vs_heuristic_best": float(champ_scores.max()),
                        "top_learner_vs_heuristic": float(learner_score),
                    })

                # External competence panel: evaluate only the most_different
                # shortlist, so a diverse-but-externally-weak learner is never
                # frozen as a champion.
                panel_scores = {}
                if not args.no_eval and cfg.train.league_external_panel:
                    from paradigm_a.training.league_metrics import evaluate_vs_panel

                    shortlist = population.most_different_shortlist(league_metrics["score"])
                    if shortlist:
                        worst = evaluate_vs_panel(
                            [population.members[i].network for i in shortlist],
                            cfg, cfg.train.league_external_panel, cfg.train.league_panel_hands,
                            encoder, device=device, seed=eval_seed,
                        )
                        panel_scores = dict(zip(shortlist, worst))

                report = population.manage(
                    league_metrics["score"], diversity=league_metrics["diversity"],
                    external_scores=external_scores, panel_scores=panel_scores,
                )
                for slot in report.rebuilt_slots:
                    if population.members[slot].trainable:
                        buffers[slot] = new_buffer()
                        trainers[slot] = new_trainer(slot)

                # Veterans that blew the drift budget go back to their snapshot.
                # Slots manage() just replaced are skipped: the drift reading
                # above predates the network now sitting in them.
                drift = league_metrics["veteran_drift"]
                snapped = population.enforce_veteran_drift(
                    drift, skip=report.rebuilt_slots
                )
                for slot in snapped:
                    buffers[slot] = new_buffer()
                    trainers[slot] = new_trainer(slot)
                veterans = population.veteran_indices()
                if veterans:
                    values = [drift[i] for i in veterans if np.isfinite(drift[i])]
                    print(
                        "  veteran drift KL(live||snapshot): "
                        + " ".join(f"{drift[i]:.4f}" for i in veterans)
                        + (f"  |  max {max(values):.4f}" if values else "")
                        + f" / budget {cfg.train.league_veteran_max_kl:g}"
                        + (f"  -> reset {snapped} to snapshot" if snapped else ""),
                        flush=True,
                    )
                    history.append({
                        "iteration": iteration,
                        "veteran_drift": [float(drift[i]) for i in veterans],
                        "veterans_reset_to_snapshot": snapped,
                    })

                over = int(gauntlet["over_specialised"].sum())
                print(population.summary(league_metrics, top=5), flush=True)
                print(
                    f"  gauntlet: mean generations beaten="
                    f"{gauntlet['beats'].mean():.1f}/{len(gauntlet['generations'])}  "
                    f"over-specialised learners={over}", flush=True
                )
                if heuristic_line is not None:
                    print(heuristic_line, flush=True)
                print(f"  manage: {report.describe()}", flush=True)

                # Approximate exploitability of the strongest champion (absolute
                # yardstick): a trained best-responder's win rate against it.
                every = int(cfg.train.league_exploitability_every)
                final = iteration >= cfg.train.iterations
                if not args.no_eval and int(cfg.train.league_exploitability_iters) > 0 and (
                    final or (every > 0 and (iteration // cfg.train.league_manage_every) % every == 0)
                ):
                    from paradigm_a.training.exploitability import estimate_exploitability

                    champions = population.champion_generations()
                    finite = [c for c in champions if np.isfinite(external_scores[c])]
                    if finite:
                        best_champ = max(finite, key=lambda c: external_scores[c])
                        exploit = estimate_exploitability(
                            population.members[best_champ].network, cfg, encoder,
                            iters=int(cfg.train.league_exploitability_iters),
                            device=device, seed=eval_seed,
                        )
                        print(
                            f"  exploitability of best champion (slot {best_champ}, "
                            f"{external_scores[best_champ]:+.0f} vs heuristic): a best-responder "
                            f"wins {exploit:+.0f} bb/100  (0 = unexploitable)", flush=True
                        )
                        history.append({"iteration": iteration, "exploitability_bb_per_100": exploit,
                                        "exploited_champion_vs_heuristic": float(external_scores[best_champ])})
                print("", flush=True)

            if cfg.train.checkpoint_every > 0 and iteration % cfg.train.checkpoint_every == 0:
                for slot, member in enumerate(population.members):
                    save_checkpoint(
                        os.path.join(cfg.train.checkpoint_dir, f"latest_slot{slot:02d}.pt"),
                        member.network, cfg,
                        extra={"iteration": iteration, "slot": slot, "role": member.role.value},
                    )
                print(f"  saved {len(population)} league checkpoints", flush=True)

    except KeyboardInterrupt:
        print("\ninterrupted; saving the league")

    for slot, member in enumerate(population.members):
        save_checkpoint(
            os.path.join(cfg.train.checkpoint_dir, f"latest_slot{slot:02d}.pt"),
            member.network, cfg,
            extra={"iteration": iteration, "slot": slot, "role": member.role.value},
        )
    with open(os.path.join(cfg.train.checkpoint_dir, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    print(
        f"done: {iteration} iterations in {time.time() - started:.1f}s; "
        f"{len(population)} league networks saved to {cfg.train.checkpoint_dir}"
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    if args.resume and args.network_size:
        raise SystemExit(
            "--network-size cannot be combined with --resume: the resumed "
            "checkpoint carries its own model config"
        )
    device = resolve_device(cfg.train.device)
    set_seeds(cfg.train.seed)

    encoder = ObservationEncoder.from_config(cfg)
    print(encoder.describe())

    # The role-structured league is its own loop: learners + frozen champions +
    # explorers, with behaviour-aware promotion and culling.
    if cfg.train.league_population:
        if cfg.train.population_self_play or cfg.train.num_policies > 1:
            raise SystemExit(
                "--league is mutually exclusive with --population and --num-policies > 1"
            )
        if args.resume:
            raise SystemExit(
                "--resume is not supported for the league; each slot has its own checkpoint"
            )
        train_league(cfg, args, encoder, device)
        return

    # Co-evolving population is a distinct training loop: n live networks, all
    # optimised at once.  It shares nothing with the single-network path below.
    if cfg.train.num_policies > 1:
        if cfg.train.population_self_play:
            raise SystemExit(
                "num_policies>1 (co-evolution) and --population are mutually exclusive"
            )
        if args.resume:
            raise SystemExit(
                "--resume is not supported for co-evolution (num_policies>1); "
                "each network has its own checkpoint (latest_net<k>.pt)"
            )
        train_coevolution(cfg, args, encoder, device)
        return

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

    population = bool(cfg.train.population_self_play)
    use_league = (not population) and "checkpoint" in cfg.train.opponent_pool
    if population:
        worker = PopulationSelfPlayWorker(
            cfg, network, encoder, device=device, seed=cfg.train.seed
        )
        print(
            f"self-play: population (library<= {cfg.train.population_library_size}, "
            f"{cfg.train.league_weighting} recency, snapshot every "
            f"{cfg.train.population_snapshot_every} iters, entry-street mix "
            f"{tuple(round(p, 2) for p in cfg.train.entry_street_probs)})"
        )
    else:
        pool = build_opponent_pool(cfg, network, device=device)
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
    total_transitions = 0
    total_hands = 0
    total_updates = 0
    started = time.time()
    iteration = start_iteration

    try:
        for iteration in range(start_iteration + 1, cfg.train.iterations + 1):
            tic = time.time()
            if population:
                worker.maybe_snapshot()
            elif use_league and iteration % max(1, cfg.train.league_snapshot_every) == 0:
                worker.add_snapshot(snapshot_agent(cfg, network, device=device))

            transitions, sp_stats = worker.generate(cfg.train.hands_per_iteration)
            buffer.extend(transitions)
            total_transitions += len(transitions)
            total_hands += cfg.train.hands_per_iteration

            # Number of gradient updates is driven by the replay-ratio knob
            # (new transitions per update), so the replay ratio stays fixed as
            # hands-per-iteration or decisions-per-hand vary.
            num_updates = updates_for_transitions(cfg.train, len(transitions))
            metrics = trainer.train_iteration(buffer, num_updates=num_updates, rng=sample_rng)
            total_updates += num_updates if metrics else 0
            elapsed = time.time() - tic
            replay_ratio = (total_updates * cfg.train.batch_size) / max(1, total_transitions)

            if iteration % cfg.train.log_every == 0:
                status = (
                    f"[{iteration:>5}/{cfg.train.iterations}] "
                    f"hands={total_hands/1e3:6.0f}k  buffer={len(buffer):>8,}  "
                    f"hands/s={cfg.train.hands_per_iteration / max(elapsed, 1e-6):5.0f}  "
                    f"upd/it={num_updates:>2}  rr={replay_ratio:4.1f}  "
                    f"showdown={sp_stats['showdown_rate']:.2f}"
                )
                if metrics:
                    status += "  " + format_metrics(metrics)
                else:
                    status += "  (filling buffer)"
                print(status, flush=True)
                history.append({
                    "iteration": iteration,
                    "total_hands": total_hands,
                    "total_transitions": total_transitions,
                    "total_updates": total_updates,
                    "replay_ratio": replay_ratio,
                    **sp_stats,
                    **metrics,
                })

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
