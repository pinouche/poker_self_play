#!/usr/bin/env python3
"""Benchmark harness: train a matrix of configurations, then score them.

Three subcommands, so cells can be parallelised across processes:

    python benchmark.py train    --config full --seed 0 --root runs/
    python benchmark.py evaluate --config full --seed 0 --root runs/
    python benchmark.py report   --root runs/

`report` archives every cell's per-iteration history alongside the tables, so
the figures can be restyled later without the run directories (which are large
and get deleted) and without re-training:

    python benchmark.py replot   --report runs/report

The matrix is an **ablation from a single reference configuration** rather than
a grid: every row changes one thing relative to `full`, so a difference is
attributable.  `legacy` is the original configuration from before any of the
fixes, included as the "before" reference point.

Scoring uses duplicate deals (each deal replayed with the hero in all three
seats, same deck order) against the three fixed baselines, and always reports a
95% interval.  bb/100 on independent hands is far too noisy to compare
configurations without one.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, List

import numpy as np

from config import Config, EnvConfig, ModelConfig, ObsConfig
from environment.state import action_space_for

# --- configuration matrix --------------------------------------------------
FINE_BETS = (0.20, 0.40, 0.60, 0.85, 1.25)
FINE_RAISES = (2.0, 2.5, 3.0, 4.0, 5.5)

POOL = ("heuristic", "loose_passive", "calling_station", "checkpoint")


def _reference(cfg: Config) -> None:
    """Everything the diagnosis and roadmap produced, switched on."""
    cfg.train.alpha = 0.05
    cfg.train.beta = 0.5
    cfg.train.opponent_pool = POOL
    cfg.train.opponent_mix_prob = 0.4
    cfg.train.opponent_heuristic_samples = 25
    cfg.train.league_snapshot_every = 40
    cfg.train.league_recency_decay = 0.5
    cfg.train.self_play_envs = 64
    cfg.obs.use_equity_feature = True
    cfg.env.all_in_ev_runout = True


def _full(cfg: Config) -> None:
    _reference(cfg)


def _no_equity(cfg: Config) -> None:
    _reference(cfg)
    cfg.obs.use_equity_feature = False


def _no_ev_runout(cfg: Config) -> None:
    _reference(cfg)
    cfg.env.all_in_ev_runout = False


def _no_pool(cfg: Config) -> None:
    _reference(cfg)
    cfg.train.opponent_pool = ()
    cfg.train.opponent_mix_prob = 0.0


def _fine_abstraction(cfg: Config) -> None:
    _reference(cfg)
    cfg.env = EnvConfig(
        **{
            **cfg.env.__dict__,
            "bet_fractions": FINE_BETS,
            "raise_multipliers": FINE_RAISES,
        }
    )


def _alpha_010(cfg: Config) -> None:
    _reference(cfg)
    cfg.train.alpha = 0.10


def _alpha_002(cfg: Config) -> None:
    _reference(cfg)
    cfg.train.alpha = 0.02


def _sweep_reference(cfg: Config) -> None:
    """Reference for the alpha sweep, with the benchmark's lessons applied.

    Three deliberate differences from `full`:

    * **The pool holds only self-snapshots.**  The earlier pool contained
      `calling_station` and heuristic-family bots, which are also evaluation
      baselines -- so those columns were partly training on the test opponent.
      With league play alone, all three baselines are genuinely held out.
    * **Equity off.**  It was not demonstrated (+22 bb/100 against random, pure
      noise) and costs 3-4x self-play throughput.
    * **Finer bet abstraction.**  Never worse in the ablation, and it is the
      lever most likely to raise the ceiling.
    """
    cfg.train.beta = 0.5
    cfg.train.opponent_pool = ("checkpoint",)
    cfg.train.opponent_mix_prob = 0.4
    cfg.train.league_snapshot_every = 40
    cfg.train.league_recency_decay = 0.5
    cfg.train.self_play_envs = 64
    cfg.obs.use_equity_feature = False
    # Pin to the original 128/6 trunk so the archived sweep/capacity/population
    # results remain reproducible now that the default model is "medium".
    cfg.model = ModelConfig(**CAPACITIES["base"], bounded_q=None)
    cfg.env = EnvConfig(
        **{
            **cfg.env.__dict__,
            "bet_fractions": FINE_BETS,
            "raise_multipliers": FINE_RAISES,
            "all_in_ev_runout": True,
        }
    )


#: Alpha values swept at three seeds each.  Alpha alone sets the sharpness of
#: the converged policy (pi* ∝ exp(Q/alpha)), and it was the single largest
#: effect found so far.
SWEEP_ALPHAS = (0.01, 0.02, 0.03, 0.04, 0.05)


def _make_sweep_config(alpha: float) -> Callable[[Config], None]:
    def apply(cfg: Config) -> None:
        _sweep_reference(cfg)
        cfg.train.alpha = alpha

    return apply


#: Alpha is settled at 0.02: the sweep from 0.01 to 0.05 was flat within seed
#: noise, so it is fixed here and everything below varies something else.
SETTLED_ALPHA = 0.02


def _stage1(cfg: Config, preset: str) -> None:
    """Research-plan Stage 1: model-free self-play only, size = ``preset``.

    No population, no fixed opponents -- the size effect is isolated.  Uses the
    settled reward/critic/abstraction and the replay-ratio knob from the plan.
    """
    from config import model_config

    cfg.model = model_config(preset, bounded_q=None)
    cfg.train.alpha = SETTLED_ALPHA
    cfg.train.beta = 0.5
    cfg.train.opponent_pool = ()
    cfg.train.opponent_mix_prob = 0.0
    cfg.train.population_self_play = False
    cfg.train.self_play_envs = 64
    cfg.train.optimizer = "adamw"
    cfg.train.transitions_per_update = 512
    cfg.train.batch_size = 512
    cfg.train.replay_capacity = 1_000_000
    cfg.obs.use_equity_feature = False
    cfg.env = EnvConfig(
        **{
            **cfg.env.__dict__,
            "bet_fractions": FINE_BETS,
            "raise_multipliers": FINE_RAISES,
            "all_in_ev_runout": True,
        }
    )


def _make_stage1(preset: str) -> Callable[[Config], None]:
    return lambda cfg: _stage1(cfg, preset)

#: Trunk sizes.  `base` is the configuration everything so far was trained
#: with; it had never been varied, and the plateau at ~16k hands is as
#: consistent with a capacity ceiling as with a data one.
CAPACITIES: Dict[str, dict] = {
    "small": dict(
        hidden_dim=64, num_residual_blocks=4, head_hidden=64,
        embed_cards=32, embed_board=32, embed_players=32,
        embed_pot_history=64, embed_position=16,
    ),
    "base": dict(
        hidden_dim=128, num_residual_blocks=6, head_hidden=128,
        embed_cards=64, embed_board=64, embed_players=64,
        embed_pot_history=128, embed_position=32,
    ),
    "large": dict(
        hidden_dim=256, num_residual_blocks=8, head_hidden=256,
        embed_cards=128, embed_board=128, embed_players=128,
        embed_pot_history=256, embed_position=64,
    ),
    "xlarge": dict(
        hidden_dim=384, num_residual_blocks=10, head_hidden=384,
        embed_cards=192, embed_board=192, embed_players=192,
        embed_pot_history=384, embed_position=96,
    ),
}


def _make_capacity_config(size: str) -> Callable[[Config], None]:
    def apply(cfg: Config) -> None:
        _sweep_reference(cfg)
        cfg.train.alpha = SETTLED_ALPHA
        cfg.model = ModelConfig(**CAPACITIES[size], bounded_q=None)

    return apply


def _make_population_config(size: str = "base") -> Callable[[Config], None]:
    """Population self-play: the feature under test.

    Villains are sampled from a library of frozen checkpoints (linear recency
    weighting), games run start-to-finish from randomised stacks and blinds, and
    a minority of hands have the learner enter on a later street.  No fixed bot
    is ever in the library, so all three baselines stay held out.
    """

    def apply(cfg: Config) -> None:
        _make_capacity_config(size)(cfg)
        cfg.train.population_self_play = True
        cfg.train.league_weighting = "linear"
        cfg.train.population_library_size = 20
        cfg.train.population_snapshot_every = 10
        cfg.train.entry_street_probs = (0.75, 0.13, 0.08, 0.04)

    return apply


def _make_broad_league_config(size: str = "base") -> Callable[[Config], None]:
    """More snapshots, taken more often, seated more often.

    Still self-snapshots only -- adding fixed bots would put an evaluation
    baseline back into training, which is exactly the contamination the sweep
    removed.  "Broader" here means a deeper and fresher league, not a more
    varied cast.
    """

    def apply(cfg: Config) -> None:
        _make_capacity_config(size)(cfg)
        cfg.train.league_size = 12
        cfg.train.league_snapshot_every = 15
        cfg.train.opponent_mix_prob = 0.6

    return apply


def _legacy(cfg: Config) -> None:
    """The original configuration, before any of the fixes."""
    cfg.train.alpha = 0.5
    cfg.train.beta = 0.5
    cfg.train.opponent_pool = ()
    cfg.train.opponent_mix_prob = 0.0
    cfg.train.self_play_envs = 1
    cfg.obs.use_equity_feature = False
    cfg.env.all_in_ev_runout = False


CONFIGS: Dict[str, Callable[[Config], None]] = {
    "full": _full,
    "no_equity": _no_equity,
    "no_ev_runout": _no_ev_runout,
    "no_pool": _no_pool,
    "fine_abstraction": _fine_abstraction,
    "alpha_0.10": _alpha_010,
    "alpha_0.02": _alpha_002,
    "legacy": _legacy,
    **{f"sweep_a{alpha:g}": _make_sweep_config(alpha) for alpha in SWEEP_ALPHAS},
    # Capacity sweep at the settled alpha.
    **{f"cap_{size}": _make_capacity_config(size) for size in CAPACITIES},
    # Same configurations, run 4x longer to separate a capacity ceiling from a
    # data ceiling.  Selected by passing --iterations 1000.
    "long_base": _make_capacity_config("base"),
    "long_large": _make_capacity_config("large"),
    "league_broad": _make_broad_league_config("base"),
    "population": _make_population_config("base"),
    "long_population": _make_population_config("base"),
    # Research-plan Stage 1: size comparison, self-play only.
    "stage1_tiny": _make_stage1("tiny"),
    "stage1_medium": _make_stage1("medium"),
    "stage1_large": _make_stage1("large"),
}

#: Order used in reports; `full` first so ablations read as deltas from it.
CONFIG_ORDER = list(CONFIGS)


def build_config(name: str, seed: int, iterations: int) -> Config:
    if name not in CONFIGS:
        raise SystemExit(f"unknown config {name!r}; choose from {', '.join(CONFIGS)}")
    cfg = Config()
    cfg.obs = ObsConfig()
    CONFIGS[name](cfg)
    cfg.train.seed = seed
    cfg.train.iterations = iterations
    cfg.train.hands_per_iteration = 64
    cfg.train.updates_per_iteration = 32
    cfg.train.batch_size = 256
    cfg.train.min_buffer_before_training = 3000
    cfg.train.device = "cpu"
    cfg.train.eval_every = 0
    cfg.train.checkpoint_every = 0
    cfg.model.bounded_q = None
    return cfg


def cell_dir(root: str, name: str, seed: int) -> str:
    return os.path.join(root, f"{name}__seed{seed}")


# --- training --------------------------------------------------------------
def run_training(name: str, seed: int, root: str, iterations: int) -> None:
    import torch

    from model.network import build_network, save_checkpoint
    from representation.observation_encoder import ObservationEncoder
    from training.replay_buffer import ReplayBuffer
    from training.self_play import (
        BatchedSelfPlayWorker,
        SelfPlayWorker,
        build_opponent_pool,
        snapshot_agent,
    )
    from training.trainer import Trainer

    cfg = build_config(name, seed, iterations)
    out = cell_dir(root, name, seed)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    encoder = ObservationEncoder.from_config(cfg)
    network = build_network(cfg)
    pool = build_opponent_pool(cfg, network)
    use_league = "checkpoint" in cfg.train.opponent_pool

    if cfg.train.self_play_envs > 1:
        worker = BatchedSelfPlayWorker(
            cfg, network, encoder, seed=seed, num_envs=cfg.train.self_play_envs,
            opponent_pool=pool,
        )
    else:
        worker = SelfPlayWorker(cfg, network, encoder, seed=seed, opponent_pool=pool)

    buffer = ReplayBuffer(
        cfg.train.replay_capacity, encoder.observation_dim, network.num_actions
    )
    trainer = Trainer(cfg, network, device="cpu")
    rng = np.random.default_rng(seed)

    history: List[dict] = []
    started = time.time()
    for iteration in range(1, iterations + 1):
        if use_league and iteration % cfg.train.league_snapshot_every == 0:
            worker.add_snapshot(snapshot_agent(cfg, network))
        transitions, sp_stats = worker.generate(cfg.train.hands_per_iteration)
        buffer.extend(transitions)
        metrics = trainer.train_iteration(buffer, rng=rng)
        history.append({"iteration": iteration, **sp_stats, **metrics})
        if iteration % 25 == 0:
            print(
                f"[{name} seed{seed}] {iteration}/{iterations} "
                f"entropy={metrics.get('entropy', float('nan')):.3f} "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )

    save_checkpoint(os.path.join(out, "latest.pt"), network, cfg,
                    extra={"iteration": iterations, "config_name": name, "seed": seed})
    cfg.save(os.path.join(out, "config.json"))
    with open(os.path.join(out, "history.json"), "w") as fh:
        json.dump(history, fh)
    print(f"[{name} seed{seed}] done in {time.time() - started:.0f}s -> {out}", flush=True)


# --- evaluation ------------------------------------------------------------
def run_evaluation(name: str, seed: int, root: str, deals: int) -> None:
    from evaluation.evaluate import (
        bb_per_100_interval,
        duplicate_deal_scores,
        make_network_agent,
    )
    from evaluation.heuristic_agent import tight_aggressive
    from evaluation.random_agent import CallingStationAgent, RandomAgent
    from model.network import load_checkpoint
    from representation.observation_encoder import ObservationEncoder

    out = cell_dir(root, name, seed)
    network, cfg, _ = load_checkpoint(os.path.join(out, "latest.pt"))
    hero = make_network_agent(network, cfg, temperature=0.0, name=name)

    # A network can only be scored in the game it was trained to observe: the
    # observation width depends on its feature set (equity on/off) and on its
    # action space, so the scoring environment must inherit both from the
    # checkpoint.  What *is* held constant across configurations is everything
    # that determines the money -- blinds, stacks, and the deal seeds -- so
    # chip results stay comparable and the deals stay paired.
    scoring = Config.from_dict(cfg.to_dict())
    # Settle every evaluation the same way.  EV runouts are unbiased, so this
    # does not shift any mean; it just removes runout luck from the measurement
    # for every configuration equally, including those trained without it.
    scoring.env = EnvConfig(**{**scoring.env.__dict__, "all_in_ev_runout": True})
    encoder = ObservationEncoder.from_config(scoring)
    space = action_space_for(scoring.env)
    seeds = list(range(1, deals + 1))

    opponents = {
        "random": RandomAgent(),
        "calling_station": CallingStationAgent(),
        "heuristic": tight_aggressive(seed=1, samples=40, action_space=space),
    }

    results = {}
    for label, opponent in opponents.items():
        scores = duplicate_deal_scores(hero, [opponent, opponent], scoring, seeds, encoder)
        results[label] = bb_per_100_interval(scores, scoring.env.big_blind)
        np.save(os.path.join(out, f"scores_{label}.npy"), scores)
        print(
            f"[{name} seed{seed}] vs {label:<16}"
            f"{results[label]['bb_per_100']:+8.1f} +-{results[label]['ci_half_width']:5.1f}",
            flush=True,
        )

    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)


# --- reporting -------------------------------------------------------------
def histories_path(out_prefix: str) -> str:
    return f"{out_prefix}_histories.json"


def _compact(record: dict) -> dict:
    """Round floats for the archive; 6 significant digits is well below plot resolution."""
    return {
        key: float(f"{value:.6g}") if isinstance(value, float) else value
        for key, value in record.items()
    }


def curves_from_archive(archive: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """Pick the curves to draw: seed 0 only, so there is one line per config."""
    curves = {}
    for cell, history in archive.items():
        name, _, seed = cell.partition("__seed")
        if seed == "0":
            curves[name] = history
    return curves


def build_report(root: str, out_prefix: str) -> None:
    from training.plotting import load_history

    cells: Dict[str, List[dict]] = {}
    # Every seed is archived, not just the one that gets plotted: the run
    # directories are large and routinely deleted, and re-training a cell to
    # recover its curve costs hours.  The archive is what `replot` reads.
    archive: Dict[str, List[dict]] = {}
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if not os.path.isdir(path) or "__seed" not in entry:
            continue
        name, seed = entry.split("__seed")
        results_path = os.path.join(path, "results.json")
        if os.path.exists(results_path):
            with open(results_path) as fh:
                cells.setdefault(name, []).append({"seed": int(seed), **json.load(fh)})
        try:
            archive[entry] = [_compact(record) for record in load_history(path)]
        except FileNotFoundError:
            pass

    ordered = [n for n in CONFIG_ORDER if n in cells] + [
        n for n in sorted(cells) if n not in CONFIG_ORDER
    ]

    # The ablation note only applies when `full` is present to be a baseline.
    note = (
        "Rows below `full` are single-change ablations from it."
        if "full" in cells
        else "Every row differs only in the swept parameter."
    )
    lines = [
        "# Benchmark results",
        "",
        "Duplicate-deal scoring: every deal is replayed with the hero in all three",
        "seats on the same deck order. `*` marks a 95% interval that excludes zero,",
        "but note it covers evaluation noise only -- the seed spread is wider and is",
        "shown as whiskers in the results plot.",
        note,
        "",
        "| config | seeds | vs random | vs calling station | vs heuristic |",
        "|---|---:|---:|---:|---:|",
    ]

    summary: Dict[str, Dict[str, float]] = {}
    for name in ordered:
        rows = sorted(cells[name], key=lambda r: r["seed"])
        cell_text = []
        summary[name] = {}
        for opponent in ("random", "calling_station", "heuristic"):
            means = [r[opponent]["bb_per_100"] for r in rows]
            halves = [r[opponent]["ci_half_width"] for r in rows]
            mean = float(np.mean(means))
            # Pooled interval across seeds (independent evaluations).
            half = float(np.sqrt(np.sum(np.square(halves))) / len(halves))
            summary[name][opponent] = mean
            star = "*" if abs(mean) > half else ""
            cell_text.append(f"{mean:+.0f} ±{half:.0f}{star}")
        lines.append(f"| `{name}` | {len(rows)} | " + " | ".join(cell_text) + " |")

    if "full" in summary:
        lines += ["", "## Ablation deltas (config minus `full`)", "",
                  "| config | vs random | vs calling station | vs heuristic |",
                  "|---|---:|---:|---:|"]
        for name in ordered:
            if name == "full":
                continue
            deltas = [
                f"{summary[name][o] - summary['full'][o]:+.0f}"
                for o in ("random", "calling_station", "heuristic")
            ]
            lines.append(f"| `{name}` | " + " | ".join(deltas) + " |")

    report = "\n".join(lines)
    with open(f"{out_prefix}.md", "w") as fh:
        fh.write(report + "\n")
    with open(f"{out_prefix}.json", "w") as fh:
        json.dump({"summary": summary, "cells": cells}, fh, indent=2)
    print(report)

    if archive:
        # No indent: this is bulk data, and indenting it triples the file.
        with open(histories_path(out_prefix), "w") as fh:
            json.dump(archive, fh)
        print(f"wrote {histories_path(out_prefix)} ({len(archive)} cells)")

    draw_plots(summary, cells, archive, out_prefix)


def draw_plots(summary, cells, archive, out_prefix: str) -> None:
    """Render both figures.  Shared by the `report` and `replot` commands."""
    from training.plotting import plot_runs

    if summary:
        _plot_results(summary, cells, f"{out_prefix}_results.png")

    curves = curves_from_archive(archive)
    if curves:
        plot_runs(
            curves,
            metrics=["entropy", "kl_target_vs_policy", "q_loss", "showdown_rate"],
            window=9,
            out=f"{out_prefix}_curves.png",
            title="Training curves by configuration (seed 0)",
            reference_lines=_entropy_reference(list(curves)),
        )


def replot(prefix: str, out_prefix: str) -> None:
    """Redraw the figures from an existing report, with no run directories.

    `report.json` and `report_histories.json` together hold everything the
    figures are built from, so they can be restyled long after the runs that
    produced them have been deleted.
    """
    summary: Dict[str, Dict[str, float]] = {}
    cells: Dict[str, List[dict]] = {}
    if os.path.exists(f"{prefix}.json"):
        with open(f"{prefix}.json") as fh:
            report = json.load(fh)
        summary, cells = report.get("summary", {}), report.get("cells", {})
    else:
        print(f"no {prefix}.json; skipping the results figure")

    archive: Dict[str, List[dict]] = {}
    if os.path.exists(histories_path(prefix)):
        with open(histories_path(prefix)) as fh:
            archive = json.load(fh)
    else:
        print(
            f"no {histories_path(prefix)}; skipping the curves figure.  Reports "
            "written before history archiving did not keep the per-iteration "
            "metrics; re-run `report` from the run directories to create it."
        )

    if not summary and not archive:
        raise SystemExit(f"nothing to plot from {prefix}.*")
    draw_plots(summary, cells, archive, out_prefix)


def _entropy_reference(names: List[str]) -> Dict[str, tuple]:
    """Uniform-entropy line for the abstraction these runs actually used.

    The bound is log(mean number of *legal* actions), which is ~4.5 for the
    default sizings and ~5.4 for the finer grid -- reading a fine-abstraction
    run against the default line would overstate how sharp its policy is.
    """
    import math

    fine = all(
        action_space_for(build_config(name, 0, 1).env).num_actions > 10
        for name in names
        if name in CONFIGS
    )
    legal = 5.4 if fine else 4.5
    return {"entropy": (f"uniform over ~{legal:g} legal actions", math.log(legal))}


def _plot_results(summary, cells, out_path: str) -> None:
    """Bar chart of final strength against each fixed baseline.

    Error bars are the **seed spread** (min..max over seeds), not the
    evaluation interval.  Seed-to-seed variation is the larger of the two here,
    so it is the honest thing to show.
    """
    import matplotlib

    if not os.environ.get("DISPLAY") and os.environ.get("MPLBACKEND") is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in CONFIG_ORDER if n in summary]
    opponents = ("random", "calling_station", "heuristic")
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2), squeeze=False)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, opponent in enumerate(opponents):
        axis = axes[0][index]
        means, lows, highs = [], [], []
        for name in names:
            values = [row[opponent]["bb_per_100"] for row in cells[name]]
            mean = float(np.mean(values))
            means.append(mean)
            lows.append(mean - min(values))
            highs.append(max(values) - mean)
        positions = np.arange(len(names))
        bars = axis.barh(
            positions,
            means,
            xerr=[lows, highs],
            color=[colours[i % len(colours)] for i in range(len(names))],
            alpha=0.85,
            capsize=3,
        )
        for bar, mean in zip(bars, means):
            axis.annotate(
                f"{mean:+.0f}",
                xy=(mean, bar.get_y() + bar.get_height() / 2),
                xytext=(4 if mean >= 0 else -4, 0),
                textcoords="offset points",
                va="center",
                ha="left" if mean >= 0 else "right",
                fontsize=8,
            )
        axis.axvline(0, color="0.2", linewidth=1.0)
        axis.set_yticks(positions)
        axis.set_yticklabels(names if index == 0 else [])
        axis.invert_yaxis()
        axis.set_title(f"vs {opponent}")
        axis.set_xlabel("bb/100")
        axis.grid(axis="x", alpha=0.25, linewidth=0.6)

    figure.suptitle("Final strength against the fixed baselines (bars = seed mean, whiskers = seed range)")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


# --- cli -------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--root", default="benchmark_runs")
    train.add_argument("--iterations", type=int, default=250)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--root", default="benchmark_runs")
    evaluate.add_argument("--deals", type=int, default=800)

    report = sub.add_parser("report")
    report.add_argument("--root", default="benchmark_runs")
    report.add_argument("--out", default="benchmark_runs/report")

    replot_cmd = sub.add_parser(
        "replot", help="redraw the figures from a saved report, without the run dirs"
    )
    replot_cmd.add_argument(
        "--report", default="benchmark_runs/report",
        help="prefix of an existing report (reads <prefix>.json and <prefix>_histories.json)",
    )
    replot_cmd.add_argument(
        "--out", default=None, help="output prefix; defaults to --report (redraw in place)"
    )

    sub.add_parser("list").add_argument("--dummy", action="store_true")

    args = parser.parse_args()
    if args.command == "train":
        run_training(args.config, args.seed, args.root, args.iterations)
    elif args.command == "evaluate":
        run_evaluation(args.config, args.seed, args.root, args.deals)
    elif args.command == "report":
        build_report(args.root, args.out)
    elif args.command == "replot":
        replot(args.report, args.out or args.report)
    elif args.command == "list":
        print("\n".join(CONFIG_ORDER))


if __name__ == "__main__":
    main()
