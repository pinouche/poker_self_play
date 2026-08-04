"""Match play and reporting.

Three-handed poker is not zero-sum between any *pair* of players, so a single
seat assignment is not a fair comparison: the button has an edge, and a bot in
the big blind bleeds chips regardless of skill.  Every matchup is therefore
played in all three seat rotations and reported both per seat and per agent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from paradigm_a.config import Config, reward_scale
from paradigm_a.environment.poker_env import PokerEnv
from paradigm_a.representation.observation_encoder import ObservationEncoder
from paradigm_a.training.self_play import Agent, NetworkAgent, play_hand


@dataclass
class Stats:
    hands: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    reward_sum: float = 0.0
    chip_sum: float = 0.0

    def update(self, reward: float, chips: int) -> None:
        self.hands += 1
        self.reward_sum += reward
        self.chip_sum += chips
        if chips > 0:
            self.wins += 1
        elif chips < 0:
            self.losses += 1
        else:
            self.ties += 1

    def summary(self, big_blind: int) -> Dict[str, float]:
        hands = max(1, self.hands)
        return {
            "hands": self.hands,
            "win_rate": self.wins / hands,
            "loss_rate": self.losses / hands,
            "tie_rate": self.ties / hands,
            "avg_reward": self.reward_sum / hands,
            "avg_chips": self.chip_sum / hands,
            "bb_per_100": (self.chip_sum / hands) / max(1, big_blind) * 100.0,
        }


def evaluate_match(
    agents: Sequence[Agent],
    cfg: Config,
    num_hands: int = 300,
    seed: int = 0,
    rotate_seats: bool = True,
    encoder: Optional[ObservationEncoder] = None,
) -> Dict[str, object]:
    """Play ``num_hands`` between three agents and collect statistics.

    With ``rotate_seats`` the hands are split evenly across the three cyclic
    seat assignments so that positional advantage cancels out.
    """
    num_players = cfg.env.num_players
    if len(agents) != num_players:
        raise ValueError(f"expected {num_players} agents, got {len(agents)}")

    encoder = encoder or ObservationEncoder.from_config(cfg)
    env = PokerEnv(cfg.env, cfg.obs, seed=seed)
    rng = random.Random(seed)

    rotations = range(num_players) if rotate_seats else (0,)
    hands_per_rotation = max(1, num_hands // len(list(rotations)))

    seat_stats: Dict[int, Stats] = {seat: Stats() for seat in range(num_players)}
    agent_stats: Dict[str, Stats] = {}
    showdowns = 0
    total_hands = 0

    for rotation in rotations:
        # seat s is played by agents[(s + rotation) % num_players]
        seating: List[Agent] = [
            agents[(seat + rotation) % num_players] for seat in range(num_players)
        ]
        for agent in seating:
            agent_stats.setdefault(_agent_key(agent), Stats())

        for _ in range(hands_per_rotation):
            result = play_hand(env, seating, encoder, rng, collect=False)
            showdowns += int(result.went_to_showdown)
            total_hands += 1
            for seat in range(num_players):
                reward = result.rewards[seat]
                chips = result.chip_deltas[seat]
                seat_stats[seat].update(reward, chips)
                agent_stats[_agent_key(seating[seat])].update(reward, chips)

    big_blind = cfg.env.big_blind
    return {
        "hands": total_hands,
        "showdown_rate": showdowns / max(1, total_hands),
        "per_seat": {seat: stats.summary(big_blind) for seat, stats in seat_stats.items()},
        "per_agent": {name: stats.summary(big_blind) for name, stats in agent_stats.items()},
        "agents": [_agent_key(a) for a in agents],
    }


def _agent_key(agent: Agent) -> str:
    return getattr(agent, "name", agent.__class__.__name__)


# --- standard evaluation suites -------------------------------------------
def make_network_agent(
    network, cfg: Config, temperature: float = 0.0, device: Optional[str] = None, name: str = "network"
) -> NetworkAgent:
    """Wrap a network for evaluation (temperature 0 = deterministic play)."""
    agent = NetworkAgent(
        network,
        alpha=cfg.train.alpha,
        beta=cfg.train.beta,
        temperature=temperature,
        device=device,
        q_scale=reward_scale(cfg.env),
    )
    agent.name = name
    return agent


def evaluate_suite(
    network,
    cfg: Config,
    num_hands: int = 300,
    seed: int = 0,
    device: Optional[str] = None,
    opponents: Optional[Dict[str, Agent]] = None,
    include_heuristic: bool = True,
) -> Dict[str, Dict]:
    """Run the standard battery of matchups against the baselines."""
    from .heuristic_agent import tight_aggressive
    from .random_agent import CallingStationAgent, RandomAgent

    encoder = ObservationEncoder.from_config(cfg)
    hero = make_network_agent(network, cfg, temperature=0.0, device=device)

    results: Dict[str, Dict] = {}

    # network vs network vs network -- the self-play population itself.
    results["self_play"] = evaluate_match(
        [hero, hero, hero], cfg, num_hands, seed=seed, encoder=encoder
    )

    # network vs baseline vs baseline
    suites: Dict[str, Agent] = {
        "vs_random": RandomAgent(),
        "vs_calling_station": CallingStationAgent(),
    }
    if include_heuristic:
        suites["vs_heuristic"] = tight_aggressive(seed=seed)
    if opponents:
        suites.update({f"vs_{k}": v for k, v in opponents.items()})

    for label, opponent in suites.items():
        results[label] = evaluate_match(
            [hero, opponent, opponent], cfg, num_hands, seed=seed + 1, encoder=encoder
        )
    return results


def evaluate_against_checkpoint(
    network,
    opponent_network,
    cfg: Config,
    num_hands: int = 300,
    seed: int = 0,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Current network in one seat against an older checkpoint in the other two."""
    encoder = ObservationEncoder.from_config(cfg)
    hero = make_network_agent(network, cfg, temperature=0.0, device=device, name="current")
    old = make_network_agent(
        opponent_network, cfg, temperature=0.0, device=device, name="checkpoint"
    )
    return evaluate_match([hero, old, old], cfg, num_hands, seed=seed, encoder=encoder)


# --- reporting -------------------------------------------------------------
def format_results(results: Dict[str, Dict]) -> str:
    lines: List[str] = []
    for label, result in results.items():
        lines.append(f"\n{label}  ({result['hands']} hands, "
                     f"showdown rate {result['showdown_rate']:.2f})")
        lines.append(
            f"  {'agent':<18}{'hands':>7}{'win':>8}{'loss':>8}{'tie':>8}"
            f"{'reward':>10}{'chips':>10}{'bb/100':>10}"
        )
        for name, stats in result["per_agent"].items():
            lines.append(
                f"  {name:<18}{stats['hands']:>7}{stats['win_rate']:>8.3f}"
                f"{stats['loss_rate']:>8.3f}{stats['tie_rate']:>8.3f}"
                f"{stats['avg_reward']:>10.3f}{stats['avg_chips']:>10.1f}"
                f"{stats['bb_per_100']:>10.1f}"
            )
        lines.append("  per seat:")
        for seat, stats in result["per_seat"].items():
            lines.append(
                f"    seat {seat:<13}{stats['hands']:>7}{stats['win_rate']:>8.3f}"
                f"{stats['loss_rate']:>8.3f}{stats['tie_rate']:>8.3f}"
                f"{stats['avg_reward']:>10.3f}{stats['avg_chips']:>10.1f}"
                f"{stats['bb_per_100']:>10.1f}"
            )
    return "\n".join(lines)


def summarize_headline(results: Dict[str, Dict], agent_name: str = "network") -> Dict[str, float]:
    """One number per matchup, for training logs."""
    headline: Dict[str, float] = {}
    for label, result in results.items():
        stats = result["per_agent"].get(agent_name)
        if stats:
            headline[f"{label}/bb_per_100"] = stats["bb_per_100"]
            headline[f"{label}/win_rate"] = stats["win_rate"]
    return headline


# --- duplicate-deal scoring ------------------------------------------------
def duplicate_deal_scores(
    hero: Agent,
    opponents: Sequence[Agent],
    cfg: Config,
    deal_seeds: Sequence[int],
    encoder: Optional[ObservationEncoder] = None,
) -> np.ndarray:
    """Per-deal chip result for ``hero``, averaged over all three seats.

    bb/100 measured on independent hands is dominated by card luck: at ~900
    hands the 95% interval is wider than any effect worth detecting.  Each deal
    here is replayed three times from the *same* deck order with the hero
    rotated through every seat, which cancels most of the "who was dealt aces"
    component.

    Because the deck order is fixed by the seed, the returned array is
    comparable deal-for-deal across different heroes -- so two agents can be
    compared as a paired difference rather than as two independent samples.
    """
    encoder = encoder or ObservationEncoder.from_config(cfg)
    scores = np.empty(len(deal_seeds), dtype=np.float64)

    for i, seed in enumerate(deal_seeds):
        total = 0.0
        for hero_seat in range(cfg.env.num_players):
            env = PokerEnv(cfg.env, cfg.obs, seed=seed)
            table = list(opponents)
            table.insert(hero_seat, hero)
            # Stateful agents must start each replay from the same point, or
            # the deal is no longer identical across heroes.
            for agent in table[: cfg.env.num_players]:
                agent.reset()
            result = play_hand(
                env,
                table[: cfg.env.num_players],
                encoder,
                random.Random(seed * 7 + hero_seat),
                collect=False,
            )
            total += result.chip_deltas[hero_seat]
        scores[i] = total / cfg.env.num_players
    return scores


def bb_per_100_interval(
    scores: np.ndarray, big_blind: int, confidence: float = 1.96
) -> Dict[str, float]:
    """Mean bb/100 with a normal-approximation confidence interval."""
    per_100 = np.asarray(scores, dtype=np.float64) / max(1, big_blind) * 100.0
    mean = float(per_100.mean())
    half = float(confidence * per_100.std(ddof=1) / np.sqrt(len(per_100))) if len(per_100) > 1 else float("inf")
    return {
        "bb_per_100": mean,
        "ci_half_width": half,
        "significant": abs(mean) > half,
        "deals": int(len(per_100)),
    }


def compare_agents_paired(
    hero: Agent,
    rival: Agent,
    opponents: Sequence[Agent],
    cfg: Config,
    num_deals: int = 1000,
    seed_offset: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Score two agents on identical deals and report the paired difference."""
    encoder = ObservationEncoder(cfg.obs)
    deal_seeds = list(range(seed_offset, seed_offset + num_deals))
    hero_scores = duplicate_deal_scores(hero, opponents, cfg, deal_seeds, encoder)
    rival_scores = duplicate_deal_scores(rival, opponents, cfg, deal_seeds, encoder)
    bb = cfg.env.big_blind
    return {
        _agent_key(hero): bb_per_100_interval(hero_scores, bb),
        _agent_key(rival): bb_per_100_interval(rival_scores, bb),
        "difference": bb_per_100_interval(hero_scores - rival_scores, bb),
    }
