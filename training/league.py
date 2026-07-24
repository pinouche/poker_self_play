"""A role-structured league population.

Instead of N interchangeable learners the population is split into three roles:

* **learners** -- updated continuously; the working edge of the league.
* **champions** -- frozen historical snapshots, never updated.  They stop the
  league drifting off strategies that already worked, which is precisely the
  failure that lets naive self-play cycle: with only live members, the whole
  population can walk away from a good strategy together and never notice.
* **explorers** -- learners that are periodically wiped and reinitialised *from
  scratch*.  From scratch matters: a fresh random policy is genuinely
  off-distribution, whereas an offspring of an incumbent inherits its blind
  spots and adds little.

Management is behaviour-aware.  Members are ranked by

    score = w_strength * norm(strength) + w_diversity * norm(diversity)

(:mod:`training.league_metrics`) rather than by strength alone.  A slightly
weaker but strategically distinct network therefore keeps its slot, which is
what stops the league converging to N copies of one policy -- and is why no
crossover or mutation operators are needed.

Slots whose network is replaced (a culled learner, a reset explorer) are
reported back by :meth:`LeaguePopulation.manage`, because the caller owns the
per-slot optimiser and replay buffer and must rebuild them.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

import numpy as np

from config import Config, reward_scale
from model.network import PokerNet, build_network


class Role(str, Enum):
    LEARNER = "learner"
    CHAMPION = "champion"
    EXPLORER = "explorer"


@dataclass
class LeagueMember:
    index: int
    role: Role
    network: PokerNet
    agent: object                 # NetworkAgent (imported lazily)
    generation: int = 0           # bumped every time the slot is reinitialised
    born_at_hands: int = 0

    @property
    def trainable(self) -> bool:
        """Champions are opponents only; learners and explorers are optimised."""
        return self.role is not Role.CHAMPION


@dataclass
class ManagementReport:
    """What a management pass changed, so the caller can rebuild slot state."""

    promoted: Optional[int] = None          # learner index snapshotted into a champion slot
    champion_slot: Optional[int] = None     # champion slot that was overwritten
    culled: Optional[int] = None            # learner slot reinitialised from scratch
    reset_explorers: List[int] = field(default_factory=list)

    @property
    def rebuilt_slots(self) -> List[int]:
        """Slots whose network object changed -> optimiser and buffer are stale."""
        slots = list(self.reset_explorers)
        if self.culled is not None:
            slots.append(self.culled)
        if self.champion_slot is not None:
            slots.append(self.champion_slot)
        return sorted(set(slots))


class LeaguePopulation:
    """``league_learners + league_champions + league_explorers`` members."""

    def __init__(self, cfg: Config, device: Optional[str] = None, seed: int = 0) -> None:
        self.cfg = cfg
        self.device = device
        self.rng = random.Random(seed)
        self.total_hands = 0
        self._last_explorer_reset = 0
        self._generation = 0

        self.members: List[LeagueMember] = []
        layout = (
            (Role.LEARNER, int(cfg.train.league_learners)),
            (Role.CHAMPION, int(cfg.train.league_champions)),
            (Role.EXPLORER, int(cfg.train.league_explorers)),
        )
        for role, count in layout:
            for _ in range(max(0, count)):
                self.members.append(self._new_member(role))
        if not self.members:
            raise ValueError("league population is empty; check the league_* sizes")

    # --- construction ------------------------------------------------------
    def _build_agent(self, network: PokerNet):
        from training.self_play import NetworkAgent

        return NetworkAgent(
            network,
            alpha=self.cfg.train.alpha,
            beta=self.cfg.train.beta,
            temperature=self.cfg.train.sampling_temperature,
            device=self.device,
            q_scale=reward_scale(self.cfg.env),
        )

    def _new_network(self) -> PokerNet:
        network = build_network(self.cfg)
        if self.device:
            network.to(self.device)
        return network

    def _new_member(self, role: Role, index: Optional[int] = None) -> LeagueMember:
        network = self._new_network()
        if role is Role.CHAMPION:
            network = self._freeze(network)
        self._generation += 1
        return LeagueMember(
            index=len(self.members) if index is None else index,
            role=role,
            network=network,
            agent=self._build_agent(network),
            generation=self._generation,
            born_at_hands=self.total_hands,
        )

    def _freeze(self, network: PokerNet) -> PokerNet:
        """An independent, non-trainable copy that cannot drift with the live net."""
        frozen = build_network(self.cfg)
        frozen.load_state_dict(copy.deepcopy(network.state_dict()))
        frozen.eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        if self.device:
            frozen.to(self.device)
        return frozen

    # --- views -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.members)

    @property
    def networks(self) -> List[PokerNet]:
        return [m.network for m in self.members]

    @property
    def agents(self) -> List[object]:
        return [m.agent for m in self.members]

    def indices(self, role: Role) -> List[int]:
        return [i for i, m in enumerate(self.members) if m.role is role]

    def trainable_indices(self) -> List[int]:
        return [i for i, m in enumerate(self.members) if m.trainable]

    def role_counts(self) -> dict:
        return {role.value: len(self.indices(role)) for role in Role}

    # --- play --------------------------------------------------------------
    def sample_seats(self, num_players: int) -> List[int]:
        """One member index per seat, uniform with replacement.

        Every role is seatable -- champions are exactly the strong, static
        opponents the learners need -- but only ``trainable`` seats are ever
        collected, which the caller enforces.
        """
        return [self.rng.randrange(len(self.members)) for _ in range(num_players)]

    # --- management --------------------------------------------------------
    def _reset_slot(self, index: int) -> None:
        """Reinitialise a slot from scratch, keeping its role."""
        member = self.members[index]
        self.members[index] = self._new_member(member.role, index=index)

    def maybe_reset_explorers(self) -> List[int]:
        """Wipe the explorers once ``league_explorer_reset_hands`` have passed."""
        every = int(self.cfg.train.league_explorer_reset_hands)
        if every <= 0:
            return []
        if self.total_hands - self._last_explorer_reset < every:
            return []
        self._last_explorer_reset = self.total_hands
        reset = self.indices(Role.EXPLORER)
        for index in reset:
            self._reset_slot(index)
        return reset

    def promote(self, scores: Sequence[float]) -> tuple:
        """Freeze the best-scoring learner into the oldest champion slot.

        Returns ``(promoted_index, champion_slot)``, or ``(None, None)`` when
        there is nothing to promote into.
        """
        learners = self.indices(Role.LEARNER)
        champions = self.indices(Role.CHAMPION)
        if not learners or not champions:
            return None, None
        best = max(learners, key=lambda i: scores[i])
        oldest = min(champions, key=lambda i: self.members[i].born_at_hands)

        self._generation += 1
        frozen = self._freeze(self.members[best].network)
        self.members[oldest] = LeagueMember(
            index=oldest,
            role=Role.CHAMPION,
            network=frozen,
            agent=self._build_agent(frozen),
            generation=self._generation,
            born_at_hands=self.total_hands,
        )
        return best, oldest

    def cull(self, scores: Sequence[float]) -> Optional[int]:
        """Reinitialise the lowest-*scoring* learner from scratch.

        Scoring, not strength: culling on strength alone is what would delete
        the weaker-but-different members and collapse the league.
        """
        learners = self.indices(Role.LEARNER)
        if len(learners) < 2:
            return None
        worst = min(learners, key=lambda i: scores[i])
        self._reset_slot(worst)
        return worst

    def manage(self, scores: Sequence[float], cull_worst: bool = True) -> ManagementReport:
        """One management pass: promote a champion, cull the weakest learner."""
        if len(scores) != len(self.members):
            raise ValueError("scores must have one entry per league member")
        promoted, champion_slot = self.promote(scores)
        culled = self.cull(scores) if cull_worst else None
        return ManagementReport(
            promoted=promoted,
            champion_slot=champion_slot,
            culled=culled,
            reset_explorers=[],
        )

    # --- metrics -----------------------------------------------------------
    def evaluate(self, encoder, seed: int = 0) -> dict:
        """Strength, diversity, coverage and the management score.

        Returns a dict of ``[n]`` arrays plus the pairwise matrices, ready to
        feed :meth:`manage` and to log.
        """
        from training.league_metrics import (
            behavioural_diversity,
            population_scores,
            sample_validation_states,
            strength_and_coverage,
        )

        train = self.cfg.train
        strength, coverage, pair_mean, hands = strength_and_coverage(
            self.agents, self.cfg, int(train.league_strength_hands), encoder, seed=seed
        )
        observations, masks = sample_validation_states(
            self.agents, self.cfg, int(train.league_diversity_states), encoder, seed=seed + 1
        )
        diversity, kl = behavioural_diversity(
            self.networks, observations, masks, device=self.device
        )
        score, norm_strength, norm_diversity = population_scores(
            strength, diversity,
            strength_weight=train.league_strength_weight,
            diversity_weight=train.league_diversity_weight,
        )
        return {
            "strength": strength,
            "diversity": diversity,
            "coverage": coverage,
            "score": score,
            "normalised_strength": norm_strength,
            "normalised_diversity": norm_diversity,
            "pair_mean": pair_mean,
            "kl_matrix": kl,
            "hands_played": hands,
            "roles": [m.role.value for m in self.members],
        }

    def summary(self, metrics: dict, top: int = 5) -> str:
        """A compact, sorted view of the population for training logs."""
        order = np.argsort(-np.asarray(metrics["score"]))
        lines = [
            f"{'slot':>5}{'role':>10}{'score':>8}{'strength':>10}{'diversity':>11}{'coverage':>10}"
        ]
        for index in order[:top]:
            lines.append(
                f"{index:>5}{self.members[index].role.value:>10}"
                f"{metrics['score'][index]:>8.3f}{metrics['strength'][index]:>10.4f}"
                f"{metrics['diversity'][index]:>11.4f}{int(metrics['coverage'][index]):>10}"
            )
        return "\n".join(lines)
