"""A role-structured league population with behaviour-aware management.

The population is split into three roles:

* **learners** -- updated continuously; the working edge of the league.
* **champions** -- historical snapshots.  They are the cheap fix for four
  problems at once: catastrophic forgetting, meta-game collapse, strategy
  cycling, and unstable evaluation.  With only live members the whole population
  can walk away from a good strategy together and never notice; a ladder of past
  elites makes that impossible to hide.
* **explorers** -- learners periodically wiped and reinitialised *from scratch*.
  From scratch matters: a fresh random policy is genuinely off-distribution,
  whereas an offspring of an incumbent inherits its blind spots.

Under ``champion_mode="frozen"`` every champion is a hard snapshot.  Under
``"anchored"`` the pool splits into **anchors** (still hard-frozen, and the only
slots the gauntlet reads -- a measuring stick that moves measures nothing) and
**veterans**, which keep training.  A veteran exists because promotion can only
freeze strategies the learner pool *currently holds*: a champion encoding a line
the learners have abandoned is the one thing here that cannot be recreated, and
frozen it merely rots until eviction.  Its improvement target is anchored to its
own frozen snapshot (``reference_policy="snapshot"``), so ``beta`` bounds the
distance from that snapshot rather than the size of one step from wherever it
drifted to; the remaining drift is *measured* (:meth:`veteran_drift`) and reset
when it exceeds the budget, rather than assumed small.

Two rules keep the champion pool useful rather than random:

1. **Promotion is earned, never scheduled.**  A learner must sustain a real
   win rate over a large sample of actual league play *and* beat most existing
   champion generations.  The generation gate is the anti-over-specialisation
   check: a learner that beats the newest champion but loses badly to an old one
   has learned the current meta, not the game.
2. **Freeze different kinds.**  Each cycle promotes the highest-EV learner *and*
   the most behaviourally different one, so the pool accumulates strong
   strategies and unusual ones instead of eight variants of a single idea.

Champions still go stale -- one that loses badly to everyone is marked and is
first in line to be evicted, so the pool rolls rather than carrying corpses.

Ranking everywhere uses ``score = w_s * norm(strength) + w_d * norm(diversity)``
(:mod:`training.league_metrics`), not strength alone.  That is what stops the
league collapsing into N copies of one policy, and why no crossover or mutation
operators are needed.

Slots whose network is replaced are reported by :meth:`LeaguePopulation.manage`,
because the caller owns the per-slot optimiser and replay buffer.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

import numpy as np

from paradigm_a.config import Config, reward_scale
from paradigm_a.model.network import PokerNet, build_network


class Role(str, Enum):
    LEARNER = "learner"
    CHAMPION = "champion"
    EXPLORER = "explorer"


class MatchType(str, Enum):
    LEARNER_VS_LEARNER = "learner_vs_learner"
    LEARNER_VS_CHAMPION = "learner_vs_champion"
    MIXED = "mixed"


@dataclass
class LeagueMember:
    index: int
    role: Role
    network: PokerNet
    agent: object                 # NetworkAgent (imported lazily)
    generation: int = 0           # bumped every time the slot is reinitialised
    born_at_hands: int = 0        # league hands elapsed when this slot was filled
    trained_at_hands: int = 0     # for champions: how much training the snapshot had
    reset_at_pass: int = -10**9   # management pass a learner was last reset (cull grace)

    # Running record from *real* league play, so promotion is judged on the games
    # actually played rather than on a separate evaluation pass.
    hands_played: int = 0
    return_sum: float = 0.0       # sum of chip delta / own starting stack
    stale: bool = False           # champion flagged for eviction

    # Veterans only: the frozen copy taken when this slot was filled.  It is both
    # the reference policy their training target is anchored to and the baseline
    # drift is measured against, so its presence is what marks a champion as a
    # veteran rather than an anchor.
    snapshot: Optional[PokerNet] = None

    @property
    def trainable(self) -> bool:
        """Anchored champions are opponents only; everything else is optimised."""
        if self.role is not Role.CHAMPION:
            return True
        return self.snapshot is not None

    @property
    def is_anchor(self) -> bool:
        """A hard-frozen champion -- part of the gauntlet's fixed measuring stick."""
        return self.role is Role.CHAMPION and self.snapshot is None

    @property
    def label(self) -> str:
        """Role for logs.  Frozen champions keep reading "champion" so that the
        default mode's output is unchanged; only veterans need a new word."""
        if self.role is Role.CHAMPION and not self.is_anchor:
            return "veteran"
        return self.role.value

    @property
    def strength(self) -> float:
        """Mean normalised chip return per hand over this slot's lifetime."""
        return self.return_sum / self.hands_played if self.hands_played else 0.0


@dataclass
class ManagementReport:
    """What a management pass changed, so the caller can rebuild slot state."""

    promoted: List[tuple] = field(default_factory=list)   # (learner, champion_slot, reason)
    retired: List[int] = field(default_factory=list)      # stale champion slots evicted
    culled: Optional[int] = None                          # learner reinitialised from scratch
    reset_explorers: List[int] = field(default_factory=list)

    @property
    def rebuilt_slots(self) -> List[int]:
        """Slots whose network object changed -> optimiser and buffer are stale."""
        slots = list(self.reset_explorers)
        if self.culled is not None:
            slots.append(self.culled)
        slots.extend(slot for _, slot, _ in self.promoted)
        slots.extend(self.retired)
        return sorted(set(slots))

    def describe(self) -> str:
        parts = [f"promote learner {i} -> champion slot {s} ({why})"
                 for i, s, why in self.promoted]
        if self.retired:
            # Flagged only -- a stale champion leaves the pool when a promotion
            # needs its slot, so this line must not read as an eviction.
            parts.append(f"flag stale champions {self.retired} (evicted on next promotion)")
        if self.culled is not None:
            parts.append(f"cull learner {self.culled}")
        if self.reset_explorers:
            parts.append(f"reset explorers {self.reset_explorers}")
        return "; ".join(parts) if parts else "no changes"


class LeaguePopulation:
    """``league_learners + league_champions + league_explorers`` members."""

    def __init__(self, cfg: Config, device: Optional[str] = None, seed: int = 0) -> None:
        self.cfg = cfg
        self.device = device
        self.rng = random.Random(seed)
        self.total_hands = 0
        self._last_explorer_reset = 0
        self._generation = 0
        self._manage_pass = 0

        self.champion_mode = str(cfg.train.champion_mode).lower()
        if self.champion_mode not in ("frozen", "anchored"):
            raise ValueError(
                f"unknown champion_mode {cfg.train.champion_mode!r}; "
                "choose from 'frozen', 'anchored'"
            )

        learners = max(0, int(cfg.train.league_learners))
        champions = max(0, int(cfg.train.league_champions))
        # The layout below lays champion slots out contiguously right after the
        # learners, which is what makes these two ranges well defined.  Under
        # "frozen" every champion is an anchor, so every anchor-aware code path
        # below reduces exactly to the original behaviour.
        anchors = (
            champions if self.champion_mode == "frozen"
            else min(champions, max(0, int(cfg.train.league_champion_anchors)))
        )
        self.anchor_slots = set(range(learners, learners + anchors))
        self.veteran_slots = set(range(learners + anchors, learners + champions))

        self.members: List[LeagueMember] = []
        layout = (
            (Role.LEARNER, learners),
            (Role.CHAMPION, champions),
            (Role.EXPLORER, int(cfg.train.league_explorers)),
        )
        for role, count in layout:
            for _ in range(max(0, count)):
                self.members.append(self._new_member(role))
        if not self.members:
            raise ValueError("league population is empty; check the league_* sizes")

        size = len(self.members)
        self._pair_sum = np.zeros((size, size), dtype=np.float64)
        self._pair_count = np.zeros((size, size), dtype=np.float64)

        weights = np.array([
            max(0.0, cfg.train.league_match_learner_vs_learner),
            max(0.0, cfg.train.league_match_learner_vs_champion),
            max(0.0, cfg.train.league_match_mixed),
        ], dtype=np.float64)
        total = weights.sum()
        self.match_weights = (weights / total) if total > 0 else np.array([1.0, 0.0, 0.0])

    # --- construction ------------------------------------------------------
    def _build_agent(self, network: PokerNet):
        from paradigm_a.training.self_play import NetworkAgent

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
        slot = len(self.members) if index is None else index
        network = self._new_network()
        snapshot = None
        if role is Role.CHAMPION:
            network, snapshot = self._champion_networks(network, slot)
        self._generation += 1
        return LeagueMember(
            index=slot,
            role=role,
            network=network,
            agent=self._build_agent(network),
            generation=self._generation,
            born_at_hands=self.total_hands,
            trained_at_hands=self.total_hands,
            snapshot=snapshot,
        )

    def _champion_networks(self, source: PokerNet, slot: int) -> tuple:
        """``(live_network, snapshot)`` for a champion filling ``slot``.

        Anchor slots get a single frozen network and no snapshot.  Veteran slots
        get a *trainable* copy to play and train with, plus an independent frozen
        copy kept as the anchor for their improvement target and the baseline for
        drift.  Whether a slot is an anchor is a property of the slot, not of who
        fills it, so the gauntlet spine survives every promotion and eviction.
        """
        if slot not in self.veteran_slots:
            return self._freeze(source), None
        return self._clone(source), self._freeze(source)

    def _clone(self, network: PokerNet) -> PokerNet:
        """An independent *trainable* copy of ``network``."""
        clone = build_network(self.cfg)
        clone.load_state_dict(copy.deepcopy(network.state_dict()))
        if self.device:
            clone.to(self.device)
        return clone

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

    def contender_indices(self) -> List[int]:
        """Trainable non-champions -- the members promotion is judged over.

        Veterans train but are already champions, so they belong in neither the
        gauntlet's rows nor the promotion pool.
        """
        return [i for i, m in enumerate(self.members)
                if m.trainable and m.role is not Role.CHAMPION]

    def veteran_indices(self) -> List[int]:
        """Champions that keep training (empty under ``champion_mode="frozen"``)."""
        return [i for i, m in enumerate(self.members) if m.role is Role.CHAMPION and m.trainable]

    def anchor_indices(self) -> List[int]:
        """Hard-frozen champions -- the gauntlet's fixed measuring stick."""
        return [i for i, m in enumerate(self.members) if m.is_anchor]

    def role_counts(self) -> dict:
        counts = {role.value: len(self.indices(role)) for role in Role}
        counts["anchor"] = len(self.anchor_indices())
        counts["veteran"] = len(self.veteran_indices())
        return counts

    # --- matchmaking -------------------------------------------------------
    def sample_match(self) -> MatchType:
        return MatchType(
            self.rng.choices(list(MatchType), weights=list(self.match_weights), k=1)[0]
        )

    def _pick(self, role: Role) -> int:
        """One member of ``role``, falling back to a learner then to anyone."""
        pool = self.indices(role) or self.indices(Role.LEARNER) or list(range(len(self.members)))
        return self.rng.choice(pool)

    def sample_seats(self, num_players: int) -> tuple:
        """``(seats, match_type)`` -- one member index per seat.

        Seat *order* is shuffled after the roles are chosen: without that the
        learner would always sit in the same position and inherit that seat's
        positional edge, which would quietly bias every strength estimate.
        """
        match = self.sample_match()
        if match is MatchType.LEARNER_VS_LEARNER:
            seats = [self._pick(Role.LEARNER) for _ in range(num_players)]
        elif match is MatchType.LEARNER_VS_CHAMPION:
            seats = [self._pick(Role.LEARNER)]
            seats += [self._pick(Role.CHAMPION) for _ in range(num_players - 1)]
        else:  # mixed: learner + champion + explorer, then top up with learners
            wanted = [Role.LEARNER, Role.CHAMPION, Role.EXPLORER][:num_players]
            seats = [self._pick(role) for role in wanted]
            seats += [self._pick(Role.LEARNER) for _ in range(num_players - len(seats))]
        self.rng.shuffle(seats)
        return seats, match

    # --- running record from real play -------------------------------------
    def record_hand(self, seated: Sequence[int], normalised_returns: Sequence[float]) -> None:
        """Fold one played hand into the running per-member and pairwise records."""
        for seat, member in enumerate(seated):
            value = float(normalised_returns[seat])
            entry = self.members[member]
            entry.hands_played += 1
            entry.return_sum += value
            for other in seated:
                if other != member:  # never your own opponent
                    self._pair_sum[member, other] += value
                    self._pair_count[member, other] += 1
        self.total_hands += 1

    def _clear_stats(self, index: int) -> None:
        """Forget a slot's record -- a replaced network did not earn it."""
        member = self.members[index]
        member.hands_played = 0
        member.return_sum = 0.0
        member.stale = False
        self._pair_sum[index, :] = 0.0
        self._pair_sum[:, index] = 0.0
        self._pair_count[index, :] = 0.0
        self._pair_count[:, index] = 0.0

    def running_strength(self) -> np.ndarray:
        return np.array([m.strength for m in self.members], dtype=np.float64)

    def running_pair_mean(self) -> np.ndarray:
        return self._pair_sum / np.maximum(self._pair_count, 1.0)

    def running_coverage(self) -> np.ndarray:
        pair_mean = self.running_pair_mean()
        return ((pair_mean > 0.0) & (self._pair_count > 0)).sum(axis=1).astype(np.float64)

    def strength_mbb_per_100(self) -> np.ndarray:
        from paradigm_a.training.league_metrics import strength_to_mbb_per_100

        return strength_to_mbb_per_100(self.running_strength(), self.cfg.env)

    # --- champion gauntlet -------------------------------------------------
    def champion_generations(self) -> List[int]:
        """Champion slots ordered oldest -> newest snapshot."""
        return sorted(self.indices(Role.CHAMPION), key=lambda i: self.members[i].trained_at_hands)

    def gauntlet_generations(self) -> List[int]:
        """Anchor slots, oldest -> newest: the generations the gauntlet judges on.

        Veterans are deliberately excluded.  They drift toward beating the
        current learner meta, so scoring learners against them would erode the
        very generational spread the over-specialisation test reads.  Under
        ``champion_mode="frozen"`` every champion is an anchor and this is
        :meth:`champion_generations`.
        """
        return sorted(self.anchor_indices(), key=lambda i: self.members[i].trained_at_hands)

    def champion_gauntlet(self, min_hands: int = 1) -> dict:
        """How every learner fares against each anchor *generation*.

        A learner that beats the newest champion but loses badly to an older one
        is over-specialised: it has learned the current meta rather than the
        game.  ``beats`` counts generations with a positive result, and
        ``over_specialised`` flags exactly that pattern.
        """
        generations = self.gauntlet_generations()
        learners = self.contender_indices()
        pair_mean = self.running_pair_mean()
        results = np.zeros((len(learners), len(generations)), dtype=np.float64)
        seen = np.zeros_like(results, dtype=bool)

        for row, learner in enumerate(learners):
            for column, champion in enumerate(generations):
                results[row, column] = pair_mean[learner, champion]
                seen[row, column] = self._pair_count[learner, champion] >= min_hands

        beats = ((results > 0) & seen).sum(axis=1)
        played = seen.sum(axis=1)
        over_specialised = np.zeros(len(learners), dtype=bool)
        if len(generations) >= 2:
            newest, oldest = -1, 0
            over_specialised = (
                seen[:, newest] & seen[:, oldest]
                & (results[:, newest] > 0) & (results[:, oldest] < 0)
            )
        return {
            "learners": learners,
            "generations": generations,
            "results": results,
            "evaluated": seen,
            "beats": beats,
            "generations_played": played,
            "over_specialised": over_specialised,
        }

    # --- management --------------------------------------------------------
    def _reset_slot(self, index: int) -> None:
        """Reinitialise a slot from scratch, keeping its role."""
        role = self.members[index].role
        self.members[index] = self._new_member(role, index=index)
        self.members[index].reset_at_pass = self._manage_pass  # start the cull grace window
        self._clear_stats(index)

    def maybe_reset_explorers(self) -> List[int]:
        """Wipe the explorers once ``league_explorer_reset_hands`` have passed."""
        every = int(self.cfg.train.league_explorer_reset_hands)
        if every <= 0 or self.total_hands - self._last_explorer_reset < every:
            return []
        self._last_explorer_reset = self.total_hands
        reset = self.indices(Role.EXPLORER)
        for index in reset:
            self._reset_slot(index)
        return reset

    def mark_stale_champions(self, external_scores: Optional[Sequence[float]] = None) -> List[int]:
        """Flag champions to evict first, so the pool rolls rather than rots.

        A champion is stale if it loses badly *in-league* OR (when
        ``external_scores`` -- bb/100 vs the held-out heuristic per member -- is
        given) falls below ``league_champion_retire_vs_heuristic``.  The external
        gate catches champions that are catastrophic against a competent outside
        opponent yet survive in-league because the league has not learned to
        punish them: in-league strength is relative, the heuristic is absolute.
        """
        train = self.cfg.train
        threshold = float(train.league_champion_retire_mbb_per_100)
        min_hands = int(train.league_champion_min_hands)
        external_threshold = float(train.league_champion_retire_vs_heuristic)
        mbb = self.strength_mbb_per_100()
        stale = []
        for index in self.indices(Role.CHAMPION):
            member = self.members[index]
            in_league_bad = member.hands_played >= min_hands and mbb[index] <= threshold
            external_bad = (
                external_scores is not None
                and index < len(external_scores)
                and np.isfinite(external_scores[index])
                and external_scores[index] <= external_threshold
            )
            if in_league_bad or external_bad:
                member.stale = True
                stale.append(index)
        return stale

    def _eviction_slot(
        self, exclude: Sequence[int] = (), diversity: Optional[Sequence[float]] = None
    ) -> Optional[int]:
        """Which champion to make room for a promotion.

        Stale champions go first (they are bad opponents).  Within the eviction
        pool, prefer the *most redundant* champion -- lowest behavioural
        diversity to the rest, when ``diversity`` is given -- rather than the
        oldest, so distinct strategies (including old, unusual ones) survive and
        the ladder keeps its spread.  Falls back to oldest when no diversity is
        available (e.g. before any hands are played).
        """
        champions = [i for i in self.indices(Role.CHAMPION) if i not in exclude]
        if not champions:
            return None
        stale = [i for i in champions if self.members[i].stale]
        if stale:
            # Among bad champions, diversity must NOT buy survival -- a distinct
            # but catastrophic champion is still a bad opponent.  Evict oldest.
            return min(stale, key=lambda i: self.members[i].trained_at_hands)
        # Healthy pool: sacrifice the most redundant, keeping distinct strategies.
        if self.cfg.train.league_evict_most_redundant and diversity is not None:
            return min(champions, key=lambda i: diversity[i])
        return min(champions, key=lambda i: self.members[i].trained_at_hands)

    def promotion_candidates(self, gauntlet: Optional[dict] = None) -> List[int]:
        """Learners that have *earned* promotion.

        Gates, all on real league play: a sustained win rate over enough hands,
        beating at least the *median* champion generation it has faced (a
        relative bar, so a strengthening ladder cannot lock promotions out), and
        not being over-specialised.
        """
        train = self.cfg.train
        mbb = self.strength_mbb_per_100()
        if gauntlet is None:
            gauntlet = self.champion_gauntlet(min_hands=int(train.league_gauntlet_min_hands))
        generations = len(gauntlet["generations"])
        fraction = float(train.league_promotion_min_generations)
        reject_over = bool(train.league_promotion_reject_over_specialised)

        candidates = []
        for row, index in enumerate(gauntlet["learners"]):
            member = self.members[index]
            if member.role is not Role.LEARNER:
                continue
            if member.hands_played < int(train.league_promotion_hands):
                continue
            if mbb[index] < float(train.league_promotion_mbb_per_100):
                continue
            if generations:
                played = gauntlet["generations_played"][row]
                # Beat the median generation actually faced (>= fraction of them),
                # rather than a fixed count of all eight.
                if played < 1 or gauntlet["beats"][row] < fraction * played:
                    continue
                if reject_over and gauntlet["over_specialised"][row]:
                    continue
            candidates.append(index)
        return candidates

    def promote(
        self,
        index: int,
        reason: str = "elite",
        diversity: Optional[Sequence[float]] = None,
        exclude: Sequence[int] = (),
    ) -> Optional[int]:
        """Snapshot a learner into a champion slot.

        Into an anchor slot that is forever; into a veteran slot the snapshot is
        kept as the anchor for the copy that carries on training.
        """
        slot = self._eviction_slot(exclude=exclude, diversity=diversity)
        if slot is None:
            return None
        self._generation += 1
        network, snapshot = self._champion_networks(self.members[index].network, slot)
        self.members[slot] = LeagueMember(
            index=slot,
            role=Role.CHAMPION,
            network=network,
            agent=self._build_agent(network),
            generation=self._generation,
            born_at_hands=self.total_hands,
            trained_at_hands=self.total_hands,
            snapshot=snapshot,
        )
        self._clear_stats(slot)
        return slot

    def cull(self, scores: Sequence[float]) -> Optional[int]:
        """Reinitialise the lowest-*scoring* eligible learner from scratch.

        Scoring, not strength: culling on strength alone would delete the
        weaker-but-different members and collapse the league.  A freshly reset
        learner is protected for ``league_cull_grace_passes`` management passes,
        so it develops instead of being re-culled every pass while it is still
        the youngest network at the table.
        """
        grace = int(self.cfg.train.league_cull_grace_passes)
        learners = self.indices(Role.LEARNER)
        if len(learners) < 2:
            return None
        eligible = [
            i for i in learners
            if self._manage_pass - self.members[i].reset_at_pass >= grace
        ]
        if not eligible:
            return None
        worst = min(eligible, key=lambda i: scores[i])
        self._reset_slot(worst)
        return worst

    def most_different_shortlist(self, scores: Sequence[float]) -> List[int]:
        """Top-K learners by score that clear the in-league strength floor.

        The caller evaluates this shortlist against the external panel, so the
        expensive panel eval touches only a handful of networks.  The floor
        (``league_most_different_min_mbb_per_100``) keeps a near-random, freshly
        reset learner out even though a fresh policy scores high on diversity.
        """
        train = self.cfg.train
        min_hands = int(train.league_promotion_hands)
        min_mbb = float(train.league_most_different_min_mbb_per_100)
        mbb = self.strength_mbb_per_100()
        eligible = [
            i for i in self.indices(Role.LEARNER)
            if self.members[i].hands_played >= min_hands and mbb[i] >= min_mbb
        ]
        ranked = sorted(eligible, key=lambda i: scores[i], reverse=True)
        return ranked[: int(train.league_most_different_top_k)]

    def _most_different_pool(
        self, scores: Sequence[float], panel_scores: Optional[dict] = None
    ) -> List[int]:
        """The most_different shortlist, filtered by the external competence panel.

        Diversity is only worth freezing from a member that is not catastrophic
        against standard opponents -- otherwise the diversity slot freezes
        in-league-strong-but-externally-weak champions (the late-run
        oscillation).  ``panel_scores[i]`` is member i's worst-case bb/100 across
        the panel; a member missing from the dict is not filtered.
        """
        shortlist = self.most_different_shortlist(scores)
        if not panel_scores:
            return shortlist
        threshold = float(self.cfg.train.league_most_different_min_panel_bb_per_100)
        return [i for i in shortlist if panel_scores.get(i, float("inf")) >= threshold]

    def manage(
        self,
        scores: Sequence[float],
        diversity: Optional[Sequence[float]] = None,
        external_scores: Optional[Sequence[float]] = None,
        panel_scores: Optional[dict] = None,
        cull_worst: bool = True,
    ) -> ManagementReport:
        """One management pass.

        Order matters: flag stale champions first so promotions evict them, then
        promote the elite (strict gate) and the most-different (relaxed top-K
        pool), then cull the weakest eligible learner.
        """
        if len(scores) != len(self.members):
            raise ValueError("scores must have one entry per league member")
        train = self.cfg.train
        self._manage_pass += 1

        report = ManagementReport()
        report.retired = self.mark_stale_champions(external_scores)

        chosen: List[tuple] = []
        gauntlet = self.champion_gauntlet(min_hands=int(train.league_gauntlet_min_hands))
        candidates = self.promotion_candidates(gauntlet)
        if train.league_promote_elite and candidates:
            mbb = self.strength_mbb_per_100()
            chosen.append((max(candidates, key=lambda i: mbb[i]), "elite"))
        if train.league_promote_most_different and diversity is not None:
            taken = {c for c, _ in chosen}
            pool = [i for i in self._most_different_pool(scores, panel_scores) if i not in taken]
            if pool:
                chosen.append((max(pool, key=lambda i: diversity[i]), "most_different"))
        promoted_slots: List[int] = []
        for index, reason in chosen:
            slot = self.promote(index, reason, diversity=diversity, exclude=promoted_slots)
            if slot is not None:
                promoted_slots.append(slot)
                report.promoted.append((index, slot, reason))

        if cull_worst:
            report.culled = self.cull(scores)
        return report

    # --- veteran drift ------------------------------------------------------
    def veteran_drift(self, observations: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Mean ``KL(veteran || its own snapshot)``; NaN for non-veterans.

        This is the number anchored mode lives or dies on.  Anchoring the
        improvement target *bounds* drift rather than eliminating it, and a bound
        nobody measures is a bound nobody has -- so it is reported every
        management pass and enforced by :meth:`enforce_veteran_drift`.
        """
        from paradigm_a.training.league_metrics import mean_kl, policy_distributions

        drift = np.full(len(self.members), np.nan, dtype=np.float64)
        for index in self.veteran_indices():
            member = self.members[index]
            live = policy_distributions(member.network, observations, masks, device=self.device)
            anchor = policy_distributions(member.snapshot, observations, masks, device=self.device)
            drift[index] = mean_kl(live, anchor, masks)
        return drift

    def enforce_veteran_drift(
        self, drift: Sequence[float], skip: Sequence[int] = ()
    ) -> List[int]:
        """Reset veterans past ``league_veteran_max_kl`` back to their snapshot.

        The returned slots need their optimiser and replay buffer rebuilt by the
        caller: Adam's moment estimates survive a bare ``load_state_dict`` and
        would shove the veteran straight back off the anchor it was just returned
        to, and the buffer holds the drifted policy's experience.  ``skip`` is
        for slots already replaced this pass, whose drift reading predates the
        network now sitting in them.
        """
        budget = float(self.cfg.train.league_veteran_max_kl)
        if budget <= 0:
            return []
        skipped = set(skip)
        reset = []
        for index in self.veteran_indices():
            value = drift[index] if index < len(drift) else np.nan
            if index in skipped or not np.isfinite(value) or value <= budget:
                continue
            member = self.members[index]
            member.network.load_state_dict(copy.deepcopy(member.snapshot.state_dict()))
            # The record was earned by the drifted policy, not by this one.
            self._clear_stats(index)
            reset.append(index)
        return reset

    # --- metrics -----------------------------------------------------------
    def evaluate(self, encoder, seed: int = 0, offline_strength: bool = False) -> dict:
        """Strength, diversity, coverage and the management score.

        Strength and coverage come from the running record of real league play
        (free, and the sample the promotion rules care about).  Set
        ``offline_strength`` to re-measure them with a dedicated round-robin
        instead -- useful before any hands have been played.
        """
        from paradigm_a.training.league_metrics import (
            behavioural_diversity,
            population_scores,
            sample_validation_states,
            strength_and_coverage,
            strength_to_mbb_per_100,
        )

        train = self.cfg.train
        if offline_strength or self.total_hands == 0:
            strength, coverage, pair_mean, hands = strength_and_coverage(
                self.agents, self.cfg, int(train.league_strength_hands), encoder, seed=seed
            )
        else:
            strength = self.running_strength()
            coverage = self.running_coverage()
            pair_mean = self.running_pair_mean()
            hands = np.array([m.hands_played for m in self.members], dtype=np.float64)

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
            "strength_mbb_per_100": strength_to_mbb_per_100(strength, self.cfg.env),
            "diversity": diversity,
            "coverage": coverage,
            "score": score,
            "normalised_strength": norm_strength,
            "normalised_diversity": norm_diversity,
            "pair_mean": pair_mean,
            "kl_matrix": kl,
            "hands_played": hands,
            "roles": [m.role.value for m in self.members],
            # Free here: the validation bank is already sampled and the live
            # policies already computed, so drift costs one forward pass per
            # veteran snapshot.
            "veteran_drift": self.veteran_drift(observations, masks),
        }

    def summary(self, metrics: dict, top: int = 5) -> str:
        """A compact, sorted view of the population for training logs."""
        order = np.argsort(-np.asarray(metrics["score"]))
        lines = [f"{'slot':>5}{'role':>10}{'score':>8}{'mbb/100':>12}{'diversity':>11}{'cover':>7}"]
        for index in order[:top]:
            lines.append(
                f"{index:>5}{self.members[index].label:>10}"
                f"{metrics['score'][index]:>8.3f}{metrics['strength_mbb_per_100'][index]:>12.0f}"
                f"{metrics['diversity'][index]:>11.4f}{int(metrics['coverage'][index]):>7}"
            )
        return "\n".join(lines)
