"""Self-play across a role-structured league.

Seats are filled by :meth:`LeaguePopulation.sample_seats`, so the matchmaking
mix (learners vs learners / learners vs champions / mixed) is decided by the
population rather than by uniform sampling.  Two things follow from that:

* **Only trainable seats are collected.**  Champions play but are never
  optimised, exactly like a fixed opponent -- their decisions were not drawn
  from a policy being improved, so they must not become training data.
* **Every hand is recorded.**  The chip result of *all* seats, champions
  included, is folded into the population's running record, because that record
  is what the promotion and retirement rules are judged on.

Decisions are grouped per member network and batched, so a league of 32 costs
one forward pass per distinct network per step rather than one per seat.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from paradigm_a.config import Config
from paradigm_a.environment.poker_env import PokerEnv
from paradigm_a.representation.observation_encoder import ObservationEncoder
from paradigm_a.training.league import LeaguePopulation, MatchType
from paradigm_a.training.replay_buffer import Transition
from paradigm_a.training.self_play import _Step, build_transitions, network_action_choice


class LeagueSelfPlayWorker:
    """Plays league hands and returns transitions grouped by member."""

    def __init__(
        self,
        cfg: Config,
        population: LeaguePopulation,
        encoder: ObservationEncoder,
        device: Optional[str] = None,
        seed: int = 0,
        num_envs: int = 1,
    ) -> None:
        self.cfg = cfg
        self.population = population
        self.encoder = encoder
        self.device = device
        self.num_envs = max(1, num_envs)
        self.envs = [
            PokerEnv(cfg.env, cfg.obs, seed=seed * 1000 + i) for i in range(self.num_envs)
        ]
        self.rngs = [random.Random(seed * 7919 + i) for i in range(self.num_envs)]

    def _act_params(self, member: int) -> tuple:
        agent = self.population.members[member].agent
        return agent.alpha, agent.beta, agent.temperature, agent.q_scale

    def _start_hand(self, index: int) -> dict:
        self.envs[index].reset()
        seats, match = self.population.sample_seats(self.cfg.env.num_players)
        return {
            "seats": seats,
            "match": match,
            "chains": {seat: [] for seat in range(self.cfg.env.num_players)},
        }

    def generate(self, num_hands: int) -> Tuple[List[List[Transition]], dict]:
        """``(per_member_transitions, stats)``; champions' lists stay empty."""
        cfg = self.cfg.train
        size = len(self.population)
        per_member: List[List[Transition]] = [[] for _ in range(size)]
        completed = showdowns = decisions = collected = 0
        match_counts: Dict[str, int] = {m.value: 0 for m in MatchType}

        active = min(self.num_envs, num_hands)
        slots: List[Optional[dict]] = [None] * self.num_envs
        for index in range(active):
            slots[index] = self._start_hand(index)
        started = active

        while any(slot is not None for slot in slots):
            pending = []  # (env index, seat, flat, mask, member)
            for index, slot in enumerate(slots):
                if slot is None or self.envs[index].is_terminal:
                    continue
                env = self.envs[index]
                seat = env.to_act
                observation = env.get_observation(seat)
                mask = observation["legal_action_mask"]
                if mask.sum() <= 0:
                    raise RuntimeError(f"seat {seat} has no legal actions")
                flat = self.encoder.encode_flat(observation)
                pending.append((index, seat, flat, mask, slot["seats"][seat]))

            if not pending:
                break

            # One batched forward pass per distinct member network.
            outputs = {}
            by_member: Dict[int, List[tuple]] = {}
            for row in pending:
                by_member.setdefault(row[4], []).append(row)
            for member, rows in by_member.items():
                observations = np.stack([row[2] for row in rows])
                logits, q_values = self.population.members[member].network.infer_batch(
                    observations, device=self.device
                )
                for offset, row in enumerate(rows):
                    outputs[(row[0], row[1])] = (logits[offset], q_values[offset])

            for index, seat, flat, mask, member in pending:
                logit_row, q_row = outputs[(index, seat)]
                alpha, beta, temperature, q_scale = self._act_params(member)
                choice = network_action_choice(
                    logit_row, q_row, mask, alpha, beta, temperature, q_scale, self.rngs[index]
                )
                if self.population.members[member].trainable:
                    slots[index]["chains"][seat].append(
                        _Step(
                            observation=flat,
                            legal_mask=mask.copy(),
                            action=choice.action,
                            policy=choice.policy.copy(),
                            value=choice.value,
                        )
                    )
                decisions += 1
                self.envs[index].step(choice.action)

            for index, slot in enumerate(slots):
                if slot is None or not self.envs[index].is_terminal:
                    continue
                env = self.envs[index]
                rewards = env.terminal_rewards()
                stacks = env.state.initial_stacks
                deltas = env.chip_deltas()

                # Every seat's result feeds the running record, champions too:
                # promotion and retirement are judged on real games.
                normalised = [
                    deltas[seat] / max(float(stacks[seat]), 1.0)
                    for seat in range(self.cfg.env.num_players)
                ]
                self.population.record_hand(slot["seats"], normalised)

                for seat, chain in slot["chains"].items():
                    if not chain:
                        continue
                    member = slot["seats"][seat]
                    per_member[member].extend(
                        build_transitions(
                            {seat: chain}, rewards, cfg.gamma, cfg.lam, cfg.store_next_obs
                        )
                    )
                    collected += len(chain)

                showdowns += int(env.state.went_to_showdown)
                match_counts[slot["match"].value] += 1
                completed += 1
                if started < num_hands:
                    slots[index] = self._start_hand(index)
                    started += 1
                else:
                    slots[index] = None

        transitions_per_member = [len(group) for group in per_member]
        stats = {
            "hands": completed,
            "transitions": sum(transitions_per_member),
            "transitions_per_member": transitions_per_member,
            "decisions_per_hand": decisions / max(1, completed),
            "collected_fraction": collected / max(1, decisions),
            "showdown_rate": showdowns / max(1, completed),
            "match_mix": {k: v / max(1, completed) for k, v in match_counts.items()},
            "league_size": size,
            "total_hands": self.population.total_hands,
        }
        return per_member, stats
