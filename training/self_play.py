"""Self-play episode generation.

All three seats are driven by the *same* network object.  The only thing that
differs between them is the observation, which is canonicalised to whichever
seat is acting.

Transition construction is the part worth reading carefully.  Each seat gets
its own chain of decisions; the lambda-return for a decision bootstraps from
that seat's *next own* decision, and the chain terminates with that seat's own
terminal reward.  A seat that folds on the flop terminates there -- its outcome
is already determined -- while the remaining seats keep acting.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import Config
from environment.poker_env import PokerEnv
from model.network import PokerNet
from representation.action_encoder import apply_temperature, masked_softmax, sample_action
from representation.observation_encoder import ObservationEncoder

from .policy_improvement import expected_value, improved_policy_np
from .replay_buffer import Transition
from .returns import lambda_returns


@dataclass
class ActionChoice:
    """What an agent decided, plus the statistics needed for training."""

    action: int
    policy: np.ndarray                  # distribution actually sampled from
    value: float                        # V(s) under that distribution
    q_values: Optional[np.ndarray] = None


class Agent:
    """Anything that can pick an action given a canonical observation."""

    name: str = "agent"

    def act(
        self,
        observation: Dict[str, np.ndarray],
        flat_observation: np.ndarray,
        legal_mask: np.ndarray,
        rng: random.Random,
    ) -> ActionChoice:
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover - most agents are stateless
        pass


class NetworkAgent(Agent):
    """Acts with the shared network plus the policy-improvement operator."""

    name = "network"

    def __init__(
        self,
        network: PokerNet,
        alpha: float = 1.0,
        beta: float = 1.0,
        temperature: float = 1.0,
        device: Optional[str] = None,
    ) -> None:
        self.network = network
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.device = device

    def act(self, observation, flat_observation, legal_mask, rng) -> ActionChoice:
        logits, q_values = self.network.infer(flat_observation, device=self.device)

        # The network's own (masked) policy is the reference for improvement.
        reference = masked_softmax(logits, legal_mask, temperature=1.0)
        improved = improved_policy_np(
            q_values, reference, legal_mask, self.alpha, self.beta, temperature=1.0
        )

        # Exploration temperature is applied on top; at temperature 1.0 the
        # sampling distribution and the stored behaviour policy are identical.
        sampling = (
            improved
            if self.temperature == 1.0
            else apply_temperature(improved, legal_mask, self.temperature)
        )

        if self.temperature <= 0:
            action = int(np.argmax(np.where(legal_mask > 0, sampling, -np.inf)))
        else:
            action = sample_action(sampling, rng)

        return ActionChoice(
            action=action,
            policy=sampling.astype(np.float32),
            value=expected_value(sampling, q_values, legal_mask),
            q_values=np.asarray(q_values, dtype=np.float32),
        )


@dataclass
class _Step:
    observation: np.ndarray
    legal_mask: np.ndarray
    action: int
    policy: np.ndarray
    value: float


@dataclass
class HandResult:
    transitions: List[Transition]
    rewards: List[float]
    chip_deltas: List[int]
    went_to_showdown: bool
    num_decisions: int
    dealer: int


def play_hand(
    env: PokerEnv,
    agents: Sequence[Agent],
    encoder: ObservationEncoder,
    rng: random.Random,
    gamma: float = 0.99,
    lam: float = 0.95,
    dealer: Optional[int] = None,
    collect: bool = True,
    store_next_obs: bool = True,
    max_decisions: int = 500,
) -> HandResult:
    """Play one hand to completion and build per-seat training transitions."""
    state = env.reset(dealer=dealer)
    num_players = state.num_players
    chains: Dict[int, List[_Step]] = {seat: [] for seat in range(num_players)}
    decisions = 0

    while not env.is_terminal:
        decisions += 1
        if decisions > max_decisions:  # pragma: no cover - guards against rule bugs
            raise RuntimeError("hand exceeded the maximum number of decisions")

        seat = env.to_act
        assert seat is not None
        observation = env.get_observation(seat)
        legal_mask = observation["legal_action_mask"]
        if legal_mask.sum() <= 0:
            raise RuntimeError(f"seat {seat} has no legal actions")

        flat = encoder.encode_flat(observation)
        choice = agents[seat].act(observation, flat, legal_mask, rng)

        if not legal_mask[choice.action]:
            raise RuntimeError(
                f"agent {agents[seat].name} chose illegal action {choice.action}"
            )

        if collect:
            chains[seat].append(
                _Step(
                    observation=flat,
                    legal_mask=legal_mask.copy(),
                    action=choice.action,
                    policy=choice.policy.copy(),
                    value=choice.value,
                )
            )
        env.step(choice.action)

    rewards = env.terminal_rewards()
    transitions = (
        build_transitions(chains, rewards, gamma, lam, store_next_obs) if collect else []
    )

    return HandResult(
        transitions=transitions,
        rewards=rewards,
        chip_deltas=env.chip_deltas(),
        went_to_showdown=env.state.went_to_showdown,
        num_decisions=decisions,
        dealer=state.dealer,
    )


def build_transitions(
    chains: Dict[int, List[_Step]],
    rewards: Sequence[float],
    gamma: float,
    lam: float,
    store_next_obs: bool = True,
) -> List[Transition]:
    """Turn per-seat decision chains into transitions with lambda-return targets.

    Every transition carries the reward of *the seat that acted*.  Mixing
    perspectives here is the single easiest way to silently break three-player
    training, so the seat is threaded through explicitly and stored.
    """
    transitions: List[Transition] = []

    for seat, chain in chains.items():
        if not chain:
            continue
        terminal_reward = float(rewards[seat])
        values = [step.value for step in chain]
        targets = lambda_returns(values, terminal_reward, gamma, lam)

        for t, step in enumerate(chain):
            is_last = t == len(chain) - 1
            next_step = None if is_last else chain[t + 1]
            transitions.append(
                Transition(
                    observation=step.observation,
                    legal_action_mask=step.legal_mask,
                    action=step.action,
                    reward=terminal_reward if is_last else 0.0,
                    next_observation=(
                        next_step.observation if (next_step and store_next_obs) else None
                    ),
                    next_legal_action_mask=(
                        next_step.legal_mask if next_step is not None else None
                    ),
                    done=is_last,
                    player_perspective=seat,
                    old_policy=step.policy,
                    q_target=float(targets[t]),
                    value=float(step.value),
                )
            )
    return transitions


class SelfPlayWorker:
    """Repeatedly plays hands with the shared network in every seat."""

    def __init__(
        self,
        cfg: Config,
        network: PokerNet,
        encoder: ObservationEncoder,
        device: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self.network = network
        self.encoder = encoder
        self.rng = random.Random(seed)
        self.env = PokerEnv(cfg.env, cfg.obs, seed=seed)
        self.agent = NetworkAgent(
            network,
            alpha=cfg.train.alpha,
            beta=cfg.train.beta,
            temperature=cfg.train.sampling_temperature,
            device=device,
        )
        self.agents = [self.agent] * cfg.env.num_players

    def generate(self, num_hands: int) -> Tuple[List[Transition], dict]:
        transitions: List[Transition] = []
        showdowns = 0
        decisions = 0
        reward_sums = np.zeros(self.cfg.env.num_players, dtype=np.float64)
        chip_sums = np.zeros(self.cfg.env.num_players, dtype=np.float64)

        for _ in range(num_hands):
            result = play_hand(
                self.env,
                self.agents,
                self.encoder,
                self.rng,
                gamma=self.cfg.train.gamma,
                lam=self.cfg.train.lam,
                store_next_obs=self.cfg.train.store_next_obs,
            )
            transitions.extend(result.transitions)
            showdowns += int(result.went_to_showdown)
            decisions += result.num_decisions
            reward_sums += result.rewards
            chip_sums += result.chip_deltas

        stats = {
            "hands": num_hands,
            "transitions": len(transitions),
            "decisions_per_hand": decisions / max(1, num_hands),
            "showdown_rate": showdowns / max(1, num_hands),
            "mean_reward_per_seat": (reward_sums / max(1, num_hands)).tolist(),
            "mean_chips_per_seat": (chip_sums / max(1, num_hands)).tolist(),
        }
        return transitions, stats
