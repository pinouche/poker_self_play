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

from config import Config, reward_scale
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


def network_action_choice(
    logits: np.ndarray,
    q_values: np.ndarray,
    legal_mask: np.ndarray,
    alpha: float,
    beta: float,
    temperature: float,
    q_scale: float,
    rng: random.Random,
) -> ActionChoice:
    """Turn one network output into an action choice.

    Factored out so the sequential agent and the batched worker cannot drift
    apart: batching must change throughput, never behaviour.
    """
    reference = masked_softmax(logits, legal_mask, temperature=1.0)
    improved = improved_policy_np(
        q_values, reference, legal_mask, alpha, beta, temperature=1.0, q_scale=q_scale
    )
    sampling = (
        improved if temperature == 1.0 else apply_temperature(improved, legal_mask, temperature)
    )
    if temperature <= 0:
        action = int(np.argmax(np.where(legal_mask > 0, sampling, -np.inf)))
    else:
        action = sample_action(sampling, rng)
    return ActionChoice(
        action=action,
        policy=sampling.astype(np.float32),
        value=expected_value(sampling, q_values, legal_mask),
        q_values=np.asarray(q_values, dtype=np.float32),
    )


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
        q_scale: float = 1.0,
    ) -> None:
        self.network = network
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.device = device
        self.q_scale = q_scale

    def act(self, observation, flat_observation, legal_mask, rng) -> ActionChoice:
        logits, q_values = self.network.infer(flat_observation, device=self.device)
        # Exploration temperature is applied on top of the improved policy; at
        # temperature 1.0 the sampling distribution and the stored behaviour
        # policy are identical.
        return network_action_choice(
            logits,
            q_values,
            legal_mask,
            self.alpha,
            self.beta,
            self.temperature,
            self.q_scale,
            rng,
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
    collect_seats: Optional[Sequence[int]] = None,
) -> HandResult:
    """Play one hand to completion and build per-seat training transitions.

    ``collect_seats`` restricts collection to the seats played by the learner.
    When fixed opponents share the table their decisions must not become
    training data -- they were not drawn from the policy being improved.
    """
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

        if collect and (collect_seats is None or seat in collect_seats):
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


def build_opponent_pool(
    cfg: Config, network: PokerNet, device: Optional[str] = None
) -> List[Agent]:
    """Instantiate the configured fixed opponents.

    Frozen learner snapshots ("checkpoint") are added separately by the
    training loop via :func:`snapshot_agent`, since they change over time.
    """
    from evaluation.heuristic_agent import HEURISTIC_AGENTS
    from evaluation.random_agent import BASELINE_AGENTS

    pool: List[Agent] = []
    for index, name in enumerate(cfg.train.opponent_pool):
        if name == "checkpoint":
            continue
        if name in BASELINE_AGENTS:
            pool.append(BASELINE_AGENTS[name]())
        elif name in HEURISTIC_AGENTS:
            pool.append(
                HEURISTIC_AGENTS[name](
                    seed=index, samples=cfg.train.opponent_heuristic_samples
                )
            )
        else:
            raise ValueError(f"unknown opponent: {name!r}")
    return pool


def snapshot_agent(cfg: Config, network: PokerNet, device: Optional[str] = None) -> Agent:
    """A frozen copy of the current network, for league play."""
    import copy

    from model.network import build_network

    frozen = build_network(cfg)
    frozen.load_state_dict(copy.deepcopy(network.state_dict()))
    frozen.eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    if device:
        frozen.to(device)
    agent = NetworkAgent(
        frozen,
        alpha=cfg.train.alpha,
        beta=cfg.train.beta,
        temperature=cfg.train.sampling_temperature,
        device=device,
        q_scale=reward_scale(cfg.env),
    )
    agent.name = "checkpoint"
    return agent


class SelfPlayWorker:
    """Repeatedly plays hands with the shared network in every seat."""

    def __init__(
        self,
        cfg: Config,
        network: PokerNet,
        encoder: ObservationEncoder,
        device: Optional[str] = None,
        seed: int = 0,
        opponent_pool: Optional[Sequence[Agent]] = None,
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
            q_scale=reward_scale(cfg.env),
        )
        self.agents = [self.agent] * cfg.env.num_players
        self.opponent_pool: List[Agent] = list(opponent_pool or [])
        self.opponent_mix_prob = cfg.train.opponent_mix_prob

    def add_snapshot(self, agent: Agent) -> None:
        """Append a frozen self-snapshot, evicting the oldest past league_size."""
        self.opponent_pool.append(agent)
        snapshots = [a for a in self.opponent_pool if getattr(a, "name", "") == "checkpoint"]
        for stale in snapshots[: -self.cfg.train.league_size]:
            self.opponent_pool.remove(stale)

    def _opponent_weights(self) -> List[float]:
        """Sampling weights over the pool.

        Fixed bots get weight 1.  Self-snapshots are weighted by recency:
        the newest gets 1, the one before it `decay`, then `decay**2`, ...
        so the learner mostly faces recent -- and therefore stronger --
        versions of itself, while older ones stay in the mix to avoid
        cycling against a single opponent.
        """
        decay = self.cfg.train.league_recency_decay
        snapshots = [
            i for i, a in enumerate(self.opponent_pool)
            if getattr(a, "name", "") == "checkpoint"
        ]
        weights = [1.0] * len(self.opponent_pool)
        for age, index in enumerate(reversed(snapshots)):  # age 0 = newest
            weights[index] = decay ** age
        return weights

    def _choose_opponent(self) -> Agent:
        weights = self._opponent_weights()
        return self.rng.choices(self.opponent_pool, weights=weights, k=1)[0]

    def _seat_agents(self) -> Tuple[List[Agent], List[int]]:
        """Build a table, optionally seating opponents from the pool.

        Pure self-play only ever shows the critic its own policy, so Q is
        calibrated against the self-play population and nothing else.  Mixing in
        fixed bots and frozen past checkpoints widens that distribution.
        """
        n = self.cfg.env.num_players
        agents: List[Agent] = [self.agent] * n
        learner_seats = list(range(n))
        if self.opponent_pool and self.rng.random() < self.opponent_mix_prob:
            num_opponents = self.rng.choice([1, 2])
            seats = self.rng.sample(range(n), num_opponents)
            agents = list(agents)
            for seat in seats:
                agents[seat] = self._choose_opponent()
            learner_seats = [s for s in range(n) if s not in seats]
        return agents, learner_seats

    def generate(self, num_hands: int) -> Tuple[List[Transition], dict]:
        transitions: List[Transition] = []
        showdowns = 0
        decisions = 0
        reward_sums = np.zeros(self.cfg.env.num_players, dtype=np.float64)
        chip_sums = np.zeros(self.cfg.env.num_players, dtype=np.float64)

        mixed_hands = 0
        for _ in range(num_hands):
            agents, learner_seats = self._seat_agents()
            mixed_hands += len(learner_seats) < self.cfg.env.num_players
            for agent in agents:
                agent.reset()
            result = play_hand(
                self.env,
                agents,
                self.encoder,
                self.rng,
                gamma=self.cfg.train.gamma,
                lam=self.cfg.train.lam,
                store_next_obs=self.cfg.train.store_next_obs,
                collect_seats=learner_seats,
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
            "mixed_opponent_rate": mixed_hands / max(1, num_hands),
            "mean_reward_per_seat": (reward_sums / max(1, num_hands)).tolist(),
            "mean_chips_per_seat": (chip_sums / max(1, num_hands)).tolist(),
        }
        return transitions, stats


class BatchedSelfPlayWorker:
    """Self-play that steps many hands in lockstep to batch the network.

    Sequential self-play spends most of its time on single-row forward passes.
    Here ``num_envs`` independent hands advance together: at each step every
    hand that needs a *network* decision is encoded, the batch goes through the
    network once, and the results are distributed back.

    Semantics are identical to :class:`SelfPlayWorker` -- the hands are
    independent, each carries its own RNG, and action selection goes through the
    same :func:`network_action_choice`.  Only throughput changes.
    """

    def __init__(
        self,
        cfg: Config,
        network: PokerNet,
        encoder: ObservationEncoder,
        device: Optional[str] = None,
        seed: int = 0,
        num_envs: int = 32,
        opponent_pool: Optional[Sequence[Agent]] = None,
    ) -> None:
        self.cfg = cfg
        self.network = network
        self.encoder = encoder
        self.device = device
        self.num_envs = max(1, num_envs)
        self.q_scale = reward_scale(cfg.env)
        self.rng = random.Random(seed)
        self.opponent_pool: List[Agent] = list(opponent_pool or [])
        self.opponent_mix_prob = cfg.train.opponent_mix_prob

        self.envs = [
            PokerEnv(cfg.env, cfg.obs, seed=seed * 1000 + i) for i in range(self.num_envs)
        ]
        self.rngs = [random.Random(seed * 7919 + i) for i in range(self.num_envs)]

    # Reuse the pool machinery from the sequential worker.
    add_snapshot = SelfPlayWorker.add_snapshot
    _opponent_weights = SelfPlayWorker._opponent_weights
    _choose_opponent = SelfPlayWorker._choose_opponent
    _seat_agents = SelfPlayWorker._seat_agents

    @property
    def agent(self) -> Agent:
        """A sequential-style agent view, used by ``_seat_agents``."""
        if not hasattr(self, "_agent"):
            self._agent = NetworkAgent(
                self.network,
                alpha=self.cfg.train.alpha,
                beta=self.cfg.train.beta,
                temperature=self.cfg.train.sampling_temperature,
                device=self.device,
                q_scale=self.q_scale,
            )
        return self._agent

    def generate(self, num_hands: int) -> Tuple[List[Transition], dict]:
        cfg = self.cfg.train
        transitions: List[Transition] = []
        completed = showdowns = decisions = mixed = 0
        reward_sums = np.zeros(self.cfg.env.num_players, dtype=np.float64)
        chip_sums = np.zeros(self.cfg.env.num_players, dtype=np.float64)

        # Never run more hands than were asked for: several envs can finish in
        # the same step, so the number started is tracked explicitly rather than
        # inferred from the number completed.
        active_envs = min(self.num_envs, num_hands)
        slots: List[Optional[dict]] = [None] * self.num_envs
        for index in range(active_envs):
            slots[index] = self._start_hand(index)
        started = active_envs

        while any(slot is not None for slot in slots):
            pending = []  # (slot index, seat, observation, flat, mask)
            for index, slot in enumerate(slots):
                if slot is None:
                    continue
                env = self.envs[index]
                if env.is_terminal:
                    continue
                seat = env.to_act
                observation = env.get_observation(seat)
                mask = observation["legal_action_mask"]
                if mask.sum() <= 0:
                    raise RuntimeError(f"seat {seat} has no legal actions")
                pending.append(
                    (index, seat, observation, self.encoder.encode_flat(observation), mask)
                )

            if not pending:
                break

            # One forward pass for every hand needing a network decision.
            network_rows = [p for p in pending if p[1] in slots[p[0]]["learner_seats"]]
            batched_outputs = {}
            if network_rows:
                observations = np.stack([row[3] for row in network_rows])
                logits, q_values = self.network.infer_batch(observations, device=self.device)
                for offset, row in enumerate(network_rows):
                    batched_outputs[(row[0], row[1])] = (logits[offset], q_values[offset])

            for index, seat, observation, flat, mask in pending:
                slot = slots[index]
                if (index, seat) in batched_outputs:
                    logit_row, q_row = batched_outputs[(index, seat)]
                    choice = network_action_choice(
                        logit_row,
                        q_row,
                        mask,
                        cfg.alpha,
                        cfg.beta,
                        cfg.sampling_temperature,
                        self.q_scale,
                        self.rngs[index],
                    )
                    slot["chains"][seat].append(
                        _Step(
                            observation=flat,
                            legal_mask=mask.copy(),
                            action=choice.action,
                            policy=choice.policy.copy(),
                            value=choice.value,
                        )
                    )
                else:
                    choice = slot["agents"][seat].act(
                        observation, flat, mask, self.rngs[index]
                    )
                decisions += 1
                self.envs[index].step(choice.action)

            for index, slot in enumerate(slots):
                if slot is None or not self.envs[index].is_terminal:
                    continue
                env = self.envs[index]
                rewards = env.terminal_rewards()
                transitions.extend(
                    build_transitions(
                        slot["chains"], rewards, cfg.gamma, cfg.lam, cfg.store_next_obs
                    )
                )
                showdowns += int(env.state.went_to_showdown)
                mixed += slot["mixed"]
                reward_sums += rewards
                chip_sums += env.chip_deltas()
                completed += 1
                if started < num_hands:
                    slots[index] = self._start_hand(index)
                    started += 1
                else:
                    slots[index] = None

        stats = {
            "hands": completed,
            "transitions": len(transitions),
            "decisions_per_hand": decisions / max(1, completed),
            "showdown_rate": showdowns / max(1, completed),
            "mixed_opponent_rate": mixed / max(1, completed),
            "mean_reward_per_seat": (reward_sums / max(1, completed)).tolist(),
            "mean_chips_per_seat": (chip_sums / max(1, completed)).tolist(),
            "num_envs": self.num_envs,
        }
        return transitions, stats

    def _start_hand(self, index: int) -> dict:
        agents, learner_seats = self._seat_agents()
        for agent in agents:
            agent.reset()
        self.envs[index].reset()
        return {
            "agents": agents,
            "learner_seats": set(learner_seats),
            "chains": {seat: [] for seat in range(self.cfg.env.num_players)},
            "mixed": int(len(learner_seats) < self.cfg.env.num_players),
        }


# ============================================================================
# Population self-play
# ============================================================================
def play_population_hand(
    env: PokerEnv,
    learner: Agent,
    villains: Sequence[Agent],
    standin: Optional[Agent],
    setup,
    learner_seat: int,
    entry_street,
    encoder: ObservationEncoder,
    rng: random.Random,
    gamma: float,
    lam: float,
    store_next_obs: bool = True,
    max_decisions: int = 500,
) -> HandResult:
    """Play one full game and collect transitions for the learner seat only.

    The learner occupies ``learner_seat``.  Before ``entry_street`` that seat is
    controlled by ``standin`` (a population policy), so the state the learner
    inherits at its entry street was produced by real play rather than sampled
    synthetically.  The other seats are controlled by ``villains`` throughout.
    Only decisions the learner itself makes become training transitions.
    """
    env.reset(
        dealer=setup.dealer,
        stacks=setup.stacks,
        small_blind=setup.small_blind,
        big_blind=setup.big_blind,
    )
    num_players = env.state.num_players

    # Map the two non-learner seats to the villain policies in order.
    villain_by_seat = {}
    vi = 0
    for seat in range(num_players):
        if seat != learner_seat:
            villain_by_seat[seat] = villains[vi % len(villains)]
            vi += 1

    chain: List[_Step] = []
    decisions = 0
    while not env.is_terminal:
        decisions += 1
        if decisions > max_decisions:  # pragma: no cover - rule-bug guard
            raise RuntimeError("hand exceeded the maximum number of decisions")

        seat = env.to_act
        observation = env.get_observation(seat)
        legal_mask = observation["legal_action_mask"]
        if legal_mask.sum() <= 0:
            raise RuntimeError(f"seat {seat} has no legal actions")
        flat = encoder.encode_flat(observation)

        is_learner_turn = seat == learner_seat and int(env.state.street) >= int(entry_street)
        if is_learner_turn:
            controller = learner
        elif seat == learner_seat:
            controller = standin if standin is not None else learner
        else:
            controller = villain_by_seat[seat]

        choice = controller.act(observation, flat, legal_mask, rng)
        if not legal_mask[choice.action]:
            raise RuntimeError(f"agent {controller.name} chose illegal action {choice.action}")

        if is_learner_turn:
            chain.append(
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
    transitions = build_transitions(
        {learner_seat: chain}, rewards, gamma, lam, store_next_obs
    )
    return HandResult(
        transitions=transitions,
        rewards=rewards,
        chip_deltas=env.chip_deltas(),
        went_to_showdown=env.state.went_to_showdown,
        num_decisions=decisions,
        dealer=env.state.dealer,
    )


class PopulationSelfPlayWorker:
    """Self-play against a library of frozen checkpoint policies.

    Every hand seats the learner once and fills the other two seats from the
    library (recency-weighted).  Games start from a randomised setup and run to
    showdown; a configurable fraction begin with the learner entering on a later
    street, the earlier streets played out by the population.

    The library is seeded with a snapshot of the initial network so villains
    exist from the first hand.  ``maybe_snapshot`` adds the current network on a
    schedule.  Because the seats hold heterogeneous policies, decisions are not
    batched -- each acts with its own forward pass.
    """

    def __init__(
        self,
        cfg: Config,
        network: PokerNet,
        encoder: ObservationEncoder,
        device: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        from training.population import PolicyLibrary, freeze_policy

        self.cfg = cfg
        self.network = network
        self.encoder = encoder
        self.device = device
        self.rng = random.Random(seed)
        self.env = PokerEnv(cfg.env, cfg.obs, seed=seed)
        self._freeze = freeze_policy

        self.learner = NetworkAgent(
            network,
            alpha=cfg.train.alpha,
            beta=cfg.train.beta,
            temperature=cfg.train.sampling_temperature,
            device=device,
            q_scale=reward_scale(cfg.env),
        )
        self.library = PolicyLibrary(
            capacity=cfg.train.population_library_size,
            weighting=cfg.train.league_weighting,
            decay=cfg.train.league_recency_decay,
        )
        self.library.add(self._freeze(cfg, network, device))  # seed at iter 0
        self._iteration = 0

    def maybe_snapshot(self) -> None:
        """Add the current network to the library on the configured schedule."""
        self._iteration += 1
        every = max(1, self.cfg.train.population_snapshot_every)
        if self._iteration % every == 0:
            self.library.add(self._freeze(self.cfg, self.network, self.device))

    def generate(self, num_hands: int) -> Tuple[List[Transition], dict]:
        from training.population import sample_entry_street, sample_game_setup

        transitions: List[Transition] = []
        showdowns = decisions = collected_hands = 0
        entry_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        num_players = self.cfg.env.num_players

        for _ in range(num_hands):
            setup = sample_game_setup(self.rng, self.cfg)
            learner_seat = self.rng.randrange(num_players)
            entry_street = sample_entry_street(self.rng, self.cfg)
            villains = self.library.sample_many(num_players - 1, self.rng)
            standin = self.library.sample(self.rng) if int(entry_street) > 0 else None
            for agent in (self.learner, standin, *villains):
                if agent is not None:
                    agent.reset()

            result = play_population_hand(
                self.env, self.learner, villains, standin, setup, learner_seat,
                entry_street, self.encoder, self.rng,
                gamma=self.cfg.train.gamma, lam=self.cfg.train.lam,
                store_next_obs=self.cfg.train.store_next_obs,
            )
            transitions.extend(result.transitions)
            showdowns += int(result.went_to_showdown)
            decisions += result.num_decisions
            collected_hands += int(bool(result.transitions))
            entry_counts[int(entry_street)] += 1

        stats = {
            "hands": num_hands,
            "transitions": len(transitions),
            "decisions_per_hand": decisions / max(1, num_hands),
            "showdown_rate": showdowns / max(1, num_hands),
            "hands_with_learner_transitions": collected_hands / max(1, num_hands),
            "library_size": len(self.library),
            "entry_street_fractions": [entry_counts[i] / max(1, num_hands) for i in range(4)],
        }
        return transitions, stats


# ============================================================================
# Co-evolving population self-play
# ============================================================================
class CoevolutionSelfPlayWorker:
    """Self-play among a co-evolving population of *live* networks.

    The other workers in this module drive all three seats with a single network
    (plus, optionally, frozen opponents) and improve only that one network.  Here
    every one of ``num_policies`` networks is being optimised at once.  Each hand
    samples one network per seat, uniformly and **with replacement**, so a
    network faces other members of the population as well as fresh copies of
    itself.  Every seat's decisions become training data for the network that
    produced them, and :meth:`generate` returns the transitions grouped by
    owning network so the caller can route each group to that network's own
    replay buffer and optimiser.

    ``num_envs`` hands are advanced in lockstep as in
    :class:`BatchedSelfPlayWorker`.  Seats can hold different networks, so a
    single forward pass no longer covers a whole step; instead the step's pending
    decisions are grouped by network and each network runs one batched pass over
    the rows it owns.  Semantics do not depend on ``num_envs`` -- the hands are
    independent and each carries its own RNG -- only throughput does.
    """

    def __init__(
        self,
        cfg: Config,
        networks: Sequence[PokerNet],
        encoder: ObservationEncoder,
        device: Optional[str] = None,
        seed: int = 0,
        num_envs: int = 1,
    ) -> None:
        if len(networks) < 1:
            raise ValueError("co-evolution needs at least one network")
        self.cfg = cfg
        self.networks = list(networks)
        self.num_policies = len(self.networks)
        self.encoder = encoder
        self.device = device
        self.num_envs = max(1, num_envs)
        # alpha/beta/temperature/q_scale are shared config, applied directly in
        # ``network_action_choice`` -- the seats differ only in which network's
        # forward pass feeds them, so no per-network ``NetworkAgent`` is needed.
        self.q_scale = reward_scale(cfg.env)
        self.rng = random.Random(seed)
        self.envs = [
            PokerEnv(cfg.env, cfg.obs, seed=seed * 1000 + i) for i in range(self.num_envs)
        ]
        self.rngs = [random.Random(seed * 7919 + i) for i in range(self.num_envs)]

    def _sample_seat_owners(self) -> List[int]:
        """One network index per seat, drawn uniformly with replacement.

        With replacement means the same network can occupy several seats and thus
        play against copies of itself.
        """
        n = self.cfg.env.num_players
        return [self.rng.randrange(self.num_policies) for _ in range(n)]

    def _start_hand(self, index: int) -> dict:
        self.envs[index].reset()
        return {
            "seat_owner": self._sample_seat_owners(),
            "chains": {seat: [] for seat in range(self.cfg.env.num_players)},
        }

    def generate(self, num_hands: int) -> Tuple[List[List[Transition]], dict]:
        """Play ``num_hands`` and return transitions grouped by owning network.

        The return value is ``(per_network_transitions, stats)`` where
        ``per_network_transitions[k]`` holds the transitions produced by seats
        that network ``k`` played this batch.
        """
        cfg = self.cfg.train
        per_network: List[List[Transition]] = [[] for _ in range(self.num_policies)]
        completed = showdowns = decisions = 0
        reward_sums = np.zeros(self.cfg.env.num_players, dtype=np.float64)
        chip_sums = np.zeros(self.cfg.env.num_players, dtype=np.float64)

        active_envs = min(self.num_envs, num_hands)
        slots: List[Optional[dict]] = [None] * self.num_envs
        for index in range(active_envs):
            slots[index] = self._start_hand(index)
        started = active_envs

        while any(slot is not None for slot in slots):
            # Gather every pending decision across the active envs.  Every seat is
            # network-controlled here, so each row needs a (network) forward pass.
            pending = []  # (index, seat, flat, mask, owner)
            for index, slot in enumerate(slots):
                if slot is None:
                    continue
                env = self.envs[index]
                if env.is_terminal:
                    continue
                seat = env.to_act
                observation = env.get_observation(seat)
                mask = observation["legal_action_mask"]
                if mask.sum() <= 0:
                    raise RuntimeError(f"seat {seat} has no legal actions")
                flat = self.encoder.encode_flat(observation)
                pending.append((index, seat, flat, mask, slot["seat_owner"][seat]))

            if not pending:
                break

            # One batched forward pass per network, over the rows it owns.
            outputs = {}  # (index, seat) -> (logits, q_values)
            for net_idx in range(self.num_policies):
                rows = [row for row in pending if row[4] == net_idx]
                if not rows:
                    continue
                observations = np.stack([row[2] for row in rows])
                logits, q_values = self.networks[net_idx].infer_batch(
                    observations, device=self.device
                )
                for offset, row in enumerate(rows):
                    outputs[(row[0], row[1])] = (logits[offset], q_values[offset])

            for index, seat, flat, mask, _owner in pending:
                logit_row, q_row = outputs[(index, seat)]
                choice = network_action_choice(
                    logit_row,
                    q_row,
                    mask,
                    cfg.alpha,
                    cfg.beta,
                    cfg.sampling_temperature,
                    self.q_scale,
                    self.rngs[index],
                )
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
                # Route each seat's transitions to the network that played it.
                for seat, chain in slot["chains"].items():
                    if not chain:
                        continue
                    owner = slot["seat_owner"][seat]
                    per_network[owner].extend(
                        build_transitions(
                            {seat: chain}, rewards, cfg.gamma, cfg.lam, cfg.store_next_obs
                        )
                    )
                showdowns += int(env.state.went_to_showdown)
                reward_sums += rewards
                chip_sums += env.chip_deltas()
                completed += 1
                if started < num_hands:
                    slots[index] = self._start_hand(index)
                    started += 1
                else:
                    slots[index] = None

        transitions_per_network = [len(group) for group in per_network]
        stats = {
            "hands": completed,
            "transitions": sum(transitions_per_network),
            "transitions_per_network": transitions_per_network,
            "decisions_per_hand": decisions / max(1, completed),
            "showdown_rate": showdowns / max(1, completed),
            "mean_reward_per_seat": (reward_sums / max(1, completed)).tolist(),
            "mean_chips_per_seat": (chip_sums / max(1, completed)).tolist(),
            "num_policies": self.num_policies,
            "num_envs": self.num_envs,
        }
        return per_network, stats
