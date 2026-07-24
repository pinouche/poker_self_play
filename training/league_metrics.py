"""Population metrics for behaviour-aware league management.

Three quantities describe a league member, and deliberately *not* Elo: Elo
assumes a two-player, transitive game, and three-handed poker is neither.  A
network that crushes B and B that crushes C tells you very little about A vs C
when they share a table.

* **Strength** -- expected chip return per hand, normalised by the seat's own
  starting stack, averaged over many opponent triples.  Money is the metric;
  win rate is not (a policy can win most hands and still lose chips).
  Normalising by the starting stack keeps different stack depths and blind
  levels comparable.

* **Behavioural diversity** -- mean pairwise KL between action distributions on
  a *fixed* set of validation states.  Measuring on shared states is what makes
  this "acts differently in the same spot" rather than "happens to score
  differently"; two policies can have identical strength and be strategically
  unrelated.

* **Coverage** -- how many distinct opponents a member beats.  Three-handed
  there is no single opponent, so this counts opponents ``j`` against whom
  ``i``'s average chip return is positive over the hands they shared.

The management score combines the first two:

    score = w_strength * norm(strength) + w_diversity * norm(diversity)

Both terms are min-max normalised across the population *first*.  That is not
cosmetic: raw chip EV is O(0.05) stacks while raw KL is O(1) nat, so an
unnormalised 0.6/0.4 sum would be decided almost entirely by the KL term.  After
normalisation the weights mean what they look like.

Scoring on strength alone is exactly what collapses a league into N copies of
one policy; keeping a diversity term means a slightly weaker but strategically
different network still earns its slot.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

from config import Config
from environment.poker_env import PokerEnv
from representation.observation_encoder import ObservationEncoder

_EPS = 1e-9


# --- validation states ------------------------------------------------------
def sample_validation_states(
    agents: Sequence,
    cfg: Config,
    num_states: int,
    encoder: ObservationEncoder,
    seed: int = 0,
    max_hands: int = 1_000_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """A fixed bank of decision states, sampled from real self-play.

    Returns ``(observations [N, obs_dim], legal_masks [N, num_actions])``.  The
    bank must be *fixed* across the population: diversity is only meaningful if
    every member is asked about the same spots.
    """
    from training.self_play import play_hand

    env = PokerEnv(cfg.env, cfg.obs, seed=seed)
    rng = random.Random(seed)
    observations: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    hands = 0
    while len(observations) < num_states and hands < max_hands:
        env.reset()
        while not env.is_terminal and len(observations) < num_states:
            seat = env.to_act
            observation = env.get_observation(seat)
            mask = observation["legal_action_mask"]
            flat = encoder.encode_flat(observation)
            observations.append(np.asarray(flat, dtype=np.float32))
            masks.append(np.asarray(mask, dtype=np.float32))
            choice = agents[rng.randrange(len(agents))].act(observation, flat, mask, rng)
            env.step(choice.action)
        hands += 1

    if not observations:  # pragma: no cover - only with num_states = 0
        raise ValueError("collected no validation states")
    return np.stack(observations), np.stack(masks)


# --- behavioural diversity --------------------------------------------------
def masked_softmax_rows(logits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Row-wise softmax over legal actions; illegal entries are exactly zero."""
    legal = masks > 0
    scores = np.where(legal, logits.astype(np.float64), -np.inf)
    scores = scores - scores.max(axis=1, keepdims=True)
    weights = np.where(legal, np.exp(scores), 0.0)
    return weights / np.maximum(weights.sum(axis=1, keepdims=True), _EPS)


def policy_distributions(
    network,
    observations: np.ndarray,
    masks: np.ndarray,
    device: Optional[str] = None,
    batch_size: int = 4096,
) -> np.ndarray:
    """``pi(a|s)`` for one network over the validation bank, ``[N, num_actions]``.

    This is the policy head's own distribution -- what the network *would play* --
    rather than the Q-derived improved policy, so diversity measures the learned
    policy itself and does not move when alpha/beta change.
    """
    chunks = []
    for start in range(0, len(observations), batch_size):
        logits, _ = network.infer_batch(observations[start : start + batch_size], device=device)
        chunks.append(np.asarray(logits))
    return masked_softmax_rows(np.concatenate(chunks, axis=0), masks)


def pairwise_kl(policies: Sequence[np.ndarray], masks: np.ndarray) -> np.ndarray:
    """``[n, n]`` matrix of mean ``KL(pi_i || pi_j)`` over the validation states.

    Both distributions are masked to the same legal set, so the supports match
    and the divergence is finite; probabilities are floored before the log for
    numerical safety.  The diagonal is zero.
    """
    n = len(policies)
    legal = masks > 0
    out = np.zeros((n, n), dtype=np.float64)
    logs = [np.log(np.clip(p, _EPS, None)) for p in policies]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            per_state = np.where(legal, policies[i] * (logs[i] - logs[j]), 0.0).sum(axis=1)
            out[i, j] = float(per_state.mean())
    return out


def diversity_from_kl(kl_matrix: np.ndarray) -> np.ndarray:
    """Per-member diversity: mean KL to every *other* member."""
    n = len(kl_matrix)
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    off_diagonal = kl_matrix.sum(axis=1) / (n - 1)
    return off_diagonal


def behavioural_diversity(
    networks: Sequence,
    observations: np.ndarray,
    masks: np.ndarray,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(diversity [n], kl_matrix [n, n])`` for the population."""
    policies = [policy_distributions(net, observations, masks, device=device) for net in networks]
    kl = pairwise_kl(policies, masks)
    return diversity_from_kl(kl), kl


# --- strength and coverage --------------------------------------------------
def strength_and_coverage(
    agents: Sequence,
    cfg: Config,
    num_hands: int,
    encoder: ObservationEncoder,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Play ``num_hands`` over random opponent triples drawn from the population.

    Returns ``(strength [n], coverage [n], pair_mean [n, n], hands_played [n])``:

    * ``strength[i]``  -- mean chip delta per hand as a fraction of ``i``'s own
      starting stack, over every hand ``i`` played.
    * ``pair_mean[i, j]`` -- ``i``'s mean normalised chip delta over the hands
      where ``j`` also sat.  Three-handed this is the closest thing to a
      head-to-head result.
    * ``coverage[i]`` -- number of distinct opponents ``j`` with
      ``pair_mean[i, j] > 0``.

    Seats are drawn with replacement, so a member can face copies of itself; a
    member never counts as its own opponent.
    """
    from training.self_play import play_hand

    n = len(agents)
    env = PokerEnv(cfg.env, cfg.obs, seed=seed)
    rng = random.Random(seed)
    num_players = cfg.env.num_players

    total = np.zeros(n, dtype=np.float64)
    played = np.zeros(n, dtype=np.float64)
    pair_total = np.zeros((n, n), dtype=np.float64)
    pair_count = np.zeros((n, n), dtype=np.float64)

    for _ in range(num_hands):
        seated = [rng.randrange(n) for _ in range(num_players)]
        result = play_hand(env, [agents[m] for m in seated], encoder, rng, collect=False)
        stacks = env.state.initial_stacks
        for seat, member in enumerate(seated):
            normalised = result.chip_deltas[seat] / max(float(stacks[seat]), 1.0)
            total[member] += normalised
            played[member] += 1
            for other in seated:
                if other != member:  # never your own opponent
                    pair_total[member, other] += normalised
                    pair_count[member, other] += 1

    strength = total / np.maximum(played, 1.0)
    pair_mean = pair_total / np.maximum(pair_count, 1.0)
    coverage = ((pair_mean > 0.0) & (pair_count > 0)).sum(axis=1).astype(np.float64)
    return strength, coverage, pair_mean, played


# --- scoring ----------------------------------------------------------------
def min_max_normalise(values: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; an all-equal population maps to 0.5 rather than 0/NaN."""
    values = np.asarray(values, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.full(values.shape, 0.5, dtype=np.float64)
    return (values - low) / (high - low)


def population_scores(
    strength: np.ndarray,
    diversity: np.ndarray,
    strength_weight: float = 0.6,
    diversity_weight: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(score, normalised_strength, normalised_diversity)``.

    Both inputs are min-max normalised across the population before weighting --
    without that the weights are meaningless, because chip EV and KL live on
    completely different scales.
    """
    normalised_strength = min_max_normalise(strength)
    normalised_diversity = min_max_normalise(diversity)
    score = strength_weight * normalised_strength + diversity_weight * normalised_diversity
    return score, normalised_strength, normalised_diversity
