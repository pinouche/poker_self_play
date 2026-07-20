"""Return targets for the Q head.

The subtlety in a three-player game is *whose* timeline a return is computed
along.  A transition belongs to the player who acted, and that player's "next
state" is their **next own decision point** -- not the state that immediately
follows, which usually belongs to a different seat.  Self-play therefore builds
one chain per seat and the recursions below run along a single chain.

Rewards are terminal-only (poker pays out at the end of the hand), so with
``lam = 1`` the target is exactly the Monte-Carlo outcome and with ``lam < 1``
it blends in the network's own bootstrap values.
"""

from __future__ import annotations

from typing import List, Optional, Sequence


def lambda_returns(
    values: Sequence[float],
    terminal_reward: float,
    gamma: float,
    lam: float,
    rewards: Optional[Sequence[float]] = None,
) -> List[float]:
    """lambda-returns along one player's own chain of decisions.

    ``values[t]`` is V(s_t) as estimated when the action was taken; only
    ``values[t+1]`` is ever used as a bootstrap, so ``values[0]`` is irrelevant
    to the result.  ``rewards[t]`` is the immediate reward collected after step
    ``t`` (zero everywhere in poker); the final step additionally receives
    ``terminal_reward``.

    Recursion, for t < T:
        G_t = r_t + gamma * [ (1 - lam) * V(s_{t+1}) + lam * G_{t+1} ]
        G_T = r_T + terminal_reward
    """
    length = len(values)
    if length == 0:
        return []
    if rewards is None:
        rewards = [0.0] * length
    if len(rewards) != length:
        raise ValueError("rewards and values must have the same length")

    targets = [0.0] * length
    targets[-1] = float(rewards[-1]) + float(terminal_reward)
    for t in range(length - 2, -1, -1):
        bootstrap = (1.0 - lam) * float(values[t + 1]) + lam * targets[t + 1]
        targets[t] = float(rewards[t]) + gamma * bootstrap
    return targets


def n_step_returns(
    values: Sequence[float],
    terminal_reward: float,
    gamma: float,
    n: int,
    rewards: Optional[Sequence[float]] = None,
) -> List[float]:
    """n-step returns along one player's chain (alternative to lambda-returns)."""
    length = len(values)
    if length == 0:
        return []
    if rewards is None:
        rewards = [0.0] * length

    targets = []
    for t in range(length):
        total = 0.0
        discount = 1.0
        horizon = min(n, length - t)
        for k in range(horizon):
            total += discount * float(rewards[t + k])
            discount *= gamma
        last = t + horizon
        if last >= length:
            # The chain ended: the terminal reward lands on the final step.
            total += (gamma ** (length - 1 - t)) * float(terminal_reward)
        else:
            total += discount * float(values[last])
        targets.append(total)
    return targets


def monte_carlo_returns(
    length: int, terminal_reward: float, gamma: float
) -> List[float]:
    """Undiscounted-to-terminal baseline used for sanity checks."""
    return [float(terminal_reward) * (gamma ** (length - 1 - t)) for t in range(length)]
