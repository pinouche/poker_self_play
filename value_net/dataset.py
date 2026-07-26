"""Training data for the value network.

Two ways to get belief states to learn from, and the difference between them is
the difference between stage 2 and stage 3.

**Sampled** (this module's :func:`sample_pbs`): draw a public state and a pair
of ranges at random.  Broad coverage, no dependence on any policy, and it makes
the network's accuracy measurable in isolation — supervised regression against
the exact solver.

**Self-generated** (``rebel/``): the belief states the agent's own search
actually visits.  Narrower, but on the distribution that matters, and it is the
only way to reach the fixed point where the network predicts the values of the
policy the search produces.

Ranges are drawn from a Dirichlet whose concentration is itself randomised, so
the mixture spans everything from "one hand, certain" to "uniform", with some
hands zeroed outright to imitate the sharp ranges real betting produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from belief.public_tree import PublicState, build_public_tree
from belief.ranges import NUM_HANDS, NUM_PLAYERS, PBS, board_mask
from cfr.tabular_cfr import CFRConfig
from value_net.features import encode_batch
from value_net.values import exact_pbs_values


def leaf_public_states(depth_limit: int = 1) -> List[PublicState]:
    """Every belief state a depth-limited search can bottom out in.

    In Leduc these are the thirty starts of round two: five ways for the first
    round to end, times six board cards.
    """
    tree = build_public_tree(depth_limit=depth_limit)
    return [leaf.public for leaf in tree.leaves()]


def random_range(
    rng: np.random.Generator, board: int, sparsity: float = 0.3
) -> np.ndarray:
    """A plausible range: Dirichlet, sometimes with hands knocked out."""
    concentration = float(np.exp(rng.uniform(np.log(0.15), np.log(6.0))))
    weights = rng.dirichlet(np.full(NUM_HANDS, concentration))
    if rng.random() < sparsity:
        keep = rng.random(NUM_HANDS) > 0.35
        if keep.any():
            weights = weights * keep
    weights = weights * board_mask(board)
    total = weights.sum()
    if total <= 0.0:
        weights = board_mask(board)
        total = weights.sum()
    return weights / total


def sample_pbs(
    rng: np.random.Generator, publics: Optional[Sequence[PublicState]] = None
) -> PBS:
    """A random public belief state at a depth-limit boundary."""
    publics = publics if publics is not None else leaf_public_states()
    public = publics[rng.integers(len(publics))]
    ranges = np.stack([random_range(rng, public.board) for _ in range(NUM_PLAYERS)])
    return PBS(public=public, ranges=ranges)


@dataclass
class ValueDataset:
    """Encoded belief states and the values the exact solver gives them."""

    features: np.ndarray  # (n, INPUT_DIM)
    targets: np.ndarray  # (n, 2, NUM_HANDS)

    def __len__(self) -> int:
        return len(self.features)

    def split(self, fraction: float = 0.2) -> Tuple["ValueDataset", "ValueDataset"]:
        cut = int(len(self) * (1.0 - fraction))
        return (
            ValueDataset(self.features[:cut], self.targets[:cut]),
            ValueDataset(self.features[cut:], self.targets[cut:]),
        )


def build_dataset(
    num_samples: int,
    rng: np.random.Generator,
    iterations: int = 150,
    config: CFRConfig | None = None,
) -> ValueDataset:
    """Sample belief states and solve each one exactly."""
    publics = leaf_public_states()
    states = [sample_pbs(rng, publics) for _ in range(num_samples)]
    targets = np.empty((num_samples, NUM_PLAYERS, NUM_HANDS))
    for i, pbs in enumerate(states):
        values = exact_pbs_values(pbs, iterations=iterations, config=config)
        targets[i] = values * board_mask(pbs.public.board)
    return ValueDataset(features=encode_batch(states), targets=targets)
