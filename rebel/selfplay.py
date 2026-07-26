"""ReBeL self play: search at a belief state, record its values, descend, repeat.

This is Algorithm 1 of Brown et al. 2020, and it is the imperfect-information
analogue of AlphaZero's self play.  Where AlphaZero runs MCTS from a world state
and trains a value network on the search result, ReBeL runs CFR from a *public
belief state* and trains a value network on the counterfactual values the search
produced.

The loop at each belief state:

1. Build the depth-limited subgame and run CFR in it, with the current network
   supplying values at the depth limit.
2. Record ``(belief state, root values)`` as a training example.  The target is
   the search's own answer, which is better than the network's guess — the same
   bootstrap that makes AlphaZero work.
3. Sample one CFR iteration ``t`` at random and descend into a leaf reached by
   *that* iteration's policy, not the average.  This is the detail the paper is
   emphatic about: the average policy is the thing that converges, but the
   network must be accurate at the belief states every iterate visits, because
   those are what the next search will query.  Descending only through the
   average would leave the rest of belief space untrained and the search would
   walk straight into it.

The fixed point this converges to is the point: a network that predicts the
values of the policy its own search produces.  Feed search a value function that
describes some *other* continuation — even an exactly-solved one — and the trunk
optimises against assumptions the continuation will not honour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from belief.public_tree import PublicState, build_public_tree
from belief.ranges import NUM_HANDS, NUM_PLAYERS, PAIR_CORRECTION, PBS, initial_reach
from rebel.config import SelfPlayConfig
from search.evaluate import leaf_reaches
from search.policy import StrategyMap
from search.subgame import LeafValueFn, SubgameSolver
from value_net.features import encode_pbs


@dataclass
class TrainingExample:
    """One belief state and the values search assigned to it."""

    pbs: PBS
    values: np.ndarray  # (2, NUM_HANDS), for normalised ranges

    def encode(self) -> Tuple[np.ndarray, np.ndarray]:
        return encode_pbs(self.pbs), self.values


def normalise_root_values(values: np.ndarray, reach: np.ndarray) -> Optional[np.ndarray]:
    """Convert counterfactual values into the network's per-hand convention.

    Search works with counterfactual values, which carry the opponent's reach
    mass inside them; the network is trained on normalised ranges, so that mass
    is divided back out.  ``None`` when a player cannot reach the belief state at
    all, in which case there is nothing to learn here.
    """
    masses = reach.sum(axis=1)
    if masses.min() <= 0.0:
        return None
    out = np.empty((NUM_PLAYERS, NUM_HANDS))
    out[0] = values[0] / (PAIR_CORRECTION * masses[1])
    out[1] = values[1] / (PAIR_CORRECTION * masses[0])
    return out


def collect_trajectory(
    leaf_value_fn: LeafValueFn,
    config: SelfPlayConfig,
    rng: np.random.Generator,
    root: PublicState | None = None,
    reach: np.ndarray | None = None,
) -> List[TrainingExample]:
    """Play one self-play trajectory through belief space."""
    public = root if root is not None else PublicState()
    reach = initial_reach() if reach is None else np.asarray(reach, dtype=np.float64)
    examples: List[TrainingExample] = []

    while True:
        tree = build_public_tree(public, depth_limit=config.depth_limit)
        solver = SubgameSolver(
            tree,
            leaf_value_fn=leaf_value_fn if tree.leaves() else None,
            config=config.cfr,
        )
        solver.solve(
            reach=reach,
            iterations=config.search_iterations,
            store_iteration_leaves=bool(tree.leaves()),
        )

        values = normalise_root_values(solver.root_values(), reach)
        if values is not None:
            examples.append(
                TrainingExample(pbs=PBS.from_reach(public, reach), values=values)
            )

        if not tree.leaves():
            return examples  # the subgame ran to real terminals

        if rng.random() < config.exploration:
            # Explore by *playing* differently rather than by picking a
            # different leaf: descend under a uniform random profile, which
            # produces the wide, flat ranges equilibrium play never generates
            # and which the network would otherwise never be trained on.
            frontier = leaf_reaches(tree, _uniform_strategies(tree), reach)
        else:
            frontier = _sample_iteration(solver, config, rng)
        chosen = _sample_leaf(frontier, config, rng)
        if chosen is None:
            return examples
        leaf, reach = chosen
        public = leaf.public


def _sample_iteration(
    solver: SubgameSolver, config: SelfPlayConfig, rng: np.random.Generator
) -> Sequence[Tuple]:
    """Pick one CFR iteration's leaf frontier, skipping the warm-up."""
    stored = solver.iteration_leaves
    first = int(len(stored) * config.warmup_fraction)
    index = int(rng.integers(first, len(stored))) if len(stored) > first else -1
    return stored[index]


def _uniform_strategies(tree) -> StrategyMap:
    return {
        node.public: np.full((NUM_HANDS, node.num_actions), 1.0 / node.num_actions)
        for node in tree.decision_nodes()
    }


def _sample_leaf(
    frontier: Sequence[Tuple], config: SelfPlayConfig, rng: np.random.Generator
):
    """Sample a leaf in proportion to how often play actually arrives there."""
    candidates = []
    weights = []
    for leaf, reach in frontier:
        weight = _arrival_probability(reach)
        if weight <= 0.0:
            continue
        candidates.append((leaf, reach))
        weights.append(weight)
    if not candidates:
        return None
    probabilities = np.asarray(weights) / float(np.sum(weights))
    return candidates[int(rng.choice(len(candidates), p=probabilities))]


def _arrival_probability(reach: np.ndarray) -> float:
    """Total probability of reaching a belief state, over all legal deals.

    The product of the two reach masses minus the impossible diagonal where both
    players hold the same card, with the usual card-removal correction.
    """
    masses = reach.sum(axis=1)
    diagonal = float((reach[0] * reach[1]).sum())
    return float(PAIR_CORRECTION * (masses[0] * masses[1] - diagonal))
