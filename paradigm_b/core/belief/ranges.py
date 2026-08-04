"""Ranges, public belief states, and Bayesian range propagation.

The representation paradigm A never had.  A *range* is a vector over the six
physical Leduc cards saying how likely a player is to hold each one; a **public
belief state** (PBS) is the common-knowledge pair of ranges together with the
public state.  Every action updates a range by Bayes' rule — a raise multiplies
the range by the raising probability of each hand and renormalises, so betting
narrows a range and checking widens it — and the board card zeroes the hands it
uses up.

Two conventions run through this module and everything built on it:

*Reach vectors* are unnormalised.  ``reach[i][h]`` is the probability that
player ``i`` was dealt ``h`` *and* played to here, so it starts at 1/6 per hand
and only ever shrinks.  Counterfactual values are linear in the opponent's
reach vector, which is what makes vector CFR one matrix product per node.

*Ranges* inside a PBS are normalised per player, because that is what a value
network can be trained on: the pair (my range, your range) is scale-free, and
the reach mass is multiplied back in afterwards.

Card removal is exact.  The two players are dealt distinct cards, so the joint
distribution is *not* the product of the marginals: the product spreads
1/36 over 36 ordered pairs while the truth spreads 1/30 over the 30 distinct
ones.  Hence :data:`PAIR_CORRECTION` = 6/5 wherever an opponent's mass is
summed, and the diagonal of every hand-vs-hand matrix is zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from paradigm_b.core.game.leduc import NUM_CARDS, rank_of, showdown_winner

NUM_HANDS = NUM_CARDS  # in Leduc a hand *is* a single card
NUM_PLAYERS = 2

# The joint deal is uniform over ordered *distinct* pairs (1/30 each) while a
# product of marginals gives 1/36; every opponent-mass sum is scaled to fix it.
PAIR_CORRECTION = NUM_CARDS / (NUM_CARDS - 1)

UNIFORM_RANGE = np.full(NUM_HANDS, 1.0 / NUM_HANDS)


@lru_cache(maxsize=None)
def showdown_matrix(board: int) -> np.ndarray:
    """``W[h, h']`` is +1 if hand ``h`` beats ``h'`` on ``board``, -1, or 0.

    The zero diagonal does double duty: two players cannot hold the same card,
    and two cards of the same rank split the pot — both contribute nothing.
    """
    matrix = np.zeros((NUM_HANDS, NUM_HANDS))
    for h in range(NUM_HANDS):
        for other in range(NUM_HANDS):
            if h == other or h == board or other == board:
                continue
            winner = showdown_winner(h, other, board)
            matrix[h, other] = 0.0 if winner < 0 else (1.0 if winner == 0 else -1.0)
    return matrix


@lru_cache(maxsize=None)
def board_mask(board: int) -> np.ndarray:
    """Hands still possible once ``board`` is on the table."""
    mask = np.ones(NUM_HANDS)
    if board >= 0:
        mask[board] = 0.0
    return mask


def initial_reach() -> np.ndarray:
    """Reach vectors at the root: every hand dealt with probability 1/6."""
    return np.full((NUM_PLAYERS, NUM_HANDS), 1.0 / NUM_HANDS)


def normalize(vector: np.ndarray) -> Tuple[np.ndarray, float]:
    """Split a reach vector into a normalised range and its total mass."""
    mass = float(vector.sum())
    if mass <= 0.0:
        return np.zeros_like(vector), 0.0
    return vector / mass, mass


def opponent_mass_excluding_self(reach: np.ndarray) -> np.ndarray:
    """``PAIR_CORRECTION * (sum(reach) - reach[h])`` for every hand ``h``.

    The mass of opponent hands compatible with holding ``h`` — the weight that
    multiplies a card-independent payoff such as a fold.
    """
    return PAIR_CORRECTION * (reach.sum() - reach)


@dataclass(frozen=True)
class PBS:
    """A public belief state: what is public, plus both normalised ranges.

    This is the "state" the solver reasons about and the value network is
    conditioned on.  Note what is *absent*: anybody's actual cards.
    """

    public: "PublicState"  # noqa: F821 - avoids a circular import
    ranges: np.ndarray  # (2, NUM_HANDS), each row sums to 1

    def __post_init__(self) -> None:
        if self.ranges.ndim != 2 or self.ranges.shape[0] != NUM_PLAYERS:
            raise ValueError("ranges must be (2, num_hands)")

    @staticmethod
    def initial(public: "PublicState") -> "PBS":  # noqa: F821
        return PBS(public=public, ranges=initial_reach())

    @staticmethod
    def from_reach(
        public: "PublicState", reach: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> "PBS":  # noqa: F821
        """Normalise a pair of reach vectors, falling back to uniform.

        A player whose reach is zero everywhere has no belief to speak of --
        the profile never lets them arrive here at all.  Their values are still
        needed (regret is counterfactual: it asks what would have happened had
        they arrived), so the fallback keeps the network input well defined.
        """
        ranges = np.empty_like(np.asarray(reach, dtype=np.float64))
        mask = board_mask(public.board) if mask is None else mask
        for player in range(NUM_PLAYERS):
            normalised, mass = normalize(reach[player])
            if mass <= 0.0:
                normalised = mask / mask.sum()
            ranges[player] = normalised
        return PBS(public=public, ranges=ranges)

    def joint(self) -> np.ndarray:
        """The common-knowledge posterior over ``(hand0, hand1)`` pairs.

        The outer product with its diagonal removed and renormalised: both
        players holding the same physical card is impossible.
        """
        joint = np.outer(self.ranges[0], self.ranges[1])
        np.fill_diagonal(joint, 0.0)
        total = joint.sum()
        return joint / total if total > 0.0 else joint

    def marginal(self, player: int) -> np.ndarray:
        """Player ``player``'s posterior *after* the card-removal coupling.

        Not the same vector as ``ranges[player]``: holding a hand the opponent
        is likely to hold is itself evidence, so the marginal of the joint
        differs from the stored (independent) range.
        """
        joint = self.joint()
        return joint.sum(axis=1) if player == 0 else joint.sum(axis=0)

    def with_ranges(self, ranges: np.ndarray) -> "PBS":
        return PBS(public=self.public, ranges=ranges)


# --- Bayesian propagation --------------------------------------------------
def propagate_action(
    ranges: np.ndarray, player: int, action_probs: np.ndarray
) -> np.ndarray:
    """Update ``player``'s range after they take an action.

    ``action_probs[h]`` is the probability their strategy takes that action
    holding ``h``.  This is Bayes' rule with a uniform prior over what we
    already believed: posterior ∝ prior x likelihood.
    """
    updated = ranges.copy()
    posterior = ranges[player] * action_probs
    total = posterior.sum()
    if total > 0.0:
        updated[player] = posterior / total
    return updated


def propagate_board(ranges: np.ndarray, board: int) -> np.ndarray:
    """Update both ranges when ``board`` is revealed: that card is now used."""
    mask = board_mask(board)
    updated = ranges * mask
    totals = updated.sum(axis=1, keepdims=True)
    safe = np.where(totals > 0.0, totals, 1.0)
    fallback = np.broadcast_to(mask / mask.sum(), updated.shape)
    return np.where(totals > 0.0, updated / safe, fallback)


def rank_probabilities(range_vector: np.ndarray) -> np.ndarray:
    """Collapse a range over cards onto ranks, for readable diagnostics."""
    ranks = np.zeros(3)
    for hand in range(NUM_HANDS):
        ranks[rank_of(hand)] += range_vector[hand]
    return ranks
