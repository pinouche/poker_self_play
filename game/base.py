"""Extensive-form game interface for the game-theoretic solver (paradigm B).

Paradigm A treats poker as a POMDP and learns a policy from experience.  The
solver needs the opposite view: an explicit **game tree** whose nodes can be
enumerated, whose chance events carry known probabilities, and whose decision
nodes are grouped into **information sets** — the states a player cannot tell
apart.  Regret and best-response reasoning live on that structure.

States are immutable: ``apply`` returns a new state.  That keeps tree walks
free of undo bugs and lets nodes be cached and shared.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Hashable, Sequence, Tuple

# Sentinel returned by ``current_player`` at a chance (dealing) node.
CHANCE = -1


class State(ABC):
    """One history in the game tree."""

    @abstractmethod
    def is_terminal(self) -> bool: ...

    @abstractmethod
    def current_player(self) -> int:
        """Seat to act, or ``CHANCE``.  Undefined at terminal states."""

    @abstractmethod
    def legal_actions(self) -> Sequence[int]:
        """Actions available to ``current_player`` (decision nodes only)."""

    @abstractmethod
    def chance_outcomes(self) -> Sequence[Tuple[int, float]]:
        """``(outcome, probability)`` pairs (chance nodes only)."""

    @abstractmethod
    def apply(self, action: int) -> "State":
        """The state reached by taking ``action``.  ``self`` is unchanged."""

    @abstractmethod
    def returns(self) -> Sequence[float]:
        """Per-seat payoff in chips (terminal states only)."""

    @abstractmethod
    def infoset_key(self) -> Hashable:
        """Identifier of the acting player's information set.

        Two histories share a key exactly when the acting player cannot
        distinguish them.  The key must determine the legal action list.
        """


class Game(ABC):
    """A game definition: the root, the seat count, and the action alphabet."""

    num_players: int = 2
    # Upper bound on ``len(legal_actions())`` anywhere in the tree.
    max_actions: int = 3

    @abstractmethod
    def new_initial_state(self) -> State: ...

    def action_name(self, action: int) -> str:
        return str(action)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return type(self).__name__
