"""Tabular policies over a :class:`GameTree`.

A policy is a dense ``(num_infosets, max_actions)`` matrix; row ``i`` holds the
distribution over ``tree.infoset_actions[i]`` and is zero past that infoset's
action count.  Dense rows keep regret matching and the tree walks branch-free,
and the ragged truth is recovered by slicing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable

import numpy as np

from game.tree import GameTree


@dataclass
class TabularPolicy:
    tree: GameTree
    probs: np.ndarray  # (num_infosets, max_actions)

    @staticmethod
    def uniform(tree: GameTree) -> "TabularPolicy":
        probs = np.zeros((tree.num_infosets, tree.game.max_actions))
        for i in range(tree.num_infosets):
            n = tree.num_actions(i)
            probs[i, :n] = 1.0 / n
        return TabularPolicy(tree=tree, probs=probs)

    def action_probs(self, infoset: int) -> np.ndarray:
        return self.probs[infoset, : self.tree.num_actions(infoset)]

    def copy(self) -> "TabularPolicy":
        return TabularPolicy(tree=self.tree, probs=self.probs.copy())

    def to_dict(self) -> Dict[Hashable, Dict[str, float]]:
        """Human-readable ``{infoset key: {action name: probability}}``."""
        game = self.tree.game
        out: Dict[Hashable, Dict[str, float]] = {}
        for i, key in enumerate(self.tree.infoset_keys):
            actions = self.tree.infoset_actions[i]
            out[key] = {
                game.action_name(a): float(self.probs[i, j])
                for j, a in enumerate(actions)
            }
        return out

    def check_valid(self, tol: float = 1e-9) -> None:
        """Raise if any row is not a distribution over its legal actions."""
        for i in range(self.tree.num_infosets):
            n = self.tree.num_actions(i)
            row = self.probs[i]
            if row[n:].any():
                raise ValueError(f"infoset {i} puts mass on illegal actions")
            if abs(row[:n].sum() - 1.0) > tol or (row[:n] < -tol).any():
                raise ValueError(f"infoset {i} is not a probability distribution")
