"""Materialise a :class:`Game` into an explicit tree once, then reuse it.

CFR walks the whole tree on every iteration.  Rebuilding immutable states each
time dominates the cost, so the tree is expanded once into plain nodes with
integer information-set ids and numpy payoff vectors; the solvers then index
arrays instead of hashing keys.  Leduc is ~7k nodes and 288 information sets,
so the whole thing fits comfortably in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np

from game.base import CHANCE, Game, State

TERMINAL = "terminal"
CHANCE_NODE = "chance"
DECISION = "decision"


@dataclass
class Node:
    """A history in the expanded tree."""

    index: int
    kind: str
    player: int = CHANCE
    # Decision nodes.
    infoset: int = -1
    actions: Tuple[int, ...] = ()
    # Chance nodes.
    chance_probs: Optional[np.ndarray] = None
    # Terminal nodes.
    payoffs: Optional[np.ndarray] = None
    children: Tuple["Node", ...] = ()
    state: Optional[State] = None

    @property
    def is_terminal(self) -> bool:
        return self.kind == TERMINAL

    @property
    def is_chance(self) -> bool:
        return self.kind == CHANCE_NODE

    @property
    def is_decision(self) -> bool:
        return self.kind == DECISION


@dataclass
class GameTree:
    """The expanded tree plus its information-set index."""

    game: Game
    root: Node
    nodes: List[Node]
    infoset_keys: List[Hashable] = field(default_factory=list)
    infoset_player: List[int] = field(default_factory=list)
    infoset_actions: List[Tuple[int, ...]] = field(default_factory=list)
    infoset_nodes: List[List[Node]] = field(default_factory=list)
    index_of_key: Dict[Hashable, int] = field(default_factory=dict)

    @property
    def num_infosets(self) -> int:
        return len(self.infoset_keys)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def infosets_of(self, player: int) -> List[int]:
        return [i for i, p in enumerate(self.infoset_player) if p == player]

    def num_actions(self, infoset: int) -> int:
        return len(self.infoset_actions[infoset])


def build_tree(game: Game) -> GameTree:
    """Expand ``game`` depth-first into a :class:`GameTree`."""
    tree = GameTree(game=game, root=None, nodes=[])  # type: ignore[arg-type]
    tree.root = _expand(game, game.new_initial_state(), tree)
    return tree


def _expand(game: Game, state: State, tree: GameTree) -> Node:
    index = len(tree.nodes)

    if state.is_terminal():
        node = Node(
            index=index,
            kind=TERMINAL,
            payoffs=np.asarray(state.returns(), dtype=np.float64),
            state=state,
        )
        tree.nodes.append(node)
        return node

    player = state.current_player()

    if player == CHANCE:
        outcomes = state.chance_outcomes()
        node = Node(
            index=index,
            kind=CHANCE_NODE,
            actions=tuple(a for a, _ in outcomes),
            chance_probs=np.asarray([p for _, p in outcomes], dtype=np.float64),
            state=state,
        )
        tree.nodes.append(node)
        node.children = tuple(_expand(game, state.apply(a), tree) for a, _ in outcomes)
        return node

    key = state.infoset_key()
    actions = tuple(state.legal_actions())
    infoset = tree.index_of_key.get(key)
    if infoset is None:
        infoset = len(tree.infoset_keys)
        tree.index_of_key[key] = infoset
        tree.infoset_keys.append(key)
        tree.infoset_player.append(player)
        tree.infoset_actions.append(actions)
        tree.infoset_nodes.append([])
    elif tree.infoset_actions[infoset] != actions or tree.infoset_player[infoset] != player:
        raise ValueError(
            f"information set {key!r} is inconsistent: it must always belong to "
            "the same player and offer the same actions"
        )

    node = Node(
        index=index,
        kind=DECISION,
        player=player,
        infoset=infoset,
        actions=actions,
        state=state,
    )
    tree.nodes.append(node)
    tree.infoset_nodes[infoset].append(node)
    node.children = tuple(_expand(game, state.apply(a), tree) for a in actions)
    return node
