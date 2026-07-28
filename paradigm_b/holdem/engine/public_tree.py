"""The public tree of a hold'em turn endgame.

Same shape as the Leduc public tree — decision, chance, terminal, leaf — with a
board that is a tuple of cards rather than one card, and a chance node that
deals 44 rivers instead of 4 boards.  The node and tree containers are shared;
only the state and the expansion differ.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

from paradigm_b.core.belief.public_tree import CHANCE, DECISION, LEAF, TERMINAL, PublicNode, PublicTree
from paradigm_b.holdem.engine.betting import Betting
from paradigm_b.holdem.engine.combos import NUM_CARDS, cards_to_str


@dataclass(frozen=True)
class PublicState:
    """Common knowledge in a hold'em endgame: the board and the betting."""

    betting: Betting = Betting()
    board: Tuple[int, ...] = ()

    @property
    def is_terminal(self) -> bool:
        return self.betting.is_terminal

    @property
    def awaiting_board(self) -> bool:
        return self.betting.awaiting_board

    @property
    def betting_round(self) -> int:
        return self.betting.betting_round

    @property
    def pot(self) -> int:
        return self.betting.pot

    def to_move(self) -> int:
        return self.betting.to_move()

    def legal_actions(self) -> Tuple[int, ...]:
        return self.betting.legal_actions()

    def apply(self, action: int) -> "PublicState":
        return replace(self, betting=self.betting.apply(action))

    def with_board(self, card: int) -> "PublicState":
        return PublicState(betting=self.betting.deal_board(), board=self.board + (card,))

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{cards_to_str(self.board)} | {self.betting}>"


def build_endgame_tree(
    root: PublicState, depth_limit: Optional[int] = None
) -> PublicTree:
    """Expand the endgame below ``root``; ``depth_limit`` counts betting rounds.

    Works from any street: a three-card root is a flop endgame with two cards
    still to come, a four-card root is the turn endgame.  Only the depth limit
    keeps a flop-rooted tree affordable — expanded in full it would branch 49
    ways at the turn and 48 again at the river.
    """
    tree = PublicTree(root=None, nodes=[], depth_limit=depth_limit, node_of_public={})  # type: ignore[arg-type]
    last_round = None if depth_limit is None else root.betting_round + depth_limit - 1
    tree.root = _expand(root, tree, last_round)
    return tree


def _expand(public: PublicState, tree: PublicTree, last_round: Optional[int]) -> PublicNode:
    index = len(tree.nodes)

    if public.is_terminal:
        node = PublicNode(index=index, kind=TERMINAL, public=public)
        tree.nodes.append(node)
        tree.node_of_public[public] = node
        return node

    if public.awaiting_board:
        beyond_limit = last_round is not None and public.betting_round > last_round
        rivers = tuple(c for c in range(NUM_CARDS) if c not in public.board)
        node = PublicNode(index=index, kind=CHANCE, public=public, boards=rivers)
        tree.nodes.append(node)
        tree.node_of_public[public] = node
        children = []
        for card in rivers:
            child_public = public.with_board(card)
            if child_public.is_terminal:
                child = _expand(child_public, tree, last_round)
            elif beyond_limit:
                child = PublicNode(
                    index=len(tree.nodes), kind=LEAF, public=child_public
                )
                tree.nodes.append(child)
                tree.node_of_public[child_public] = child
            else:
                child = _expand(child_public, tree, last_round)
            children.append(child)
        node.children = tuple(children)
        return node

    actions = public.legal_actions()
    node = PublicNode(
        index=index, kind=DECISION, public=public, player=public.to_move(), actions=actions
    )
    tree.nodes.append(node)
    tree.node_of_public[public] = node
    node.children = tuple(_expand(public.apply(a), tree, last_round) for a in actions)
    return node


# The turn endgame is the two-round case; the name is kept for existing callers.
build_turn_tree = build_endgame_tree
