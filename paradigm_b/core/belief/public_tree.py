"""The public tree: the game as seen by a spectator who knows no cards.

Where the world tree of stage 0 has one node per (cards, betting) history, the
public tree has one node per *public* history — the betting plus the board.
Private information lives in the range vectors attached to a node rather than
in the node itself, so a single traversal of this tree handles every hand at
once.  That is the whole reason search is affordable: 372 public nodes instead
of 9,457 world nodes, with numpy doing all six hands in one operation.

A tree can be truncated: ``depth_limit`` counts betting rounds, and a truncated
branch ends in a ``LEAF`` node whose value has to come from somewhere else — a
value network, or a recursive solve.  That is the depth-limited subgame ReBeL
and DeepStack search in.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from paradigm_b.core.game.leduc import NUM_CARDS, Betting, card_name

DECISION = "decision"
CHANCE = "chance"
TERMINAL = "terminal"
LEAF = "leaf"


@dataclass(frozen=True)
class PublicState:
    """Common knowledge: the betting so far and the board if it is out."""

    betting: Betting = Betting()
    board: int = -1

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

    def with_board(self, board: int) -> "PublicState":
        return PublicState(betting=self.betting.deal_board(), board=board)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        board = card_name(self.board) if self.board >= 0 else "--"
        return f"<{board} {self.betting} pot={self.pot}>"


@dataclass
class PublicNode:
    index: int
    kind: str
    public: PublicState
    player: int = -1
    actions: Tuple[int, ...] = ()
    children: Tuple["PublicNode", ...] = ()
    # Chance nodes only: ``children[b]`` is the subtree for board card ``b``.
    boards: Tuple[int, ...] = ()

    @property
    def is_decision(self) -> bool:
        return self.kind == DECISION

    @property
    def is_chance(self) -> bool:
        return self.kind == CHANCE

    @property
    def is_terminal(self) -> bool:
        return self.kind == TERMINAL

    @property
    def is_leaf(self) -> bool:
        return self.kind == LEAF

    @property
    def num_actions(self) -> int:
        return len(self.actions)


@dataclass
class PublicTree:
    root: PublicNode
    nodes: List[PublicNode]
    depth_limit: Optional[int]
    node_of_public: Dict[PublicState, PublicNode]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def decision_nodes(self) -> List[PublicNode]:
        return [n for n in self.nodes if n.is_decision]

    def leaves(self) -> List[PublicNode]:
        return [n for n in self.nodes if n.is_leaf]

    def round_starts(self) -> List[PublicNode]:
        """Nodes that begin a betting round below the root.

        The points a continually re-solving agent stops and solves again.  With
        a one-round depth limit these coincide with the leaves; with a deeper
        limit they are strictly more, which is the difference between "solve
        once and follow the plan" and DeepStack's continual re-solving.
        """
        root_round = self.root.public.betting_round
        return [
            node
            for node in self.nodes
            if node is not self.root
            and not node.is_terminal
            and not node.public.awaiting_board
            and node.public.betting_round > root_round
            and not node.public.betting.history[node.public.betting_round]
        ]


def build_public_tree(
    root: PublicState | None = None, depth_limit: Optional[int] = None
) -> PublicTree:
    """Expand the public tree below ``root``.

    ``depth_limit`` is a number of betting rounds: 1 expands the current round
    and stops at the start of the next one (after the board is dealt, so each
    leaf knows its board), ``None`` expands to real terminals.
    """
    root = root if root is not None else PublicState()
    tree = PublicTree(root=None, nodes=[], depth_limit=depth_limit, node_of_public={})  # type: ignore[arg-type]
    last_round = None if depth_limit is None else root.betting_round + depth_limit - 1
    tree.root = _expand(root, tree, last_round)
    return tree


def _expand(
    public: PublicState, tree: PublicTree, last_round: Optional[int]
) -> PublicNode:
    index = len(tree.nodes)

    if public.is_terminal:
        node = PublicNode(index=index, kind=TERMINAL, public=public)
        tree.nodes.append(node)
        tree.node_of_public[public] = node
        return node

    if public.awaiting_board:
        beyond_limit = last_round is not None and public.betting_round > last_round
        kind = CHANCE
        node = PublicNode(
            index=index, kind=kind, public=public, boards=tuple(range(NUM_CARDS))
        )
        tree.nodes.append(node)
        tree.node_of_public[public] = node
        children = []
        for board in node.boards:
            child_public = public.with_board(board)
            if beyond_limit:
                child_index = len(tree.nodes)
                child = PublicNode(index=child_index, kind=LEAF, public=child_public)
                tree.nodes.append(child)
                tree.node_of_public[child_public] = child
            else:
                child = _expand(child_public, tree, last_round)
            children.append(child)
        node.children = tuple(children)
        return node

    actions = public.legal_actions()
    node = PublicNode(
        index=index,
        kind=DECISION,
        public=public,
        player=public.to_move(),
        actions=actions,
    )
    tree.nodes.append(node)
    tree.node_of_public[public] = node
    node.children = tuple(_expand(public.apply(a), tree, last_round) for a in actions)
    return node
