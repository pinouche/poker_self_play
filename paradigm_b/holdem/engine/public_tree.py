"""The public tree of a hold'em hand, or of any endgame inside it.

Same shape as the Leduc public tree — decision, chance, terminal, leaf — with a
board that is a tuple of cards rather than one card, and a chance node that
deals 44 rivers instead of 4 boards.  The node and tree containers are shared;
only the state and the expansion differ.

**Where the depth limit falls.**  A truncated branch ends at the *end* of its
betting round, before the board is dealt — ReBeL's subgame boundary, and the
reason its value network has to learn six layers of values (the end of each
round as well as the start of the next) where DeepStack learned three.  The
alternative, expanding the chance node and truncating after the deal, is what
this tree used to do; it is exact but it costs one network evaluation per board
per frontier line, and preflop it is not available at all, because the flop deal
has C(50,3) = 19,600 outcomes against the river's 44.  Since a value network is
queried at the truncation point and trained at subgame roots, moving the
boundary means both layers now need training labels — see
``holdem/selfplay.py``, which emits them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

from paradigm_b.core.belief.public_tree import CHANCE, DECISION, LEAF, TERMINAL, PublicNode, PublicTree
from paradigm_b.holdem.engine.betting import Betting
from paradigm_b.holdem.engine.combos import NUM_CARDS, cards_to_str

FLOP_CARDS = 3


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
        return self.with_cards((card,))

    def with_cards(self, cards: Tuple[int, ...]) -> "PublicState":
        """Reveal a whole chance outcome at once — one card, or a three-card flop."""
        return PublicState(
            betting=self.betting.deal_board(), board=self.board + tuple(cards)
        )

    def undealt_cards(self) -> Tuple[int, ...]:
        """Cards the deck can still turn over, ignoring anyone's hole cards."""
        return tuple(c for c in range(NUM_CARDS) if c not in self.board)

    @property
    def cards_to_deal(self) -> int:
        """How many cards the next chance outcome turns over.

        Three preflop and one thereafter, which is the whole reason a preflop
        subgame cannot enumerate its chance node: C(50,3) is 19,600 boards
        against 48 for the turn.
        """
        return FLOP_CARDS if not self.board else 1

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
        if last_round is not None and public.betting_round > last_round:
            # The depth limit, and it falls here rather than past the deal: the
            # betting of this round is finished, the board is not yet out, and
            # what this belief state is worth is exactly what the value network
            # is for.  Stopping in front of the chance node instead of behind
            # it is what makes a preflop subgame possible.
            node = PublicNode(index=index, kind=LEAF, public=public)
            tree.nodes.append(node)
            tree.node_of_public[public] = node
            return node

        if public.cards_to_deal != 1:
            raise ValueError(
                f"expanding this chance node enumerates a {public.cards_to_deal}"
                "-card deal (19,600 flops); a subgame that reaches the flop deal "
                "must stop in front of it, which is what depth_limit=1 does"
            )
        cards = tuple(c for c in range(NUM_CARDS) if c not in public.board)
        node = PublicNode(index=index, kind=CHANCE, public=public, boards=cards)
        tree.nodes.append(node)
        tree.node_of_public[public] = node
        node.children = tuple(
            _expand(public.with_board(card), tree, last_round) for card in cards
        )
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
