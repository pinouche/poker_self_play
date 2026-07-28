"""The toy games and their expanded trees (paradigm B, stage 0)."""

from __future__ import annotations

import pytest

from paradigm_b.core.game import KuhnPoker, LeducHoldem, build_tree
from paradigm_b.core.game.actions import CALL, FOLD, RAISE
from paradigm_b.core.game.base import CHANCE
from paradigm_b.core.game.leduc import Betting, LeducState, showdown_winner


# --- structure -------------------------------------------------------------
def test_kuhn_tree_has_twelve_infosets():
    tree = build_tree(KuhnPoker())
    assert tree.num_infosets == 12  # 3 cards x 4 betting histories
    assert len(tree.infosets_of(0)) == 6
    assert len(tree.infosets_of(1)) == 6


def test_leduc_tree_has_288_infosets():
    """The canonical count for Leduc with suits collapsed."""
    tree = build_tree(LeducHoldem())
    assert tree.num_infosets == 288
    assert len(tree.infosets_of(0)) == 144
    assert len(tree.infosets_of(1)) == 144


@pytest.mark.parametrize("game", [KuhnPoker(), LeducHoldem()])
def test_tree_is_well_formed(game):
    tree = build_tree(game)
    for node in tree.nodes:
        if node.is_terminal:
            assert node.payoffs is not None
            assert node.payoffs.sum() == pytest.approx(0.0), "poker is zero-sum"
            assert node.children == ()
        elif node.is_chance:
            assert node.chance_probs.sum() == pytest.approx(1.0)
            assert len(node.children) == len(node.chance_probs)
        else:
            assert len(node.children) == len(node.actions) >= 2
            assert 0 <= node.player < game.num_players
            assert node.infoset >= 0


@pytest.mark.parametrize("game", [KuhnPoker(), LeducHoldem()])
def test_infoset_grouping_is_consistent(game):
    """Every history in an information set agrees on player and actions."""
    tree = build_tree(game)
    for infoset, nodes in enumerate(tree.infoset_nodes):
        assert nodes, "no empty information sets"
        players = {n.player for n in nodes}
        actions = {n.actions for n in nodes}
        assert players == {tree.infoset_player[infoset]}
        assert actions == {tree.infoset_actions[infoset]}


# --- Leduc rules -----------------------------------------------------------
def test_leduc_deals_two_private_cards_then_a_board():
    state = LeducHoldem().new_initial_state()
    assert state.current_player() == CHANCE
    assert len(state.chance_outcomes()) == 6
    state = state.apply(0)
    assert len(state.chance_outcomes()) == 5, "card removal"
    state = state.apply(2)
    assert state.current_player() == 0, "player 0 acts first"

    state = state.apply(CALL).apply(CALL)  # check-check ends round one
    assert state.betting_round == 1
    assert state.current_player() == CHANCE
    assert len(state.chance_outcomes()) == 4, "board comes from the same deck"


def test_leduc_betting_is_capped_at_two_raises_per_round():
    state = LeducHoldem().new_initial_state().apply(0).apply(2)
    state = state.apply(RAISE).apply(RAISE)
    assert RAISE not in state.legal_actions(), "two raises is the cap"
    assert set(state.legal_actions()) == {FOLD, CALL}
    state = state.apply(CALL)
    assert state.betting_round == 1
    assert state.contributions == (5, 5), "ante 1 + bet 2 + raise 2"


def test_leduc_raise_sizes_are_two_then_four():
    state = LeducHoldem().new_initial_state().apply(0).apply(2)
    state = state.apply(RAISE)
    assert state.contributions == (3, 1)
    state = state.apply(CALL).apply(4)  # close round one, deal the board
    state = state.apply(RAISE)
    assert state.contributions == (7, 3), "round two raises cost 4"


def test_leduc_cannot_fold_when_nothing_to_call():
    state = LeducHoldem().new_initial_state().apply(0).apply(2)
    assert FOLD not in state.legal_actions()


def test_leduc_fold_pays_the_caller_what_the_folder_put_in():
    state = LeducHoldem().new_initial_state().apply(0).apply(2)
    state = state.apply(RAISE).apply(FOLD)
    assert state.is_terminal()
    assert list(state.returns()) == [1.0, -1.0], "player 1 loses only the ante"


def test_leduc_showdown_pairs_beat_high_cards():
    # Cards are ids: rank = id // 2, so 0,1 are jacks; 2,3 queens; 4,5 kings.
    assert showdown_winner(0, 4, 1) == 0, "a paired jack beats a king"
    assert showdown_winner(0, 4, 3) == 1, "otherwise the king wins"
    assert showdown_winner(0, 1, 2) == -1, "same rank splits"


def test_leduc_showdown_payout_matches_contributions():
    betting = Betting(betting_round=1, contributions=(5, 5))
    state = LeducState(cards=(4, 0), board=1, betting=betting)
    state = state.apply(CALL).apply(CALL)
    assert state.is_terminal()
    # Player 1's jack pairs the board and beats the king.
    assert list(state.returns()) == [-5.0, 5.0]


def test_leduc_reaches_every_showdown_and_fold():
    """Terminal payoffs are symmetric across the whole tree."""
    tree = build_tree(LeducHoldem())
    terminals = [n for n in tree.nodes if n.is_terminal]
    assert len(terminals) > 1000
    stakes = {abs(n.payoffs[0]) for n in terminals}
    # A player's contribution is 1, 3 or 5 after round one and 4 or 8 more after
    # round two, so those (plus 0 for a split) are the only possible stakes.
    assert stakes == {0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0}


def test_kuhn_payoffs():
    game = KuhnPoker()
    state = game.new_initial_state().apply(2).apply(0)  # K vs J
    assert list(state.apply(CALL).apply(CALL).returns()) == [1.0, -1.0]
    assert list(state.apply(RAISE).apply(FOLD).returns()) == [1.0, -1.0]
    assert list(state.apply(RAISE).apply(CALL).returns()) == [2.0, -2.0]
    state = game.new_initial_state().apply(0).apply(2)  # J vs K
    assert list(state.apply(CALL).apply(RAISE).apply(FOLD).returns()) == [-1.0, 1.0]


def test_states_are_immutable():
    state = LeducHoldem().new_initial_state().apply(0).apply(2)
    before = state.contributions
    state.apply(RAISE)
    assert state.contributions == before
    with pytest.raises(Exception):
        state.contributions = (99, 99)  # frozen dataclass
