"""Public belief states and range-form CFR (paradigm B, stage 1).

The claim being tested is that reasoning over *ranges* on the public tree is
not an approximation of reasoning over the world tree — it is the same thing,
reorganised.  So every number here is checked against the stage-0 machinery or
against brute-force enumeration of all 30 deals:

* counterfactual values from one range traversal == per-hand expected values
  from the world tree,
* Bayes-propagated beliefs == posteriors enumerated over deals, card removal
  included,
* the extracted policy is a Nash equilibrium by the exact best-response metric.
"""

from __future__ import annotations

import numpy as np
import pytest

from belief import (
    PBS,
    PublicState,
    build_public_tree,
    initial_reach,
    propagate_action,
    propagate_board,
    showdown_matrix,
)
from belief.ranges import PAIR_CORRECTION, board_mask, rank_probabilities
from cfr import CFRConfig, exploitability, expected_values, expected_values_at
from game import LeducHoldem, build_tree
from game.actions import CALL, FOLD, RAISE
from game.leduc import NUM_CARDS
from search import (
    SubgameSolver,
    extract_tabular_policy,
    suit_symmetry_error,
    terminal_values,
    world_infoset_key,
)

LEDUC_VALUE = -0.085606424078
SOLVE_ITERATIONS = 300


@pytest.fixture(scope="module")
def world_tree():
    return build_tree(LeducHoldem())


@pytest.fixture(scope="module")
def solved():
    """A full-game range-form solve, shared by the tests below."""
    solver = SubgameSolver(build_public_tree(), config=CFRConfig.dcfr())
    solver.solve(iterations=SOLVE_ITERATIONS)
    return solver


@pytest.fixture(scope="module")
def solved_policy(solved, world_tree):
    return extract_tabular_policy(solved, world_tree)


# --- the public tree -------------------------------------------------------
def test_public_tree_is_much_smaller_than_the_world_tree(world_tree):
    tree = build_public_tree()
    assert tree.num_nodes == 465
    assert len(tree.decision_nodes()) == 186
    assert tree.leaves() == [], "a full tree ends in real terminals"
    assert tree.num_nodes < world_tree.num_nodes / 20


def test_depth_limited_tree_stops_at_the_next_round():
    tree = build_public_tree(depth_limit=1)
    assert tree.leaves(), "round one closing must produce leaf belief states"
    for leaf in tree.leaves():
        assert leaf.public.board >= 0, "a leaf knows its board card"
        assert leaf.public.betting_round == 1
    # Five ways for round one to close, times six board cards.
    assert len(tree.leaves()) == 30
    for node in tree.nodes:
        assert node.public.betting_round == 0 or node.is_leaf or node.is_chance


def test_public_states_do_not_mention_cards():
    public = PublicState().apply(RAISE).apply(CALL)
    assert public.awaiting_board
    assert public.board == -1
    assert public.pot == 6
    with_board = public.with_board(3)
    assert with_board.board == 3
    assert not with_board.awaiting_board
    assert with_board.to_move() == 0


# --- range propagation ----------------------------------------------------
def test_a_raise_narrows_a_range_by_bayes_rule():
    ranges = initial_reach()
    # Bets kings, folds jacks, and checks queens: cards 4,5 are kings.
    raise_probs = np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])
    updated = propagate_action(ranges, 0, raise_probs)
    assert updated[0].sum() == pytest.approx(1.0)
    ranks = rank_probabilities(updated[0])
    assert ranks[0] == pytest.approx(0.0), "jacks are gone from the range"
    assert ranks[2] == pytest.approx(2 / 3), "kings dominate what is left"
    assert np.allclose(updated[1], ranges[1]), "the other range is untouched"


def test_a_check_widens_the_complementary_range():
    ranges = initial_reach()
    check_probs = np.array([1.0, 1.0, 0.5, 0.5, 0.0, 0.0])
    updated = propagate_action(ranges, 1, check_probs)
    ranks = rank_probabilities(updated[1])
    assert ranks[0] > ranks[1] > ranks[2]
    assert ranks[2] == pytest.approx(0.0)


def test_the_board_card_is_removed_from_both_ranges():
    updated = propagate_board(initial_reach(), board=3)
    assert updated[0][3] == 0.0 and updated[1][3] == 0.0
    assert updated[0].sum() == pytest.approx(1.0)
    assert updated[0][0] == pytest.approx(0.2), "mass spreads over five cards"


def test_a_zero_reach_player_falls_back_to_a_uniform_belief():
    """Counterfactual reasoning still needs a belief for hands never played."""
    public = PublicState().with_board(2)
    reach = initial_reach()
    reach[1] = 0.0
    pbs = PBS.from_reach(public, reach)
    assert pbs.ranges[1].sum() == pytest.approx(1.0)
    assert pbs.ranges[1][2] == 0.0, "the board card stays impossible"


def test_joint_belief_excludes_both_players_holding_one_card():
    pbs = PBS(public=PublicState(), ranges=initial_reach())
    joint = pbs.joint()
    assert joint.sum() == pytest.approx(1.0)
    assert np.allclose(np.diag(joint), 0.0)
    assert joint[0, 1] == pytest.approx(1 / 30), "30 equally likely ordered deals"


def test_card_removal_makes_the_marginal_differ_from_the_stored_range():
    ranges = initial_reach()
    ranges[1] = np.array([0.9, 0.0, 0.02, 0.02, 0.03, 0.03])
    pbs = PBS(public=PublicState(), ranges=ranges)
    marginal = pbs.marginal(0)
    assert marginal.sum() == pytest.approx(1.0)
    assert marginal[0] < marginal[1], (
        "holding the card the opponent probably has is itself unlikely"
    )


# --- values agree with the world tree -------------------------------------
def deal_reach(policy, world_tree, public: PublicState) -> np.ndarray:
    """Brute-force reach probability of ``public`` for every deal.

    Independent of the range code: it walks the world tree's information sets
    directly, one deal at a time, multiplying action probabilities.
    """
    boards = [public.board] if public.board >= 0 else [-1]
    reach = np.zeros((NUM_CARDS, NUM_CARDS))
    for hand0 in range(NUM_CARDS):
        for hand1 in range(NUM_CARDS):
            if hand0 == hand1:
                continue
            for board in boards:
                if board in (hand0, hand1):
                    continue
                probability = 1.0 / 30.0
                if public.board >= 0:
                    probability *= 1.0 / 4.0
                state = PublicState()
                for rnd, actions in enumerate(public.betting.history):
                    if rnd == 1 and state.awaiting_board:
                        state = state.with_board(board)
                    for action in actions:
                        player = state.to_move()
                        hand = hand0 if player == 0 else hand1
                        key = world_infoset_key(state, hand)
                        infoset = world_tree.index_of_key[key]
                        index = world_tree.infoset_actions[infoset].index(action)
                        probability *= policy.probs[infoset, index]
                        state = state.apply(action)
                reach[hand0, hand1] += probability
    return reach


def test_root_counterfactual_values_match_per_hand_world_tree_values(
    solved, solved_policy, world_tree
):
    """cfv_0[h] is the expected payoff to player 0 given they hold h."""
    values = solved.evaluate()
    for hand in range(NUM_CARDS):
        deal_node = world_tree.root.children[hand]
        expected = expected_values_at(deal_node, solved_policy)[0]
        assert values[0, hand] == pytest.approx(expected, abs=1e-12)


def test_range_values_average_to_the_game_value(solved, solved_policy, world_tree):
    values = solved.evaluate()
    game_value = expected_values(world_tree, solved_policy)[0]
    assert values[0].mean() == pytest.approx(game_value, abs=1e-12)
    assert values[1].mean() == pytest.approx(-game_value, abs=1e-12)


def test_beliefs_match_brute_force_posteriors(solved, solved_policy, world_tree):
    """Bayes propagation through actions and the board, checked by enumeration."""
    reach = initial_reach()
    public = PublicState()
    tested = 0
    # A raise and a call close round one; the board then lands, and betting
    # resumes -- so this walks a posterior through both kinds of update.
    for action in (RAISE, CALL, None, RAISE, CALL):
        if action is None:
            board = 2
            reach = reach * board_mask(board)
            public = public.with_board(board)
        else:
            node = solved.tree.node_of_public[public]
            strategy = solved.average_strategy(node)
            index = node.actions.index(action)
            player = node.player
            reach = reach.copy()
            reach[player] = reach[player] * strategy[:, index]
            public = public.apply(action)

        deals = deal_reach(solved_policy, world_tree, public)
        expected = deals.sum(axis=1) / deals.sum()  # posterior over player 0's card
        pbs = PBS.from_reach(public, reach)
        assert np.allclose(pbs.marginal(0), expected, atol=1e-12), public
        tested += 1
    assert tested == 5


def test_normalised_ranges_are_the_reach_vectors_rescaled():
    reach = initial_reach()
    reach[0] = np.array([0.1, 0.1, 0.02, 0.02, 0.0, 0.0])
    pbs = PBS.from_reach(PublicState(), reach)
    assert np.allclose(pbs.ranges[0], reach[0] / reach[0].sum())


# --- terminal values -------------------------------------------------------
def test_fold_values_are_card_independent():
    public = PublicState().apply(RAISE).apply(FOLD)
    reach = initial_reach()
    values = terminal_values(public, reach)
    stake = 1.0  # player 1 folded having posted only the ante
    mass = PAIR_CORRECTION * (1.0 - 1.0 / NUM_CARDS)
    assert np.allclose(values[0], stake * mass)
    assert np.allclose(values[1], -stake * mass)


def test_showdown_values_follow_the_win_matrix():
    public = PublicState().apply(CALL).apply(CALL).with_board(0)
    public = public.apply(CALL).apply(CALL)
    assert public.is_terminal
    reach = initial_reach() * board_mask(0)
    values = terminal_values(public, reach)
    matrix = showdown_matrix(0)
    expected = PAIR_CORRECTION * 1.0 * (matrix @ reach[1])
    assert np.allclose(values[0], expected)
    # The other jack pairs the board on a jack, so it beats everything.
    assert values[0, 1] == max(values[0])
    assert values[0].sum() + values[1].sum() == pytest.approx(0.0)


def test_showdown_matrix_is_antisymmetric_with_a_zero_diagonal():
    for board in range(NUM_CARDS):
        matrix = showdown_matrix(board)
        assert np.allclose(matrix, -matrix.T)
        assert np.allclose(np.diag(matrix), 0.0)
        assert np.allclose(matrix[board], 0.0), "nobody holds the board card"


# --- the stage-1 milestone -------------------------------------------------
def test_range_form_cfr_solves_leduc(solved, solved_policy, world_tree):
    """Range CFR reaches equilibrium, measured by exact best response."""
    assert exploitability(world_tree, solved_policy) < 6e-3
    assert expected_values(world_tree, solved_policy)[0] == pytest.approx(
        LEDUC_VALUE, abs=1e-3
    )


def test_solution_is_exactly_suit_symmetric(solved):
    assert suit_symmetry_error(solved) == 0.0
