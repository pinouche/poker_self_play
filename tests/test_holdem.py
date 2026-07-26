"""Hold'em turn endgames (paradigm B, stage 4).

Real cards this time: 1,326 two-card combinations, exact card removal, and a
no-limit betting abstraction.  The things worth testing are the ones Leduc never
exercised — blocking arithmetic, the linear-time showdown, and whether the
solver written for six one-card hands really does work unchanged on this.
"""

from __future__ import annotations

import numpy as np
import pytest

from cfr import CFRConfig
from holdem.betting import ALL_IN, BET_POT, CALL, FOLD, Betting
from holdem.combos import (
    COMBO_CARDS,
    NUM_COMBOS,
    board_mask,
    combo_index,
    compatible_mass,
    num_legal_combos,
    pair_correction,
)
from holdem.public_tree import PublicState, build_turn_tree
from holdem.showdown import showdown_values, showdown_values_brute_force
from holdem.space import TurnEndgameSpace
from holdem.strength import hand_ranks
from search import SubgameSolver, strategy_map
from search.best_response import subgame_exploitability

TURN_BOARD = (51, 47, 22, 6)  # As Ks 7h 3h
RIVER_BOARD = TURN_BOARD + (34,)


def endgame_root(max_raises: int = 1) -> PublicState:
    return PublicState(
        betting=Betting(starting_pot=20, stack=100, max_raises=max_raises),
        board=TURN_BOARD,
    )


# --- combinations and blocking ---------------------------------------------
def test_combo_counts():
    assert NUM_COMBOS == 1326
    assert int(board_mask(TURN_BOARD).sum()) == num_legal_combos(4) == 1128
    assert int(board_mask(RIVER_BOARD).sum()) == num_legal_combos(5) == 1081


def test_a_hand_blocks_every_combination_sharing_a_card():
    reach = board_mask(TURN_BOARD).copy()
    mine = combo_index(0, 1)
    blocked = [
        i for i in range(NUM_COMBOS) if set(COMBO_CARDS[i]) & {0, 1} and reach[i] > 0
    ]
    assert len(blocked) == 2 * 46 + 1, "each card blocks 46 others, plus my own hand"
    assert compatible_mass(reach)[mine] == pytest.approx(reach.sum() - len(blocked))


def test_pair_correction_matches_the_true_deal():
    """The constant that turns independent ranges into a legal joint deal."""
    free = 52 - 4
    assert pair_correction(4) == pytest.approx(
        (free * (free - 1) / 2) / ((free - 2) * (free - 3) / 2)
    )


# --- hand strength and showdowns -------------------------------------------
def test_strength_ordering_is_real_poker():
    ranks = hand_ranks(RIVER_BOARD)  # As Ks 7h 3h Th
    nut_flush = combo_index(50, 46)  # Ah Kh
    trip_sevens = combo_index(23, 21)  # 7s 7d
    ace_high = combo_index(49, 2)  # Ad 2c
    assert ranks[nut_flush] > ranks[trip_sevens] > ranks[ace_high]
    assert ranks[combo_index(TURN_BOARD[0], 5)] == -1, "board cards cannot be held"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_showdown_matches_the_definition(seed):
    """The linear-time version against the O(n^2) one it replaces."""
    rng = np.random.default_rng(seed)
    reach = rng.random(NUM_COMBOS) * board_mask(RIVER_BOARD)
    if seed == 1:
        reach *= rng.random(NUM_COMBOS) > 0.6  # a sharp, sparse range
    fast = showdown_values(RIVER_BOARD, reach, 12.0)
    slow = showdown_values_brute_force(RIVER_BOARD, reach, 12.0)
    assert np.abs(fast - slow).max() < 1e-9


def test_showdown_is_zero_sum_over_a_shared_range():
    reach = board_mask(RIVER_BOARD) / board_mask(RIVER_BOARD).sum()
    values = showdown_values(RIVER_BOARD, reach, 1.0)
    assert float(reach @ values) == pytest.approx(0.0, abs=1e-12)


# --- no-limit betting ------------------------------------------------------
def test_betting_sizes_and_pot():
    betting = Betting(starting_pot=20, stack=100, max_raises=1)
    assert FOLD not in betting.legal_actions(), "nothing to fold to"
    after_bet = betting.apply(BET_POT)
    assert after_bet.contributions == (20, 0) and after_bet.pot == 40
    assert FOLD in after_bet.legal_actions()
    closed = after_bet.apply(CALL)
    assert closed.betting_round == 1 and closed.awaiting_board
    assert closed.pot == 60


def test_an_all_in_that_is_called_skips_the_last_betting_round():
    betting = Betting(starting_pot=20, stack=100, max_raises=1)
    both_in = betting.apply(ALL_IN).apply(CALL)
    assert both_in.all_in and both_in.awaiting_board
    dealt = both_in.deal_board()
    assert dealt.showdown, "no one has chips left to bet with"
    assert dealt.showdown_stake() == pytest.approx(110.0)


def test_folding_costs_what_you_put_in_plus_your_half_of_the_pot():
    betting = Betting(starting_pot=20, stack=100, max_raises=1)
    folded = betting.apply(BET_POT).apply(FOLD)
    assert list(folded.fold_returns()) == [10.0, -10.0]


# --- the tree ---------------------------------------------------------------
def test_turn_tree_shape():
    tree = build_turn_tree(endgame_root(), depth_limit=1)
    endings = {leaf.public.betting.history for leaf in tree.leaves()}
    assert len(tree.leaves()) == 48 * len(endings), "one leaf per river per line"
    assert all(len(leaf.public.board) == 5 for leaf in tree.leaves())
    assert all(node.public.betting_round == 0 for node in tree.decision_nodes())

    full = build_turn_tree(endgame_root(), depth_limit=None)
    assert full.leaves() == []
    assert len(full.decision_nodes()) > 1000
    assert {n.public.betting_round for n in full.decision_nodes()} == {0, 1}


def test_every_river_card_is_dealt_once():
    tree = build_turn_tree(endgame_root(), depth_limit=1)
    chance = [n for n in tree.nodes if n.is_chance][0]
    assert len(chance.boards) == 48, "52 cards less the four on the turn"
    assert len(set(chance.boards)) == 48


# --- the shared solver, on hold'em -----------------------------------------
def test_terminal_values_are_zero_sum():
    space = TurnEndgameSpace(TURN_BOARD)
    reach = space.initial_reach()
    tree = build_turn_tree(endgame_root(), depth_limit=None)
    terminals = [n for n in tree.nodes if n.is_terminal]
    for node in terminals[:20]:
        values = space.terminal_values(node.public, reach)
        assert (reach * values).sum() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.slow
def test_cfr_reduces_endgame_exploitability():
    """The stage-4 milestone: the Leduc solver, unchanged, solves real hold'em."""
    space = TurnEndgameSpace(TURN_BOARD)
    root = endgame_root()
    tree = build_turn_tree(root, depth_limit=None)
    solver = SubgameSolver(tree, config=CFRConfig.dcfr(), space=space)
    reach = space.initial_reach()

    solver.solve(reach=reach, iterations=10)
    early, _ = subgame_exploitability(tree, strategy_map(solver), reach, space=space)
    solver.solve(reach=reach, iterations=40)
    later, _ = subgame_exploitability(tree, strategy_map(solver), reach, space=space)

    assert later < early / 3.0, f"{later} vs {early}"
    assert later < 1.5, "well under a tenth of the starting pot"


# --- stage C: bet abstraction and translation ------------------------------
def test_bet_abstraction_is_configurable():
    """More sizes means more actions, without touching the action alphabet."""
    coarse = Betting(starting_pot=20, stack=200, max_raises=1, bet_fractions=(0.5, 1.0))
    fine = Betting(
        starting_pot=20, stack=200, max_raises=1, bet_fractions=(0.33, 0.5, 0.75, 1.0, 1.5)
    )
    assert len(fine.legal_actions()) > len(coarse.legal_actions())
    assert coarse.all_in_action == 4 and fine.all_in_action == 7
    # Every advertised size is really that size.
    for action in fine.bet_action_ids():
        total = fine.raise_total(0, action)
        assert total == pytest.approx(round(fine.fraction_of(action) * fine.pot), abs=1)


def test_translation_is_exact_at_the_abstraction_sizes():
    from holdem.translation import bet_sizes_in_pots, translate_bet

    betting = Betting(starting_pot=20, stack=200, max_raises=1)
    for action, fraction in bet_sizes_in_pots(betting):
        amount = betting.raise_total(0, action)
        assert translate_bet(betting, amount).get(action, 0.0) == pytest.approx(1.0)


def test_translation_splits_off_tree_bets_and_stays_monotone():
    from holdem.translation import translate_bet

    betting = Betting(starting_pot=20, stack=200, max_raises=1)  # sizes: 10, 20, all-in
    split = translate_bet(betting, 15)
    assert len(split) == 2 and sum(split.values()) == pytest.approx(1.0)
    # As the real bet grows, weight moves off the small size and never back.
    weights = [translate_bet(betting, amount).get(2, 0.0) for amount in range(10, 21)]
    assert all(a >= b for a, b in zip(weights, weights[1:]))
    assert weights[0] == pytest.approx(1.0) and weights[-1] == pytest.approx(0.0)


def test_translation_leans_to_the_larger_size():
    """The property that stops 'bet just over the boundary' being free.

    A linear rule would map the midpoint 50/50; the pseudo-harmonic rule sends
    it to the larger size more often, so shading a bet just above an abstraction
    size does not get it treated as that size.
    """
    from holdem.translation import pseudo_harmonic_weight

    assert pseudo_harmonic_weight(0.5, 1.0, 0.75) < 0.5


def test_translation_is_randomised_but_reproducible():
    from holdem.translation import translate_action

    betting = Betting(starting_pot=20, stack=200, max_raises=1)
    rng = np.random.default_rng(0)
    picks = {translate_action(betting, 15, rng) for _ in range(40)}
    assert len(picks) == 2, "an off-tree bet must not always map the same way"


# --- stage A: randomised situations ----------------------------------------
def test_sampled_situations_differ():
    """The defect this fixes: every trajectory once began from the same state."""
    from holdem.features import encode_pbs
    from holdem.sampling import sample_situation

    rng = np.random.default_rng(0)
    encoded = set()
    for _ in range(12):
        space, root, reach = sample_situation(rng)
        encoded.add(encode_pbs(space.pbs(root, reach)).tobytes())
    assert len(encoded) == 12, "sampled situations must all be distinct"


def test_sampled_ranges_are_valid_and_varied():
    from holdem.sampling import RANGE_STYLES, sample_board, sample_range

    rng = np.random.default_rng(1)
    board = sample_board(rng, 4)
    mask = board_mask(board)
    for style in RANGE_STYLES:
        weights = sample_range(rng, board, style)
        assert weights.sum() == pytest.approx(1.0)
        assert (weights * (1 - mask)).sum() == 0.0, "no mass on blocked combos"
        assert (weights >= 0).all()


def test_held_out_boards_are_excluded_from_training_draws():
    from holdem.sampling import SituationConfig, held_out_boards, sample_situation

    rng = np.random.default_rng(2)
    reserved = held_out_boards(rng, 4)
    config = SituationConfig(excluded_boards=reserved)
    drawn = {tuple(sorted(sample_situation(rng, config)[1].board)) for _ in range(30)}
    assert not (drawn & {tuple(sorted(b)) for b in reserved})


# --- stage B: three streets and suit isomorphism ---------------------------
def flop_root(flop=(51, 47, 22)) -> PublicState:
    return PublicState(
        betting=Betting(starting_pot=20, stack=100, max_raises=1, num_rounds=3),
        board=flop,
    )


def test_a_flop_rooted_tree_reaches_the_river_one_street_at_a_time():
    """Depth-limited search chains: flop -> turn belief state -> river."""
    from holdem.public_tree import build_endgame_tree

    flop = build_endgame_tree(flop_root(), depth_limit=1)
    assert len(flop.leaves()) == 49 * len(
        {leaf.public.betting.history for leaf in flop.leaves()}
    )
    assert all(len(leaf.public.board) == 4 for leaf in flop.leaves())

    turn = build_endgame_tree(flop.leaves()[0].public, depth_limit=1)
    assert all(len(leaf.public.board) == 5 for leaf in turn.leaves())

    river = build_endgame_tree(turn.leaves()[0].public, depth_limit=1)
    assert river.leaves() == [], "the river runs to real showdowns"
    assert any(node.is_terminal for node in river.nodes)


def test_chance_weights_match_the_cards_actually_left():
    from holdem.public_tree import build_endgame_tree

    space = TurnEndgameSpace((51, 47, 22))
    root = flop_root()
    # 52 less three board cards and four hole cards.
    assert space.chance_weight(root) == pytest.approx(1 / 45)
    turn = build_endgame_tree(root, depth_limit=1).leaves()[0].public
    assert space.chance_weight(turn) == pytest.approx(1 / 44)


def test_a_flop_endgame_solves_at_the_river():
    from holdem.public_tree import build_endgame_tree

    flop = build_endgame_tree(flop_root(), depth_limit=1)
    turn = build_endgame_tree(flop.leaves()[0].public, depth_limit=1)
    river_public = turn.leaves()[0].public
    space = TurnEndgameSpace((51, 47, 22))
    tree = build_endgame_tree(river_public, depth_limit=None)
    reach = space.initial_reach() * board_mask(tuple(river_public.board))
    reach /= reach.sum(axis=1, keepdims=True)
    solver = SubgameSolver(tree, config=CFRConfig.dcfr(), space=space)
    solver.solve(reach=reach, iterations=60)
    total, _ = subgame_exploitability(tree, strategy_map(solver), reach, space=space)
    assert total < 2.0


def test_isomorphic_boards_share_a_canonical_form():
    from holdem.isomorphism import canonical_board

    spades = (51, 47, 22)  # As Ks 7h
    hearts = (50, 46, 21)  # Ah Kh 7d — the same board, suits renamed
    assert canonical_board(spades)[0] == canonical_board(hearts)[0]
    assert canonical_board((51, 47, 22))[0] != canonical_board((51, 47, 23))[0]


def test_there_are_1755_distinct_flops():
    """The canonical count, and the reason isomorphism is not optional."""
    from holdem.isomorphism import count_canonical

    assert count_canonical(3) == 1755


def test_strengths_and_ranges_survive_relabelling():
    from holdem.isomorphism import canonical_board, canonicalise_range, combo_permutation

    board = (51, 47, 22, 6, 34)
    canonical, relabelling = canonical_board(board)
    permutation = combo_permutation(relabelling)
    legal = np.flatnonzero(board_mask(board))
    original, renamed = hand_ranks(board), hand_ranks(canonical)
    assert np.array_equal(
        np.argsort(original[legal]), np.argsort(renamed[permutation[legal]])
    ), "relabelling suits cannot change which hand is stronger"

    rng = np.random.default_rng(0)
    weights = rng.random(NUM_COMBOS) * board_mask(board)
    moved = canonicalise_range(weights, relabelling)
    assert moved.sum() == pytest.approx(weights.sum())
    assert (moved * (1 - board_mask(canonical))).sum() == pytest.approx(0.0)
