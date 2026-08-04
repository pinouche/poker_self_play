"""Regret minimisation and exact exploitability (paradigm B, stage 0).

These are the tests that prove the regret machinery before any neural network
touches it.  Both toy games have published equilibrium values, so the numbers
here are checked against the literature and not against ourselves:

* Kuhn poker: value -1/18 to the first player, and a known one-parameter
  family of equilibria (bluff frequency alpha in [0, 1/3]).
* Leduc hold'em: value -0.085606 chips to the first player.
"""

from __future__ import annotations

import pytest

from paradigm_b.core.cfr import (
    BestResponse,
    CFRConfig,
    CFRSolver,
    TabularPolicy,
    best_response_value,
    expected_values,
    exploitability,
)
from paradigm_b.core.game import KuhnPoker, LeducHoldem, build_tree
from paradigm_b.core.game.actions import CALL, FOLD, RAISE

KUHN_VALUE = -1.0 / 18.0
LEDUC_VALUE = -0.085606424078  # exact, sequence-form LP (Southey et al. 2005)


@pytest.fixture(scope="module")
def kuhn_tree():
    return build_tree(KuhnPoker())


@pytest.fixture(scope="module")
def leduc_tree():
    return build_tree(LeducHoldem())


def kuhn_equilibrium(tree, alpha: float) -> TabularPolicy:
    """The analytic Kuhn equilibrium family, parameterised by bluff rate alpha."""
    assert 0.0 <= alpha <= 1.0 / 3.0
    spec = {
        # Player 0, opening.
        (0, ""): {CALL: 1 - alpha, RAISE: alpha},
        (1, ""): {CALL: 1.0, RAISE: 0.0},
        (2, ""): {CALL: 1 - 3 * alpha, RAISE: 3 * alpha},
        # Player 0, checked then faced a bet.
        (0, "cr"): {FOLD: 1.0, CALL: 0.0},
        (1, "cr"): {FOLD: 2 / 3 - alpha, CALL: 1 / 3 + alpha},
        (2, "cr"): {FOLD: 0.0, CALL: 1.0},
        # Player 1, checked to.
        (0, "c"): {CALL: 2 / 3, RAISE: 1 / 3},
        (1, "c"): {CALL: 1.0, RAISE: 0.0},
        (2, "c"): {CALL: 0.0, RAISE: 1.0},
        # Player 1, facing a bet.
        (0, "r"): {FOLD: 1.0, CALL: 0.0},
        (1, "r"): {FOLD: 2 / 3, CALL: 1 / 3},
        (2, "r"): {FOLD: 0.0, CALL: 1.0},
    }
    policy = TabularPolicy.uniform(tree)
    for i, key in enumerate(tree.infoset_keys):
        for j, action in enumerate(tree.infoset_actions[i]):
            policy.probs[i, j] = spec[key][action]
    policy.check_valid()
    return policy


# --- the exploitability routine itself -------------------------------------
@pytest.mark.parametrize("alpha", [0.0, 1 / 6, 1 / 3])
def test_known_kuhn_equilibria_are_unexploitable(kuhn_tree, alpha):
    """Zero exploitability on strategies we know to be optimal, to 1e-12."""
    policy = kuhn_equilibrium(kuhn_tree, alpha)
    assert exploitability(kuhn_tree, policy) == pytest.approx(0.0, abs=1e-12)
    assert expected_values(kuhn_tree, policy)[0] == pytest.approx(KUHN_VALUE, abs=1e-12)


def test_best_response_beats_a_uniform_random_opponent(kuhn_tree):
    uniform = TabularPolicy.uniform(kuhn_tree)
    for player in (0, 1):
        assert best_response_value(kuhn_tree, uniform, player) > 0.0
    assert exploitability(kuhn_tree, uniform) > 0.1


def test_best_response_is_at_least_the_equilibrium_value(kuhn_tree):
    """A best response can never do worse than the game value."""
    policy = kuhn_equilibrium(kuhn_tree, 1 / 3)
    assert best_response_value(kuhn_tree, policy, 0) >= KUHN_VALUE - 1e-12
    assert best_response_value(kuhn_tree, policy, 1) >= -KUHN_VALUE - 1e-12


def test_best_response_policy_realises_its_value(kuhn_tree):
    """Playing out the returned BR table reproduces the BR value."""
    uniform = TabularPolicy.uniform(kuhn_tree)
    for player in (0, 1):
        br = BestResponse(kuhn_tree, uniform, player)
        realised = expected_values(kuhn_tree, br.policy_table())[player]
        assert realised == pytest.approx(br.value, abs=1e-12)


def test_best_response_never_folds_the_nuts(kuhn_tree):
    uniform = TabularPolicy.uniform(kuhn_tree)
    br = BestResponse(kuhn_tree, uniform, 1).policy_table().to_dict()
    assert br[(2, "r")]["check/call"] == 1.0, "the king always calls"
    assert br[(0, "r")]["fold"] == 1.0, "the jack always folds"


# --- CFR convergence -------------------------------------------------------
def test_cfr_plus_solves_kuhn(kuhn_tree):
    solver = CFRSolver(kuhn_tree, CFRConfig.cfr_plus())
    solver.iterate(500)
    policy = solver.average_policy()
    policy.check_valid()
    assert exploitability(kuhn_tree, policy) < 1e-3
    assert expected_values(kuhn_tree, policy)[0] == pytest.approx(KUHN_VALUE, abs=1e-3)


def test_exploitability_keeps_falling_with_more_iterations(kuhn_tree):
    """Not monotone iteration by iteration, but the trend is unambiguous."""
    solver = CFRSolver(kuhn_tree, CFRConfig.dcfr())
    curve = []
    for _ in range(5):
        solver.iterate(100)
        curve.append(exploitability(kuhn_tree, solver.average_policy()))
    assert curve[-1] < curve[0] / 3.0
    assert max(curve[3:]) < min(curve[:2])


@pytest.mark.parametrize(
    "variant,threshold",
    [("vanilla", 0.5), ("cfr_plus", 0.05), ("linear", 0.3), ("dcfr", 0.02)],
)
def test_every_cfr_variant_reduces_leduc_exploitability(leduc_tree, variant, threshold):
    """Each variant reaches its documented ballpark after 300 iterations."""
    solver = CFRSolver(leduc_tree, getattr(CFRConfig, variant)())
    solver.iterate(300)
    policy = solver.average_policy()
    policy.check_valid()
    assert exploitability(leduc_tree, policy) < threshold


def test_dcfr_reproduces_the_leduc_game_value(leduc_tree):
    """The stage-0 milestone: exploitability -> 0 and the value matches the LP."""
    solver = CFRSolver(leduc_tree, CFRConfig.dcfr())
    solver.iterate(600)
    policy = solver.average_policy()
    assert exploitability(leduc_tree, policy) < 3e-3
    assert expected_values(leduc_tree, policy)[0] == pytest.approx(LEDUC_VALUE, abs=1e-3)


def test_solved_leduc_strategy_is_sensible(leduc_tree):
    solver = CFRSolver(leduc_tree, CFRConfig.dcfr())
    solver.iterate(300)
    table = solver.average_policy().to_dict()
    # (rank, board rank, round-one history, round-two history); rank 2 is a king.
    king_pairs_the_board = table[(2, 2, "cc", "")]
    assert king_pairs_the_board["bet/raise"] > 0.9, "always bet a paired king"
    jack_facing_a_king_board_bet = table[(0, 2, "cc", "r")]
    assert jack_facing_a_king_board_bet["fold"] > 0.5, "a jack folds to that board"
