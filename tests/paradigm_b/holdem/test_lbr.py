"""Local best response: the responder that is allowed off the abstraction.

Two invariants carry this module, and both of them caught real bugs while it was
being written, so they are here rather than in a comment:

* with probe sizes equal to the agent's own, translation is the identity and the
  full responder *is* the exact range best response — to the last bit;
* the myopic responder can never beat the searching one, because the searching
  one is free to reproduce any myopic choice.

The first pinned a fold branch that had been hand-written without the
opponent-mass weighting every other terminal carries.  The second caught the
rollout pricing a showdown at ``min(contributions)`` — the responder's own bet
was not yet matched at that point, so it was collecting fold equity for free and
scoring four times the searching responder.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from paradigm_b.core.cfr.tabular_cfr import CFRConfig
from paradigm_b.core.search.best_response import subgame_exploitability
from paradigm_b.core.search.policy import strategy_map
from paradigm_b.core.search.subgame import SubgameSolver
from paradigm_b.holdem.arms_common.lbr import (
    LBRConfig,
    checkdown_equity,
    lbr_values,
)
from paradigm_b.holdem.engine.betting import DEFAULT_BET_FRACTIONS, Betting
from paradigm_b.holdem.engine.combos import board_mask
from paradigm_b.holdem.engine.public_tree import PublicState, build_endgame_tree
from paradigm_b.holdem.engine.showdown import showdown_values
from paradigm_b.holdem.net.value_net import HoldemValueNet, HoldemValueNetConfig
from paradigm_b.holdem.engine.space import TurnEndgameSpace

RIVER_BOARD = (51, 47, 22, 6, 34)
PROBES = (0.375, 0.75, 1.5, 3.0)


def river_setup(bet_fractions=DEFAULT_BET_FRACTIONS, pot=40, stack=200):
    """A one-round river endgame: exact terminals, no value network anywhere."""
    space = TurnEndgameSpace(RIVER_BOARD)
    root = PublicState(
        betting=Betting(
            starting_pot=pot,
            stack=stack,
            max_raises=1,
            num_rounds=1,
            bet_fractions=bet_fractions,
        ),
        board=RIVER_BOARD,
    )
    mask = space.root_mask()
    row = mask / mask.sum()
    return space, root, np.stack([row, row])


def uniform_strategies(space, tree):
    return {
        node.public: np.full((space.num_hands, node.num_actions), 1.0 / node.num_actions)
        for node in tree.decision_nodes()
    }


def solved_strategies(space, tree, reach, iterations=400):
    solver = SubgameSolver(tree, config=CFRConfig.dcfr(), space=space)
    solver.solve(reach=reach, iterations=iterations)
    return strategy_map(solver)


def test_full_responder_is_the_exact_best_response_when_probes_match():
    """The anchor: no translation, so the two must agree exactly.

    This is what ties the whole dual-state walk — real chips on one side, the
    agent's believed state on the other — back to the already-verified metric.
    Anything that breaks the walk breaks this to many decimal places.
    """
    space, root, reach = river_setup()
    tree = build_endgame_tree(root, depth_limit=None)
    strategies = uniform_strategies(space, tree)

    exact, _ = subgame_exploitability(tree, strategies, reach, space=space)
    scores = lbr_values(
        space,
        strategies,
        root,
        reach,
        LBRConfig(probe_fractions=DEFAULT_BET_FRACTIONS, include_all_in=True),
    )
    assert scores["lbr_full"] == pytest.approx(exact, rel=1e-12)


def test_myopia_never_helps_the_responder():
    """``lbr_full >= lbr_classic``, on both a solved and an unsolved agent.

    The searching responder can always replay whatever the myopic one chose, so
    a violation means the rollout is pricing something it does not have to pay
    for — which is exactly the bug this caught.
    """
    space, root, reach = river_setup()
    tree = build_endgame_tree(root, depth_limit=None)
    for strategies in (
        uniform_strategies(space, tree),
        solved_strategies(space, tree, reach, iterations=120),
    ):
        scores = lbr_values(
            space, strategies, root, reach, LBRConfig(probe_fractions=PROBES)
        )
        assert scores["lbr_full"] >= scores["lbr_classic"] - 1e-9
        assert scores["lbr_myopia_gap"] == pytest.approx(
            scores["lbr_full"] - scores["lbr_classic"]
        )


def test_off_abstraction_probes_find_what_the_exact_response_cannot():
    """The point of the metric.

    A solved agent is very nearly unexploitable *inside its own abstraction* —
    that is what CFR converged to — so the exact best response reports almost
    nothing.  A responder allowed to bet between its sizes finds a great deal
    more, and the difference is the translation loss, which is invisible to the
    existing metric by construction.
    """
    space, root, reach = river_setup()
    tree = build_endgame_tree(root, depth_limit=None)
    strategies = solved_strategies(space, tree, reach)

    exact, _ = subgame_exploitability(tree, strategies, reach, space=space)
    scores = lbr_values(
        space, strategies, root, reach, LBRConfig(probe_fractions=PROBES)
    )
    assert exact < 0.5, "a solved agent should be near-unexploitable in-abstraction"
    assert scores["lbr_full"] > 10 * exact


def test_a_wider_abstraction_translates_better():
    """The measured payoff of the four-size abstraction.

    Same board, same ranges, same probes, same solver effort — the only thing
    that changes is how many sizes the agent has to map a real bet onto.
    """
    losses = {}
    for fractions in ((0.5, 1.0), (0.25, 0.5, 1.0, 2.0)):
        space, root, reach = river_setup(bet_fractions=fractions)
        tree = build_endgame_tree(root, depth_limit=None)
        strategies = solved_strategies(space, tree, reach)
        exact, _ = subgame_exploitability(tree, strategies, reach, space=space)
        scores = lbr_values(
            space, root=root, reach=reach, strategies=strategies,
            config=LBRConfig(probe_fractions=PROBES),
        )
        losses[fractions] = scores["lbr_full"] - exact

    assert losses[(0.25, 0.5, 1.0, 2.0)] < losses[(0.5, 1.0)]


def test_the_responder_obeys_the_raise_cap():
    """A cap is a rule of the game, not a quirk of the agent's abstraction.

    With ``max_raises=1`` the responder's own bet uses the round's only raise,
    so facing the agent's re-raise it must have nothing but fold and call — a
    responder that could re-raise anyway would be scored on lines the game does
    not contain.
    """
    from paradigm_b.holdem.arms_common.lbr import _Walker

    space, root, reach = river_setup()
    walker = _Walker(space, {}, hero=0, config=LBRConfig(), myopic=False)

    opened = root.betting.apply(root.betting.bet_action_ids()[0])
    assert walker._probe_totals(opened) == [], "the raise cap must bind the responder"
    assert walker._probe_totals(root.betting), "an unraised pot must offer probes"


def test_checkdown_equity_on_a_complete_board_is_the_showdown():
    """With no cards to come there is nothing to average, so it is exact."""
    space = TurnEndgameSpace(RIVER_BOARD)
    rng = np.random.default_rng(0)
    villain = rng.random(space.num_hands) * board_mask(RIVER_BOARD)
    np.testing.assert_allclose(
        checkdown_equity(space, RIVER_BOARD, villain),
        showdown_values(RIVER_BOARD, np.ascontiguousarray(villain), 1.0),
    )


def test_checkdown_equity_averages_the_runouts_on_the_turn():
    """One term per river card, each masked by the card it deals."""
    turn = RIVER_BOARD[:4]
    space = TurnEndgameSpace(turn)
    rng = np.random.default_rng(1)
    villain = rng.random(space.num_hands) * board_mask(turn)

    expected = np.zeros(space.num_hands)
    rivers = [c for c in range(52) if c not in turn]
    for card in rivers:
        full = tuple(sorted(turn + (card,)))
        mask = board_mask(full)
        expected += showdown_values(
            full, np.ascontiguousarray(villain * mask), 1.0
        ) * mask
    expected /= len(rivers)

    np.testing.assert_allclose(
        checkdown_equity(space, turn, villain), expected, atol=1e-12
    )


def test_the_stderr_is_over_situations_and_is_absent_from_a_single_one():
    """What the ``+/-`` means here, pinned.

    ReBeL's Table 1 reports LBR as 881 +/- 94 mbb/g, and that ± is sampling
    error over *dealt hands*.  Nothing is dealt here: the responder walks the
    whole tree against full ranges and enumerates every runout, so one situation
    is exact and repeating it is bit-identical.  The dispersion that does exist
    is over which held-out situations were drawn, so it needs more than one of
    them and it is zero — not small — when there is only one.
    """
    from paradigm_b.holdem.arms_common.evaluation import (
        EvaluationConfig,
        evaluate_agent,
        make_held_out_situations,
    )
    from paradigm_b.holdem.data.sampling import SituationConfig

    net = HoldemValueNet(HoldemValueNetConfig(hidden_dim=16, num_residual_blocks=1, card_embedding_dim=8))
    net.eval()
    config = EvaluationConfig(search_iterations=2, local_best_response=True, streets=(5,))

    single = evaluate_agent(
        net,
        make_held_out_situations(np.random.default_rng(0), SituationConfig(), streets=(5,), boards_per_street=1),
        replace(config, boards_per_street=1),
    )
    assert single["river_boards"] == 1.0
    assert single["river_stderr"] == 0.0
    assert single["river_lbr_full_stderr"] == 0.0

    several = evaluate_agent(
        net,
        make_held_out_situations(np.random.default_rng(0), SituationConfig(), streets=(5,), boards_per_street=4),
        replace(config, boards_per_street=4),
    )
    assert several["river_boards"] == 4.0
    assert several["river_stderr"] > 0.0
    assert several["river_lbr_full_stderr"] > 0.0


def test_repeating_an_evaluation_returns_the_same_number():
    """The LBR figure is a computation, not a measurement: no run-to-run noise."""
    from paradigm_b.holdem.arms_common.evaluation import (
        EvaluationConfig,
        evaluate_agent,
        make_held_out_situations,
    )
    from paradigm_b.holdem.data.sampling import SituationConfig

    net = HoldemValueNet(HoldemValueNetConfig(hidden_dim=16, num_residual_blocks=1, card_embedding_dim=8))
    net.eval()
    tests = make_held_out_situations(
        np.random.default_rng(0), SituationConfig(), streets=(5,), boards_per_street=2
    )
    config = EvaluationConfig(
        boards_per_street=2, search_iterations=2, local_best_response=True, streets=(5,)
    )
    first = evaluate_agent(net, tests, config)
    second = evaluate_agent(net, tests, config)
    assert first["river_lbr_full"] == second["river_lbr_full"]
    assert first["river_lbr_classic"] == second["river_lbr_classic"]
