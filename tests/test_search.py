"""Depth-limited search and the value network (paradigm B, stage 2).

Three separable claims, tested separately because confusing them is how
depth-limited solving goes wrong:

1. The *search* is sound — given correct values at the depth limit, the
   resulting play is close to what full-depth CFR produces.
2. The *re-solving* is safe — a subgame solved from ranges alone honours what
   the solve above it promised (this is the gadget, and it is measurable: the
   unsafe version is an order of magnitude worse).
3. The *network* is accurate — it reproduces the exact solver's values well
   enough to search with.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from belief import build_public_tree, initial_reach
from belief.ranges import board_mask
from cfr import CFRConfig, exploitability
from game import LeducHoldem, build_tree
from game.actions import CALL
from search import ContinualResolver, ResolveConfig, SubgameSolver, strategy_map
from search.best_response import RangeBestResponse
from search.evaluate import leaf_reaches, range_values
from search.policy import tabular_policy_from_strategies
from search.subgame import Gadget
from value_net import (
    ExactLeafValues,
    NestedLeafValues,
    PBSValueNet,
    ValueNetConfig,
    ValueTrainConfig,
    ZeroLeafValues,
    build_dataset,
    evaluate_net,
    exact_pbs_values,
    sample_pbs,
    train_value_net,
)
from value_net.features import INPUT_DIM, encode_batch, encode_pbs
from value_net.values import NetLeafValues


@pytest.fixture(scope="module")
def world_tree():
    return build_tree(LeducHoldem())


@pytest.fixture(scope="module")
def equilibrium():
    solver = SubgameSolver(build_public_tree(), config=CFRConfig.dcfr())
    solver.solve(iterations=400)
    return strategy_map(solver)


# --- range-form best response ---------------------------------------------
def test_range_best_response_matches_the_exact_tree_best_response(
    equilibrium, world_tree
):
    """The cheap best response agrees with the stage-0 one, to 1e-9."""
    from cfr import best_response_value

    policy = tabular_policy_from_strategies(equilibrium, world_tree)
    tree = build_public_tree()
    for player in (0, 1):
        cheap = RangeBestResponse(tree, equilibrium, player).value(initial_reach())
        exact = best_response_value(world_tree, policy, player)
        assert cheap == pytest.approx(exact, abs=1e-9)


def test_a_best_responder_beats_a_checking_station():
    """Against a player who never folds, the response is to bet everything."""
    tree = build_public_tree()
    passive = {
        node.public: np.tile(
            np.eye(node.num_actions)[list(node.actions).index(CALL)], (6, 1)
        )
        for node in tree.decision_nodes()
    }
    value = RangeBestResponse(tree, passive, 0).value(initial_reach())
    assert value > 0.5, "never folding is very exploitable"


# --- the value of a fixed strategy ----------------------------------------
def test_range_values_agree_with_the_solver(equilibrium):
    tree = build_public_tree()
    solver = SubgameSolver(tree, config=CFRConfig.dcfr())
    for node in tree.decision_nodes():
        solver.strategy_sum[node.index] = equilibrium[node.public].copy()
    assert np.allclose(
        range_values(tree, equilibrium, initial_reach()),
        solver.evaluate(initial_reach()),
        atol=1e-12,
    )


def test_exact_subgame_values_are_zero_sum():
    rng = np.random.default_rng(0)
    for _ in range(5):
        pbs = sample_pbs(rng)
        values = exact_pbs_values(pbs, iterations=100)
        total = (pbs.ranges * values).sum()
        assert total == pytest.approx(0.0, abs=2e-3)


# --- the re-solving gadget -------------------------------------------------
def test_unsafe_resolving_is_measurably_exploitable(equilibrium, world_tree):
    """Re-solving round two from ranges alone damages an exact equilibrium.

    The headline number for why the gadget exists: same trunk, same arriving
    ranges, and the only change is that the second round is re-solved rather
    than played from the equilibrium.
    """
    before = exploitability(
        world_tree, tabular_policy_from_strategies(equilibrium, world_tree)
    )
    unsafe = dict(equilibrium)
    safe = dict(equilibrium)
    tree = build_public_tree(depth_limit=1)

    for leaf, reach in leaf_reaches(tree, equilibrium, initial_reach()):
        if reach.sum(axis=1).min() <= 0.0:
            continue
        subtree = build_public_tree(leaf.public, depth_limit=1)
        promised = range_values(subtree, equilibrium, reach)

        plain = SubgameSolver(subtree, config=CFRConfig.dcfr())
        plain.solve(reach=reach, iterations=200)
        for node in plain.tree.decision_nodes():
            unsafe[node.public] = plain.average_strategy(node)

        for player in (0, 1):
            guarded = SubgameSolver(
                build_public_tree(leaf.public, depth_limit=1), config=CFRConfig.dcfr()
            )
            guarded.solve(
                reach=reach,
                iterations=200,
                gadget=Gadget(player=player, promised=promised[1 - player]),
            )
            for node in guarded.tree.decision_nodes():
                if node.player == player:
                    safe[node.public] = guarded.average_strategy(node)

    unsafe_value = exploitability(
        world_tree, tabular_policy_from_strategies(unsafe, world_tree)
    )
    safe_value = exploitability(
        world_tree, tabular_policy_from_strategies(safe, world_tree)
    )
    assert before < 0.01
    assert unsafe_value > 20 * before, "unsafe re-solving really does hurt"
    assert safe_value < unsafe_value / 4.0, "the gadget recovers most of it"


def test_gadget_leaves_the_opponent_no_better_than_promised(equilibrium):
    """The guarantee the gadget buys, stated directly."""
    tree = build_public_tree(depth_limit=1)
    leaf, reach = next(
        (leaf, reach)
        for leaf, reach in leaf_reaches(tree, equilibrium, initial_reach())
        if reach.sum(axis=1).min() > 0.05
    )
    subtree = build_public_tree(leaf.public, depth_limit=1)
    promised = range_values(subtree, equilibrium, reach)

    solver = SubgameSolver(subtree, config=CFRConfig.dcfr())
    solver.solve(
        reach=reach, iterations=400, gadget=Gadget(player=0, promised=promised[1])
    )
    ours = {n.public: solver.average_strategy(n) for n in subtree.decision_nodes()}
    achieved = RangeBestResponse(subtree, ours, 1).values(reach)
    # Only hands the opponent can actually hold here: the gadget has no lever on
    # a hand that never enters the subgame, and none on the board card at all.
    live = (reach[1] > 0.0) & (board_mask(leaf.public.board) > 0.0)
    slack = (achieved - promised[1])[live]
    assert slack.max() < 0.05, "no hand does much better than it was promised"


# --- depth-limited search --------------------------------------------------
def test_search_with_exact_values_beats_search_with_none(world_tree):
    """Values at the depth limit are what make depth-limited search work."""
    blind = ContinualResolver(
        ZeroLeafValues(), ResolveConfig(iterations=150, depth_limit=1)
    ).policy(world_tree)
    informed = ContinualResolver(
        ExactLeafValues(iterations=100), ResolveConfig(iterations=150, depth_limit=1)
    ).policy(world_tree)
    assert exploitability(world_tree, informed) < 0.5 * exploitability(world_tree, blind)


def test_full_lookahead_resolving_approaches_the_solved_game(world_tree):
    """Continual re-solving, safe at every round boundary, with nothing truncated.

    The strongest configuration available: no value function, no depth limit,
    but the agent still re-solves from scratch when the board lands instead of
    following the plan it made before it.  This is what DeepStack does, and it
    lands within a factor of two of solving the game outright.
    """
    policy = ContinualResolver(
        None, ResolveConfig(iterations=1000, depth_limit=2)
    ).policy(world_tree)
    assert exploitability(world_tree, policy) < 8e-3


def test_the_resolver_solves_again_at_every_round_start():
    """Re-solving is driven by round boundaries, not by the depth limit.

    Before this was true, a lookahead that reached past the end of the round
    silently produced a single solve and no re-solving at all.
    """
    _, traces = ContinualResolver(
        None, ResolveConfig(iterations=40, depth_limit=2)
    ).run()
    assert {trace.pbs.public.betting_round for trace in traces} == {0, 1}
    assert len(traces) > 1, "one solve for the first round, then one per board"


def test_more_search_iterations_reduce_exploitability(world_tree):
    """The depth-limited agent is iteration-starved, not stuck at a floor."""
    def solve(iterations):
        return exploitability(
            world_tree,
            ContinualResolver(
                NestedLeafValues(iterations_per_call=2, warmup=200),
                ResolveConfig(iterations=iterations, depth_limit=1),
            ).policy(world_tree),
        )

    assert solve(2000) < 0.8 * solve(400)


# --- the network -----------------------------------------------------------
def test_features_round_trip():
    rng = np.random.default_rng(1)
    pbs = sample_pbs(rng)
    features = encode_pbs(pbs)
    assert features.shape == (INPUT_DIM,)
    assert np.allclose(features[:12].reshape(2, 6), pbs.ranges)
    assert features[12 + pbs.public.board] == 1.0
    assert encode_batch([pbs, pbs]).shape == (2, INPUT_DIM)


def test_network_output_is_zero_sum_and_masks_impossible_hands():
    rng = np.random.default_rng(2)
    states = [sample_pbs(rng) for _ in range(16)]
    net = PBSValueNet(ValueNetConfig(hidden_dim=32, num_residual_blocks=1))
    features = torch.as_tensor(encode_batch(states), dtype=torch.float32)
    with torch.no_grad():
        values = net(features).numpy()
    assert values.shape == (16, 2, 6)
    for pbs, value in zip(states, values):
        assert value[:, pbs.public.board] == pytest.approx(0.0)
        assert (pbs.ranges * value).sum() == pytest.approx(0.0, abs=1e-5)


def test_network_learns_the_exact_values():
    """The stage-2 milestone: the net regresses CFR values."""
    rng = np.random.default_rng(3)
    data = build_dataset(500, rng, iterations=80)
    train, validation = data.split(0.2)
    net = PBSValueNet(ValueNetConfig(hidden_dim=128, num_residual_blocks=2))
    train_value_net(
        net,
        train,
        config=ValueTrainConfig(epochs=120, learning_rate=1e-3, batch_size=128),
    )
    scores = evaluate_net(net, validation)
    assert scores["r2"] > 0.8, scores
    assert scores["mae"] < 0.6, scores


def test_net_leaf_values_have_the_solver_contract():
    rng = np.random.default_rng(4)
    states = [sample_pbs(rng) for _ in range(8)]
    net = PBSValueNet(ValueNetConfig(hidden_dim=32, num_residual_blocks=1))
    values = NetLeafValues(net)(states)
    assert values.shape == (8, 2, 6)
    assert values.dtype == np.float64
