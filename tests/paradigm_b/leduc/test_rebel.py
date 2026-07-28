"""ReBeL self play (paradigm B, stage 3).

The loop's own invariants — what a trajectory produces, how values are converted
between the solver's convention and the network's — plus the end-to-end claim
that self play plus search actually reduces exploitability.
"""

from __future__ import annotations

import numpy as np
import pytest

from paradigm_b.core.belief import PBS, PublicState, initial_reach
from paradigm_b.core.belief.ranges import PAIR_CORRECTION
from paradigm_b.core.cfr import exploitability
from paradigm_b.core.game import LeducHoldem, build_tree
from paradigm_b.leduc.rebel import (
    ReBeLConfig,
    SelfPlayConfig,
    ValueReplayBuffer,
    collect_trajectory,
    evaluate_agent,
    normalise_root_values,
    train_rebel,
)
from paradigm_b.leduc.rebel.selfplay import TrainingExample, _arrival_probability
from paradigm_b.core.search import ContinualResolver, ResolveConfig
from paradigm_b.leduc.value_net import PBSValueNet, ValueNetConfig
from paradigm_b.leduc.value_net.values import NetLeafValues


@pytest.fixture(scope="module")
def world_tree():
    return build_tree(LeducHoldem())


# --- value conventions -----------------------------------------------------
def test_normalising_root_values_undoes_the_reach_scaling():
    reach = initial_reach() * 0.5
    counterfactual = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [-1.0] * 6])
    normalised = normalise_root_values(counterfactual, reach)
    masses = reach.sum(axis=1)
    assert normalised[0] == pytest.approx(
        counterfactual[0] / (PAIR_CORRECTION * masses[1])
    )
    assert normalised[1] == pytest.approx(
        counterfactual[1] / (PAIR_CORRECTION * masses[0])
    )


def test_unreachable_belief_states_produce_no_training_example():
    reach = initial_reach()
    reach[1] = 0.0
    assert normalise_root_values(np.zeros((2, 6)), reach) is None


def test_arrival_probability_excludes_the_impossible_diagonal():
    reach = initial_reach()  # every hand at 1/6 for both players
    # 30 legal ordered deals out of 36 pairs, each at 1/36, corrected by 6/5.
    assert _arrival_probability(reach) == pytest.approx(1.0)


# --- trajectories ----------------------------------------------------------
def test_a_trajectory_visits_both_rounds_and_stops():
    rng = np.random.default_rng(0)
    net = PBSValueNet(ValueNetConfig(hidden_dim=32, num_residual_blocks=1))
    config = SelfPlayConfig(search_iterations=20)
    examples = collect_trajectory(NetLeafValues(net), config, rng)
    assert len(examples) == 2, "one belief state per betting round in Leduc"
    assert examples[0].pbs.public.betting_round == 0
    assert examples[1].pbs.public.betting_round == 1
    assert examples[1].pbs.public.board >= 0
    for example in examples:
        assert example.values.shape == (2, 6)
        assert np.isfinite(example.values).all()


def test_trajectory_targets_are_close_to_zero_sum():
    """A belief state's values must balance: one player's chips are the other's."""
    rng = np.random.default_rng(1)
    net = PBSValueNet(ValueNetConfig(hidden_dim=32, num_residual_blocks=1))
    for _ in range(4):
        for example in collect_trajectory(
            NetLeafValues(net), SelfPlayConfig(search_iterations=40), rng
        ):
            total = (example.pbs.ranges * example.values).sum()
            assert total == pytest.approx(0.0, abs=0.05)


def test_exploration_reaches_belief_states_the_policy_avoids():
    rng = np.random.default_rng(2)
    net = PBSValueNet(ValueNetConfig(hidden_dim=32, num_residual_blocks=1))
    boards = set()
    for _ in range(20):
        examples = collect_trajectory(
            NetLeafValues(net), SelfPlayConfig(search_iterations=20), rng
        )
        boards.add(examples[-1].pbs.public.board)
    assert len(boards) >= 4, "self play should not always land on the same board"


# --- the replay buffer -----------------------------------------------------
def test_buffer_overwrites_oldest_first():
    buffer = ValueReplayBuffer(capacity=3)
    public = PublicState()
    for i in range(5):
        buffer.add(
            [
                TrainingExample(
                    pbs=PBS(public=public, ranges=initial_reach()),
                    values=np.full((2, 6), float(i)),
                )
            ]
        )
    assert len(buffer) == 3
    assert sorted(buffer.targets[:, 0, 0]) == [2.0, 3.0, 4.0]


def test_buffer_sampling_shapes():
    buffer = ValueReplayBuffer(capacity=10)
    public = PublicState()
    buffer.add(
        [
            TrainingExample(PBS(public=public, ranges=initial_reach()), np.zeros((2, 6)))
            for _ in range(6)
        ]
    )
    features, targets = buffer.sample(4, np.random.default_rng(0))
    assert features.shape[0] == 4 and targets.shape == (4, 2, 6)


# --- the stage-3 milestone -------------------------------------------------
@pytest.mark.slow
def test_self_play_reduces_exploitability(world_tree):
    """Search plus self play beats search with an untrained network."""
    config = ReBeLConfig(
        iterations=12,
        evaluate_every=100,  # only the final evaluation
        updates_per_iteration=150,
        self_play=SelfPlayConfig(trajectories_per_iteration=48, search_iterations=60),
        evaluation=ResolveConfig(iterations=200, depth_limit=1),
    )
    untrained = ContinualResolver(
        NetLeafValues(PBSValueNet(config.value_net)), config.evaluation
    ).policy(world_tree)
    before = exploitability(world_tree, untrained)

    run = train_rebel(config, world_tree=world_tree)
    after = run.history[-1]["exploitability"]

    assert after < before / 3.0, f"{after} vs {before}"
    assert after < 0.25
    assert run.history[-1]["value"] == pytest.approx(-0.0856, abs=0.15)


def test_evaluating_an_agent_reports_exploitability(world_tree):
    config = ReBeLConfig(evaluation=ResolveConfig(iterations=30, depth_limit=1))
    scores = evaluate_agent(PBSValueNet(config.value_net), config, world_tree)
    assert set(scores) == {"exploitability", "value"}
    assert scores["exploitability"] > 0.0
