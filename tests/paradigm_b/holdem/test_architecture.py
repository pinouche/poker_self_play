"""Paper-aligned architecture checks for the hold'em nets and self-play loop."""

import numpy as np
import pytest
import torch

from paradigm_b.core.belief import PBS, PublicState, initial_reach
from paradigm_b.holdem.engine.combos import NUM_COMBOS, board_mask
from paradigm_b.holdem.net.features import (
    BOARD_OFFSET,
    CARD_SET_DIM,
    HISTORY_OFFSET,
    MAX_ACTIONS_PER_ROUND,
    encode_pbs as encode_holdem_pbs,
    encode_public as encode_holdem_public,
)
from paradigm_b.holdem.net.value_net import HoldemValueNet, HoldemValueNetConfig
from paradigm_b.holdem.net.policy import (
    HoldemPolicyNet,
    HoldemPolicyNetConfig,
    PolicyExample,
    PolicyReplayBuffer,
    train_policy_net,
)
from paradigm_b.holdem.selfplay import (
    HoldemReBeLConfig,
    HoldemSelfPlayConfig,
    _arrival_probability,
    collect_trajectory,
    train,
)
from paradigm_b.holdem.data.sampling import SituationConfig, sample_situation
from paradigm_b.holdem.engine.space import TurnEndgameSpace
from paradigm_b.holdem.net.leaf_values import ZeroLeafValues
from paradigm_b.holdem.engine.public_tree import PublicState as HoldemPublicState
from paradigm_b.holdem.engine.betting import BET_HALF, CALL, Betting


def test_holdem_public_cards_are_grouped_by_street_and_permutation_invariant():
    original = HoldemPublicState(board=(51, 47, 22, 6))
    permuted_flop = HoldemPublicState(board=(22, 51, 47, 6))
    moved_to_turn = HoldemPublicState(board=(51, 47, 6, 22))

    original_features = encode_holdem_public(original)
    permuted_features = encode_holdem_public(permuted_flop)
    moved_features = encode_holdem_public(moved_to_turn)

    np.testing.assert_array_equal(original_features, permuted_features)
    assert not np.array_equal(
        original_features[: 3 * CARD_SET_DIM], moved_features[: 3 * CARD_SET_DIM]
    )
    assert original_features[51] == 1.0
    assert original_features[CARD_SET_DIM + 6] == 1.0


def test_holdem_betting_history_preserves_action_order_and_sizes():
    betting = Betting(max_raises=2).apply(BET_HALF).apply(CALL)
    public = HoldemPublicState(betting=betting, board=(51, 47, 22, 6))

    features = encode_holdem_public(public)
    history = features[HISTORY_OFFSET - BOARD_OFFSET :].reshape(
        3, MAX_ACTIONS_PER_ROUND, 2
    )

    assert np.count_nonzero(history[0]) == 0
    np.testing.assert_allclose(history[1, :2], [[1.0, 0.1], [1.0, 0.1]])
    assert np.count_nonzero(history[1, 2:]) == 0
    assert np.count_nonzero(history[2]) == 0


def test_holdem_betting_history_keeps_rounds_separate():
    betting = Betting(max_raises=2).apply(BET_HALF).apply(CALL)
    betting = betting.deal_board().apply(CALL).apply(CALL)
    public = HoldemPublicState(betting=betting, board=(51, 47, 22, 6, 3))

    features = encode_holdem_public(public)
    history = features[HISTORY_OFFSET - BOARD_OFFSET :].reshape(
        3, MAX_ACTIONS_PER_ROUND, 2
    )

    assert np.count_nonzero(history[0]) == 0
    np.testing.assert_allclose(history[1, :2], [[1.0, 0.1], [1.0, 0.1]])
    np.testing.assert_allclose(history[2, :2], [[1.0, 0.0], [1.0, 0.0]])


def test_holdem_betting_history_rejects_more_than_six_actions_per_round():
    betting = Betting(history=((CALL,) * 7, (), ()))

    with pytest.raises(ValueError, match="maximum is 6"):
        encode_holdem_public(HoldemPublicState(betting=betting, board=(51, 47, 22, 6)))


def test_holdem_policy_is_per_hand_and_normalised_over_legal_actions():
    public = HoldemPublicState(betting=Betting(), board=(51, 47, 22, 6))
    ranges = torch.as_tensor(board_mask(public.board), dtype=torch.float32)
    ranges = (ranges / ranges.sum()).repeat(2, 1).numpy()
    pbs = type("PBS", (), {"public": public, "ranges": ranges})()
    features = torch.as_tensor(encode_holdem_pbs(pbs)[None], dtype=torch.float32)
    legal = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32)
    net = HoldemPolicyNet(HoldemPolicyNetConfig(hidden_dim=16, num_hidden_layers=1))

    probabilities = net(features, agent_index=torch.tensor([0]), legal_mask=legal)

    assert probabilities.shape == (1, NUM_COMBOS, 9)
    torch.testing.assert_close(probabilities[..., :5].sum(dim=-1), torch.ones((1, NUM_COMBOS)))
    assert torch.count_nonzero(probabilities[..., 5:]) == 0


def test_holdem_self_play_records_a_quantizable_root_policy_target():
    public = HoldemPublicState(betting=Betting(), board=(51, 47, 22, 6))
    examples = collect_trajectory(
        ZeroLeafValues(),
        TurnEndgameSpace(public.board),
        public,
        HoldemSelfPlayConfig(search_iterations=2, river_iterations=2),
        np.random.default_rng(0),
    )

    policy = examples[0].policy
    assert policy is not None
    assert policy.target.shape == (NUM_COMBOS, 9)
    np.testing.assert_allclose(policy.target.sum(axis=-1), 1.0)
    assert set(np.flatnonzero(policy.legal_mask)) == set(public.legal_actions())


def test_holdem_self_play_follows_a_sampled_flop_through_every_street():
    rng = np.random.default_rng(3)
    space, public, reach = sample_situation(
        rng, SituationConfig(board_cards=3, max_raises=0)
    )

    examples = collect_trajectory(
        ZeroLeafValues(),
        space,
        public,
        HoldemSelfPlayConfig(
            search_iterations=1,
            river_iterations=1,
            warmup_fraction=0.0,
            exploration=0.0,
        ),
        rng,
        reach=reach,
    )

    assert [int(example.mask.sum()) for example in examples] == [1176, 1128, 1081]


@pytest.mark.parametrize(
    ("board_cards", "expected_rounds", "expected_examples"),
    [(3, 3, 3), (4, 2, 2), (5, 1, 1)],
)
def test_sampled_trajectory_covers_every_remaining_postflop_street(
    board_cards, expected_rounds, expected_examples
):
    rng = np.random.default_rng(board_cards)
    space, public, reach = sample_situation(
        rng, SituationConfig(board_cards=board_cards, max_raises=0)
    )

    examples = collect_trajectory(
        ZeroLeafValues(),
        space,
        public,
        HoldemSelfPlayConfig(search_iterations=1, river_iterations=1),
        rng,
        reach=reach,
    )

    assert public.betting.num_rounds == expected_rounds
    assert len(examples) == expected_examples


@pytest.mark.parametrize("board_cards", [0, 2, 6])
def test_sampling_rejects_non_postflop_board_sizes(board_cards):
    with pytest.raises(ValueError, match="board_cards"):
        sample_situation(
            np.random.default_rng(0), SituationConfig(board_cards=board_cards)
        )


def test_holdem_self_play_defaults_to_algorithm_one_sampling():
    config = HoldemSelfPlayConfig()
    # No policy warm start, so Algorithm 1 samples the descent leaf uniformly
    # over every CFR iterate.
    assert config.warmup_fraction == 0.0
    # Appendix E: "for all experiments we set the probability to explore a
    # random action to eps = 25%".
    assert config.exploration == 0.25


def test_leaf_arrival_probability_excludes_overlapping_private_hands():
    overlapping = np.zeros((2, NUM_COMBOS))
    overlapping[:, 0] = 1.0
    disjoint = overlapping.copy()
    disjoint[1] = 0.0
    disjoint[1, -1] = 1.0

    assert _arrival_probability(overlapping) == pytest.approx(0.0)
    assert _arrival_probability(disjoint) > 0.0


def test_quantized_policy_replay_trains_with_probability_mse():
    public = HoldemPublicState(betting=Betting(), board=(51, 47, 22, 6))
    ranges = board_mask(public.board).astype(np.float32)
    pbs = type("PBS", (), {"public": public, "ranges": np.stack((ranges / ranges.sum(),) * 2)})()
    target = np.zeros((NUM_COMBOS, 9), dtype=np.float32)
    target[:, 1] = 1.0
    example = PolicyExample(
        features=encode_holdem_pbs(pbs),
        agent_index=0,
        legal_mask=np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        target=target,
    )
    buffer = PolicyReplayBuffer(2)
    buffer.add([example])
    net = HoldemPolicyNet(HoldemPolicyNetConfig(hidden_dim=16, num_hidden_layers=1))

    loss = train_policy_net(net, buffer, 1, 1, 1e-3, np.random.default_rng(0))
    _, _, _, restored = buffer.sample(1, np.random.default_rng(1))

    assert np.isfinite(loss)
    np.testing.assert_allclose(restored, target[None])


def test_holdem_self_play_trains_and_returns_the_policy_network():
    public = HoldemPublicState(betting=Betting(), board=(51, 47, 22, 6))
    value_config = HoldemValueNetConfig(hidden_dim=16, num_hidden_layers=1)
    policy_config = HoldemPolicyNetConfig(hidden_dim=16, num_hidden_layers=1)
    config = HoldemReBeLConfig(
        iterations=1,
        buffer_size=4,
        updates_per_iteration=1,
        batch_size=1,
        policy_buffer_size=2,
        policy_updates_per_iteration=1,
        value_net=value_config,
        policy_net=policy_config,
        self_play=HoldemSelfPlayConfig(
            trajectories_per_iteration=1, search_iterations=2, river_iterations=2
        ),
    )

    value_net, history = train(TurnEndgameSpace(public.board), public, config)

    assert isinstance(value_net.policy_net, HoldemPolicyNet)
    assert np.isfinite(history[0]["policy_loss"])