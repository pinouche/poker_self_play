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
    NUM_BETTING_ROUNDS,
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
from paradigm_b.core.cfr.tabular_cfr import CFRConfig
from paradigm_b.core.search.subgame import SubgameSolver
from paradigm_b.holdem.selfplay import (
    HoldemReBeLConfig,
    HoldemSelfPlayConfig,
    _arrival_probability,
    collect_trajectory,
    exploration_profile,
    initialize_policy,
    sample_iteration,
    sample_leaf,
    train,
)
from paradigm_b.holdem.data.sampling import SituationConfig, sample_situation
from paradigm_b.holdem.engine.space import TurnEndgameSpace
from paradigm_b.holdem.net.leaf_values import ZeroLeafValues
from paradigm_b.holdem.engine.public_tree import (
    PublicState as HoldemPublicState,
    build_turn_tree,
)
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
        NUM_BETTING_ROUNDS, MAX_ACTIONS_PER_ROUND, 2
    )

    # Four slots now that a hand can start preflop, and a two-round endgame
    # right-aligns into the last two of them.
    assert np.count_nonzero(history[:2]) == 0
    np.testing.assert_allclose(history[2, :2], [[1.0, 0.1], [1.0, 0.1]])
    assert np.count_nonzero(history[2, 2:]) == 0
    assert np.count_nonzero(history[3]) == 0


def test_holdem_betting_history_keeps_rounds_separate():
    betting = Betting(max_raises=2).apply(BET_HALF).apply(CALL)
    betting = betting.deal_board().apply(CALL).apply(CALL)
    public = HoldemPublicState(betting=betting, board=(51, 47, 22, 6, 3))

    features = encode_holdem_public(public)
    history = features[HISTORY_OFFSET - BOARD_OFFSET :].reshape(
        NUM_BETTING_ROUNDS, MAX_ACTIONS_PER_ROUND, 2
    )

    assert np.count_nonzero(history[:2]) == 0
    np.testing.assert_allclose(history[2, :2], [[1.0, 0.1], [1.0, 0.1]])
    np.testing.assert_allclose(history[3, :2], [[1.0, 0.0], [1.0, 0.0]])


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


def test_holdem_self_play_records_a_quantizable_policy_target_per_public_state():
    """``for beta in G: add {beta, pi_bar(beta)} to D_pi`` — every state, not the root."""
    public = HoldemPublicState(betting=Betting(), board=(51, 47, 22, 6))
    space = TurnEndgameSpace(public.board)
    tree = build_turn_tree(public, depth_limit=1)
    examples = collect_trajectory(
        ZeroLeafValues(),
        space,
        public,
        HoldemSelfPlayConfig(search_iterations=2, river_iterations=2),
        np.random.default_rng(0),
    )

    policies = examples[0].policies
    assert 1 < len(policies) <= len(tree.decision_nodes())
    assert {policy.agent_index for policy in policies} == {0, 1}
    for policy in policies:
        assert policy.target.shape == (NUM_COMBOS, 9)
        np.testing.assert_allclose(policy.target.sum(axis=-1), 1.0)
    assert set(np.flatnonzero(policies[0].legal_mask)) == set(public.legal_actions())


def test_policy_targets_are_dropped_when_no_policy_network_consumes_them():
    public = HoldemPublicState(betting=Betting(), board=(51, 47, 22, 6))
    examples = collect_trajectory(
        ZeroLeafValues(),
        TurnEndgameSpace(public.board),
        public,
        HoldemSelfPlayConfig(
            search_iterations=2, river_iterations=2, policy_targets=False
        ),
        np.random.default_rng(0),
    )

    assert all(example.policies == () for example in examples)


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
            exploration=0.0,
        ),
        rng,
        reach=reach,
    )

    # Two layers per street: each subgame's root, then the pre-deal belief
    # state it stopped in front of, back-filled once the next one is solved.
    # Hence flop, turn, flop-again, river, turn-again.
    assert [int(example.mask.sum()) for example in examples] == [
        1176,
        1128,
        1176,
        1081,
        1128,
    ]


@pytest.mark.parametrize(
    ("board_cards", "expected_rounds", "expected_examples"),
    # 2n - 1 labels from n betting rounds: a root apiece, plus a pre-deal one
    # for every deal the trajectory crosses.
    [(3, 3, 5), (4, 2, 3), (5, 1, 1)],
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


def test_holdem_self_play_defaults_to_algorithm_two_settings():
    config = HoldemSelfPlayConfig()
    # "t_warm = 0 and pi_0 is uniform if no warm start" — the branch of
    # INITIALIZE_POLICY this repo runs by default.
    assert config.warm_start_iterations == 0
    # Appendix E: "for all experiments we set the probability to explore a
    # random action to eps = 25%".
    assert config.exploration == 0.25
    # Algorithm 2 is *Linear* CFR-D: regret weighted by t, both averages
    # weighted by t+1, and no separate discounting of the averages on top.
    assert config.cfr == CFRConfig.linear_cfr_d()
    assert (config.cfr.regret_alpha, config.cfr.regret_beta) == (1.0, 1.0)
    assert config.cfr.strategy_gamma is None
    assert config.cfr.linear_averaging and config.cfr.average_offset == 1


def test_descent_iteration_is_sampled_proportionally_to_t():
    """``t_sample ~ linear{t_warm+1, ..., T}``, not uniform."""
    rng = np.random.default_rng(0)
    drawn = np.array([sample_iteration(0, 4, rng) for _ in range(20_000)])

    assert set(np.unique(drawn)) == {1, 2, 3, 4}
    frequencies = np.array([(drawn == t).mean() for t in (1, 2, 3, 4)])
    np.testing.assert_allclose(frequencies, np.array([1, 2, 3, 4]) / 10, atol=0.01)


def test_sampled_iteration_never_precedes_the_warm_start():
    rng = np.random.default_rng(0)
    drawn = {sample_iteration(6, 8, rng) for _ in range(200)}

    assert drawn == {7, 8}


def _deterministic_profile(tree, space):
    """A strategy list that always plays the first legal action."""
    strategies = [None] * tree.num_nodes
    for node in tree.decision_nodes():
        strategy = np.zeros((space.num_hands, node.num_actions))
        strategy[:, 0] = 1.0
        strategies[node.index] = strategy
    return strategies


def test_sample_leaf_explores_with_one_player_not_both():
    """SAMPLE_LEAF mixes eps of uniform into i* only; 1 - i* follows pi exactly."""
    public = HoldemPublicState(betting=Betting(), board=(51, 47, 22, 6))
    space = TurnEndgameSpace(public.board)
    tree = build_turn_tree(public, depth_limit=1)
    strategies = _deterministic_profile(tree, space)

    profile = exploration_profile(tree, space, strategies, explorer=0, epsilon=0.25)

    for node in tree.decision_nodes():
        played = profile[node.public]
        if node.player == 0:
            expected = 0.75 * strategies[node.index] + 0.25 / node.num_actions
            np.testing.assert_allclose(played, expected)
        else:
            np.testing.assert_array_equal(played, strategies[node.index])
        np.testing.assert_allclose(played.sum(axis=1), 1.0)


def test_sample_leaf_dilutes_one_players_range_and_leaves_the_others_alone():
    """The consequence of exploring for i* only: one range widens, not both."""
    public = HoldemPublicState(betting=Betting(), board=(51, 47, 22, 6))
    space = TurnEndgameSpace(public.board)
    tree = build_turn_tree(public, depth_limit=1)
    strategies = _deterministic_profile(tree, space)
    reach = space.initial_reach()
    exploring = HoldemSelfPlayConfig(exploration=0.25)
    on_policy = HoldemSelfPlayConfig(exploration=0.0)

    explorers = set()
    for seed in range(30):
        _, leaf_reach = sample_leaf(
            tree, space, strategies, reach, exploring, np.random.default_rng(seed)
        )
        # Both ranges lose the combos the dealt river uses, so the two masses
        # are only comparable to each other — and one of them is strictly
        # smaller, because exactly one player deviated from the deterministic
        # profile with probability eps.
        masses = leaf_reach.sum(axis=1)
        assert masses.min() < masses.max()
        explorers.add(int(np.argmin(masses)))

    assert explorers == {0, 1}  # i* ~ unif{1, N}, redrawn every subgame

    for seed in range(5):
        _, leaf_reach = sample_leaf(
            tree, space, strategies, reach, on_policy, np.random.default_rng(seed)
        )
        masses = leaf_reach.sum(axis=1)
        assert masses[0] == pytest.approx(masses[1])


def test_initialize_policy_is_uniform_and_unweighted_without_a_policy_network():
    """"t_warm = 0 and pi_0 is uniform if no warm start"."""
    public = HoldemPublicState(betting=Betting(), board=(51, 47, 22, 6))
    space = TurnEndgameSpace(public.board)
    tree = build_turn_tree(public, depth_limit=1)

    assert (
        initialize_policy(
            tree, space, space.initial_reach(), HoldemSelfPlayConfig(), None
        )
        is None
    )
    # A network on its own is not enough: t_warm has to be asked for.
    assert (
        initialize_policy(
            tree,
            space,
            space.initial_reach(),
            HoldemSelfPlayConfig(warm_start_iterations=0),
            HoldemPolicyNet(HoldemPolicyNetConfig(hidden_dim=8, num_hidden_layers=1)),
        )
        is None
    )


def test_initialize_policy_warm_starts_every_decision_node_from_theta_pi():
    public = HoldemPublicState(betting=Betting(), board=(51, 47, 22, 6))
    space = TurnEndgameSpace(public.board)
    tree = build_turn_tree(public, depth_limit=1)
    policy_net = HoldemPolicyNet(HoldemPolicyNetConfig(hidden_dim=8, num_hidden_layers=1))
    config = HoldemSelfPlayConfig(warm_start_iterations=7)

    warm = initialize_policy(tree, space, space.initial_reach(), config, policy_net)

    assert warm.iterations == 7
    assert all(warm.strategies[node.index] is not None for node in tree.decision_nodes())
    for node in tree.decision_nodes():
        strategy = warm.strategies[node.index]
        assert strategy.shape == (space.num_hands, node.num_actions)
        np.testing.assert_allclose(strategy.sum(axis=1), 1.0)
        # Each node was asked about its own belief state, so the ranges the
        # initialiser carried down are the ones pi_0 itself produces.
        assert warm.reaches[node.index].shape == (2, space.num_hands)

    # Regret matching on the seeded regrets has to give pi_0 straight back, or
    # the first CFR iteration throws the warm start away.
    solver = SubgameSolver(
        tree, leaf_value_fn=ZeroLeafValues(), config=config.cfr, space=space
    )
    solver.solve(reach=space.initial_reach(), iterations=7, warm_start=warm)
    assert solver.iteration == 7  # t starts at t_warm, so this ran nothing
    for node in tree.decision_nodes():
        np.testing.assert_allclose(
            solver.current_strategy(node), warm.strategies[node.index]
        )
        # The average is reach-weighted, so it only reproduces pi_0 for the
        # hands that can actually be held here — the board blocks the rest, and
        # a zero-reach row falls back to uniform as it always does.
        held = warm.reaches[node.index][node.player] > 0.0
        np.testing.assert_allclose(
            solver.average_strategy(node)[held], warm.strategies[node.index][held]
        )


def test_root_value_average_follows_the_linear_recursion():
    """``v <- (t/(t+2))*v + (2/(t+2))*EV(pi_t)``, seeded with ``EV(pi_0)``."""
    public = HoldemPublicState(betting=Betting(), board=(51, 47, 22, 6))
    space = TurnEndgameSpace(public.board)
    tree = build_turn_tree(public, depth_limit=None)
    reach = space.initial_reach()
    solver = SubgameSolver(tree, config=CFRConfig.linear_cfr_d(), space=space)

    solver.solve(
        reach=reach, iterations=6, store_iteration_values=True, seed_root_value=True
    )

    replayed = SubgameSolver(tree, config=CFRConfig.linear_cfr_d(), space=space)
    expected = replayed.evaluate(reach, average=False)  # COMPUTE_EV(G, pi_0)
    for t, values in enumerate(solver.iteration_values, start=1):
        expected = (t / (t + 2)) * expected + (2 / (t + 2)) * values

    np.testing.assert_allclose(solver.root_values(), expected)


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

def test_cached_encodings_cannot_be_corrupted_by_a_caller():
    """``encode_public`` and ``deal_mask`` hand out shared arrays.

    Both are cached because a depth-limited solve asks for the same public
    states and the same dealt cards on every CFR iteration.  That makes the
    returned arrays shared, so they must be read-only — a caller that scaled one
    in place would silently poison every later solve rather than fail here.
    """
    from paradigm_b.holdem.engine.space import TurnEndgameSpace as Space

    public = HoldemPublicState(betting=Betting(starting_pot=20, stack=100), board=(51, 47, 22, 6))
    features = encode_holdem_public(public)
    assert not features.flags.writeable
    with pytest.raises(ValueError):
        features[0] = 5.0
    # The cache returns the same object, so identity is the point of the test.
    assert encode_holdem_public(public) is features

    mask = Space((51, 47, 22, 6)).deal_mask(34)
    assert not mask.flags.writeable
    with pytest.raises(ValueError):
        mask[0] = 7.0

    # And the values still come out right after all that poking.
    np.testing.assert_array_equal(encode_holdem_public(public), features)


def _two_pass_forward(net, features, possible):
    """The two-trunk-pass forward, kept here as what the fused one must match.

    Deliberately a test-local copy rather than a production code path: it exists
    only to pin the fused implementation, so keeping it in the module would be
    dead weight that someone would eventually have to wonder about.
    """
    from paradigm_b.holdem.net.features import (
        BOARD_OFFSET, CARD_SET_DIM, NUM_CARD_SETS, POT_CHIPS_INDEX, RANGE_DIM,
        SCALAR_OFFSET,
    )

    batch = features.shape[0]
    ranges = features[..., :RANGE_DIM].reshape(-1, 2, NUM_COMBOS)
    pot = features[..., POT_CHIPS_INDEX].reshape(-1, 1, 1) * net.pot_scale
    board = features[..., BOARD_OFFSET:SCALAR_OFFSET].reshape(
        batch, NUM_CARD_SETS, CARD_SET_DIM
    )

    def one(player):
        index = torch.full((batch,), player, device=features.device, dtype=torch.long)
        encoded = torch.cat(
            (
                features[..., :RANGE_DIM],
                net.board_embedding(board).reshape(batch, -1),
                features[..., SCALAR_OFFSET:],
                net.agent_embedding(index),
            ),
            dim=-1,
        )
        return net.head(net.trunk(encoded))

    raw = torch.stack([one(player) for player in range(2)], dim=1)
    values = raw * possible.unsqueeze(1) * pot
    excess = (ranges * values).sum(dim=(-2, -1), keepdim=True)
    return values - 0.5 * excess * possible.unsqueeze(1)


def test_fused_forward_is_algebraically_the_two_pass_forward():
    """One trunk pass over 2B rows against two passes over B.

    The two are the *same arithmetic on the same inputs*, but not bit-identical:
    torch picks a different GEMM tiling for a 2B-row matrix than a B-row one, and
    float32 addition is not associative.  So the exactness claim is made in
    float64, where the accumulation-order difference falls below double
    precision -- that is what actually establishes the algebra is unchanged.
    Float32 is then only asked to agree to float32's own accuracy.
    """
    from paradigm_b.holdem.net.features import INPUT_DIM

    config = HoldemValueNetConfig(
        hidden_dim=32, num_residual_blocks=2, card_embedding_dim=8
    )
    # Relative, because the head's output is scaled by the pot and so lands in
    # the thousands -- an absolute bound would be a bound on the pot, not on the
    # arithmetic.  Both limits sit many orders below the ~1e-1 relative error a
    # crossed player pair or a transposed reshape would produce.
    for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 1e-4)):
        torch.manual_seed(0)
        net = HoldemValueNet(config).eval().to(dtype)
        features = torch.randn(5, INPUT_DIM, dtype=dtype)
        possible = (torch.rand(5, NUM_COMBOS) > 0.2).to(dtype)

        with torch.no_grad():
            fused = net(features, possible)
            reference = _two_pass_forward(net, features, possible)

        assert fused.shape == (5, 2, NUM_COMBOS)
        relative = (
            (fused - reference).abs().max() / reference.abs().max()
        ).item()
        assert relative < tolerance, f"{dtype}: relative error {relative:.2e}"


def test_fused_forward_keeps_the_players_distinct_and_ordered():
    """A transposed reshape would swap the two players and still look sane."""
    from paradigm_b.holdem.net.features import INPUT_DIM

    torch.manual_seed(1)
    net = HoldemValueNet(
        HoldemValueNetConfig(hidden_dim=32, num_residual_blocks=2, card_embedding_dim=8)
    ).eval()
    features = torch.randn(4, INPUT_DIM)
    possible = torch.ones(4, NUM_COMBOS)

    with torch.no_grad():
        values = net(features, possible)
    # The agent embedding is the only asymmetry, so the two players' rows must
    # differ -- if they matched, the embedding was not reaching the trunk.
    assert not torch.allclose(values[:, 0], values[:, 1])


def test_fused_forward_gradients_match_two_passes():
    """The fused path is used in training too, so backward must agree as well.

    In float64, for the same reason the forward comparison is: the two differ
    only by float32 accumulation order, and asserting that away in double
    precision is what shows the gradients are the same gradients.
    """
    from paradigm_b.holdem.net.features import INPUT_DIM

    config = HoldemValueNetConfig(
        hidden_dim=32, num_residual_blocks=2, card_embedding_dim=8
    )
    features = torch.randn(3, INPUT_DIM, dtype=torch.float64)
    possible = torch.ones(3, NUM_COMBOS, dtype=torch.float64)
    target = torch.randn(3, 2, NUM_COMBOS, dtype=torch.float64)

    grads = []
    for forward in (lambda n: n(features, possible),
                    lambda n: _two_pass_forward(n, features, possible)):
        torch.manual_seed(2)
        net = HoldemValueNet(config).to(torch.float64)
        loss = ((forward(net) - target) ** 2).mean()
        loss.backward()
        grads.append([p.grad.clone() for p in net.parameters()])

    for fused, reference in zip(*grads):
        torch.testing.assert_close(fused, reference, rtol=1e-7, atol=1e-6)
