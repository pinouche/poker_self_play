"""Network shapes, checkpointing, and action masking."""

import os
import tempfile

import numpy as np
import pytest
import torch

from config import Config, ModelConfig, ObsConfig
from environment.state import NUM_ACTIONS
from model.network import (
    build_network,
    load_checkpoint,
    masked_log_softmax,
    masked_softmax,
    save_checkpoint,
)
from representation.action_encoder import (
    apply_temperature,
    masked_softmax as np_masked_softmax,
    probs_to_dict,
    q_values_to_dict,
    sample_action,
)
from representation.observation_encoder import ObservationEncoder, build_spec
from training.policy_improvement import improved_policy_np


def small_config() -> Config:
    cfg = Config()
    cfg.model = ModelConfig(hidden_dim=32, num_residual_blocks=2, head_hidden=32)
    return cfg


# --- shapes ----------------------------------------------------------------
@pytest.mark.parametrize("batch_size", [1, 7, 64])
def test_forward_returns_logits_and_q_values_per_action(batch_size):
    cfg = small_config()
    network = build_network(cfg)
    spec = build_spec(cfg.obs)

    observations = torch.randn(batch_size, spec.total_dim)
    policy_logits, q_values = network(observations)

    assert policy_logits.shape == (batch_size, NUM_ACTIONS)
    assert q_values.shape == (batch_size, NUM_ACTIONS)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(q_values).all()


@pytest.mark.parametrize(
    "reward_mode,expected_scale", [("binary", 1.0), ("normalized_chip_return", 2.0)]
)
def test_q_head_is_bounded_by_the_reward_range(reward_mode, expected_scale):
    """The squash must reach the full return range, not an arbitrary +-1.

    Three-handed normalised chip returns reach +2, so a plain tanh would make
    the largest legitimate wins unrepresentable by the critic.
    """
    cfg = small_config()
    cfg.env.reward_mode = reward_mode
    network = build_network(cfg)
    assert network.q_head.bounded
    assert network.q_head.scale == expected_scale

    _, q_values = network(torch.randn(64, network.spec.total_dim) * 50)
    assert q_values.min() >= -expected_scale
    assert q_values.max() <= expected_scale


def test_unbounded_reward_modes_get_a_linear_q_head():
    cfg = small_config()
    cfg.env.reward_mode = "bb_normalized"
    assert not build_network(cfg).q_head.bounded


def test_bounded_q_can_be_overridden_explicitly():
    cfg = small_config()
    cfg.model.bounded_q = False
    network = build_network(cfg)
    _, q_values = network(torch.randn(32, network.spec.total_dim))
    assert q_values.shape == (32, NUM_ACTIONS)  # simply must not be squashed


def test_observation_dim_mismatch_is_rejected():
    network = build_network(small_config())
    with pytest.raises(ValueError):
        network(torch.randn(4, network.spec.total_dim + 1))


def test_infer_accepts_a_single_observation():
    network = build_network(small_config())
    logits, q_values = network.infer(np.zeros(network.spec.total_dim, dtype=np.float32))
    assert logits.shape == (NUM_ACTIONS,)
    assert q_values.shape == (NUM_ACTIONS,)


# --- masking ---------------------------------------------------------------
def random_mask(rng, num_legal=None):
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    count = num_legal or rng.integers(1, NUM_ACTIONS + 1)
    mask[rng.choice(NUM_ACTIONS, size=count, replace=False)] = 1.0
    return mask


def test_illegal_actions_receive_zero_probability_torch():
    rng = np.random.default_rng(0)
    masks = np.stack([random_mask(rng) for _ in range(32)])
    logits = torch.randn(32, NUM_ACTIONS) * 10
    mask_tensor = torch.as_tensor(masks)

    probs = masked_softmax(logits, mask_tensor)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(32), atol=1e-5)
    assert (probs[mask_tensor == 0] == 0).all()

    log_probs = masked_log_softmax(logits, mask_tensor)
    assert (log_probs[mask_tensor == 0] < -1e6).all()


def test_illegal_actions_receive_zero_probability_numpy():
    rng = np.random.default_rng(1)
    for _ in range(50):
        mask = random_mask(rng)
        logits = rng.normal(size=NUM_ACTIONS) * 10
        probs = np_masked_softmax(logits, mask)
        assert probs[mask == 0].sum() == 0.0
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)


def test_improved_policy_is_zero_on_illegal_actions():
    rng = np.random.default_rng(2)
    for _ in range(50):
        mask = random_mask(rng)
        q_values = rng.normal(size=NUM_ACTIONS)
        reference = np_masked_softmax(rng.normal(size=NUM_ACTIONS), mask)
        probs = improved_policy_np(q_values, reference, mask, alpha=0.5, beta=0.5)
        assert probs[mask == 0].sum() == 0.0
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)


def test_improved_policy_prefers_higher_q_values():
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)
    reference = np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float32)
    q_values = np.linspace(-1.0, 1.0, NUM_ACTIONS)
    probs = improved_policy_np(q_values, reference, mask, alpha=0.5, beta=0.5)
    assert np.argmax(probs) == NUM_ACTIONS - 1
    assert np.all(np.diff(probs) > 0)  # monotone in Q when the reference is flat


def test_improved_policy_collapses_to_argmax_at_zero_temperature():
    mask = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 1], dtype=np.float32)
    q_values = np.array([0.1, 9.0, 0.5, 9.0, -0.3, 9.0, 9.0, 9.0, 9.0, 0.2])
    reference = np_masked_softmax(np.zeros(NUM_ACTIONS), mask)
    probs = improved_policy_np(q_values, reference, mask, 0.5, 0.5, temperature=0.0)
    assert probs[2] == 1.0  # best *legal* action, ignoring the illegal 9.0s
    assert probs.sum() == 1.0


def test_beta_keeps_the_improved_policy_near_the_reference():
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)
    q_values = np.linspace(-1.0, 1.0, NUM_ACTIONS)
    reference = np_masked_softmax(np.arange(NUM_ACTIONS, dtype=float), mask)

    near = improved_policy_np(q_values, reference, mask, alpha=0.01, beta=50.0)
    far = improved_policy_np(q_values, reference, mask, alpha=0.01, beta=0.01)
    assert np.abs(near - reference).sum() < np.abs(far - reference).sum()


def test_sampling_never_returns_an_illegal_action():
    import random

    rng = random.Random(0)
    np_rng = np.random.default_rng(3)
    for _ in range(200):
        mask = random_mask(np_rng)
        probs = np_masked_softmax(np_rng.normal(size=NUM_ACTIONS), mask)
        assert mask[sample_action(probs, rng)] == 1.0


def test_temperature_sharpens_and_flattens():
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)
    probs = np_masked_softmax(np.linspace(-2, 2, NUM_ACTIONS), mask)
    sharp = apply_temperature(probs, mask, 0.1)
    flat = apply_temperature(probs, mask, 10.0)
    assert sharp.max() > probs.max() > flat.max()
    assert sharp.sum() == pytest.approx(1.0, abs=1e-6)
    assert flat.sum() == pytest.approx(1.0, abs=1e-6)


# --- public API views ------------------------------------------------------
def test_public_dicts_report_zero_and_null_for_illegal_actions():
    mask = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    probs = np_masked_softmax(np.zeros(NUM_ACTIONS), mask)
    prob_dict = probs_to_dict(probs, mask)
    q_dict = q_values_to_dict(np.arange(NUM_ACTIONS, dtype=float), mask)

    assert prob_dict["CHECK"] == 0.0
    assert q_dict["CHECK"] is None
    assert q_dict["FOLD"] == 0.0
    assert sum(prob_dict.values()) == pytest.approx(1.0, abs=1e-6)


# --- checkpointing ---------------------------------------------------------
def test_checkpoint_roundtrip_preserves_outputs():
    cfg = small_config()
    network = build_network(cfg)
    observation = np.random.default_rng(0).normal(size=network.spec.total_dim).astype(np.float32)
    expected = network.infer(observation)

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "net.pt")
        save_checkpoint(path, network, cfg, extra={"iteration": 12})
        restored, restored_cfg, extra = load_checkpoint(path)

    assert extra["iteration"] == 12
    assert restored_cfg.obs == cfg.obs
    actual = restored.infer(observation)
    np.testing.assert_allclose(expected[0], actual[0], atol=1e-6)
    np.testing.assert_allclose(expected[1], actual[1], atol=1e-6)


def test_observation_spec_matches_the_encoder():
    cfg = ObsConfig()
    encoder = ObservationEncoder(cfg)
    network = build_network(small_config())
    assert network.spec.total_dim == encoder.observation_dim
    assert sum(encoder.spec.group_dims.values()) == encoder.observation_dim


# --- reward-scale invariance of the improvement operator -------------------
def test_improvement_operator_is_invariant_to_reward_rescaling():
    """alpha and beta must be dimensionless.

    Rescaling the reward (and therefore Q) must not change the improved policy,
    otherwise switching reward modes silently changes the operator temperature.
    """
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)
    reference = np_masked_softmax(np.linspace(-1, 1, NUM_ACTIONS), mask)
    q_unit = np.linspace(-1.0, 1.0, NUM_ACTIONS)

    base = improved_policy_np(q_unit, reference, mask, 0.5, 0.5, q_scale=1.0)
    for factor in (10.0, 50.0, 1000.0):
        scaled = improved_policy_np(
            q_unit * factor, reference, mask, 0.5, 0.5, q_scale=factor
        )
        np.testing.assert_allclose(base, scaled, atol=1e-9)


def test_unscaled_large_q_values_collapse_the_policy():
    """Documents the failure the q_scale correction prevents."""
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)
    reference = np_masked_softmax(np.zeros(NUM_ACTIONS), mask)
    q_big = np.linspace(-50.0, 50.0, NUM_ACTIONS)

    collapsed = improved_policy_np(q_big, reference, mask, 0.5, 0.5, q_scale=1.0)
    healthy = improved_policy_np(q_big, reference, mask, 0.5, 0.5, q_scale=50.0)

    assert collapsed.max() > 0.999          # effectively deterministic
    assert healthy.max() < 0.6              # still a distribution
    assert _entropy(healthy) > _entropy(collapsed) * 50


def _entropy(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, None)
    return float(-(p * np.log(p)).sum())


def test_reward_scale_matches_the_reward_mode():
    from config import Config, reward_scale

    cfg = Config()
    assert reward_scale(cfg.env) == 1.0  # normalized_chip_return
    cfg.env.reward_mode = "binary"
    assert reward_scale(cfg.env) == 1.0
    cfg.env.reward_mode = "bb_normalized"
    assert reward_scale(cfg.env) == cfg.env.starting_stack / cfg.env.big_blind
    cfg.env.reward_mode = "chip_return"
    assert reward_scale(cfg.env) == float(cfg.env.starting_stack)
