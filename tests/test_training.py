"""Replay buffer behaviour and a training smoke test."""

import numpy as np
import pytest
import torch

from config import Config, ModelConfig
from environment.state import NUM_ACTIONS
from model.network import build_network
from representation.observation_encoder import ObservationEncoder
from training.replay_buffer import ReplayBuffer, Transition
from training.self_play import SelfPlayWorker
from training.trainer import Trainer


def small_config() -> Config:
    cfg = Config()
    cfg.model = ModelConfig(hidden_dim=32, num_residual_blocks=2, head_hidden=32)
    cfg.train.batch_size = 64
    cfg.train.min_buffer_before_training = 64
    return cfg


def make_transition(obs_dim, rng, action=0):
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    mask[[0, 2, 9]] = 1.0
    policy = mask / mask.sum()
    return Transition(
        observation=rng.normal(size=obs_dim).astype(np.float32),
        legal_action_mask=mask,
        action=action,
        reward=0.0,
        next_observation=rng.normal(size=obs_dim).astype(np.float32),
        next_legal_action_mask=mask,
        done=False,
        player_perspective=1,
        old_policy=policy.astype(np.float32),
        q_target=0.5,
        value=0.25,
    )


# --- replay buffer ---------------------------------------------------------
def test_buffer_roundtrip_preserves_fields():
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(capacity=10, observation_dim=8, num_actions=NUM_ACTIONS)
    transition = make_transition(8, rng, action=2)
    buffer.add(transition)

    batch = buffer.get([0])
    assert len(batch) == 1
    assert batch.actions[0] == 2
    assert batch.player_perspectives[0] == 1
    assert batch.q_targets[0] == pytest.approx(0.5)
    np.testing.assert_allclose(
        batch.observations[0], transition.observation, atol=1e-2  # float16 storage
    )
    np.testing.assert_allclose(batch.old_policies[0], transition.old_policy, atol=1e-6)


def test_buffer_evicts_oldest_when_full():
    rng = np.random.default_rng(1)
    buffer = ReplayBuffer(capacity=4, observation_dim=8, num_actions=NUM_ACTIONS)
    for i in range(10):
        buffer.add(make_transition(8, rng, action=i % NUM_ACTIONS))
    assert len(buffer) == 4
    assert buffer.total_added == 10


def test_sampling_from_an_empty_buffer_is_an_error():
    buffer = ReplayBuffer(capacity=4, observation_dim=8, num_actions=NUM_ACTIONS)
    with pytest.raises(ValueError):
        buffer.sample(2)


def test_buffer_can_omit_next_observations():
    rng = np.random.default_rng(2)
    buffer = ReplayBuffer(
        capacity=4, observation_dim=8, num_actions=NUM_ACTIONS, store_next_obs=False
    )
    buffer.add(make_transition(8, rng))
    assert buffer.sample(1).next_observations is None


# --- training smoke test ---------------------------------------------------
def test_training_on_one_thousand_transitions_is_numerically_stable():
    cfg = small_config()
    network = build_network(cfg)
    encoder = ObservationEncoder(cfg.obs)
    worker = SelfPlayWorker(cfg, network, encoder, seed=3)
    buffer = ReplayBuffer(
        capacity=5000,
        observation_dim=encoder.observation_dim,
        num_actions=NUM_ACTIONS,
    )

    while len(buffer) < 1000:
        transitions, _ = worker.generate(40)
        buffer.extend(transitions)
    assert len(buffer) >= 1000

    trainer = Trainer(cfg, network, device="cpu")
    rng = np.random.default_rng(0)

    losses = []
    for _ in range(40):
        metrics = trainer.train_step(buffer.sample(cfg.train.batch_size, rng))
        losses.append(metrics["loss"])
        for name, value in metrics.items():
            assert np.isfinite(value), f"{name} became non-finite"

    for parameter in network.parameters():
        assert torch.isfinite(parameter).all()

    early = float(np.mean(losses[:10]))
    late = float(np.mean(losses[-10:]))
    # Stable or improving; a diverging run is the failure we care about.
    assert late <= early + 0.25
    assert all(np.isfinite(losses))


def test_q_loss_falls_when_overfitting_a_single_batch():
    cfg = small_config()
    cfg.train.learning_rate = 3e-3
    network = build_network(cfg)
    encoder = ObservationEncoder(cfg.obs)
    worker = SelfPlayWorker(cfg, network, encoder, seed=4)
    buffer = ReplayBuffer(
        capacity=2000, observation_dim=encoder.observation_dim, num_actions=NUM_ACTIONS
    )
    while len(buffer) < 256:
        transitions, _ = worker.generate(30)
        buffer.extend(transitions)

    trainer = Trainer(cfg, network, device="cpu")
    batch = buffer.sample(128, np.random.default_rng(0))

    first = trainer.train_step(batch)["q_loss"]
    for _ in range(60):
        last = trainer.train_step(batch)["q_loss"]
    assert last < first


def test_train_iteration_waits_for_a_minimum_buffer():
    cfg = small_config()
    cfg.train.min_buffer_before_training = 10_000
    network = build_network(cfg)
    encoder = ObservationEncoder(cfg.obs)
    buffer = ReplayBuffer(
        capacity=1000, observation_dim=encoder.observation_dim, num_actions=NUM_ACTIONS
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        buffer.add(make_transition(encoder.observation_dim, rng))

    trainer = Trainer(cfg, network, device="cpu")
    assert trainer.train_iteration(buffer, num_updates=2) == {}


def test_entropy_weight_pushes_the_policy_toward_uniform():
    """A large entropy bonus must raise the measured policy entropy."""
    cfg = small_config()
    cfg.train.entropy_weight = 5.0
    cfg.train.q_weight = 0.0
    cfg.train.policy_weight = 0.0
    cfg.train.learning_rate = 3e-3

    network = build_network(cfg)
    encoder = ObservationEncoder(cfg.obs)
    worker = SelfPlayWorker(cfg, network, encoder, seed=5)
    buffer = ReplayBuffer(
        capacity=2000, observation_dim=encoder.observation_dim, num_actions=NUM_ACTIONS
    )
    while len(buffer) < 256:
        transitions, _ = worker.generate(30)
        buffer.extend(transitions)

    trainer = Trainer(cfg, network, device="cpu")
    batch = buffer.sample(128, np.random.default_rng(0))
    first = trainer.train_step(batch)["entropy"]
    for _ in range(40):
        last = trainer.train_step(batch)["entropy"]
    assert last > first


def test_reference_policy_modes_both_run():
    cfg = small_config()
    network = build_network(cfg)
    encoder = ObservationEncoder(cfg.obs)
    worker = SelfPlayWorker(cfg, network, encoder, seed=6)
    buffer = ReplayBuffer(
        capacity=2000, observation_dim=encoder.observation_dim, num_actions=NUM_ACTIONS
    )
    while len(buffer) < 128:
        transitions, _ = worker.generate(30)
        buffer.extend(transitions)
    batch = buffer.sample(64, np.random.default_rng(0))

    for mode in ("current", "behavior"):
        cfg.train.reference_policy = mode
        metrics = Trainer(cfg, build_network(cfg), device="cpu").train_step(batch)
        assert np.isfinite(metrics["loss"])

    cfg.train.reference_policy = "nonsense"
    with pytest.raises(ValueError):
        Trainer(cfg, build_network(cfg), device="cpu").train_step(batch)
