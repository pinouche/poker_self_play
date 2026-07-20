"""lambda-return arithmetic along a single player's decision chain."""

import pytest

from training.returns import lambda_returns, monte_carlo_returns, n_step_returns


def test_terminal_step_receives_the_terminal_reward():
    assert lambda_returns([0.4], terminal_reward=1.0, gamma=0.99, lam=0.95) == [1.0]


def test_lambda_one_and_gamma_one_is_the_monte_carlo_outcome():
    values = [0.9, -0.3, 0.5, 0.1]
    targets = lambda_returns(values, terminal_reward=-1.0, gamma=1.0, lam=1.0)
    assert targets == pytest.approx([-1.0, -1.0, -1.0, -1.0])


def test_lambda_zero_bootstraps_from_the_next_own_decision():
    values = [0.0, 0.5, -0.25]
    gamma = 0.9
    targets = lambda_returns(values, terminal_reward=1.0, gamma=gamma, lam=0.0)
    assert targets[-1] == pytest.approx(1.0)
    assert targets[1] == pytest.approx(gamma * values[2])
    assert targets[0] == pytest.approx(gamma * values[1])


def test_intermediate_lambda_blends_bootstrap_and_outcome():
    values = [0.0, 0.5]
    gamma, lam = 1.0, 0.5
    targets = lambda_returns(values, terminal_reward=1.0, gamma=gamma, lam=lam)
    # G_0 = (1 - lam) * V(s_1) + lam * G_1
    assert targets[0] == pytest.approx(0.5 * 0.5 + 0.5 * 1.0)


def test_discounting_applies_along_the_chain():
    values = [0.0, 0.0, 0.0]
    targets = lambda_returns(values, terminal_reward=1.0, gamma=0.5, lam=1.0)
    assert targets == pytest.approx([0.25, 0.5, 1.0])


def test_first_value_is_never_used():
    a = lambda_returns([99.0, 0.5, 0.1], 1.0, 0.99, 0.5)
    b = lambda_returns([-99.0, 0.5, 0.1], 1.0, 0.99, 0.5)
    assert a == pytest.approx(b)


def test_empty_chain_returns_nothing():
    assert lambda_returns([], 1.0, 0.99, 0.95) == []


def test_mismatched_reward_length_is_rejected():
    with pytest.raises(ValueError):
        lambda_returns([0.0, 0.0], 1.0, 0.99, 0.95, rewards=[0.0])


def test_monte_carlo_returns_discount_backwards_from_the_terminal_reward():
    assert monte_carlo_returns(3, 1.0, 0.5) == pytest.approx([0.25, 0.5, 1.0])


def test_n_step_returns_reach_the_terminal_reward_within_the_horizon():
    values = [0.2, 0.4, 0.6]
    targets = n_step_returns(values, terminal_reward=1.0, gamma=1.0, n=10)
    assert targets == pytest.approx([1.0, 1.0, 1.0])
