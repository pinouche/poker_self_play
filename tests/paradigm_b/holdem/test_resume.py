"""Stopping a long run and continuing it must not restart the experiment.

A 12-hour generation run that dies at hour 11 has to be resumable, and the
thing that makes that non-trivial is everything *besides* the weights: Adam's
moment estimates, the replay buffer, the spend the budgets are measured
against, and the RNG.  Drop any one of them and the resumed run is a different
experiment that happens to share a checkpoint.

The load-bearing test here is :func:`test_resuming_matches_an_uninterrupted_run`
— split a run in two and it must end where the whole one did, exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from paradigm_b.holdem.arm2_iterative.student import (
    OnlineStudentConfig,
    fit_online_student,
)
from paradigm_b.holdem.arms_common.storage import load_run_state, save_run_state
from paradigm_b.holdem.data.sampling import SituationConfig
from paradigm_b.holdem.net.value_net import HoldemValueNet, HoldemValueNetConfig
from paradigm_b.holdem.arms_common.situations import StreetMix
from paradigm_b.holdem.selfplay import Buffer, Example, HoldemSelfPlayConfig

TINY_NET = HoldemValueNetConfig(hidden_dim=32, num_residual_blocks=1, card_embedding_dim=8)


def tiny_config(**overrides) -> OnlineStudentConfig:
    base = dict(
        trajectories_per_iteration=1,
        updates_per_iteration=2,
        buffer_size=64,
        batch_size=4,
        value_net=TINY_NET,
        situations=SituationConfig(board_cards=5),
        # River only, so a trajectory is exactly one label and the budgets
        # below fall on iteration boundaries.  The default mixture yields
        # 1, 3 or 5 labels depending on the street it draws.
        street_mix=StreetMix(river=1.0, turn=0.0, flop=0.0),
        preflop_start=False,
        self_play=HoldemSelfPlayConfig(
            search_iterations=2, river_iterations=2, exploration=0.0
        ),
        seed=0,
        actors=0,
    )
    base.update(overrides)
    return OnlineStudentConfig(**base)


def weights(net) -> dict:
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def test_resuming_matches_an_uninterrupted_run(tmp_path):
    """Two halves and one whole must end in the same place, to the bit.

    This is the only check that covers the pieces individually easy to forget —
    the optimiser moments and the RNG stream — because either one being reset
    shows up as diverged weights and nothing else.

    The budgets are deliberately in the ratio the loop actually consumes them
    (one river label and two updates per iteration), so the split falls on an
    iteration boundary.  Halve only the labels and the first run's *update*
    budget binds early, its last iterations take no gradient steps, and the two
    runs interleave generation and training differently — which shows up as
    diverged weights and looks exactly like a broken resume.
    """
    torch.manual_seed(0)
    whole = fit_online_student(tiny_config(), label_budget=6, update_budget=12)

    torch.manual_seed(0)
    first = fit_online_student(
        tiny_config(state_every=1),
        label_budget=3,
        update_budget=6,
        state_path=tmp_path / "state",
    )
    second = fit_online_student(
        tiny_config(resume_from=str(tmp_path / "state")),
        label_budget=6,
        update_budget=12,
    )

    assert second.spend.labels == whole.spend.labels
    assert second.spend.updates == whole.spend.updates
    for key, value in weights(whole.net).items():
        torch.testing.assert_close(weights(second.net)[key], value, rtol=0, atol=0)


def test_budgets_are_totals_not_increments(tmp_path):
    """A resumed run finishes the original budget rather than spending it again."""
    torch.manual_seed(0)
    first = fit_online_student(
        tiny_config(state_every=1),
        label_budget=3,
        update_budget=4,
        state_path=tmp_path / "state",
    )
    assert first.spend.labels >= 3

    second = fit_online_student(
        tiny_config(resume_from=str(tmp_path / "state")),
        label_budget=6,
        update_budget=8,
    )
    assert second.spend.labels == 6
    assert second.spend.updates == 8


def test_the_buffer_survives_with_its_eviction_order(tmp_path):
    """A wrapped buffer must come back oldest-first, not raw.

    ``purge_oldest`` reconstructs age from the write pointer, so a buffer that
    is saved as a raw dump and reloaded with the pointer at zero would purge
    the *newest* half.  Saving in oldest-first order is what makes that safe,
    and this pins it.
    """
    rng = np.random.default_rng(0)
    buffer = Buffer(8)
    for value in range(12):  # wraps: 12 examples into 8 slots
        buffer.add(
            [
                Example(
                    features=np.full(buffer.features.shape[1], value, dtype=np.float32),
                    mask=np.ones(buffer.masks.shape[1], dtype=np.float32),
                    values=np.full(buffer.targets.shape[1:], value, dtype=np.float32),
                )
            ]
        )
    assert buffer.size == 8 and buffer._next == 4

    net = HoldemValueNet(TINY_NET)
    optimiser = torch.optim.Adam(net.parameters())
    config = tiny_config(buffer_size=8)
    save_run_state(
        tmp_path / "state",
        config=config,
        spend=type("S", (), {"to_dict": lambda self: {}})(),
        iteration=1,
        trajectory_id=1,
        rng=rng,
        net=net,
        optimiser=optimiser,
        buffer=buffer,
        initial_state=weights(net),
    )

    restored = Buffer(8)
    load_run_state(
        tmp_path / "state",
        config=config,
        net=net,
        optimiser=optimiser,
        buffer=restored,
        rng=np.random.default_rng(1),
    )
    # Oldest (4) first, newest (11) last -- the order age is read off.
    assert [float(row[0]) for row in restored.features[: restored.size]] == list(
        range(4, 12)
    )
    assert restored._next == 0, "a full buffer's pointer wraps back to the start"


def test_resume_refuses_a_different_abstraction(tmp_path):
    """Action ids are positional, so a buffer is only readable under its own sizes."""
    net = HoldemValueNet(TINY_NET)
    optimiser = torch.optim.Adam(net.parameters())
    config = tiny_config()
    save_run_state(
        tmp_path / "state",
        config=config,
        spend=type("S", (), {"to_dict": lambda self: {}})(),
        iteration=1,
        trajectory_id=0,
        rng=np.random.default_rng(0),
        net=net,
        optimiser=optimiser,
        buffer=Buffer(64),
        initial_state=weights(net),
    )

    wider = tiny_config(
        situations=SituationConfig(board_cards=5, bet_fractions=(0.5, 1.0))
    )
    with pytest.raises(ValueError, match="refusing to resume"):
        load_run_state(
            tmp_path / "state",
            config=wider,
            net=net,
            optimiser=optimiser,
            buffer=Buffer(64),
            rng=np.random.default_rng(0),
        )
