"""Played-hand local best response regressions."""

import numpy as np

from paradigm_b.holdem.arms_common import lbr_match
from paradigm_b.holdem.engine.betting import CALL


def test_play_hand_reveals_the_flop_then_each_remaining_street(monkeypatch):
    """The advanced betting round names the board that is now due."""
    observed = []

    class RecordingAgent:
        def __init__(self, *args, **kwargs):
            self.space = object()
            self.solves = 0

        def act(self):
            return CALL, 2

        def observe(self, action, total):
            pass

        def observe_board(self, cards):
            observed.append(tuple(cards))

    class CallingLbr:
        def __init__(self, *args, **kwargs):
            self.probes = 0

        def act(self, real, board, agent):
            return CALL, real.contributions[1 - real.to_move()]

    monkeypatch.setattr(lbr_match, "ResolvingAgent", RecordingAgent)
    monkeypatch.setattr(lbr_match, "LbrPlayer", CallingLbr)

    outcome = lbr_match.play_hand(object(), 0, np.random.default_rng(0))

    assert [len(cards) for cards in observed] == [3, 1, 1]
    assert tuple(card for cards in observed for card in cards) == outcome.board