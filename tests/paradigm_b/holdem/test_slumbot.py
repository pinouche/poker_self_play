"""The Slumbot harness, played against the local stand-in rather than the server.

Everything here runs on :class:`~.slumbot_fake.FakeSlumbotClient`, so no test
touches the network.  What is being asserted is the *harness*: that a session
counts what it played, that playing it on four threads counts the same things
as playing it on one, and that the concurrency does not quietly cost the
balanced seating a reported mbb/g depends on.
"""

from __future__ import annotations

import json
import threading

import pytest

from paradigm_b.holdem.arms_common.play import PlayConfig
from paradigm_b.holdem.arms_common.progress import (
    ProgressConfig,
    ProgressEvaluator,
    evaluate_against_slumbot,
)
from paradigm_b.holdem.arms_common.slumbot import (
    call_policy,
    fold_policy,
    play_session,
    play_session_parallel,
    split_hands,
)
from paradigm_b.holdem.arms_common.slumbot_fake import FakeSlumbotClient
from paradigm_b.holdem.net.value_net import HoldemValueNet, HoldemValueNetConfig

TINY_NET = HoldemValueNetConfig(hidden_dim=32, num_residual_blocks=1, card_embedding_dim=8)
# Two CFR iterations resolve nothing; this asserts the wiring, not the poker.
QUICK_PLAY = PlayConfig(search_iterations=2, river_iterations=2)


def seeded_clients():
    """A fresh fake per worker, seeded by worker so they deal different hands."""
    return lambda worker: FakeSlumbotClient(seed=worker)


def test_the_parallel_session_plays_exactly_the_hands_it_was_asked_for():
    summary = play_session_parallel(
        40, policy_factory=lambda _: call_policy, workers=4, client_factory=seeded_clients()
    )
    assert summary.hands == 40


def test_each_worker_gets_its_own_client_and_never_shares_a_token():
    """Two sessions sharing a token would be one player answering twice."""
    seen = []
    lock = threading.Lock()

    def client_factory(worker):
        client = FakeSlumbotClient(seed=worker)
        with lock:
            seen.append(client)
        return client

    play_session_parallel(
        24, policy_factory=lambda _: call_policy, workers=4, client_factory=client_factory
    )
    assert len(seen) == 4
    assert len({id(client) for client in seen}) == 4
    assert sum(client.hands_played for client in seen) == 24


def test_each_worker_gets_its_own_policy():
    """A re-solving policy carries a hand's belief state; sharing one corrupts it."""
    made = []

    def policy_factory(worker):
        made.append(worker)
        return call_policy

    play_session_parallel(
        24, policy_factory=policy_factory, workers=4, client_factory=seeded_clients()
    )
    assert sorted(made) == [0, 1, 2, 3]


def test_four_threads_count_what_the_same_four_sessions_count_alone():
    """The concurrency must not touch the chips.

    Four workers on seeds 0-3 deal exactly the hands four separate sequential
    sessions on seeds 0-3 deal, so the totals have to agree exactly.  If they
    do not, the shared bookkeeping has dropped or double-counted a hand — which
    is the failure mode threading actually has here, and the one a "roughly
    equal" assertion would let through.
    """
    parallel = play_session_parallel(
        16, policy_factory=lambda _: call_policy, workers=4, client_factory=seeded_clients()
    )
    alone = [
        play_session(4, call_policy, client=FakeSlumbotClient(seed=worker))
        for worker in range(4)
    ]
    assert parallel.hands == sum(summary.hands for summary in alone) == 16
    assert parallel.total_chips == sum(summary.total_chips for summary in alone)


def test_the_sequential_path_still_runs_without_a_thread():
    """``workers=1`` must not spawn anything, so failures keep their stack."""
    before = threading.active_count()
    play_session(8, call_policy, client=FakeSlumbotClient(seed=0))
    assert threading.active_count() == before


def test_a_worker_failure_stops_the_session_and_reaches_the_caller():
    class Boom(RuntimeError):
        pass

    def policy_factory(worker):
        def policy(state):
            if worker == 2:
                raise Boom("worker 2 fell over")
            return call_policy(state)

        return policy

    with pytest.raises(Boom):
        play_session_parallel(
            80, policy_factory=policy_factory, workers=4, client_factory=seeded_clients()
        )


def test_an_illegal_action_is_still_reported_rather_than_swallowed():
    """The fake rejects a fold with nothing to call; that must not pass silently."""

    with pytest.raises(RuntimeError, match="slumbot rejected"):
        play_session_parallel(
            8,
            policy_factory=lambda _: (lambda state: "f"),
            workers=2,
            client_factory=seeded_clients(),
        )


def test_the_log_holds_one_line_per_hand_with_every_index_once(tmp_path):
    """Lines interleave across workers, so each must carry its own identity."""
    path = tmp_path / "session.jsonl"
    play_session_parallel(
        40,
        policy_factory=lambda _: call_policy,
        workers=4,
        log_path=path,
        client_factory=seeded_clients(),
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 40
    assert sorted(row["hand"] for row in rows) == list(range(40))


def test_on_hand_sees_every_hand_exactly_once():
    seen = []
    play_session_parallel(
        40,
        policy_factory=lambda _: call_policy,
        workers=4,
        client_factory=seeded_clients(),
        on_hand=lambda result, summary: seen.append((result.hand, summary.hands)),
    )
    assert sorted(hand for hand, _ in seen) == list(range(40))
    # The running summary is handed out under the lock, so its count is the
    # number of hands recorded so far -- 1..40 with no repeats or gaps.
    assert sorted(count for _, count in seen) == list(range(1, 41))


@pytest.mark.parametrize("hands,workers", [(1000, 4), (200, 3), (40, 8), (7, 4), (2, 4)])
def test_the_split_covers_the_hands_and_keeps_the_chunks_even(hands, workers):
    split = split_hands(hands, workers)
    assert sum(count for _, count in split) == hands
    assert [start for start, _ in split] == [
        sum(c for _, c in split[:index]) for index in range(len(split))
    ]
    # At most one worker plays an odd number, and only when ``hands`` is odd --
    # otherwise the session ends up biased toward the big blind.
    odd = [count for _, count in split if count % 2]
    assert len(odd) == hands % 2


def test_the_split_degrades_rather_than_dealing_empty_chunks():
    assert split_hands(3, 8) == [(0, 3)]
    assert split_hands(0, 4) == []


def test_the_resolving_agent_survives_being_run_on_several_workers():
    """The real policy, four at a time, sharing one frozen network.

    This is the case the threading exists for and the one the trivial policies
    cannot exercise: each worker holds a belief state across a hand, and they
    all call ``forward`` on the same weights.  Four hands is enough to catch a
    shared-state bug — the agent raises rather than misplaying when its cursor
    and Slumbot's action string disagree.
    """
    net = HoldemValueNet(TINY_NET)
    net.eval()
    record = evaluate_against_slumbot(
        net,
        hands=4,
        play=QUICK_PLAY,
        client_factory=lambda worker: FakeSlumbotClient(seed=worker),
        workers=4,
    )
    assert record["slumbot_hands"] == 4.0
    assert record["slumbot_seconds"] > 0.0


def test_the_seats_stay_balanced_across_workers():
    """Every token starts in the big blind, so the split has to compensate."""
    seats = []
    play_session_parallel(
        60,
        policy_factory=lambda _: fold_policy,
        workers=4,
        client_factory=seeded_clients(),
        on_hand=lambda result, _summary: seats.append(result.client_pos),
    )
    assert abs(seats.count(0) - seats.count(1)) <= 1


def test_the_wall_clock_cadence_fires_on_the_clock_not_the_iteration_count():
    """``every_seconds`` is the knob a five-hour cadence actually needs.

    Iterations are not a fixed amount of work on the actor path, so the
    evaluator has to be able to key off the clock.  A zero-length interval
    stands in for "the interval has elapsed" without making the test sleep.
    """
    evaluator = ProgressEvaluator(
        ProgressConfig(every_seconds=1e-9, slumbot_hands=0),
        net_config=TINY_NET,
    )
    assert evaluator.due(1)
    # Iteration 0 is never due: there is nothing to measure before the first
    # drain, whatever the clock says.
    assert not evaluator.due(0)


def test_a_skipped_evaluation_does_not_count_a_skip_every_iteration():
    """The timer resets on any due tick, launched or skipped.

    Without that, one overrun leaves the interval permanently elapsed and
    ``skipped`` climbs once per iteration — five a second — which buries the
    signal that the cadence is too tight.
    """
    evaluator = ProgressEvaluator(
        ProgressConfig(every_seconds=3600.0, slumbot_hands=0),
        net_config=TINY_NET,
    )
    evaluator._last_launched -= 7200.0  # an interval has gone by
    blocking = threading.Event()
    evaluator._thread = threading.Thread(target=blocking.wait)
    evaluator._thread.start()
    try:
        assert not evaluator.maybe_start(1, HoldemValueNet(TINY_NET))
        assert evaluator.skipped == 1
        for iteration in range(2, 20):
            evaluator.maybe_start(iteration, HoldemValueNet(TINY_NET))
        assert evaluator.skipped == 1
    finally:
        blocking.set()
        evaluator._thread.join()


def test_the_two_cadences_coexist():
    config = ProgressConfig(every=10, every_seconds=3600.0)
    assert config.enabled
    evaluator = ProgressEvaluator(config, net_config=TINY_NET)
    assert evaluator.due(10)  # the iteration bound, well inside the hour
    assert not evaluator.due(11)
    assert not ProgressConfig().enabled


def test_the_evaluation_prints_the_slumbot_line_it_writes(tmp_path, capsys):
    """A run's log must show the two numbers, not just the file.

    ``progress.jsonl`` is what gets plotted, but a run whose only visible output
    is a file nobody is tailing looks identical to one that stopped measuring.
    """
    evaluator = ProgressEvaluator(
        ProgressConfig(
            every=1,
            slumbot_hands=4,
            slumbot_workers=2,
            slumbot_play=QUICK_PLAY,
        ),
        net_config=TINY_NET,
        tests=None,
        path=tmp_path / "progress.jsonl",
        slumbot_client_factory=lambda worker: FakeSlumbotClient(seed=worker),
    )
    assert evaluator.maybe_start(1, HoldemValueNet(TINY_NET))
    evaluator.join(timeout=120)

    printed = capsys.readouterr().out
    assert "progress iter 1" in printed
    assert "slumbot" in printed and "mbb/g" in printed and "over 4 hands" in printed
    assert "ERROR" not in printed
    written = json.loads((tmp_path / "progress.jsonl").read_text().strip())
    assert written["slumbot_hands"] == 4.0
