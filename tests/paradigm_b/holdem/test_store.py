"""The replay buffer, in memory and on disk.

The buffer is the one structure whose size the paper sets and the hardware
refuses: 120,000,000 raw examples at the old float32 encoding-plus-mask is
3.3 TB.  What these tests pin is the three things that bring it into reach —
one canonical row per solve, 11,037 bytes a row, and shards on a drive — and,
more importantly, that none of them quietly changed what the buffer *means*.

The load-bearing tests are :func:`test_sampling_is_uniform_over_live_rows`,
because a sharded buffer that samples its shards rather than its rows would be
faster and biased, and
:func:`test_a_sharded_buffer_holds_the_same_rows_as_an_in_memory_one`, which
holds the two implementations to the same answers.
"""

from __future__ import annotations

import numpy as np
import pytest

from paradigm_b.holdem.data.store import (
    MAX_BOARD_CARDS,
    NO_CARD,
    ROW_BYTES,
    ReplayBuffer,
    ShardedReplayBuffer,
    build_buffer,
    check_storable,
    encode_board,
)
from paradigm_b.holdem.data.augmentation import board_masks
from paradigm_b.holdem.engine.combos import NUM_COMBOS, board_mask
from paradigm_b.holdem.net.features import INPUT_DIM
from paradigm_b.holdem.selfplay import Example

BOARD = (0, 1, 2, 3, 4)


def label(value: int, board=BOARD) -> Example:
    """A row whose every number is ``value``, so it identifies itself."""
    return Example(
        features=np.full(INPUT_DIM, float(value), dtype=np.float32),
        mask=board_mask(board).astype(np.float32),
        values=np.full((2, NUM_COMBOS), float(value), dtype=np.float32),
        board=board,
    )


def live(buffer) -> list:
    """The values in the buffer, oldest first."""
    features, _, _ = buffer.raw(np.arange(len(buffer)))
    return [int(row[0]) for row in features]


def buffers(tmp_path, capacity, **kwargs):
    """The same buffer both ways, so a test can hold them to one answer."""
    return [
        ReplayBuffer(capacity, augment=False, **kwargs),
        ShardedReplayBuffer(
            capacity, tmp_path / "shards", shard_rows=8, hot_rows=5, augment=False, **kwargs
        ),
    ]


# --- the row ----------------------------------------------------------------
def test_the_row_costs_what_the_capacity_arithmetic_assumes():
    assert ROW_BYTES == INPUT_DIM * 2 + 2 * NUM_COMBOS * 2 + MAX_BOARD_CARDS
    assert ROW_BYTES == 11_037


def test_a_board_shorter_than_the_river_pads_rather_than_shifts():
    """One fixed-width field for all three streets is what makes boards batch."""
    assert list(encode_board((7, 8, 9))) == [7, 8, 9, NO_CARD, NO_CARD]
    assert list(encode_board(())) == [NO_CARD] * MAX_BOARD_CARDS
    assert list(encode_board(BOARD)) == list(BOARD)


def test_a_target_float16_cannot_hold_is_refused_not_stored():
    """An overflowing cast is silent, and one ``inf`` target ruins a network.

    Reaching this needs a belief state with almost no reach mass on one side,
    since the label is divided by that mass.  Measured headroom in ordinary
    running is two orders of magnitude.
    """
    with pytest.raises(ValueError, match="float16"):
        check_storable(np.full((2, NUM_COMBOS), 1e5, dtype=np.float32))
    with pytest.raises(ValueError, match="float16"):
        check_storable(np.full((2, NUM_COMBOS), np.inf, dtype=np.float32))
    check_storable(np.full((2, NUM_COMBOS), 697.0, dtype=np.float32))  # a real one


def test_the_mask_is_not_stored_but_comes_back_exact(tmp_path):
    """19% of the old row, replaced by five bytes it is a pure function of."""
    flop = (10, 20, 30)
    for buffer in buffers(tmp_path, 16):
        buffer.add([label(1, board=BOARD), label(2, board=flop)])
        _, _, boards = buffer.raw(np.arange(2))
        rebuilt = board_masks(boards)
        np.testing.assert_array_equal(rebuilt[0], board_mask(BOARD))
        np.testing.assert_array_equal(rebuilt[1], board_mask(flop))


# --- ordering and eviction --------------------------------------------------
def test_rows_come_back_oldest_first(tmp_path):
    for buffer in buffers(tmp_path, 64):
        buffer.add([label(i) for i in range(20)])
        assert live(buffer) == list(range(20))


def test_reaching_capacity_evicts_the_oldest(tmp_path):
    for buffer in buffers(tmp_path, 12):
        buffer.add([label(i) for i in range(20)])
        assert len(buffer) == 12
        assert live(buffer) == list(range(8, 20))


def test_purging_moves_a_watermark_rather_than_compacting(tmp_path):
    """ReBeL appendix E's one-off purge, at a size where copying is not an option.

    Dropping half of a 60M-row buffer used to mean copying the surviving half
    down over the dead half — hundreds of gigabytes of memmove to *delete*
    data, and on disk a full rewrite.  Advancing the index of the oldest live
    row does the same thing in constant time.
    """
    for buffer in buffers(tmp_path, 64):
        buffer.add([label(i) for i in range(20)])
        assert buffer.purge_oldest(0.5) == 10
        assert live(buffer) == list(range(10, 20))


def test_purging_unlinks_the_shards_it_clears(tmp_path):
    """The disk form of the same thing: eviction is ``unlink``, not a rewrite."""
    directory = tmp_path / "shards"
    buffer = ShardedReplayBuffer(64, directory, shard_rows=4, hot_rows=2, augment=False)
    buffer.add([label(i) for i in range(20)])
    buffer.flush()
    before = len(list(directory.glob("shard-*-features.npy")))

    buffer.purge_oldest(0.5)

    after = len(list(directory.glob("shard-*-features.npy")))
    assert after < before, "purging half the rows must remove shard files"
    assert live(buffer) == list(range(10, 20))


def test_purging_is_a_no_op_at_the_edges(tmp_path):
    for buffer in buffers(tmp_path, 8):
        assert buffer.purge_oldest(0.5) == 0  # empty
        buffer.add([label(i) for i in range(4)])
        assert buffer.purge_oldest(0.0) == 0
        assert buffer.purge_oldest(1.0) == 0
        assert len(buffer) == 4


def test_growth_undoes_a_wrap_rather_than_leaving_one(tmp_path):
    """A watermark past the end plus a doubling is where an off-by-one lives."""
    buffer = ReplayBuffer(1_000, augment=False)
    buffer.add([label(i) for i in range(64)])  # exactly fills the first block
    buffer.purge_oldest(0.5)
    buffer.add([label(i) for i in range(64, 200)])
    assert live(buffer) == list(range(32, 200))


# --- sampling ---------------------------------------------------------------
def test_sampling_is_uniform_over_live_rows(tmp_path):
    """Not over shards.

    Drawing whole shards and shuffling within a window reads sequentially and
    is the usual choice at this size, but it biases the draw toward whatever
    happens to share a shard.  The arithmetic does not demand it — 1,024 rows
    is 11 MB — so the buffer pays for random reads and keeps the distribution.
    """
    directory = tmp_path / "shards"
    buffer = ShardedReplayBuffer(
        200, directory, shard_rows=8, hot_rows=4, augment=False
    )
    buffer.add([label(i) for i in range(64)])

    rng = np.random.default_rng(0)
    counts = np.zeros(64)
    for _ in range(200):
        features, _, _ = buffer.sample(32, rng)
        for row in features:
            counts[int(row[0])] += 1

    assert (counts > 0).all(), "some rows were never drawn"
    # 6,400 draws over 64 rows is 100 each; a shard-level sampler would show
    # eight-row blocks moving together instead.
    assert counts.std() / counts.mean() < 0.25


def test_a_sharded_buffer_holds_the_same_rows_as_an_in_memory_one(tmp_path):
    """Two implementations, one meaning.  Pending, flushed and hot rows alike."""
    memory, sharded = buffers(tmp_path, 40)
    for buffer in (memory, sharded):
        buffer.add([label(i) for i in range(50)])  # wraps both
    assert live(memory) == live(sharded) == list(range(10, 50))

    for buffer in (memory, sharded):
        buffer.purge_oldest(0.25)
    assert live(memory) == live(sharded)


def test_rows_read_the_same_whether_or_not_they_are_still_hot(tmp_path):
    """The RAM window is a cache, so it must not be able to disagree with disk."""
    directory = tmp_path / "shards"
    hot = ShardedReplayBuffer(200, directory, shard_rows=8, hot_rows=64, augment=False)
    hot.add([label(i) for i in range(40)])
    hot.flush()

    cold = ShardedReplayBuffer(200, directory, shard_rows=8, hot_rows=0, augment=False)
    cold.read_state(directory, hot._manifest())

    rows = np.arange(len(hot))
    for warm, chilly in zip(hot.raw(rows), cold.raw(rows)):
        np.testing.assert_array_equal(warm, chilly)


def test_sampling_an_empty_buffer_says_so(tmp_path):
    for buffer in buffers(tmp_path, 8):
        with pytest.raises(ValueError, match="empty"):
            buffer.sample(4, np.random.default_rng(0))


# --- construction -----------------------------------------------------------
def test_build_buffer_picks_the_form_from_the_directory(tmp_path):
    assert isinstance(build_buffer(10), ReplayBuffer)
    assert isinstance(build_buffer(10, directory=tmp_path / "s"), ShardedReplayBuffer)


def test_the_papers_capacity_costs_nothing_until_the_rows_exist():
    """Declaring 120,000,000 must not allocate 120,000,000 rows."""
    buffer = ReplayBuffer(120_000_000)
    assert buffer.capacity == 120_000_000
    assert buffer.allocated <= 64


def test_a_reopened_shard_directory_refuses_a_foreign_row_layout(tmp_path):
    buffer = ShardedReplayBuffer(16, tmp_path / "shards", shard_rows=4, augment=False)
    buffer.add([label(0)])
    meta = dict(buffer.write_state(tmp_path / "state"), row_version=999)

    with pytest.raises(ValueError, match="layout version"):
        ShardedReplayBuffer(16, tmp_path / "shards", shard_rows=4).read_state(
            tmp_path / "state", meta
        )


def test_the_hot_window_survives_a_purge_that_outruns_it(tmp_path):
    """Dropping more rows than the RAM window holds must not read stale ones.

    The window answers a logical row by its offset from ``size - filled``.
    Purging shifts every logical index down and ``size`` with it, so the offset
    is invariant — but only if the arithmetic is allowed to go negative rather
    than being clamped, which is exactly the kind of thing a later tidy-up
    breaks.
    """
    directory = tmp_path / "shards"
    hot = ShardedReplayBuffer(64, directory, shard_rows=4, hot_rows=8, augment=False)
    hot.add([label(i) for i in range(20)])
    hot.flush()
    hot.purge_oldest(0.9)  # 18 of 20, far more than the window holds

    assert live(hot) == [18, 19]

    # And against the same buffer with no window at all, which has to read
    # every row from the shards.
    cold = ShardedReplayBuffer(64, directory, shard_rows=4, hot_rows=0, augment=False)
    cold.read_state(directory, hot._manifest())
    assert live(cold) == live(hot)
