"""Where labels live between being solved and being trained on.

The replay buffer is the one structure in this implementation whose size is set
by the paper rather than by the hardware, and the two do not meet.  A row used
to cost 27,372 bytes — a 2,864-wide encoding, a 1,326 mask and a 2 x 1,326
target, all float32 — so the paper's 120,000,000 raw examples is 3.3 TB.  No
machine this runs on has that, and the honest consequence was a buffer that
never came within two orders of magnitude of the setting it claimed.

Three changes bring it into reach, and they compose::

    stored copies per solve   K = 2      -> 1        (transform on read)
    bytes per stored row      27,372     -> 11,037   (fp16, mask from board)
    where the rows live       RAM        -> shards on disk

The first two are :mod:`~paradigm_b.holdem.data.augmentation` and the row layout
below; together they are 5x, which turns the paper's 120M into 660 GB of
60M canonical rows.  The third is :class:`ShardedReplayBuffer`, for when even
that does not fit in memory.

The row
-------

A stored row is **canonical**: the untransformed encoding, at chip scale 1.0,
with no mask.

=============  ===========================  =======
field          dtype and shape              bytes
=============  ===========================  =======
``features``   float16 ``(2864,)``          5,728
``targets``    float16 ``(2, 1326)``        5,304
``boards``     int8 ``(5,)``                5
=============  ===========================  =======

float16 is not a compromise here in the way it would be for weights.  Ranges
are probabilities and targets are counterfactual values already normalised by
pot; fp16 carries ~5e-4 relative error on both, which is orders of magnitude
below the noise in a label produced by 250 CFR iterations of a single sampled
trajectory.  The masks are gone entirely because a mask is a pure function of
the board — 5 bytes against 5,304 — and the boards are kept as card ids with
``-1`` for a slot that has not been dealt.

Ordering and eviction
---------------------

Both buffers here are circular and both track age by a **watermark** rather
than by compaction.  ``purge_oldest`` used to copy the surviving half of the
buffer down over the dead half; at 60M rows that is a memmove of hundreds of
gigabytes to delete data.  Advancing the index of the oldest live row does the
same thing in constant time, and is the only form that works at all once the
rows are on disk, where the equivalent copy would be a full rewrite.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from paradigm_b.holdem.data.augmentation import transform_batch
from paradigm_b.holdem.engine.combos import NUM_COMBOS
from paradigm_b.holdem.net.features import INPUT_DIM

NUM_PLAYERS = 2
# Flop, turn and river in one fixed-width field, so a row's board costs the
# same whatever street produced it and a batch of boards is one array.
MAX_BOARD_CARDS = 5
NO_CARD = -1

FEATURE_DTYPE = np.float16
TARGET_DTYPE = np.float16
BOARD_DTYPE = np.int8

# Bumped whenever the on-disk row layout changes.  A shard directory records
# it, and reopening one written under a different layout is refused rather than
# reinterpreted.
ROW_VERSION = 1

ROW_BYTES = (
    INPUT_DIM * np.dtype(FEATURE_DTYPE).itemsize
    + NUM_PLAYERS * NUM_COMBOS * np.dtype(TARGET_DTYPE).itemsize
    + MAX_BOARD_CARDS * np.dtype(BOARD_DTYPE).itemsize
)


def encode_board(board: Sequence[int]) -> np.ndarray:
    """Board cards as a fixed-width row, ``-1`` for undealt slots."""
    out = np.full(MAX_BOARD_CARDS, NO_CARD, dtype=BOARD_DTYPE)
    for slot, card in enumerate(board[:MAX_BOARD_CARDS]):
        out[slot] = card
    return out


def _augment_stream(rng: Optional[np.random.Generator]) -> np.random.Generator:
    """The generator read-time transforms are drawn from.

    Deliberately not the caller's.  A run with augmentation off and a run with
    it on must draw the same *rows* in the same order, and in the synchronous
    loop the training generator is also the generation generator, so a transform
    drawn from it would shift which trajectories the run searches.  Callers pass
    the spawned stream the run state already saves; the default is for tests.
    """
    return np.random.default_rng() if rng is None else rng


FP16_MAX = 65504.0


def check_storable(values: np.ndarray) -> np.ndarray:
    """Refuse a target float16 cannot hold, rather than storing ``inf``.

    Measured over 1,771 preflop-rooted labels, the largest ``|target|`` was 697
    and the largest ``|feature|`` was 3, so the headroom is two orders of
    magnitude and this never fires in ordinary running.  It exists because the
    one thing that *can* reach the ceiling is not bounded by the game:
    :func:`~paradigm_b.holdem.selfplay.normalise` divides counterfactual values by
    the opponent's reach mass, and a belief state concentrated on a handful of
    combinations makes that divisor arbitrarily small.

    An overflowing cast is silent — ``float32(1e5).astype(float16)`` is ``inf``
    — and one ``inf`` target destroys the network on the first batch that draws
    it, hours after the row was written and with nothing in the history to say
    which row it was.  A pass over 2,652 floats to turn that into a message
    costs a fraction of a percent of generation, which is the right trade.
    """
    values = np.asarray(values)
    largest = np.abs(values).max() if values.size else 0.0
    if not np.isfinite(largest) or largest > FP16_MAX:
        raise ValueError(
            f"label has |value| up to {largest:.4g}, which float16 storage "
            f"cannot hold (max {FP16_MAX:.0f}).  This normally means a belief "
            f"state with almost no reach mass on one side, since the label is "
            f"divided by that mass"
        )
    return values


def _empty_rows(count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((count, INPUT_DIM), dtype=FEATURE_DTYPE),
        np.zeros((count, NUM_PLAYERS, NUM_COMBOS), dtype=TARGET_DTYPE),
        np.full((count, MAX_BOARD_CARDS), NO_CARD, dtype=BOARD_DTYPE),
    )


class ReplayBuffer:
    """Canonical rows in memory, oldest tracked by watermark.

    The default buffer, and the one every small run uses.  ``capacity`` is
    logical: physical storage doubles toward it as rows arrive, so declaring the
    paper's 120,000,000 costs nothing until the rows exist.

    Two indices describe the live region.  ``_start`` is the physical slot of
    the oldest live row and ``size`` is how many follow it; the write position
    is ``(_start + size) % allocated`` and is derived rather than stored.  Every
    operation that drops rows moves ``_start``, so none of them copy.
    """

    def __init__(
        self,
        capacity: int,
        *,
        augment: bool = True,
        augment_rng: Optional[np.random.Generator] = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("buffer capacity must be positive")
        self.capacity = int(capacity)
        self.augment = bool(augment)
        self.augment_rng = _augment_stream(augment_rng)
        allocated = min(self.capacity, 64)
        self.features, self.targets, self.boards = _empty_rows(allocated)
        self.size = 0
        self._start = 0

    # -- shape ------------------------------------------------------------
    def __len__(self) -> int:
        return self.size

    @property
    def allocated(self) -> int:
        return len(self.features)

    @property
    def _next(self) -> int:
        """Where the next row lands.  Derived; kept for readability at call sites."""
        return (self._start + self.size) % max(self.allocated, 1)

    def physical(self, logical: np.ndarray) -> np.ndarray:
        """Physical slots for logical row numbers, ``0`` being the oldest live row."""
        return (self._start + np.asarray(logical)) % max(self.allocated, 1)

    # -- writing ----------------------------------------------------------
    def add(self, examples: Sequence) -> None:
        for example in examples:
            if self.size < self.capacity:
                self._ensure_allocated(self.size + 1)
            slot = self._next
            self.features[slot] = example.features
            self.targets[slot] = check_storable(example.values)
            self.boards[slot] = encode_board(getattr(example, "board", ()))
            if self.size < self.capacity:
                self.size += 1
            else:  # full: the row just written replaced the oldest one
                self._start = (self._start + 1) % self.allocated

    def _ensure_allocated(self, required: int) -> None:
        """Grow physical storage, normalising the live region to slot zero.

        Growing is also the moment the wrap is undone: rows are copied out in
        logical order into the larger arrays, which leaves ``_start`` at zero
        and means the copy happens once per doubling rather than once per
        watermark move.
        """
        current = self.allocated
        if required <= current:
            return
        allocated = min(self.capacity, max(required, current * 2))
        features, targets, boards = _empty_rows(allocated)
        if self.size:
            order = self.physical(np.arange(self.size))
            features[: self.size] = self.features[order]
            targets[: self.size] = self.targets[order]
            boards[: self.size] = self.boards[order]
        self.features, self.targets, self.boards = features, targets, boards
        self._start = 0

    # -- reading ----------------------------------------------------------
    def raw(self, logical: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stored rows, untransformed, in the order asked for."""
        slots = self.physical(logical)
        return self.features[slots], self.targets[slots], self.boards[slots]

    def sample(self, batch_size: int, rng: np.random.Generator):
        """One training batch: uniform over live rows, transformed on the way out."""
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        logical = rng.integers(0, self.size, size=min(batch_size, self.size))
        features, targets, boards = self.raw(logical)
        # Which rows are drawn comes from the caller's stream; how they are
        # transformed comes from a stream of the buffer's own.  That separation
        # is what keeps ``suit_augmentations=0`` and the default drawing the
        # *same* rows and, in the synchronous loop where one generator feeds
        # both generation and training, searching the same trajectories.
        return transform_batch(
            features,
            targets,
            boards,
            self.augment_rng,
            relabel=self.augment,
            rescale=self.augment,
        )

    # -- eviction ---------------------------------------------------------
    def purge_oldest(self, fraction: float = 0.5) -> int:
        """Drop the oldest ``fraction`` of the buffer; return how many went.

        ReBeL's appendix E: *"As initial data is produced with a random value
        network, we remove half of the data from the replay buffer after 20
        epochs."*  Without this, labels written by an essentially random
        network keep being sampled at full weight for the rest of the run —
        and in a buffer that never fills nothing is ever evicted by the
        ordinary circular churn either, so *every* early mistake survives to
        the last gradient step.

        This moves ``_start`` and nothing else.  The rows are still in the
        array; they are simply no longer addressable, and the next wrap writes
        over them.
        """
        if not 0.0 < fraction < 1.0 or self.size == 0:
            return 0
        dropped = int(self.size * fraction)
        if dropped <= 0 or dropped >= self.size:
            return 0
        self._start = (self._start + dropped) % self.allocated
        self.size -= dropped
        return dropped

    # -- run state --------------------------------------------------------
    def write_state(self, directory: Path, name: str = "buffer") -> Dict[str, Any]:
        """Save the live rows, oldest first, and describe them.

        Oldest-first is what makes the saved form independent of where the
        watermark happened to be, so a resumed buffer evicts in the same order
        it would have.  The arrays are canonical fp16, which is why this is now
        2.5x smaller than the float32-with-masks form it replaces.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        order = self.physical(np.arange(self.size))
        np.save(directory / f"{name}-features.npy", self.features[order])
        np.save(directory / f"{name}-targets.npy", self.targets[order])
        np.save(directory / f"{name}-boards.npy", self.boards[order])
        return {
            "kind": "memory",
            "row_version": ROW_VERSION,
            "size": int(self.size),
            "capacity": int(self.capacity),
        }

    def read_state(
        self, directory: Path, meta: Mapping[str, Any], name: str = "buffer"
    ) -> None:
        directory = Path(directory)
        _check_row_version(meta)
        size = int(meta["size"])
        if size > self.capacity:
            raise ValueError(
                f"saved buffer holds {size} rows but this run's capacity is "
                f"{self.capacity}; raise buffer_size or the oldest data would "
                f"be silently dropped on resume"
            )
        self._ensure_allocated(max(size, 1))
        self.features[:size] = np.load(directory / f"{name}-features.npy")
        self.targets[:size] = np.load(directory / f"{name}-targets.npy")
        self.boards[:size] = np.load(directory / f"{name}-boards.npy")
        self.size = size
        self._start = 0


# --- the same buffer, on disk -----------------------------------------------
#
# Past a few million rows the arrays stop fitting and the question becomes what
# shape the disk traffic takes.  Rows are written once, read many times at
# random, and deleted oldest-first, which is exactly an append-only log with a
# moving tail -- so that is the layout: fixed-size shards of ``.npy``, memory
# mapped for reading, unlinked whole for eviction.
#
# Two things fall out of the shape for free.  Eviction is ``os.unlink`` instead
# of a rewrite, and a state save is a manifest instead of a copy: the shards
# already are the buffer's durable form, so ``--state-every`` no longer writes
# a second copy of every row it is trying to protect.
#
# Sampling stays exactly uniform over live rows.  The alternative -- draw whole
# shards and shuffle within a window -- reads sequentially and is the usual
# choice, but it biases the draw, and the arithmetic does not demand it: 1,024
# rows is 11 MB, which even at one random read per row is well inside what an
# NVMe delivers.  What it does demand is that the reads within a shard go in
# file order, which ``_gather`` does.


@dataclass
class _Shard:
    index: int
    rows: int


class ShardedReplayBuffer:
    """Canonical rows in append-only shards on disk, with a hot window in RAM.

    ``directory`` is where the shards go and is expected to be fast: the whole
    design assumes NVMe, and a spinning disk will not serve random row reads at
    any useful rate.  It may sit outside the run directory — an external drive
    is the normal case at this size — and the run state records the path rather
    than the contents.

    ``hot_rows`` of the most recent rows are mirrored in memory.  This is a
    cache, not a tier: sampling remains uniform over every live row, and the
    window only decides which of them are answered without touching the disk.
    Recent rows are also the ones a run reads most, because they are the ones
    that have not been evicted yet, so the hit rate is higher than the size
    ratio suggests.
    """

    MANIFEST = "manifest.json"

    def __init__(
        self,
        capacity: int,
        directory: os.PathLike | str,
        *,
        shard_rows: int = 16_384,
        hot_rows: int = 262_144,
        augment: bool = True,
        augment_rng: Optional[np.random.Generator] = None,
        open_shards: int = 64,
    ) -> None:
        if capacity <= 0:
            raise ValueError("buffer capacity must be positive")
        if shard_rows <= 0:
            raise ValueError("shard_rows must be positive")
        self.augment_rng = _augment_stream(augment_rng)
        self.capacity = int(capacity)
        self.directory = Path(directory)
        self.shard_rows = int(shard_rows)
        self.hot_rows = max(int(hot_rows), 0)
        self.augment = bool(augment)
        self._open_shards = max(int(open_shards), 1)

        self.directory.mkdir(parents=True, exist_ok=True)
        self._shards: List[_Shard] = []
        self._head = 0  # live rows skipped at the front of the oldest shard
        self._next_index = 0
        self._pending = _empty_rows(self.shard_rows)
        self._pending_rows = 0
        self._offsets: Optional[np.ndarray] = None
        self._maps: "OrderedDict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]" = (
            OrderedDict()
        )
        self._hot = _empty_rows(self.hot_rows) if self.hot_rows else None
        self._hot_next = 0
        self._hot_filled = 0

    # -- shape ------------------------------------------------------------
    def __len__(self) -> int:
        return self.size

    @property
    def size(self) -> int:
        return sum(shard.rows for shard in self._shards) - self._head + self._pending_rows

    @property
    def num_shards(self) -> int:
        return len(self._shards)

    @property
    def nbytes(self) -> int:
        """What the live rows occupy on disk, near enough for a progress line."""
        return self.size * ROW_BYTES

    def _shard_paths(self, index: int) -> Tuple[Path, Path, Path]:
        stem = self.directory / f"shard-{index:08d}"
        return (
            stem.with_name(stem.name + "-features.npy"),
            stem.with_name(stem.name + "-targets.npy"),
            stem.with_name(stem.name + "-boards.npy"),
        )

    def _boundaries(self) -> np.ndarray:
        """Logical row where each shard starts, then the pending block, then size.

        Cached because it changes only when rows are added or evicted, and it is
        consulted once per sampled batch.
        """
        if self._offsets is None:
            rows = [shard.rows for shard in self._shards]
            if rows:
                rows[0] -= self._head
            rows.append(self._pending_rows)
            self._offsets = np.concatenate([[0], np.cumsum(rows)])
        return self._offsets

    # -- writing ----------------------------------------------------------
    def add(self, examples: Sequence) -> None:
        for example in examples:
            slot = self._pending_rows
            self._pending[0][slot] = example.features
            self._pending[1][slot] = check_storable(example.values)
            self._pending[2][slot] = encode_board(getattr(example, "board", ()))
            self._pending_rows += 1
            self._offsets = None
            self._remember(slot)
            if self._pending_rows == self.shard_rows:
                self._flush_pending()
        self._evict_to_capacity()

    def _remember(self, pending_slot: int) -> None:
        """Mirror the row just written into the hot window."""
        if self._hot is None:
            return
        for destination, source in zip(self._hot, self._pending):
            destination[self._hot_next] = source[pending_slot]
        self._hot_next = (self._hot_next + 1) % self.hot_rows
        self._hot_filled = min(self._hot_filled + 1, self.hot_rows)

    def _flush_pending(self) -> None:
        if self._pending_rows == 0:
            return
        rows = self._pending_rows
        for path, array in zip(self._shard_paths(self._next_index), self._pending):
            _atomic_save(path, array[:rows])
        self._shards.append(_Shard(index=self._next_index, rows=rows))
        self._next_index += 1
        self._pending_rows = 0
        self._offsets = None

    def flush(self) -> None:
        """Make every live row durable, including a partly filled shard."""
        self._flush_pending()
        self._write_manifest()

    # -- reading ----------------------------------------------------------
    def _map(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._maps.get(index)
        if cached is not None:
            self._maps.move_to_end(index)
            return cached
        maps = tuple(
            np.load(path, mmap_mode="r") for path in self._shard_paths(index)
        )
        self._maps[index] = maps
        while len(self._maps) > self._open_shards:
            self._maps.popitem(last=False)
        return maps

    def raw(self, logical: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stored rows for logical row numbers, gathered from wherever they live."""
        logical = np.asarray(logical)
        features, targets, boards = _empty_rows(len(logical))
        if len(logical) == 0:
            return features, targets, boards

        destination = (features, targets, boards)

        # The hot window first, and globally: a row inside it is in RAM whether
        # or not it has also been flushed, so answering here keeps whole shards
        # from being mapped for rows that never needed the disk.
        cold = np.arange(len(logical))
        if self._hot is not None and self._hot_filled:
            floor = self.size - self._hot_filled
            hot = np.nonzero(logical >= floor)[0]
            if hot.size:
                self._read_hot(logical[hot], floor, destination, hot)
                cold = np.nonzero(logical < floor)[0]
        if cold.size == 0:
            return features, targets, boards

        bounds = self._boundaries()
        blocks = np.searchsorted(bounds, logical[cold], side="right") - 1
        for block in np.unique(blocks):
            here = cold[blocks == block]
            local = logical[here] - bounds[block]
            if block == len(self._shards):  # the pending, unflushed rows
                source = self._pending
            else:
                if block == 0:
                    local = local + self._head
                source = self._map(self._shards[block].index)
            # Sorted so the reads walk the mapping forward.  On a memory mapped
            # shard this is the difference between one pass over the pages the
            # batch needs and a random walk through the page cache.
            order = np.argsort(local, kind="stable")
            _gather(source, local[order], destination, here[order])
        return features, targets, boards

    def _read_hot(
        self,
        logical: np.ndarray,
        floor: int,
        destination: Tuple[np.ndarray, np.ndarray, np.ndarray],
        rows: np.ndarray,
    ) -> None:
        oldest = (self._hot_next - self._hot_filled) % self.hot_rows
        slots = (oldest + (logical - floor)) % self.hot_rows
        _gather(self._hot, slots, destination, rows)

    def sample(self, batch_size: int, rng: np.random.Generator):
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        logical = rng.integers(0, self.size, size=min(batch_size, self.size))
        features, targets, boards = self.raw(logical)
        # Which rows are drawn comes from the caller's stream; how they are
        # transformed comes from a stream of the buffer's own.  That separation
        # is what keeps ``suit_augmentations=0`` and the default drawing the
        # *same* rows and, in the synchronous loop where one generator feeds
        # both generation and training, searching the same trajectories.
        return transform_batch(
            features,
            targets,
            boards,
            self.augment_rng,
            relabel=self.augment,
            rescale=self.augment,
        )

    # -- eviction ---------------------------------------------------------
    def _drop_oldest(self, count: int) -> int:
        """Advance the watermark by ``count`` rows, unlinking shards it clears."""
        count = min(count, self.size)
        remaining = count
        while remaining > 0 and self._shards:
            shard = self._shards[0]
            live = shard.rows - self._head
            if live > remaining:
                self._head += remaining
                remaining = 0
                break
            for path in self._shard_paths(shard.index):
                path.unlink(missing_ok=True)
            self._maps.pop(shard.index, None)
            self._shards.pop(0)
            self._head = 0
            remaining -= live
        if remaining > 0:  # nothing but pending rows left
            keep = self._pending_rows - remaining
            for array in self._pending:
                array[:keep] = array[remaining : self._pending_rows]
            self._pending_rows = keep
            remaining = 0
        self._offsets = None
        return count

    def _evict_to_capacity(self) -> None:
        excess = self.size - self.capacity
        if excess > 0:
            self._drop_oldest(excess)

    def purge_oldest(self, fraction: float = 0.5) -> int:
        """ReBeL appendix E's one-off purge; see :meth:`ReplayBuffer.purge_oldest`.

        On disk this is where the watermark earns its keep: dropping half of a
        600 GB buffer unlinks half the shard files and moves an integer.
        """
        if not 0.0 < fraction < 1.0 or self.size == 0:
            return 0
        dropped = int(self.size * fraction)
        if dropped <= 0 or dropped >= self.size:
            return 0
        return self._drop_oldest(dropped)

    # -- run state --------------------------------------------------------
    def _manifest(self) -> Dict[str, Any]:
        return {
            "kind": "sharded",
            "row_version": ROW_VERSION,
            "directory": str(self.directory),
            "capacity": int(self.capacity),
            "shard_rows": int(self.shard_rows),
            "head": int(self._head),
            "next_index": int(self._next_index),
            "shards": [[shard.index, shard.rows] for shard in self._shards],
        }

    def _write_manifest(self) -> None:
        path = self.directory / self.MANIFEST
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._manifest(), indent=2, sort_keys=True))
        os.replace(tmp, path)

    def write_state(self, directory: Path, name: str = "buffer") -> Dict[str, Any]:
        """Describe the buffer.  Copies nothing: the shards *are* the state.

        This is the whole reason a disk-backed buffer is cheaper end to end and
        not merely larger.  The in-memory form has to write every live row into
        the state directory on each ``--state-every``, which at these sizes is
        hundreds of gigabytes staged and swapped to protect data that a crash
        would not have touched.  Here the rows are already durable and already
        outside the run's atomic swap, so the state is a manifest: the shard
        list as of this moment, and the watermark into it.
        """
        self.flush()
        return self._manifest()

    def read_state(
        self, directory: Path, meta: Mapping[str, Any], name: str = "buffer"
    ) -> None:
        _check_row_version(meta)
        recorded = Path(meta["directory"])
        if recorded != self.directory:
            # Moving the drive is legitimate; silently reading a *different*
            # buffer than the one the run was saved against is not.  Take the
            # caller's path and say what changed.
            print(
                f"note: run state was written against buffer directory {recorded}, "
                f"resuming against {self.directory}"
            )
        self._shards = [_Shard(index=int(i), rows=int(n)) for i, n in meta["shards"]]
        self._head = int(meta["head"])
        self._next_index = int(meta["next_index"])
        self._pending_rows = 0
        self._offsets = None
        self._maps.clear()
        missing = [
            shard.index
            for shard in self._shards
            if not self._shard_paths(shard.index)[0].exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} shard(s) named in the run state are missing from "
                f"{self.directory} (first: shard-{missing[0]:08d}); the buffer "
                f"directory and the run state have to be restored together"
            )
        # Shards written after the state save are ignored rather than adopted:
        # the manifest is the authority on what the run had seen.
        self._rebuild_hot()

    def _rebuild_hot(self) -> None:
        """Refill the RAM window from the newest shards after a resume."""
        if self._hot is None or self.size == 0:
            return
        wanted = min(self.hot_rows, self.size)
        logical = np.arange(self.size - wanted, self.size)
        self._hot_filled = 0  # read straight from disk while refilling
        self._hot_next = 0
        features, targets, boards = self.raw(logical)
        for destination, source in zip(self._hot, (features, targets, boards)):
            destination[:wanted] = source
        self._hot_filled = wanted
        self._hot_next = wanted % self.hot_rows


def _gather(
    source: Tuple[np.ndarray, np.ndarray, np.ndarray],
    slots: np.ndarray,
    destination: Tuple[np.ndarray, np.ndarray, np.ndarray],
    rows: np.ndarray,
) -> None:
    for out, array in zip(destination, source):
        out[rows] = array[slots]


def _atomic_save(path: Path, array: np.ndarray) -> None:
    """Write a shard so a crash cannot leave a readable-but-short file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _check_row_version(meta: Mapping[str, Any]) -> None:
    version = int(meta.get("row_version", 0))
    if version != ROW_VERSION:
        raise ValueError(
            f"buffer rows were written under layout version {version}, this build "
            f"reads version {ROW_VERSION}; the encodings are not interchangeable"
        )


def build_buffer(
    capacity: int,
    *,
    directory: Optional[os.PathLike | str] = None,
    shard_rows: int = 16_384,
    hot_rows: int = 262_144,
    augment: bool = True,
    augment_rng: Optional[np.random.Generator] = None,
):
    """The buffer a config asks for: in memory, or sharded onto ``directory``."""
    if directory is None:
        return ReplayBuffer(capacity, augment=augment, augment_rng=augment_rng)
    return ShardedReplayBuffer(
        capacity,
        directory,
        shard_rows=shard_rows,
        hot_rows=hot_rows,
        augment=augment,
        augment_rng=augment_rng,
    )
