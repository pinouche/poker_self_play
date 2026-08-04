"""Arm 2's record: every label the online loop produced, in the order it made them.

The offline arm's data is a *file*, so inspecting it afterwards is trivial.  The
online arm's data is a stream that normally evaporates — labels are generated,
trained on, and pushed out of a fixed-size replay buffer.  Keeping only the
final network would throw away the thing that actually distinguishes this arm:
that its labels **change as the network improves**.

So every trajectory is written down as it happens, tagged with the iteration
that produced it and the street it started on.  That makes the arm's central
claim checkable after the fact rather than merely asserted:

* labels for comparable belief states should drift early and settle late — if
  they never move, refreshing bought nothing and the offline arm should match;
* the drift should be largest on the flop and absent on the river, because the
  river solves to real showdowns and is exact from the very first iteration.

Written as append-only shards, one per flush, exactly like the offline artifact
— a run that dies at iteration 40 leaves 40 iterations of readable history
rather than a corrupt file.  Metadata is stored per row (iteration, street,
trajectory id, step within the trajectory) so a reader can slice by any of them
without re-deriving anything from the encoded features.

**The index is append-only too, and that is not a detail.**  Schema 1 kept the
shard list inside ``manifest.json`` and rewrote the whole file on every flush.
That is quadratic in shard count, and it does not stay small: the 12-hour run
of 2026-08-01 wrote 137,081 shards, so it re-serialised a manifest growing to
35 MB once per iteration and spent **66% of its wall clock** doing it — more
than generation and training combined, and invisible because the cost sat
between the two ``Spend`` timers.  Doubling a run quadruples that work, so it
was the thing capping run length rather than the solver.  Schema 2 appends one
JSON line per shard to ``shards.jsonl`` and leaves ``manifest.json`` as a small
O(1) header.  Schema 1 journals still read, so the runs already on disk keep
working.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple

import numpy as np

from paradigm_b.holdem.engine.combos import NUM_COMBOS
from paradigm_b.holdem.arms_common.storage import PathLike, write_json
from paradigm_b.holdem.net.features import INPUT_DIM
from paradigm_b.holdem.data.generation import Examples

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
MANIFEST_FILENAME = "manifest.json"
SHARD_LOG_FILENAME = "shards.jsonl"
NUM_PLAYERS = 2


def _read_shard_log(path: Path) -> List[Dict[str, Any]]:
    """Every complete line of a shard log, stopping at a torn one.

    A kill between ``write`` and ``flush`` can leave the final line truncated.
    Everything before it was flushed and is good, so the log is read up to the
    first line that will not parse rather than refusing to open at all — the
    same "a dead run leaves readable history" property the shards themselves
    have.
    """
    shards: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                shards.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return shards


@dataclass
class JournalEntry:
    """One labelled belief state, plus where in the run it came from."""

    features: np.ndarray
    mask: np.ndarray
    values: np.ndarray
    iteration: int
    trajectory: int
    step: int
    board_cards: int


class TrajectoryJournal:
    """Append-only, sharded log of the online arm's generated labels."""

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        # Pick up where a previous process left off rather than starting the
        # count from zero.  A resumed run rewinds ``iteration`` to the last
        # state snapshot, so iteration numbers *do* repeat across a resume;
        # carrying the shard index forward is what keeps the filenames unique
        # and the earlier history readable.  (Schema 1 could not do this: a
        # fresh journal rewrote manifest.json and dropped every shard the
        # previous process had recorded.)
        existing = self.path / SHARD_LOG_FILENAME
        self.shards: List[Dict[str, Any]] = (
            _read_shard_log(existing) if existing.exists() else []
        )
        self._pending: List[JournalEntry] = []
        self._total = sum(int(shard["count"]) for shard in self.shards)
        self._shard_log: Optional[TextIO] = existing.open("a", encoding="utf-8")
        self._write_manifest()

    def __len__(self) -> int:
        return self._total + len(self._pending)

    @property
    def flushed(self) -> int:
        return self._total

    def add(self, entries: Sequence[JournalEntry]) -> None:
        self._pending.extend(entries)

    def close(self) -> None:
        """Release the shard log and refresh the header.  Safe to call twice."""
        if self._shard_log is not None:
            self._shard_log.flush()
            self._shard_log.close()
            self._shard_log = None
        self._write_manifest()

    def __enter__(self) -> "TrajectoryJournal":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def flush(self, iteration: int) -> Optional[Dict[str, Any]]:
        """Write everything buffered as one shard tagged with ``iteration``."""
        if not self._pending:
            return None
        index = len(self.shards)
        prefix = f"iter-{iteration:05d}-{index:04d}"
        arrays = {
            "features": np.stack([e.features for e in self._pending]).astype(np.float32),
            "masks": np.stack([e.mask for e in self._pending]).astype(bool),
            "targets": np.stack([e.values for e in self._pending]).astype(np.float32),
            "meta": np.array(
                [
                    (e.iteration, e.trajectory, e.step, e.board_cards)
                    for e in self._pending
                ],
                dtype=np.int32,
            ),
        }
        names = {}
        for key, array in arrays.items():
            name = f"{prefix}-{key}.npy"
            np.save(self.path / name, array)
            names[key] = name

        entry = {
            "index": index,
            "iteration": int(iteration),
            "count": len(self._pending),
            **names,
        }
        self.shards.append(entry)
        self._total += len(self._pending)
        self._pending = []
        self._append_to_shard_log(entry)
        return entry

    def _append_to_shard_log(self, entry: Dict[str, Any]) -> None:
        """One line, one shard.  Constant work per flush, whatever the total."""
        if self._shard_log is None:
            raise ValueError("journal is closed, or was opened for reading")
        self._shard_log.write(json.dumps(entry, sort_keys=True) + "\n")
        # Flush to the OS so a reader — or a crash — sees every shard whose
        # .npy files are already on disk.  This is one small write, not a
        # rewrite of the index, which is the whole point of the format.
        self._shard_log.flush()

    def _write_manifest(self) -> None:
        """The header only.  The shard list lives in the append-only log."""
        write_json(
            {
                "schema_version": SCHEMA_VERSION,
                "total": self._total,
                "shard_log": SHARD_LOG_FILENAME,
            },
            self.path / MANIFEST_FILENAME,
        )

    # -- reading ----------------------------------------------------------
    @classmethod
    def open(cls, path: PathLike) -> "TrajectoryJournal":
        """Read a journal of either schema.

        The shard log is authoritative whenever it exists: the header is
        written at construction and refreshed at ``close``, so a run killed
        mid-flight leaves a ``total`` of zero in ``manifest.json`` while the
        log itself is complete.  Counting the log costs one pass and is always
        right, so the header's ``total`` is never trusted.
        """
        journal = cls.__new__(cls)
        journal.path = Path(path)
        journal._pending = []
        journal._shard_log = None  # read-only; ``flush`` would have nowhere to go

        manifest_path = journal.path / MANIFEST_FILENAME
        shard_log_path = journal.path / SHARD_LOG_FILENAME
        manifest: Optional[Dict[str, Any]] = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            version = int(manifest.get("schema_version", SCHEMA_VERSION))
            if version not in SUPPORTED_SCHEMA_VERSIONS:
                raise ValueError(
                    f"unsupported journal schema {version}, expected one of "
                    f"{', '.join(str(v) for v in SUPPORTED_SCHEMA_VERSIONS)}"
                )

        if shard_log_path.exists():
            journal.shards = _read_shard_log(shard_log_path)
        elif manifest is not None and "shards" in manifest:
            journal.shards = list(manifest["shards"])  # schema 1
        elif manifest is None:
            raise FileNotFoundError(
                f"no journal at {journal.path}: expected {SHARD_LOG_FILENAME} "
                f"or a schema-1 {MANIFEST_FILENAME}"
            )
        else:
            journal.shards = []

        journal._total = sum(int(shard["count"]) for shard in journal.shards)
        return journal

    def iterations(self) -> Tuple[int, ...]:
        return tuple(sorted({int(s["iteration"]) for s in self.shards}))

    def read(self, iteration: Optional[int] = None) -> Tuple[Examples, np.ndarray]:
        """Labels and their metadata, for one iteration or the whole run."""
        shards = [
            s
            for s in self.shards
            if iteration is None or int(s["iteration"]) == iteration
        ]
        if not shards:
            raise ValueError(f"journal has nothing for iteration {iteration}")
        parts, metas = [], []
        for shard in shards:
            parts.append(
                Examples(
                    features=np.load(self.path / shard["features"]).astype(np.float32),
                    masks=np.load(self.path / shard["masks"]).astype(np.float32),
                    targets=np.load(self.path / shard["targets"]).astype(np.float32),
                )
            )
            metas.append(np.load(self.path / shard["meta"]))
        return Examples.concatenate(parts), np.concatenate(metas)
