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
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from paradigm_b.holdem.engine.combos import NUM_COMBOS
from paradigm_b.holdem.arms_common.storage import PathLike, write_json
from paradigm_b.holdem.net.features import INPUT_DIM
from paradigm_b.holdem.data.generation import Examples

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
NUM_PLAYERS = 2


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
        self.shards: List[Dict[str, Any]] = []
        self._pending: List[JournalEntry] = []
        self._total = 0

    def __len__(self) -> int:
        return self._total + len(self._pending)

    @property
    def flushed(self) -> int:
        return self._total

    def add(self, entries: Sequence[JournalEntry]) -> None:
        self._pending.extend(entries)

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
        self._write_manifest()
        return entry

    def _write_manifest(self) -> None:
        write_json(
            {
                "schema_version": SCHEMA_VERSION,
                "total": self._total,
                "shards": self.shards,
            },
            self.path / MANIFEST_FILENAME,
        )

    # -- reading ----------------------------------------------------------
    @classmethod
    def open(cls, path: PathLike) -> "TrajectoryJournal":
        journal = cls.__new__(cls)
        journal.path = Path(path)
        manifest_path = journal.path / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"no journal manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported journal schema {manifest['schema_version']}, "
                f"expected {SCHEMA_VERSION}"
            )
        journal.shards = list(manifest["shards"])
        journal._pending = []
        journal._total = int(manifest["total"])
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
