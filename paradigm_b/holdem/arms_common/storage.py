"""Where a comparison run puts things, and how they get written safely.

One run is one directory, and every artifact either arm produces lives under
that arm's own folder.  Nothing is shared but the top-level config and the
final results, because the whole point is that the two arms be separately
inspectable after the fact — "which teacher produced this turn label?" and
"what did the labels look like at iteration 30?" are the questions this layout
exists to answer::

    <run>/
      config.json                  the ComparisonConfig this run was launched with
      results.json                 the final ComparisonResult
      relabel.json                 the staleness probe (see compare/relabel.py)
      held_out.json                the evaluation boards, listed once
      arm_fixed/
        dataset/                   frozen labels: sharded .npy + manifest.json
        teachers/                  the networks that produced those labels
          teacher-after-river.pt
          teacher-after-turn.pt
        student.pt                 final student weights
        history.json               per-stage training record
      arm_iterative/
        journal/                   every label the loop generated, in order
          manifest.json
          iter-00001-features.npy ...
        checkpoints/               student weights over time
          student-iter-00010.pt ...
        student.pt
        history.json

Writes go through :func:`write_json` and :func:`save_checkpoint`, which write to
a temporary sibling and :func:`os.replace` it into position.  A reader therefore
never sees a half-written file, and a crash mid-write leaves the previous
version intact — worth the few extra lines when a run may have spent hours
producing the thing being written.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import torch

PathLike = Union[str, Path]


def write_json(data: Any, path: PathLike, overwrite: bool = True) -> Path:
    """Serialise ``data`` to ``path`` atomically."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True, default=_fallback)
    _atomic_write(path, lambda handle: handle.write(payload), text=True)
    return path


def read_json(path: PathLike) -> Any:
    return json.loads(Path(path).read_text())


def save_checkpoint(state: Mapping[str, Any], path: PathLike) -> Path:
    """Persist a ``state_dict`` (or anything torch can pickle) atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, lambda handle: torch.save(dict(state), handle), text=False)
    return path


def load_checkpoint(path: PathLike, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def _fallback(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _atomic_write(path: Path, write, text: bool) -> None:
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w" if text else "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


@dataclass(frozen=True)
class RunLayout:
    """Every path in one run, derived from its root.  Creates nothing until asked."""

    root: Path

    @classmethod
    def at(cls, root: PathLike) -> "RunLayout":
        return cls(root=Path(root))

    # -- top level --------------------------------------------------------
    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def results(self) -> Path:
        return self.root / "results.json"

    @property
    def relabel(self) -> Path:
        return self.root / "relabel.json"

    @property
    def held_out(self) -> Path:
        return self.root / "held_out.json"

    # -- arm 1: fixed -----------------------------------------------------
    @property
    def fixed(self) -> Path:
        return self.root / "arm_fixed"

    @property
    def dataset(self) -> Path:
        return self.fixed / "dataset"

    @property
    def teachers(self) -> Path:
        return self.fixed / "teachers"

    def teacher(self, after_source: str) -> Path:
        return self.teachers / f"teacher-after-{after_source}.pt"

    @property
    def fixed_student(self) -> Path:
        return self.fixed / "student.pt"

    @property
    def fixed_history(self) -> Path:
        return self.fixed / "history.json"

    # -- arm 2: iterative -------------------------------------------------
    @property
    def iterative(self) -> Path:
        return self.root / "arm_iterative"

    @property
    def journal(self) -> Path:
        return self.iterative / "journal"

    @property
    def iterative_checkpoints(self) -> Path:
        return self.iterative / "checkpoints"

    def iterative_checkpoint(self, iteration: int) -> Path:
        return self.iterative_checkpoints / f"student-iter-{iteration:05d}.pt"

    @property
    def iterative_student(self) -> Path:
        return self.iterative / "student.pt"

    @property
    def iterative_history(self) -> Path:
        return self.iterative / "history.json"

    def prepare(self) -> "RunLayout":
        """Create the directory skeleton.  Safe to call on an existing run."""
        for directory in (
            self.root,
            self.fixed,
            self.teachers,
            self.iterative,
            self.iterative_checkpoints,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self
