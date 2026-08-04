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
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union

import numpy as np
import torch

from paradigm_b.holdem.arms_common.budget import SpendRecord

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


# --- resuming a long run ----------------------------------------------------
#
# A 12-hour generation run that dies at hour 11 and has to start again is the
# difference between an overnight experiment and a lost day.  Weights alone are
# not enough to continue from: restart with a fresh Adam and the first few
# hundred steps undo themselves while the moment estimates rebuild, and restart
# with an empty replay buffer and the network is briefly trained on nothing but
# the newest, most correlated labels.  So the whole learner is written down --
# both networks, both optimisers, both buffers, the spend, the iteration counter
# and the RNG.
#
# The value buffer used to dominate the bytes, and used to be copied here in
# full on every save.  It no longer is.  It owns its own persistence now
# (:meth:`paradigm_b.holdem.data.store.ReplayBuffer.write_state`), which for the
# in-memory form means canonical fp16 rows at 2.5x less than the float32
# encoding-plus-mask it replaces, and for the sharded form means a manifest and
# no copy at all -- the shards are already durable, already outside this
# directory, and staging a second copy of them would be the largest single cost
# in a long run.  The policy buffer is small and unchanged.
#
# 3: the value buffer's rows became canonical fp16 without a mask, so a state
# written under 2 holds arrays this build cannot read as its buffer.

RUN_STATE_VERSION = 3


def _buffer_order(size: int, capacity: int, next_index: int) -> np.ndarray:
    """Indices of a circular buffer's contents, oldest first."""
    if size < capacity:
        return np.arange(size)
    return np.concatenate([np.arange(next_index, capacity), np.arange(0, next_index)])


def _save_circular(directory: Path, name: str, buffer, fields: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """Write a circular buffer's live contents in oldest-first order.

    Re-ordering rather than dumping the raw arrays is what makes the saved form
    independent of where the write pointer happened to be, so a resumed buffer
    evicts and purges in the same order it would have.  ``purge_oldest`` depends
    on that ordering being recoverable, and a raw dump loses it.
    """
    order = _buffer_order(buffer.size, buffer.capacity, buffer._next)
    for field_name, array in fields.items():
        np.save(directory / f"{name}-{field_name}.npy", array[order])
    return {"size": int(buffer.size), "capacity": int(buffer.capacity)}


def _load_circular(directory: Path, name: str, buffer, fields: Sequence[str], meta: Mapping[str, Any]) -> None:
    size = int(meta["size"])
    if size > buffer.capacity:
        raise ValueError(
            f"saved {name} holds {size} examples but this run's capacity is "
            f"{buffer.capacity}; raise buffer_size or the oldest data would be "
            f"silently dropped on resume"
        )
    if hasattr(buffer, "_ensure_allocated"):
        buffer._ensure_allocated(size)
    for field_name in fields:
        array = np.load(directory / f"{name}-{field_name}.npy")
        getattr(buffer, field_name)[:size] = array
    buffer.size = size
    buffer._next = size % buffer.capacity


def run_state_fingerprint(config) -> Dict[str, Any]:
    """What must match for a resume to be meaningful rather than merely possible.

    The action abstraction is in here for a specific reason: action ids are
    positional, so a buffer of policy targets recorded under
    ``(0.5, 1.0)`` is not readable under ``(0.25, 0.5, 1.0, 2.0)`` -- id 2 means
    a different bet and all-in has moved.  Resuming across that would train on
    silently relabelled data and look like nothing was wrong.
    """
    value_net = config.value_net
    return {
        "version": RUN_STATE_VERSION,
        "hidden_dim": int(value_net.hidden_dim),
        "num_residual_blocks": int(value_net.num_residual_blocks),
        "card_embedding_dim": int(value_net.card_embedding_dim),
        "bet_fractions": [float(f) for f in config.situations.bet_fractions],
        "max_raises": int(config.situations.max_raises),
        "buffer_size": int(config.buffer_size),
        "uses_policy_net": bool(config.uses_policy_net),
    }


def save_run_state(
    directory: PathLike,
    *,
    config,
    spend,
    iteration: int,
    trajectory_id: int,
    rng,
    # The stream the buffer's read-time isomorphisms are drawn from.  Saved with
    # everything else, because a resume that reset it would re-draw the same
    # transforms over rows it has already served under them.
    augment_rng=None,
    net,
    optimiser,
    buffer,
    initial_state: Mapping[str, "torch.Tensor"],
    policy_net=None,
    policy_optimiser=None,
    policy_buffer=None,
) -> Path:
    """Write everything needed to continue this run, swapped in atomically."""
    final = Path(directory)
    staging = final.with_name(final.name + ".writing")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    torch.save(net.state_dict(), staging / "value_net.pt")
    torch.save(optimiser.state_dict(), staging / "value_optimiser.pt")
    torch.save(dict(initial_state), staging / "initial_state.pt")
    meta: Dict[str, Any] = {
        "fingerprint": run_state_fingerprint(config),
        "spend": spend.to_dict(),
        "iteration": int(iteration),
        "trajectory_id": int(trajectory_id),
        "rng": rng.bit_generator.state,
        # The buffer decides what "saving" means for it.  A sharded buffer
        # returns a manifest and writes no rows here at all, which is what keeps
        # ``state_every`` from costing a full buffer copy.
        "buffer": buffer.write_state(staging),
    }
    if augment_rng is not None:
        meta["augment_rng"] = augment_rng.bit_generator.state
    if policy_net is not None:
        torch.save(policy_net.state_dict(), staging / "policy_net.pt")
    if policy_optimiser is not None:
        torch.save(policy_optimiser.state_dict(), staging / "policy_optimiser.pt")
    if policy_buffer is not None:
        meta["policy_buffer"] = _save_circular(
            staging,
            "policy",
            policy_buffer,
            {
                "features": policy_buffer.features,
                "agent_indices": policy_buffer.agent_indices,
                "legal_masks": policy_buffer.legal_masks,
                "targets": policy_buffer.targets,
            },
        )
    write_json(meta, staging / "meta.json")

    # Swap last, so a crash at any point above leaves the previous state whole.
    previous = final.with_name(final.name + ".previous")
    if previous.exists():
        shutil.rmtree(previous)
    if final.exists():
        os.rename(final, previous)
    os.rename(staging, final)
    if previous.exists():
        shutil.rmtree(previous)
    return final


def load_run_state(
    directory: PathLike,
    *,
    config,
    net,
    optimiser,
    buffer,
    rng,
    augment_rng=None,
    policy_net=None,
    policy_optimiser=None,
    policy_buffer=None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Restore a saved run into the objects given, in place.

    Returns the scalars the caller has to carry itself: the spend record, the
    iteration and trajectory counters, and the weights the run originally
    started from.
    """
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())

    expected = run_state_fingerprint(config)
    saved = meta.get("fingerprint", {})
    if strict and saved != expected:
        differing = {
            key: (saved.get(key), expected.get(key))
            for key in set(saved) | set(expected)
            if saved.get(key) != expected.get(key)
        }
        raise ValueError(
            f"refusing to resume {directory}: the saved run does not match this "
            f"config (saved, wanted) = {differing}"
        )

    net.load_state_dict(torch.load(directory / "value_net.pt", map_location="cpu"))
    optimiser.load_state_dict(
        torch.load(directory / "value_optimiser.pt", map_location="cpu")
    )
    buffer.read_state(directory, meta["buffer"])
    if policy_net is not None and (directory / "policy_net.pt").exists():
        policy_net.load_state_dict(
            torch.load(directory / "policy_net.pt", map_location="cpu")
        )
    if policy_optimiser is not None and (directory / "policy_optimiser.pt").exists():
        policy_optimiser.load_state_dict(
            torch.load(directory / "policy_optimiser.pt", map_location="cpu")
        )
    if policy_buffer is not None and "policy_buffer" in meta:
        _load_circular(
            directory,
            "policy",
            policy_buffer,
            ("features", "agent_indices", "legal_masks", "targets"),
            meta["policy_buffer"],
        )
    rng.bit_generator.state = meta["rng"]
    if augment_rng is not None and "augment_rng" in meta:
        augment_rng.bit_generator.state = meta["augment_rng"]

    initial_state = torch.load(directory / "initial_state.pt", map_location="cpu")
    return {
        "spend": SpendRecord.from_dict(meta["spend"]),
        "iteration": int(meta["iteration"]),
        "trajectory_id": int(meta["trajectory_id"]),
        "initial_state": initial_state,
    }
