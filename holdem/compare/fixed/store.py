"""A reusable, provenance-rich dataset artifact built from :class:`Examples`.

Exact river solves and staged teacher-backed turn/flop solves are each
expensive enough that relabeling to compare training regimes would be
wasteful.  This module gives them a home on disk so they can be built once
and reused: a directory of source-separated ``.npy`` shards plus one JSON
manifest describing what is in them and how it was made.

Design choices, and why:

* **Shards, not one array.**  Each :meth:`DatasetStore.append` call writes a
  new set of files rather than rewriting the whole source.  That keeps
  generation embarrassingly appendable and lets a shard be sampled — or
  skipped — without touching its neighbours.
* **Source-separated.**  River, turn and flop data answer different
  questions and are trusted to different degrees (river is exact, the rest
  is only as good as the teacher that produced it), so keeping them apart
  mirrors :class:`holdem.curriculum.MixedBuffer` and lets a caller weight
  them explicitly rather than pooling silently.
* **A fourth source, ``self_play``.**  River/turn/flop are sampled from a
  distribution somebody designed; the online arm's data comes from belief
  states its own search actually reached.  Without a frozen copy of the
  latter, the two arms differ in *two* ways at once — stale labels and a
  different input distribution — and a win for either one cannot be
  attributed.  Filling this source lets a third arm hold the distribution
  fixed and vary only label freshness, which is the question being asked.
* **Read-only and memory-mapped on reopen.**  ``open()`` never loads a whole
  shard into RAM and never writes to it; sampling only ever pages in the
  rows it draws, which is what "sample without concatenating the full
  dataset" requires as shards grow.
* **Compact masks on disk.**  A mask is exactly 0.0 or 1.0 — legal or
  card-blocked — so it is stored as a single byte per combo (``bool``)
  rather than four (``float32``), and cast back to float on the way out so
  every consumer keeps seeing the arrays :class:`Examples` has always had.
* **Strict validation.**  A manifest is the only thing standing between a
  caller and silently training on data that does not match its own
  bookkeeping, so schema mismatches, unknown sources, missing/incomplete
  shards, empty artifacts, malformed weights and shape mismatches all raise
  rather than degrade quietly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from holdem.combos import NUM_COMBOS
from holdem.features import INPUT_DIM
from holdem.generation import Examples

SCHEMA_VERSION = 2
KNOWN_SOURCES: tuple[str, ...] = ("river", "turn", "flop", "self_play")
# The streets, lowest first: the order they must be generated in, because each
# is bootstrapped from the teacher fitted on the one below it.
STREET_SOURCES: tuple[str, ...] = ("river", "turn", "flop")
MANIFEST_FILENAME = "manifest.json"

PathLike = Union[str, Path]


@dataclass
class DatasetManifest:
    """Everything needed to reopen and trust an artifact without relabeling.

    ``sources`` holds the total example count per source; ``shards`` holds
    the on-disk bookkeeping (filenames and per-shard counts) that makes those
    totals checkable.  ``generation`` and ``seeds`` record, per source, the
    configuration and seed that produced it; ``teacher_stages`` is an ordered
    log of every teacher fit that happened between stages, so a turn or flop
    label can be traced back to the network that produced it.
    """

    schema_version: int = SCHEMA_VERSION
    sources: Dict[str, int] = field(default_factory=dict)
    generation: Dict[str, Any] = field(default_factory=dict)
    seeds: Dict[str, int] = field(default_factory=dict)
    teacher_stages: List[Dict[str, Any]] = field(default_factory=list)
    shards: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sources": dict(self.sources),
            "generation": dict(self.generation),
            "seeds": dict(self.seeds),
            "teacher_stages": [dict(stage) for stage in self.teacher_stages],
            "shards": {source: [dict(s) for s in shards] for source, shards in self.shards.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DatasetManifest":
        required = (
            "schema_version",
            "sources",
            "generation",
            "seeds",
            "teacher_stages",
            "shards",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"manifest is incomplete: missing field(s) {missing}")
        return cls(
            schema_version=int(data["schema_version"]),
            sources=dict(data["sources"]),
            generation=dict(data.get("generation", {})),
            seeds=dict(data.get("seeds", {})),
            teacher_stages=[dict(s) for s in data.get("teacher_stages", [])],
            shards={source: [dict(s) for s in shards] for source, shards in data["shards"].items()},
        )


class DatasetStore:
    """Source-separated, sharded, memory-mappable store for :class:`Examples`.

    Use :meth:`create` to start a fresh artifact and :meth:`append` to add a
    stage's examples to it; use :meth:`open` to reopen an existing one
    read-only for sampling.  Not thread-safe: one writer at a time.
    """

    def __init__(
        self,
        path: PathLike,
        manifest: DatasetManifest,
        sources: Optional[Sequence[str]] = None,
        read_only: bool = False,
    ) -> None:
        if manifest.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema version {manifest.schema_version}, "
                f"expected {SCHEMA_VERSION}"
            )
        self.path = Path(path)
        self.manifest = manifest
        self.allowed_sources = tuple(sources) if sources is not None else KNOWN_SOURCES
        referenced = set(self.manifest.sources) | set(self.manifest.shards)
        unknown = sorted(referenced - set(self.allowed_sources))
        if unknown:
            raise ValueError(f"manifest references unknown source(s): {unknown}")
        self.read_only = read_only
        self._shards: Dict[str, List[Dict[str, np.ndarray]]] = {}
        self._load_shards()

    # -- construction -------------------------------------------------
    @classmethod
    def create(
        cls,
        path: PathLike,
        manifest: Optional[DatasetManifest] = None,
        sources: Optional[Sequence[str]] = None,
    ) -> "DatasetStore":
        path = Path(path)
        if path.exists():
            if not path.is_dir() or any(path.iterdir()):
                raise FileExistsError(f"dataset artifact path already exists: {path}")
        else:
            path.mkdir(parents=True)
        manifest = manifest or DatasetManifest()
        store = cls(path, manifest, sources=sources, read_only=False)
        store._write_manifest()
        return store

    @classmethod
    def open(
        cls,
        path: PathLike,
        sources: Optional[Sequence[str]] = None,
        read_only: bool = True,
    ) -> "DatasetStore":
        path = Path(path)
        manifest_path = path / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest found at {manifest_path}")
        manifest = DatasetManifest.from_dict(json.loads(manifest_path.read_text()))
        store = cls(path, manifest, sources=sources, read_only=read_only)
        store._validate()
        return store

    # -- disk I/O -------------------------------------------------------
    def _write_manifest(self) -> None:
        (self.path / MANIFEST_FILENAME).write_text(json.dumps(self.manifest.to_dict(), indent=2))

    def _load_shards(self) -> None:
        self._shards = {}
        for source, entries in self.manifest.shards.items():
            loaded = []
            for entry in entries:
                loaded.append(
                    {
                        "count": int(entry["count"]),
                        "features": np.load(self.path / entry["features"], mmap_mode="r"),
                        "masks": np.load(self.path / entry["masks"], mmap_mode="r"),
                        "targets": np.load(self.path / entry["targets"], mmap_mode="r"),
                    }
                )
            self._shards[source] = loaded

    def _validate(self) -> None:
        if not self.manifest.sources or sum(self.manifest.sources.values()) == 0:
            raise ValueError("dataset artifact is empty: no examples in any source")
        for source, declared_count in self.manifest.sources.items():
            entries = self.manifest.shards.get(source)
            if not entries:
                raise ValueError(
                    f"manifest is incomplete/incompatible: source {source!r} has a "
                    "declared count but no shards"
                )
            shard_total = sum(int(e["count"]) for e in entries)
            if shard_total != declared_count:
                raise ValueError(
                    f"manifest is incomplete/incompatible for source {source!r}: "
                    f"declared {declared_count} example(s) but shards hold {shard_total}"
                )
            shards = self._shards[source]
            reference_shapes = None
            for entry, shard in zip(entries, shards):
                n = int(entry["count"])
                shapes = (shard["features"].shape, shard["masks"].shape, shard["targets"].shape)
                if shapes[0][0] != n or shapes[1][0] != n or shapes[2][0] != n:
                    raise ValueError(
                        f"mismatched shapes for source {source!r}: shard declares "
                        f"{n} example(s) but arrays have shapes {shapes}"
                    )
                trailing = tuple(s[1:] for s in shapes)
                if reference_shapes is None:
                    reference_shapes = trailing
                elif trailing != reference_shapes:
                    raise ValueError(
                        f"mismatched shapes across shards for source {source!r}: "
                        f"{trailing} vs {reference_shapes}"
                    )

    # -- writing ----------------------------------------------------------
    def append(
        self,
        source: str,
        examples: Examples,
        *,
        seed: Optional[int] = None,
        generation: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Persist ``examples`` as a new shard of ``source``.

        Masks are cast to ``bool`` on disk and back to ``float32`` on read;
        every other array keeps :class:`Examples`'s ``float32`` convention.
        """
        if self.read_only:
            raise RuntimeError("cannot append to a read-only dataset store")
        if source not in self.allowed_sources:
            raise ValueError(f"unknown source {source!r}; expected one of {self.allowed_sources}")
        if len(examples) == 0:
            raise ValueError("cannot append an empty batch of examples")
        self._validate_examples(examples)

        existing_shards = self._shards.get(source, [])
        if existing_shards:
            reference = existing_shards[0]
            if examples.features.shape[1:] != reference["features"].shape[1:]:
                raise ValueError(
                    f"feature shape mismatch for source {source!r}: "
                    f"{examples.features.shape[1:]} vs {reference['features'].shape[1:]}"
                )
            if examples.masks.shape[1:] != reference["masks"].shape[1:]:
                raise ValueError(
                    f"mask shape mismatch for source {source!r}: "
                    f"{examples.masks.shape[1:]} vs {reference['masks'].shape[1:]}"
                )
            if examples.targets.shape[1:] != reference["targets"].shape[1:]:
                raise ValueError(
                    f"target shape mismatch for source {source!r}: "
                    f"{examples.targets.shape[1:]} vs {reference['targets'].shape[1:]}"
                )

        index = len(self.manifest.shards.get(source, []))
        features_name = f"{source}-{index:05d}-features.npy"
        masks_name = f"{source}-{index:05d}-masks.npy"
        targets_name = f"{source}-{index:05d}-targets.npy"
        np.save(self.path / features_name, np.asarray(examples.features, dtype=np.float32))
        np.save(self.path / masks_name, np.asarray(examples.masks, dtype=np.float32) != 0.0)
        np.save(self.path / targets_name, np.asarray(examples.targets, dtype=np.float32))

        entry = {
            "index": index,
            "count": len(examples),
            "features": features_name,
            "masks": masks_name,
            "targets": targets_name,
        }
        self.manifest.shards.setdefault(source, []).append(entry)
        self.manifest.sources[source] = self.manifest.sources.get(source, 0) + len(examples)
        if seed is not None:
            self.manifest.seeds[source] = int(seed)
        if generation is not None:
            self.manifest.generation[source] = dict(generation)
        self._write_manifest()
        self._load_shards()

    @staticmethod
    def _validate_examples(examples: Examples) -> None:
        n = len(examples)
        expected = (
            (n, INPUT_DIM),
            (n, NUM_COMBOS),
            (n, 2, NUM_COMBOS),
        )
        actual = (
            examples.features.shape,
            examples.masks.shape,
            examples.targets.shape,
        )
        if actual != expected:
            raise ValueError(
                "examples do not match the expected features/masks/targets "
                f"shapes: expected {expected}, got {actual}"
            )

    def record_teacher_stage(self, info: Mapping[str, Any]) -> None:
        """Log one teacher fit in the provenance trail (e.g. "fit on river")."""
        if self.read_only:
            raise RuntimeError("cannot update a read-only dataset store")
        self.manifest.teacher_stages.append(dict(info))
        self._write_manifest()

    # -- reading ------------------------------------------------------
    def counts(self) -> Dict[str, int]:
        return dict(self.manifest.sources)

    def __len__(self) -> int:
        return sum(self.manifest.sources.values())

    def read_all(self, source: str) -> Examples:
        """Every example of one source, concatenated into memory.

        For verification and small artifacts.  Ordinary training should
        prefer :meth:`sample`, which never concatenates the full dataset.
        """
        shards = self._shards.get(source)
        if not shards:
            raise ValueError(f"source {source!r} has no examples")
        parts = [
            Examples(
                np.asarray(shard["features"], dtype=np.float32),
                np.asarray(shard["masks"], dtype=np.float32),
                np.asarray(shard["targets"], dtype=np.float32),
            )
            for shard in shards
        ]
        return Examples.concatenate(parts)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        source_weights: Optional[Mapping[str, float]] = None,
    ) -> Examples:
        """A batch drawn across sources (and across each source's shards).

        Rows are gathered with fancy indexing on the memory-mapped shard
        arrays, so only the sampled rows are ever paged into memory — the
        full dataset is never concatenated, however many shards it has.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        available = {s: c for s, c in self.manifest.sources.items() if c > 0}
        if not available:
            raise ValueError("cannot sample from an empty dataset")

        if source_weights is None:
            weights = {s: 1.0 for s in available}
        else:
            unknown = sorted(set(source_weights) - set(self.allowed_sources))
            if unknown:
                raise ValueError(f"source_weights references unknown source(s): {unknown}")
            if any(not np.isfinite(w) or w < 0 for w in source_weights.values()):
                raise ValueError("source weights must be finite and non-negative")
            weights = {s: float(w) for s, w in source_weights.items() if s in available and w > 0}
            if not weights:
                raise ValueError("source_weights has no positive weight on an available source")

        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("source weights must sum to a positive value")
        normalised = {s: w / total_weight for s, w in weights.items()}
        sources = tuple(normalised)
        drawn = rng.multinomial(batch_size, [normalised[source] for source in sources])
        counts = {
            source: int(count)
            for source, count in zip(sources, drawn)
            if count > 0
        }

        parts = [self._sample_source(source, count, rng) for source, count in counts.items()]
        return Examples.concatenate(parts)

    def gather(self, source: str, rows: Sequence[int]) -> Examples:
        """Specific rows of ``source``, by index across its shards in order.

        Lets a caller pin an exact subset of the artifact and draw only from
        it — which is what spending a *label budget* smaller than the whole
        dataset has to mean, if the number is to be comparable with an arm that
        generates exactly that many labels and no more.  Rows are gathered from
        the memory maps, so the full dataset is still never materialised.
        """
        shards = self._shards.get(source, [])
        if not shards:
            raise ValueError(f"source {source!r} has no examples")
        counts = np.array([s["count"] for s in shards], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size and (rows.min() < 0 or rows.max() >= offsets[-1]):
            raise IndexError(
                f"row index out of range for source {source!r}: "
                f"have {offsets[-1]} example(s)"
            )

        parts = []
        which = np.searchsorted(offsets, rows, side="right") - 1
        for shard_index in np.unique(which):
            local = rows[which == shard_index] - offsets[shard_index]
            order = np.argsort(local)  # mmap fancy-indexing wants sorted rows
            shard = shards[int(shard_index)]
            picked = local[order]
            parts.append(
                Examples(
                    features=np.asarray(shard["features"][picked], dtype=np.float32),
                    masks=np.asarray(shard["masks"][picked], dtype=np.float32),
                    targets=np.asarray(shard["targets"][picked], dtype=np.float32),
                )
            )
        return Examples.concatenate(parts)

    def _sample_source(self, source: str, count: int, rng: np.random.Generator) -> Examples:
        shards = self._shards.get(source, [])
        shard_counts = np.array([s["count"] for s in shards], dtype=np.float64)
        total = shard_counts.sum() if len(shard_counts) else 0.0
        if total <= 0:
            raise ValueError(f"source {source!r} has no examples to sample from")
        probabilities = shard_counts / total
        shard_choices = rng.choice(len(shards), size=count, p=probabilities)

        features, masks, targets = [], [], []
        for shard_index in np.unique(shard_choices):
            n = int((shard_choices == shard_index).sum())
            shard = shards[int(shard_index)]
            rows = rng.integers(0, shard["count"], size=n)
            features.append(np.asarray(shard["features"][rows], dtype=np.float32))
            masks.append(np.asarray(shard["masks"][rows], dtype=np.float32))
            targets.append(np.asarray(shard["targets"][rows], dtype=np.float32))
        return Examples(
            features=np.concatenate(features),
            masks=np.concatenate(masks),
            targets=np.concatenate(targets),
        )
