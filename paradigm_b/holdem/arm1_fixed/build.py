"""Arm 1, step one: build the frozen dataset, and keep the teachers that made it.

This is the "label the world once, then never solve again" regime.  It runs the
streets in order, lowest first, because each is bootstrapped from the one below::

    river   solved exactly, to real showdowns          -> ground truth
      |     fit teacher                                -> teachers/teacher-after-river.pt
    turn    depth-limited solve, river leaves priced
            by that teacher                            -> frozen labels
      |     refit teacher (river + turn)               -> teachers/teacher-after-turn.pt
    flop    depth-limited solve, turn leaves priced
            by that teacher                            -> frozen labels

**Order is not negotiable.**  A turn label built on a bad river network is
confident nonsense, and freezing it teaches the error rather than the value.

**There are two teachers, not three.**  Each one exists to price the leaves of
the street above it, and the flop is the top street — nothing is bootstrapped
from it, so no teacher is fitted after it.  The exception is the optional
``self_play`` control source, which is generated last and does need an
evaluator; enabling it adds a third.  Note also that the teacher is one network
trained cumulatively, not a fresh network per street: ``teacher-after-turn`` is
``teacher-after-river`` trained further on river *and* turn data.

**The teachers are saved, not discarded.**  They are the only record of *what
produced each label*.  Without them the staleness probe in
``compare/relabel.py`` has nothing to compare against, and a turn label is an
unattributable number: you cannot ask "how wrong was the network that wrote
this?" once that network is gone.  Each one is also logged in the store's
manifest via ``record_teacher_stage``, so the provenance survives even if the
directory is moved.

Note what happens at the end: the teachers are set aside and a **fresh** student
is trained on the frozen file alone (see ``student.py``).  The student never
sees a solver.  That is the regime being tested — not "train a network with a
curriculum", but "is a dataset of labels a reusable asset?"
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from paradigm_b.holdem.data.bootstrap import BootstrapConfig, generate_bootstrapped
from paradigm_b.holdem.arms_common.budget import CountingLeafValues, SpendRecord
from paradigm_b.holdem.arms_common.fitting import fit_value_net
from paradigm_b.holdem.arms_common.situations import StreetMix, sample_mixed_situation
from paradigm_b.holdem.arms_common.storage import PathLike, RunLayout, save_checkpoint
from paradigm_b.holdem.arm1_fixed.store import STREET_SOURCES, DatasetStore
from paradigm_b.holdem.data.generation import Examples, GenerationConfig, generate
from paradigm_b.holdem.net.value_net import HoldemValueNet, HoldemValueNetConfig
from paradigm_b.holdem.selfplay import HoldemSelfPlayConfig, collect_trajectory
from paradigm_b.holdem.data.sampling import SituationConfig
from paradigm_b.holdem.net.leaf_values import NetLeafValues

BOARD_CARDS: Dict[str, int] = {"river": 5, "turn": 4, "flop": 3}


@dataclass
class DatasetBuildConfig:
    """How large the frozen artifact is, and how good its teachers get.

    ``teacher_updates`` is the one knob that decides how strong the labels
    above the river are.  Starving it produces exactly the failure this whole
    experiment is designed to detect, so a run that wants the offline arm to
    look its best should spend generously here — it is a one-time cost that the
    artifact amortises over every student ever trained on it.
    """

    river_examples: int = 40_000
    turn_examples: int = 8_000
    flop_examples: int = 4_000
    # The control source: frozen on-policy belief states.  Left at zero the
    # artifact is river/turn/flop only, and the two arms differ in both label
    # freshness and input distribution.  See ``store.py``'s docstring.
    self_play_examples: int = 0

    teacher_updates: int = 4_000
    teacher_batch_size: int = 128
    teacher_learning_rate: float = 1e-3

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    bootstrap_cfr_iterations: int = 40
    situations_per_board: int = 8
    self_play: HoldemSelfPlayConfig = field(default_factory=HoldemSelfPlayConfig)
    situations: SituationConfig = field(default_factory=SituationConfig)
    value_net: HoldemValueNetConfig = field(default_factory=HoldemValueNetConfig)
    workers: int = 8
    seed: int = 0
    device: str = "cpu"

    def counts(self) -> Dict[str, int]:
        counts = {
            "river": self.river_examples,
            "turn": self.turn_examples,
            "flop": self.flop_examples,
            "self_play": self.self_play_examples,
        }
        return {source: n for source, n in counts.items() if n > 0}

    @property
    def total_examples(self) -> int:
        return sum(self.counts().values())


def slice_examples(examples: Examples, count: int) -> Examples:
    """The first ``count`` rows, so a stage lands on an exact size."""
    return Examples(
        features=examples.features[:count],
        masks=examples.masks[:count],
        targets=examples.targets[:count],
    )


def _generate_exactly(make, count: int) -> Examples:
    """Call ``make(remaining, attempt)`` until ``count`` examples exist.

    Neither generator promises an exact size: :func:`holdem.generation.generate`
    rounds up to whole boards, and :func:`~holdem.bootstrap.generate_bootstrapped`
    silently drops situations whose range mass collapses to zero.  The offline
    arm's budget check demands an exact total, so the shortfall is topped up
    rather than papered over with a tolerance.
    """
    parts: List[Examples] = []
    have = 0
    attempt = 0
    while have < count:
        produced = make(count - have, attempt)
        attempt += 1
        if not len(produced):
            if attempt > 20:
                raise RuntimeError(
                    f"generator produced nothing for {attempt} attempts; "
                    f"wanted {count} examples, have {have}"
                )
            continue
        parts.append(produced)
        have += len(produced)
    return slice_examples(Examples.concatenate(parts), count)


def _store_sampler(store: DatasetStore, source_weights: Optional[Dict[str, float]] = None):
    def sample(batch_size: int, rng: np.random.Generator):
        batch = store.sample(batch_size, rng, source_weights)
        return batch.features, batch.masks, batch.targets

    return sample


def _fit_teacher(
    net: HoldemValueNet,
    store: DatasetStore,
    config: DatasetBuildConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    """Bring the teacher up to date on everything generated so far."""
    optimiser = torch.optim.Adam(net.parameters(), lr=config.teacher_learning_rate)
    total, steps = fit_value_net(
        net,
        optimiser,
        nn.HuberLoss(reduction="mean"),
        _store_sampler(store),
        config.teacher_updates,
        config.teacher_batch_size,
        device,
        rng,
    )
    net.eval()
    return total / max(steps, 1)


def _bootstrap_source(
    source: str,
    count: int,
    teacher: HoldemValueNet,
    config: DatasetBuildConfig,
    excluded_boards: Tuple[Tuple[int, ...], ...],
    spend: SpendRecord,
    seed: int,
    device: str,
) -> Examples:
    board_cards = BOARD_CARDS[source]
    leaf_values = CountingLeafValues(NetLeafValues(teacher, device=device), spend)
    bootstrap = BootstrapConfig(
        board_cards=board_cards,
        situations_per_board=config.situations_per_board,
        cfr_iterations=config.bootstrap_cfr_iterations,
        num_rounds=6 - board_cards,
        excluded_boards=excluded_boards,
    )
    return _generate_exactly(
        lambda remaining, attempt: generate_bootstrapped(
            leaf_values, remaining, bootstrap, seed=seed + attempt
        ),
        count,
    )


def _self_play_source(
    count: int,
    teacher: HoldemValueNet,
    config: DatasetBuildConfig,
    excluded_boards: Tuple[Tuple[int, ...], ...],
    mix: StreetMix,
    spend: SpendRecord,
    seed: int,
    device: str,
) -> Examples:
    """Frozen on-policy belief states, drawn with the finished teacher.

    Deliberately the *same* generator the online arm uses, run once and frozen,
    so a third arm can hold the input distribution fixed and vary only whether
    the labels are refreshed.
    """
    rng = np.random.default_rng(seed)
    leaf_values = CountingLeafValues(NetLeafValues(teacher, device=device), spend)
    situations = replace(config.situations, excluded_boards=excluded_boards)
    features, masks, targets = [], [], []
    while len(features) < count:
        space, root, reach, _ = sample_mixed_situation(rng, situations, mix)
        for example in collect_trajectory(
            leaf_values, space, root, config.self_play, rng, reach=reach
        ):
            features.append(example.features)
            masks.append(example.mask)
            targets.append(example.values)
    return slice_examples(
        Examples(
            features=np.stack(features).astype(np.float32),
            masks=np.stack(masks).astype(np.float32),
            targets=np.stack(targets).astype(np.float32),
        ),
        count,
    )


def build_layered_dataset(
    path: PathLike,
    config: DatasetBuildConfig,
    excluded_boards: Tuple[Tuple[int, ...], ...] = (),
    teachers_path: Optional[PathLike] = None,
) -> Tuple[DatasetStore, SpendRecord]:
    """Generate the frozen artifact at ``path``; save its teachers beside it.

    ``excluded_boards`` is the held-out set, threaded into every generator so
    the artifact provably cannot contain a board the comparison tests on.
    """
    device = torch.device(config.device)
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    spend = SpendRecord()

    store = DatasetStore.create(path)
    teachers = Path(teachers_path) if teachers_path is not None else Path(path).parent / "teachers"
    teachers.mkdir(parents=True, exist_ok=True)

    teacher = HoldemValueNet(config.value_net).to(device)
    teacher.eval()
    counts = config.counts()
    started = time.perf_counter()
    last_teacher: Optional[str] = None

    for position, source in enumerate(STREET_SOURCES):
        count = counts.get(source, 0)
        if not count:
            continue
        seed = int(rng.integers(1 << 31))

        if source == "river":
            generation = replace(config.generation, excluded_boards=excluded_boards)
            examples = _generate_exactly(
                lambda remaining, attempt: generate(
                    remaining, generation, seed=seed + attempt, workers=config.workers
                ),
                count,
            )
            provenance = asdict(generation)
        else:
            examples = _bootstrap_source(
                source, count, teacher, config, excluded_boards, spend, seed, config.device
            )
            provenance = {
                "board_cards": BOARD_CARDS[source],
                "cfr_iterations": config.bootstrap_cfr_iterations,
                "situations_per_board": config.situations_per_board,
                "teacher": last_teacher,
            }

        store.append(source, examples, seed=seed, generation=provenance)

        # Only fit a teacher if something later actually consumes it.  The top
        # street bootstraps nothing above itself, so fitting after it would
        # spend ``teacher_updates`` gradient steps on a network that prices no
        # labels — unless the ``self_play`` control source is enabled, which is
        # generated last and does use it.
        if not _teacher_is_consumed(position, counts):
            continue
        loss = _fit_teacher(teacher, store, config, rng, device)
        checkpoint = teachers / f"teacher-after-{source}.pt"
        save_checkpoint(teacher.state_dict(), checkpoint)
        last_teacher = checkpoint.name
        store.record_teacher_stage(
            {
                "after_source": source,
                "checkpoint": checkpoint.name,
                "updates": config.teacher_updates,
                "batch_size": config.teacher_batch_size,
                "learning_rate": config.teacher_learning_rate,
                "mean_loss": loss,
                "trained_on": store.counts(),
            }
        )

    if counts.get("self_play"):
        seed = int(rng.integers(1 << 31))
        mix = StreetMix.from_counts({k: v for k, v in counts.items() if k in STREET_SOURCES})
        examples = _self_play_source(
            counts["self_play"], teacher, config, excluded_boards, mix, spend, seed, config.device
        )
        store.append(
            "self_play",
            examples,
            seed=seed,
            generation={
                "street_mix": mix.to_dict(),
                "search_iterations": config.self_play.search_iterations,
                "river_iterations": config.self_play.river_iterations,
                "teacher": last_teacher,
            },
        )

    spend.generation_seconds = time.perf_counter() - started
    spend.labels = len(store)
    return store, spend


def _teacher_is_consumed(position: int, counts: Dict[str, int]) -> bool:
    """Will anything generated after this street ask the teacher for values?

    The river's teacher prices turn leaves and the turn's prices flop leaves, so
    with the default three streets there are exactly **two** load-bearing
    teachers.  A third is fitted only when ``self_play_examples`` is set, since
    that source is generated last and needs an evaluator of its own.
    """
    later_streets = any(counts.get(s, 0) for s in STREET_SOURCES[position + 1 :])
    return later_streets or bool(counts.get("self_play", 0))
