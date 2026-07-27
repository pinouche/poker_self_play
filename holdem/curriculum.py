"""Training one value network across streets, from four sources of data.

The four sources answer different needs and none of them suffices alone:

``river``      Sampled situations solved exactly.  Ground truth, and cheap
               enough to make in bulk, so it anchors everything above it.
``turn``       Bootstrapped: a depth-limited turn solve that asks the network
               what the river is worth.  Only as good as the river layer.
``flop``       The same, one street higher.
``self_play``  Genuine ReBeL trajectories through belief space.

The first three give **coverage** — a wide sweep of boards and range shapes,
drawn from a distribution someone designed.  The last gives **relevance**: the
belief states the agent's own search actually reaches, with ranges that are the
residue of real betting rather than a Dirichlet draw.  Coverage alone trains for
a distribution that is not the game; relevance alone leaves the network blind the
moment search steps off-policy, which CFR iterates do constantly.  What the
proportion between them should be is an empirical question, so it is a knob.

**Order is not negotiable.**  Bootstrapping propagates error upward: a turn
target built on a bad river network is confident nonsense, and training on it
teaches the error rather than the value.  So the river layer is trained first and
kept in the mixture afterwards, so that it cannot drift while the streets above
it are learning from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from holdem.bootstrap import BootstrapConfig, generate_bootstrapped
from holdem.combos import NUM_COMBOS
from holdem.features import INPUT_DIM
from holdem.generation import Examples, GenerationConfig, generate
from holdem.net import HoldemValueNet, HoldemValueNetConfig
from holdem.values import NetLeafValues

NUM_PLAYERS = 2


@dataclass
class MixConfig:
    """How much of each stage's data to hold, as proportions of the buffer."""

    river: float = 0.5
    turn: float = 0.3
    flop: float = 0.0
    self_play: float = 0.2

    def normalised(self) -> Dict[str, float]:
        total = self.river + self.turn + self.flop + self.self_play
        if total <= 0:
            raise ValueError("at least one source must have weight")
        return {
            "river": self.river / total,
            "turn": self.turn / total,
            "flop": self.flop / total,
            "self_play": self.self_play / total,
        }


class MixedBuffer:
    """Per-source circular buffers, sampled together in fixed proportions.

    Keeping the sources apart rather than pooling them means a burst of cheap
    river data cannot crowd out the expensive self-play examples, and the
    mixture the network sees stays what was asked for regardless of the rate
    each source happens to be produced at.
    """

    def __init__(self, capacity: int, mix: MixConfig) -> None:
        self.mix = mix.normalised()
        self.capacity = capacity
        self._store: Dict[str, Examples] = {}

    def add(self, source: str, examples: Examples) -> None:
        if source not in self.mix:
            raise KeyError(f"unknown source {source!r}")
        if not len(examples):
            return
        existing = self._store.get(source)
        merged = (
            examples if existing is None else Examples.concatenate([existing, examples])
        )
        limit = max(int(self.capacity * self.mix[source]), 1)
        if len(merged) > limit:  # keep the most recent
            merged = Examples(
                merged.features[-limit:], merged.masks[-limit:], merged.targets[-limit:]
            )
        self._store[source] = merged

    def __len__(self) -> int:
        return sum(len(v) for v in self._store.values())

    def counts(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self._store.items()}

    def sample(self, batch_size: int, rng: np.random.Generator):
        """A batch drawn from each source in proportion, skipping empty ones."""
        available = {k: v for k, v in self._store.items() if len(v)}
        if not available:
            raise ValueError("buffer is empty")
        weights = np.array([self.mix[k] for k in available], dtype=np.float64)
        weights = weights / weights.sum()
        counts = np.maximum((weights * batch_size).astype(int), 1)

        parts = []
        for (source, examples), count in zip(available.items(), counts):
            index = rng.integers(0, len(examples), size=count)
            parts.append(
                Examples(
                    examples.features[index],
                    examples.masks[index],
                    examples.targets[index],
                )
            )
        batch = Examples.concatenate(parts)
        return batch.features, batch.masks, batch.targets


@dataclass
class CurriculumConfig:
    """A staged run: river first, then the streets that depend on it."""

    # (source, examples to generate, gradient steps to take afterwards)
    stages: Tuple[Tuple[str, int, int], ...] = (
        ("river", 40_000, 6_000),
        ("turn", 4_000, 4_000),
        ("self_play", 400, 2_000),
    )
    buffer_size: int = 400_000
    batch_size: int = 128
    learning_rate: float = 1e-3
    mix: MixConfig = field(default_factory=MixConfig)
    value_net: HoldemValueNetConfig = field(default_factory=HoldemValueNetConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    workers: int = 8
    seed: int = 0
    device: str = "cpu"


def train_curriculum(
    config: CurriculumConfig | None = None,
    net: Optional[HoldemValueNet] = None,
    evaluate: Optional[Callable[[HoldemValueNet, str], Dict[str, float]]] = None,
    verbose: bool = False,
) -> Tuple[HoldemValueNet, List[Dict[str, float]]]:
    """Generate and train stage by stage, lowest street first."""
    config = config or CurriculumConfig()
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    net = net or HoldemValueNet(config.value_net)
    net.to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    loss_fn = nn.HuberLoss(reduction="mean")
    buffer = MixedBuffer(config.buffer_size, config.mix)
    history: List[Dict[str, float]] = []

    for source, count, steps in config.stages:
        examples = _generate(source, count, net, config, rng)
        buffer.add(source, examples)

        net.train()
        total = 0.0
        for _ in range(steps):
            features, masks, targets = buffer.sample(config.batch_size, rng)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(
                net(
                    torch.as_tensor(features, device=device),
                    torch.as_tensor(masks, device=device),
                ),
                torch.as_tensor(targets, device=device),
            )
            loss.backward()
            optimiser.step()
            total += float(loss.item())

        record = {
            "stage": source,
            "generated": float(len(examples)),
            "steps": float(steps),
            "loss": total / max(steps, 1),
            **{f"buffer_{k}": float(v) for k, v in buffer.counts().items()},
        }
        if evaluate is not None:
            net.eval()
            record.update(evaluate(net, source))
        history.append(record)
        if verbose:
            print(
                "  ".join(
                    f"{k}={v}" if isinstance(v, str) else f"{k}={v:.5g}"
                    for k, v in record.items()
                ),
                flush=True,
            )
    return net, history


def _generate(
    source: str,
    count: int,
    net: HoldemValueNet,
    config: CurriculumConfig,
    rng: np.random.Generator,
) -> Examples:
    seed = int(rng.integers(1 << 31))
    if source == "river":
        return generate(count, config.generation, seed=seed, workers=config.workers)
    if source in ("turn", "flop"):
        leaf_values = NetLeafValues(net, device=config.device)
        bootstrap = BootstrapConfig(
            board_cards=4 if source == "turn" else 3,
            num_rounds=2 if source == "turn" else 3,
            excluded_boards=config.generation.excluded_boards,
        )
        return generate_bootstrapped(leaf_values, count, bootstrap, seed=seed)
    if source == "self_play":
        return _self_play(count, net, config, seed)
    raise KeyError(f"unknown source {source!r}")


def _self_play(
    count: int, net: HoldemValueNet, config: CurriculumConfig, seed: int
) -> Examples:
    """Genuine ReBeL trajectories: descend through belief space and record."""
    from holdem.rebel import HoldemSelfPlayConfig, collect_trajectory
    from holdem.sampling import SituationConfig, sample_situation

    rng = np.random.default_rng(seed)
    leaf_values = NetLeafValues(net, device=config.device)
    situations = SituationConfig(
        board_cards=3, excluded_boards=config.generation.excluded_boards
    )
    play = HoldemSelfPlayConfig()
    features, masks, targets = [], [], []
    while len(features) < count:
        space, root, reach = sample_situation(rng, situations)
        for example in collect_trajectory(
            leaf_values, space, root, play, rng, reach=reach
        ):
            features.append(example.features)
            masks.append(example.mask)
            targets.append(example.values)
    return Examples(
        features=np.stack(features).astype(np.float32),
        masks=np.stack(masks).astype(np.float32),
        targets=np.stack(targets).astype(np.float32),
    )
