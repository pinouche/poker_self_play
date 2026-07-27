"""Configuration for ReBeL training.

Paradigm B keeps its own config rather than extending the root ``config.py``:
that one describes a three-player POMDP with an observation encoder and a
policy/value network trained from experience, and none of those knobs mean
anything here.  The shared piece is ``resolve_device``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cfr.tabular_cfr import CFRConfig
from search.resolve import ResolveConfig
from value_net.net import ValueNetConfig
from value_net.train import ValueTrainConfig


@dataclass
class SelfPlayConfig:
    """How trajectories are generated."""

    trajectories_per_iteration: int = 32
    # Iterations of CFR run inside each subgame during self play.
    search_iterations: int = 100
    depth_limit: int = 1
    # Ignore the first fraction of CFR iterations when sampling which one to
    # descend from: early iterates are close to uniform and carry no signal.
    warmup_fraction: float = 0.2
    # Chance of descending under a uniform random profile instead of the
    # searched one.  Equilibrium play produces sharp ranges and visits a narrow
    # slice of belief space; the network still has to be accurate off it,
    # because the moment search strays there it is querying untrained inputs.
    exploration: float = 0.25
    cfr: CFRConfig = field(default_factory=CFRConfig.dcfr)


@dataclass
class ReBeLConfig:
    """The whole loop: generate, train, repeat."""

    iterations: int = 20
    buffer_size: int = 200_000
    updates_per_iteration: int = 120
    batch_size: int = 512
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    value_net: ValueNetConfig = field(default_factory=ValueNetConfig)
    training: ValueTrainConfig = field(
        default_factory=lambda: ValueTrainConfig(learning_rate=1e-3, batch_size=512)
    )
    # Search settings used when the agent is *evaluated*, and deliberately more
    # thorough than the ones used to generate data.  Exploitability keeps
    # falling well past the point where the training curve looks flat -- at 300
    # iterations the agent measures 0.058 and at 10,000 it measures 0.016, with
    # the same network.  Evaluating too cheaply reads as a plateau that is not
    # there.
    evaluation: ResolveConfig = field(
        default_factory=lambda: ResolveConfig(iterations=1000, depth_limit=1)
    )
    evaluate_every: int = 5
    seed: int = 0
    device: str = "cpu"
