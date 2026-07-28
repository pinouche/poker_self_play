"""The value-network gradient loop shared by both arms.

:func:`holdem.compare.fixed.student.fit_fixed_student` (batches drawn from a
frozen on-disk artifact) and
:func:`holdem.compare.iterative.student.fit_online_student` (batches drawn from
a replay buffer that is being refilled as training proceeds) reduce, at the
core, to the same thing: draw ``(features, masks, targets)`` from *something*,
take a gradient step, repeat.

Keeping that loop in one place is what makes the comparison a comparison.  If
each arm had its own training loop, any difference in the result could be the
data or could be a stray detail of the optimiser, and there would be no way to
tell which from the outside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from paradigm_b.holdem.arms_common.budget import SpendRecord
from paradigm_b.holdem.net.value_net import HoldemValueNet

# Anything that can hand back a batch given a size and an RNG.
SampleFn = Callable[[int, np.random.Generator], Tuple[np.ndarray, np.ndarray, np.ndarray]]


def fit_value_net(
    net: HoldemValueNet,
    optimiser: torch.optim.Optimizer,
    loss_fn: nn.Module,
    sample_fn: SampleFn,
    steps: int,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[float, int]:
    """Take up to ``steps`` gradient steps on batches drawn from ``sample_fn``.

    Returns ``(summed_loss, steps_taken)``; ``steps_taken`` is ``steps``
    unless ``steps <= 0``, in which case nothing happens and both are zero.
    """
    if steps <= 0:
        return 0.0, 0
    net.train()
    total = 0.0
    for _ in range(steps):
        features, masks, targets = sample_fn(batch_size, rng)
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
    return total, steps


@dataclass
class StudentResult:
    """What a comparison needs back from either arm's student trainer.

    ``initial_state`` is a snapshot of the parameters taken before any gradient
    step, so a caller can *verify* two students started from byte-identical
    weights rather than merely being told so.  ``spend`` is the full accounting
    (see :class:`~holdem.compare.common.budget.SpendRecord`); ``labels`` and
    ``updates`` are surfaced directly because they are the two quantities the
    experiment holds equal.
    """

    net: HoldemValueNet
    history: List[Dict[str, float]]
    initial_state: Dict[str, torch.Tensor]
    spend: SpendRecord = field(default_factory=SpendRecord)

    @property
    def labels(self) -> int:
        return self.spend.labels

    @property
    def updates(self) -> int:
        return self.spend.updates
