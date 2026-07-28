"""The value-network gradient loop shared by staged and online training.

:func:`holdem.curriculum.fit_stage` (buffer-backed staged/offline training)
and :func:`holdem.training.fit_online_student` (replay-buffer-backed online
ReBeL training) both reduce, at the core, to the same thing: draw a batch
from *something* that knows how to hand out ``(features, masks, targets)``,
take a gradient step, repeat.  Keeping that one loop in one place means a
fair comparison between the two regimes is comparing the same optimiser
behaviour on different data, not two subtly different training loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from holdem.net import HoldemValueNet

# Anything that can hand back a batch given a size and an RNG: a
# :class:`~holdem.curriculum.MixedBuffer`, a :class:`~holdem.rebel.Buffer`,
# or a :class:`~holdem.dataset.DatasetStore` wrapped to return raw arrays.
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
    """What a comparison needs back from either regime's student trainer.

    ``initial_state`` is a snapshot of the network's parameters taken before
    any gradient step, so a caller can verify two students actually started
    from byte-identical weights rather than merely being told so.  ``labels``
    and ``updates`` are the exact accounting of what each regime spent:
    labelled examples consumed (generated, for the online student; sampled,
    for the fixed one) and gradient updates taken.
    """

    net: HoldemValueNet
    history: List[Dict[str, float]]
    initial_state: Dict[str, torch.Tensor]
    labels: int
    updates: int
