"""How well the network predicts values — the metric exploitability cannot give.

Exploitability scores the network *as a player*.  On the flop and turn that is
the right question, because the agent's search stops early and the network
decides what the unexplored future is worth.  On the **river it asks nothing at
all**: one betting round remains, every path ends at a real fold or showdown, so
the depth-limited tree has no leaves and the network is never consulted.  Two
completely different networks produce byte-identical river exploitability.

That is not a gap in the experiment so much as a gap in the *instrument*.  The
network's river knowledge is real and it matters — it is exactly what a turn
solve consumes when it reaches a river node it cannot afford to expand.  It is
simply consumed one street up rather than used to play the river itself.

So measure it directly: give the network held-out belief states and compare its
predicted counterfactual values against known-good ones.  For the river those
targets are *exact* (solved to real showdowns), which makes this the one place
in the whole harness with unambiguous ground truth.

Reported per source:

``mae``        mean absolute error in chips, over legal hands only
``rmse``       root mean squared error, which punishes the confident misses
``r2``         fraction of the target's variance explained; <= 0 means the
               network is no better than predicting the mean
``bias``       mean signed error — systematic over- or under-valuation, which
               a magnitude-only metric hides entirely

Masked hands are excluded throughout.  A board blocks every combination using
one of its cards — 150 of the 1,326 on a flop, 198 on a turn, 245 on a river
(11-19%) — and letting those structural zeros into the average would flatter
every network by the same margin and shrink the differences the metric exists
to show.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from paradigm_b.holdem.arm1_fixed.store import STREET_SOURCES, DatasetStore
from paradigm_b.holdem.net.value_net import HoldemValueNet


@dataclass
class AccuracyResult:
    """Prediction error of one network on one source's held-out labels."""

    source: str
    examples: int
    mae: float
    rmse: float
    r2: float
    bias: float
    target_scale: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "examples": self.examples,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "bias": self.bias,
            "target_scale": self.target_scale,
        }


@torch.no_grad()
def predict(
    net: HoldemValueNet,
    features: np.ndarray,
    masks: np.ndarray,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Network outputs for a block of encoded belief states."""
    net.eval()
    net.to(torch.device(device))
    out = []
    for start in range(0, len(features), batch_size):
        stop = start + batch_size
        out.append(
            net(
                torch.as_tensor(features[start:stop], dtype=torch.float32, device=device),
                torch.as_tensor(masks[start:stop], dtype=torch.float32, device=device),
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(out) if out else np.zeros((0,))


def accuracy_on(
    net: HoldemValueNet,
    features: np.ndarray,
    masks: np.ndarray,
    targets: np.ndarray,
    source: str = "held_out",
    device: str = "cpu",
) -> Optional[AccuracyResult]:
    """Compare predictions against ``targets``, over legal hands only."""
    if not len(features):
        return None
    predicted = predict(net, features, masks, device=device)
    # (n, 2, 1326) values against an (n, 1326) per-example legality mask.
    legal = np.broadcast_to(masks[:, None, :], targets.shape) > 0.0
    if not legal.any():
        return None
    error = (predicted - targets)[legal]
    truth = targets[legal]

    variance = float(truth.var())
    r2 = 1.0 - float((error**2).mean()) / variance if variance > 0 else float("nan")
    return AccuracyResult(
        source=source,
        examples=len(features),
        mae=float(np.abs(error).mean()),
        rmse=float(np.sqrt((error**2).mean())),
        r2=r2,
        bias=float(error.mean()),
        target_scale=float(np.abs(truth).mean()),
    )


def evaluate_accuracy(
    net: HoldemValueNet,
    store: DatasetStore,
    sample_size: int = 512,
    seed: int = 0,
    device: str = "cpu",
    sources: Sequence[str] = STREET_SOURCES,
) -> Dict[str, Any]:
    """Per-source prediction error of ``net`` against an artifact's labels.

    **Only fair against a held-out artifact.**  Run this on the dataset a
    student trained from and it reports memorisation, not knowledge — the
    offline arm would score well by construction.  Build a small second
    artifact with a different seed, or score the online arm's journal, to ask
    the question honestly.  The river is the source worth trusting most: its
    labels are exact rather than bootstrapped.
    """
    rng = np.random.default_rng(seed)
    results: Dict[str, Any] = {"sample_size": sample_size, "sources": {}}
    for source in sources:
        if not store.counts().get(source):
            continue
        available = store.counts()[source]
        rows = rng.choice(available, size=min(sample_size, available), replace=False)
        batch = store.gather(source, rows)
        result = accuracy_on(
            net, batch.features, batch.masks, batch.targets, source=source, device=device
        )
        if result is not None:
            results["sources"][source] = result.to_dict()
    return results
