"""Output heads.

Two heads share the trunk:

* the **policy head** emits one logit per action;
* the **Q head** emits one scalar per action, the expected terminal reward of
  taking that action from the acting player's perspective.

The Q head deliberately does *not* collapse to a scalar state value; the
policy-improvement operator needs per-action values.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """Linear -> GeLU -> Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyHead(MLPHead):
    """Unnormalised action logits, shape ``[batch, action_count]``."""


class QHead(MLPHead):
    """Per-action Q-values, shape ``[batch, action_count]``.

    When the reward is bounded the head is squashed as ``scale * tanh(.)``,
    which keeps early training well conditioned.  ``scale`` must cover the full
    return range: three-handed normalised chip returns reach +2, so a plain
    ``tanh`` capped at 1 would make the largest wins unrepresentable.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        bounded: bool = True,
        scale: float = 1.0,
    ) -> None:
        super().__init__(in_dim, hidden_dim, out_dim)
        self.bounded = bounded
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.net(x)
        return self.scale * torch.tanh(q) if self.bounded else q
