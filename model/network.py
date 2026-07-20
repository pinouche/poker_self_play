"""The shared policy/Q network.

One network, three seats.  The observation is already canonicalised to the
acting player's perspective, so the same parameters serve hero, villain-left
and villain-right without any seat conditioning.

    flat observation
        |
        +-- split into feature groups (layout owned by ObservationSpec)
        |
    per-group encoders (cards / board / players / pot+history / position)
        |
    concat -> Linear(-> hidden) -> N residual MLP blocks
        |
        +-- policy head -> action logits
        +-- Q head      -> Q(s, a) for every action
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import Config, ModelConfig, resolve_q_head
from representation.observation_encoder import GROUP_ORDER, ObservationSpec, build_spec

from .heads import PolicyHead, QHead
from .residual_blocks import FeatureGroupEncoder, ResidualTrunk

NEG_INF = -1e9


class PokerNet(nn.Module):
    def __init__(
        self,
        spec: ObservationSpec,
        model_cfg: Optional[ModelConfig] = None,
        bounded_q: bool = True,
        q_scale: float = 1.0,
    ) -> None:
        super().__init__()
        cfg = model_cfg or ModelConfig()
        self.spec = spec
        self.cfg = cfg
        self.num_actions = spec.num_actions

        embed_dims = {
            "cards": cfg.embed_cards,
            "board": cfg.embed_board,
            "players": cfg.embed_players,
            "pot_history": cfg.embed_pot_history,
            "position": cfg.embed_position,
        }
        self.group_order = [g for g in GROUP_ORDER if spec.group_dims[g] > 0]
        self.encoders = nn.ModuleDict(
            {
                group: FeatureGroupEncoder(spec.group_dims[group], embed_dims[group])
                for group in self.group_order
            }
        )
        embedding_dim = sum(embed_dims[g] for g in self.group_order)

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.trunk = ResidualTrunk(cfg.hidden_dim, cfg.num_residual_blocks, cfg.dropout)
        self.policy_head = PolicyHead(cfg.hidden_dim, cfg.head_hidden, self.num_actions)
        self.q_head = QHead(
            cfg.hidden_dim,
            cfg.head_hidden,
            self.num_actions,
            bounded=bounded_q,
            scale=q_scale,
        )

    # --- forward -----------------------------------------------------------
    def split_groups(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Slice a flat observation batch into its feature groups."""
        if observations.shape[-1] != self.spec.total_dim:
            raise ValueError(
                f"expected observation dim {self.spec.total_dim}, got {observations.shape[-1]}"
            )
        out = {}
        for group in self.group_order:
            start, end = self.spec.group_slices[group]
            out[group] = observations[..., start:end]
        return out

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(policy_logits, q_values)``, each ``[batch, action_count]``."""
        groups = self.split_groups(observations)
        embedded = torch.cat(
            [self.encoders[group](groups[group]) for group in self.group_order], dim=-1
        )
        features = self.trunk(self.projection(embedded))
        return self.policy_head(features), self.q_head(features)

    # --- convenience -------------------------------------------------------
    @torch.no_grad()
    def infer(
        self, observation: np.ndarray, device: Optional[torch.device] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-observation forward pass returning numpy arrays."""
        was_training = self.training
        self.eval()
        device = device or next(self.parameters()).device
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        logits, q_values = self.forward(tensor)
        if was_training:
            self.train()
        return logits[0].cpu().numpy(), q_values[0].cpu().numpy()

    @torch.no_grad()
    def infer_batch(
        self, observations: np.ndarray, device: Optional[torch.device] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        was_training = self.training
        self.eval()
        device = device or next(self.parameters()).device
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        logits, q_values = self.forward(tensor)
        if was_training:
            self.train()
        return logits.cpu().numpy(), q_values.cpu().numpy()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax over legal actions only.

    Illegal entries are driven to a large negative value *before* the
    normalisation, so they receive probability zero and contribute no gradient.
    """
    masked = logits.masked_fill(mask <= 0, NEG_INF)
    return torch.log_softmax(masked, dim=-1)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.masked_fill(mask <= 0, NEG_INF), dim=-1)
    return probs * (mask > 0)


# --- checkpointing --------------------------------------------------------
def build_network(cfg: Config) -> PokerNet:
    """Construct a network matching the observation layout implied by ``cfg``.

    The Q head's bounding is derived from the reward mode so the head can
    always represent the return range it is asked to regress onto.
    """
    from environment.state import action_space_for

    bounded, scale = resolve_q_head(cfg)
    spec = build_spec(cfg.obs, action_space_for(cfg.env).num_actions)
    return PokerNet(spec, cfg.model, bounded_q=bounded, q_scale=scale)


def save_checkpoint(path: str, network: PokerNet, cfg: Config, extra: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "state_dict": network.state_dict(),
        "config": cfg.to_dict(),
        "observation_dim": network.spec.total_dim,
        "num_actions": network.num_actions,
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: str, device: str = "cpu") -> Tuple[PokerNet, Config, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(payload["config"])
    network = build_network(cfg)
    if network.spec.total_dim != payload["observation_dim"]:
        raise ValueError(
            "checkpoint observation dim does not match the config it carries: "
            f"{payload['observation_dim']} vs {network.spec.total_dim}"
        )
    network.load_state_dict(payload["state_dict"])
    network.to(device)
    network.eval()
    return network, cfg, payload.get("extra", {})
