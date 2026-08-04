"""The shared policy/Q network.

One network, three seats.  The observation is already canonicalised to the
acting player's perspective, so the same parameters serve hero, villain-left
and villain-right without any seat conditioning.

    flat observation
        |
        +-- card slots -> shared CardEmbedding (rank + suit + identity)
        |                 one bag per set: hole / flop / turn / river
        +-- everything else -> per-group encoders (cards / board / players /
        |                      pot+history / position)
        |
    concat -> Linear(-> hidden) -> N residual MLP blocks
        |
        +-- policy head -> action logits
        +-- Q head      -> Q(s, a) for every action

The card embedding is :class:`common.nets.CardEmbedding` — the same module
paradigm B's value and policy networks use.  Cards reach it as a 52-wide
multi-hot indicator, which the observation does not store directly: it stores
each slot as a rank one-hot, a suit one-hot and a present flag.  The outer
product of those two one-hots *is* the card indicator, because a card's index
is ``rank * 4 + suit`` — the very convention ``CardEmbedding`` assumes when it
factors its table into rank and suit components.  So the reconstruction is
exact and the observation layout did not have to change to feed it.

Set ``ModelConfig.card_embedding_dim = 0`` for the older architecture, where
card slots went through the same flat ``FeatureGroupEncoder`` as everything
else.  Checkpoints written before the embedding existed load that way
automatically.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from common.cards import NUM_CARDS, NUM_RANKS, NUM_SUITS
from common.nets import CardEmbedding
from paradigm_a.config import Config, ModelConfig, resolve_q_head
from paradigm_a.representation.observation_encoder import GROUP_ORDER, ObservationSpec, build_spec

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

        # Card slots are pulled out of their groups and embedded jointly; each
        # group's encoder then sees only what is left of it.  With the embedding
        # off, `card_sets` is empty and every group is encoded whole, which is
        # the pre-embedding architecture exactly.
        self.card_sets = spec.card_sets() if cfg.card_embedding_dim > 0 else []
        self.card_embedding = (
            CardEmbedding(NUM_CARDS, cfg.card_embedding_dim) if self.card_sets else None
        )
        self.encoded_dims = {
            group: (
                sum(end - start for start, end in spec.non_card_slices(group))
                if self.card_sets
                else spec.group_dims[group]
            )
            for group in GROUP_ORDER
        }
        self.group_order = [g for g in GROUP_ORDER if self.encoded_dims[g] > 0]
        self.encoders = nn.ModuleDict(
            {
                group: FeatureGroupEncoder(self.encoded_dims[group], embed_dims[group])
                for group in self.group_order
            }
        )
        embedding_dim = sum(embed_dims[g] for g in self.group_order)
        embedding_dim += len(self.card_sets) * cfg.card_embedding_dim

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
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
        """Slice a flat observation batch into the parts each encoder consumes.

        With the card embedding on, that is the group minus its card slots;
        with it off, the whole group.
        """
        if observations.shape[-1] != self.spec.total_dim:
            raise ValueError(
                f"expected observation dim {self.spec.total_dim}, got {observations.shape[-1]}"
            )
        out = {}
        for group in self.group_order:
            if self.card_sets:
                parts = [
                    observations[..., start:end]
                    for start, end in self.spec.non_card_slices(group)
                ]
                out[group] = torch.cat(parts, dim=-1)
            else:
                start, end = self.spec.group_slices[group]
                out[group] = observations[..., start:end]
        return out

    def card_indicators(self, observations: torch.Tensor) -> torch.Tensor:
        """``[batch, num_sets, 52]`` multi-hot bags, one per card set.

        Each slot stores a rank one-hot, a suit one-hot and a present flag.
        Card index is ``rank * NUM_SUITS + suit``, so the flattened outer
        product of the two one-hots is that card's indicator — and the present
        flag zeroes the slots that hold no card, whose one-hots are already all
        zero.  Summing the slots in a set makes it a bag: order within the set
        cannot matter, which is the point.
        """
        sets = []
        for _, slots in self.card_sets:
            bag = None
            for start, end in slots:
                slot = observations[..., start:end]
                ranks = slot[..., :NUM_RANKS]
                suits = slot[..., NUM_RANKS : NUM_RANKS + NUM_SUITS]
                indicator = (ranks.unsqueeze(-1) * suits.unsqueeze(-2)).flatten(-2)
                bag = indicator if bag is None else bag + indicator
            sets.append(bag)
        return torch.stack(sets, dim=-2)

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(policy_logits, q_values)``, each ``[batch, action_count]``."""
        groups = self.split_groups(observations)
        parts = [self.encoders[group](groups[group]) for group in self.group_order]
        if self.card_embedding is not None:
            embedded_cards = self.card_embedding(self.card_indicators(observations))
            parts.append(embedded_cards.flatten(-2))
        embedded = torch.cat(parts, dim=-1)
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
    from paradigm_a.environment.state import action_space_for

    bounded, scale = resolve_q_head(cfg)
    spec = build_spec(cfg.obs, action_space_for(cfg.env).num_actions)
    return PokerNet(spec, cfg.model, bounded_q=bounded, q_scale=scale)


def _with_legacy_defaults(config: dict) -> dict:
    """Fill in architecture keys a checkpoint predates, at their old values.

    A checkpoint's config describes the weights sitting next to it, so a key it
    does not carry has to default to whatever the architecture did *before* that
    key existed — not to the current dataclass default.  Otherwise the network
    is built one way and the state dict was written another, and
    ``load_state_dict`` fails on a shape mismatch with no hint as to why.

    This is deliberately not done in :meth:`Config.from_dict`, which also loads
    hand-written config files: there, a missing key should mean "I did not
    override it", and silently getting the old architecture would be a trap.
    """
    config = dict(config)
    model = dict(config.get("model", {}))
    model.setdefault("card_embedding_dim", 0)
    config["model"] = model
    return config


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
    cfg = Config.from_dict(_with_legacy_defaults(payload["config"]))
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
