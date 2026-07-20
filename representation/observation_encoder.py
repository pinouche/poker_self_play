"""Observation dictionary -> fixed-size tensor.

The encoder owns the *layout*: which feature blocks exist, how large they are,
and where they sit in the flat vector.  The network consumes the same layout to
split the flat vector back into semantic groups for its per-group encoders, so
there is exactly one definition of the observation shape in the codebase.

Group -> fields:

    cards        hole cards, (masked) opponent card slots, derived features
    board        board cards, street one-hot
    players      per-player features for SELF / LEFT / RIGHT
    pot_history  pot features, action history, legal-action mask
    position     position relative to the button
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from config import ObsConfig
from environment.state import MAX_BOARD_CARDS, NUM_ACTIONS, NUM_BETTING_STREETS, NUM_HOLE_CARDS

from .canonicalizer import (
    CARD_DIM,
    DERIVED_FEATURE_DIM,
    EQUITY_FEATURE_DIM,
    HISTORY_EVENT_DIM,
    NUM_REL_PLAYERS,
    PLAYER_FEATURE_DIM,
    POSITION_FEATURE_DIM,
    POT_FEATURE_DIM,
)

GROUP_ORDER = ["cards", "board", "players", "pot_history", "position"]

GROUP_FIELDS: Dict[str, List[str]] = {
    "cards": ["hole_cards", "opponent_cards", "derived_features", "equity_features"],
    "board": ["board", "street"],
    "players": ["players"],
    "pot_history": ["pot_features", "action_history", "legal_action_mask"],
    "position": ["position_features"],
}


@dataclass(frozen=True)
class ObservationSpec:
    """Immutable description of the observation layout."""

    field_dims: Dict[str, int]
    field_slices: Dict[str, Tuple[int, int]]
    group_dims: Dict[str, int]
    group_slices: Dict[str, Tuple[int, int]]
    total_dim: int
    num_actions: int = NUM_ACTIONS

    def describe(self) -> str:
        lines = ["Observation layout", "=" * 62]
        for group in GROUP_ORDER:
            start, end = self.group_slices[group]
            lines.append(f"  {group:<12} dim={self.group_dims[group]:>5}   [{start}:{end}]")
            for field in GROUP_FIELDS[group]:
                if field not in self.field_dims:
                    continue
                fstart, fend = self.field_slices[field]
                lines.append(
                    f"      {field:<20} dim={self.field_dims[field]:>5}   [{fstart}:{fend}]"
                )
        lines.append("-" * 62)
        lines.append(f"  TOTAL observation_dim = {self.total_dim}")
        lines.append(f"  action_count          = {self.num_actions}")
        lines.append("=" * 62)
        return "\n".join(lines)


def build_spec(obs_cfg: ObsConfig) -> ObservationSpec:
    dims: Dict[str, int] = {
        "hole_cards": NUM_HOLE_CARDS * CARD_DIM,
        "board": MAX_BOARD_CARDS * CARD_DIM,
        "street": NUM_BETTING_STREETS,
        "players": NUM_REL_PLAYERS * PLAYER_FEATURE_DIM,
        "pot_features": POT_FEATURE_DIM,
        "position_features": POSITION_FEATURE_DIM,
        "action_history": obs_cfg.action_history_length * HISTORY_EVENT_DIM,
        "legal_action_mask": NUM_ACTIONS,
    }
    if obs_cfg.include_opponent_card_slots:
        dims["opponent_cards"] = (NUM_REL_PLAYERS - 1) * NUM_HOLE_CARDS * CARD_DIM
    if obs_cfg.use_derived_card_features:
        dims["derived_features"] = DERIVED_FEATURE_DIM
    if obs_cfg.use_equity_feature:
        dims["equity_features"] = EQUITY_FEATURE_DIM

    field_slices: Dict[str, Tuple[int, int]] = {}
    group_dims: Dict[str, int] = {}
    group_slices: Dict[str, Tuple[int, int]] = {}

    cursor = 0
    for group in GROUP_ORDER:
        group_start = cursor
        for field in GROUP_FIELDS[group]:
            if field not in dims:
                continue
            field_slices[field] = (cursor, cursor + dims[field])
            cursor += dims[field]
        group_dims[group] = cursor - group_start
        group_slices[group] = (group_start, cursor)

    return ObservationSpec(
        field_dims=dims,
        field_slices=field_slices,
        group_dims=group_dims,
        group_slices=group_slices,
        total_dim=cursor,
    )


class ObservationEncoder:
    """Turns observation dictionaries into fixed-shape arrays."""

    def __init__(self, obs_cfg: ObsConfig) -> None:
        self.obs_cfg = obs_cfg
        self.spec = build_spec(obs_cfg)

    @property
    def observation_dim(self) -> int:
        return self.spec.total_dim

    def encode(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Encode into per-group vectors (structured view)."""
        groups: Dict[str, np.ndarray] = {}
        for group in GROUP_ORDER:
            parts = [
                np.asarray(observation[field], dtype=np.float32).ravel()
                for field in GROUP_FIELDS[group]
                if field in self.spec.field_dims
            ]
            vector = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
            if vector.size != self.spec.group_dims[group]:
                raise ValueError(
                    f"group {group}: expected {self.spec.group_dims[group]} features, "
                    f"got {vector.size}"
                )
            groups[group] = vector
        return groups

    def encode_flat(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Encode into a single flat vector of shape ``[observation_dim]``."""
        groups = self.encode(observation)
        flat = np.concatenate([groups[g] for g in GROUP_ORDER])
        if flat.size != self.spec.total_dim:
            raise ValueError(f"expected {self.spec.total_dim} features, got {flat.size}")
        if not np.all(np.isfinite(flat)):
            raise ValueError("observation contains non-finite values")
        return flat.astype(np.float32)

    def encode_batch(self, observations: Sequence[Dict[str, np.ndarray]]) -> np.ndarray:
        return np.stack([self.encode_flat(o) for o in observations], axis=0)

    def field(self, flat: np.ndarray, name: str) -> np.ndarray:
        """Slice a named field back out of a flat observation."""
        start, end = self.spec.field_slices[name]
        return flat[..., start:end]

    def describe(self) -> str:
        return self.spec.describe()

    def print_spec(self) -> None:
        print(self.describe())
