"""Central configuration for the 3-player poker self-play system.

Every tunable lives here so that experiments differ only by a config object.
Configs are plain dataclasses and are serialised into checkpoints, which keeps
a trained network permanently paired with the observation layout it expects.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EnvConfig:
    """Rules of the game itself."""

    num_players: int = 3
    small_blind: int = 10
    big_blind: int = 20
    starting_stack: int = 1000  # 50 big blinds

    # Bet sizings as a fraction of the current pot.
    bet_fractions: tuple = (0.33, 0.66, 1.00)
    # Raise sizings as a multiple of the current outstanding bet.
    raise_multipliers: tuple = (2.0, 3.0, 4.0)

    # Folding when nobody has bet is legal in casino poker but strictly
    # dominated by checking.  Allowing it only adds noise to self-play, so it
    # is disabled by default.  See README "Design decisions".
    allow_fold_when_check_available: bool = False

    # "normalized_chip_return" | "bb_normalized" | "chip_return" | "binary"
    #
    # Default: net chips as a fraction of the seat's own stack at the start of
    # the hand.  Money is the reward.  `binary` is retained for comparison but
    # maximises win rate rather than chip EV -- see the README.
    reward_mode: str = "normalized_chip_return"

    # Symmetric clip on the terminal reward.
    #
    # None = automatic: the tightest bound the rules actually allow, which is
    # (num_players - 1).  A seat can lose at most its own stack (-1) but can win
    # every opponent's at-risk chips (+2 three-handed).  That catches bugs and
    # side-pot errors without ever truncating a legitimate outcome.
    #
    # Do NOT set this to 1.0: measured over 12,000 random-play outcomes it
    # truncates 48% of all winning results and shifts the mean reward from
    # exactly 0.0 to -0.057, turning a zero-sum game into a negative-sum one and
    # creating a systematic fold bias.  0 disables clipping entirely.
    reward_clip: Optional[float] = None

    # Rotate the button between hands during self-play.
    rotate_dealer: bool = True


@dataclass
class ObsConfig:
    """Shape and content of the network-facing observation."""

    action_history_length: int = 32

    # Derived card features (rank/suit counts, board texture, made-hand
    # category).  Computed *only* from the acting player's hole cards and the
    # public board, so they leak nothing.
    use_derived_card_features: bool = True

    # Monte-Carlo equity.  Disabled by default per the specification.
    use_equity_feature: bool = False
    equity_samples: int = 200

    # Keep fixed (always-masked) slots for opponent hole cards.  They are zero
    # during normal play and exist so that the leakage test has an explicit
    # region to assert on.
    include_opponent_card_slots: bool = True

    # Fill those slots once the hand reaches showdown.  Off during training;
    # the flag exists for analysis and replay tooling only.
    reveal_at_showdown: bool = False


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    num_residual_blocks: int = 6
    embed_cards: int = 64
    embed_board: int = 64
    embed_players: int = 64
    embed_pot_history: int = 128
    embed_position: int = 32
    head_hidden: int = 128
    dropout: float = 0.0
    # Squash Q-values through `q_scale * tanh(.)`.  Only correct when the
    # return target is itself bounded -- rewards are terminal-only, so that
    # reduces to whether the reward is bounded.  None = automatic from the
    # reward mode; set bounded_q=False explicitly for a linear head.
    bounded_q: Optional[bool] = None
    q_scale: Optional[float] = None


@dataclass
class TrainConfig:
    # --- policy improvement -------------------------------------------------
    # Improvement operator:
    #   pi_new  proportional to  pi_ref^(beta/(alpha+beta)) * exp(Q/(alpha+beta))
    # Smaller (alpha + beta) makes the step greedier; a larger beta/alpha ratio
    # keeps it closer to the reference policy.  With binary rewards Q lives in
    # [-1, 1], so alpha + beta = 1 gives a useful spread (up to ~7x between the
    # best and worst action) while still regularising.
    alpha: float = 0.5   # entropy regularisation coefficient
    beta: float = 0.5    # reverse-KL (trust region) coefficient
    # Reference policy for the improvement operator.  "current" anchors to the
    # live network (detached) -- i.e. the policy immediately before the update.
    # "behavior" anchors to the policy stored with the transition, which grows
    # stale as the buffer ages.
    reference_policy: str = "current"

    # --- returns ------------------------------------------------------------
    gamma: float = 0.99
    lam: float = 0.95

    # --- losses -------------------------------------------------------------
    q_weight: float = 1.0
    policy_weight: float = 1.0
    entropy_weight: float = 0.01
    huber_delta: float = 1.0

    # --- optimisation -------------------------------------------------------
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    grad_clip: float = 5.0

    # --- self-play loop -----------------------------------------------------
    iterations: int = 200
    hands_per_iteration: int = 64
    updates_per_iteration: int = 32
    min_buffer_before_training: int = 2_000
    sampling_temperature: float = 1.0

    # --- replay buffer ------------------------------------------------------
    # The specification suggests 1_000_000.  That is supported, but the default
    # is smaller so that `python train.py` is comfortable on a laptop
    # (observations dominate memory: ~910 float16 per transition, x2 with
    # next-observations stored).
    replay_capacity: int = 100_000
    store_next_obs: bool = True

    # --- bookkeeping --------------------------------------------------------
    eval_every: int = 20
    eval_hands: int = 200
    checkpoint_every: int = 20
    checkpoint_dir: str = "checkpoints"
    log_every: int = 1
    seed: int = 0
    device: str = "auto"


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Config":
        return Config(
            env=EnvConfig(**d.get("env", {})),
            obs=ObsConfig(**d.get("obs", {})),
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as fh:
            return Config.from_dict(json.load(fh))


def reward_bound(env: EnvConfig) -> Optional[float]:
    """Tightest bound on |terminal reward|, or None when it is unbounded.

    Three-handed, `normalized_chip_return` lives in [-1, +2]: a seat risks at
    most its own stack but can win both opponents' stacks.  The bound is the
    *upper* magnitude, so the Q head must be scaled to reach +2 even though
    rewards never go below -1.
    """
    if env.reward_clip == 0:
        return None
    if env.reward_mode == "binary":
        return 1.0
    if env.reward_mode == "normalized_chip_return":
        if env.reward_clip is not None:
            return float(env.reward_clip)
        return float(env.num_players - 1)
    return None  # bb_normalized and chip_return are unbounded


def resolve_q_head(cfg: "Config") -> tuple:
    """(bounded, scale) for the Q head, honouring explicit overrides."""
    bound = reward_bound(cfg.env)
    bounded = cfg.model.bounded_q if cfg.model.bounded_q is not None else bound is not None
    if not bounded:
        return False, 1.0
    scale = cfg.model.q_scale if cfg.model.q_scale is not None else (bound or 1.0)
    return True, float(scale)


def resolve_device(name: str) -> str:
    """Turn ``"auto"`` into a concrete torch device string."""
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
