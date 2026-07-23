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

    # --- all-in EV runouts --------------------------------------------------
    # When betting is closed and no further decisions are possible, pay the
    # *expected* split over the remaining boards instead of dealing one random
    # runout.  Identical in expectation, far lower variance -- the single
    # largest noise reduction available in the learning signal.  A concrete
    # board is still dealt for the record; only the payout is the expectation.
    all_in_ev_runout: bool = True
    # Enumerate every completion when there are at most this many, else sample.
    all_in_ev_exact_threshold: int = 200
    all_in_ev_samples: int = 100


@dataclass
class ObsConfig:
    """Shape and content of the network-facing observation."""

    action_history_length: int = 32

    # Derived card features (rank/suit counts, board texture, made-hand
    # category).  Computed *only* from the acting player's hole cards and the
    # public board, so they leak nothing.
    use_derived_card_features: bool = True

    # Monte-Carlo equity against random opponent ranges, computed only from the
    # acting player's own cards and the public board.  Memoised on a
    # suit-isomorphic key, so the cost amortises to near zero.
    use_equity_feature: bool = True
    equity_samples: int = 120

    # Keep fixed (always-masked) slots for opponent hole cards.  They are zero
    # during normal play and exist so that the leakage test has an explicit
    # region to assert on.
    include_opponent_card_slots: bool = True

    # Fill those slots once the hand reaches showdown.  Off during training;
    # the flag exists for analysis and replay tooling only.
    reveal_at_showdown: bool = False


@dataclass
class ModelConfig:
    # Defaults are the "medium" baseline from the research plan (~1.6M params):
    # 256-wide trunk, 8 residual blocks.  Named presets live in
    # ``MODEL_PRESETS`` (tiny / medium / large) for controlled size comparisons.
    hidden_dim: int = 256
    num_residual_blocks: int = 8
    embed_cards: int = 128
    embed_board: int = 128
    embed_players: int = 128
    embed_pot_history: int = 256
    embed_position: int = 64
    head_hidden: int = 256
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
    optimizer: str = "adamw"          # "adamw" | "adam"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2         # AdamW decoupled decay; larger than Adam's
    batch_size: int = 512
    grad_clip: float = 5.0

    # Replay ratio, expressed as *new environment transitions collected per
    # gradient update* (the research plan's primary knob, target 256-2048).
    # The number of updates each iteration is derived from this and the number
    # of transitions generated.  None falls back to the fixed
    # ``updates_per_iteration`` below (legacy behaviour).
    transitions_per_update: Optional[int] = 512

    # --- self-play loop -----------------------------------------------------
    iterations: int = 200
    hands_per_iteration: int = 256
    updates_per_iteration: int = 32   # only used when transitions_per_update is None
    min_buffer_before_training: int = 2_000
    sampling_temperature: float = 1.0
    # Number of hands advanced in lockstep so their decisions batch into one
    # forward pass.  1 falls back to the sequential worker.  Behaviour is
    # identical either way; only throughput changes.
    self_play_envs: int = 64

    # --- opponent pool ------------------------------------------------------
    # Fraction of self-play hands in which one or two seats are played by an
    # external opponent instead of the learner.  Pure self-play (0.0) calibrates
    # the critic against the self-play population and nothing else.
    opponent_mix_prob: float = 0.0
    # Names from evaluation.random_agent / evaluation.heuristic_agent, plus
    # "checkpoint" for frozen snapshots of the learner itself (league play).
    opponent_pool: tuple = ()
    # Monte-Carlo samples for heuristic opponents used *during training*.
    # Evaluation bots keep their own (higher) setting.
    opponent_heuristic_samples: int = 25
    league_snapshot_every: int = 50
    league_size: int = 5
    # Recency weighting for league snapshots.
    #   "geometric": snapshot at age `a` (0 = newest) has weight
    #                `league_recency_decay ** a`.
    #   "linear":    the K retained snapshots get weights K, K-1, ..., 1 from
    #                newest to oldest -- a straight-line decline with recency.
    league_weighting: str = "geometric"
    league_recency_decay: float = 0.5

    # --- population self-play ----------------------------------------------
    # When enabled, every hand seats the learner in one seat and samples the
    # other two "villains" from a library of frozen checkpoint policies, with
    # recency weighting.  Games are played start to finish from a randomised
    # setup, and a configurable fraction begin on a later street (the earlier
    # streets played out by the population, so the starting state is realistic).
    population_self_play: bool = False
    population_library_size: int = 20
    population_snapshot_every: int = 10
    # Randomised game setup.
    population_random_setup: bool = True
    population_stack_min_bb: float = 15.0
    population_stack_max_bb: float = 200.0
    population_big_blind_choices: tuple = (10, 20, 50, 100)
    # Scenario mixture: probability a hand's learner *enters* on each street.
    # Earlier streets are still played (by the population) so ranges and pots
    # are realistic; the learner simply starts collecting from its entry street.
    # Order: preflop, flop, turn, river.  Normalised if it does not sum to 1.
    entry_street_probs: tuple = (0.75, 0.13, 0.08, 0.04)

    # --- co-evolving population (multi-network self-play) ------------------
    # Number of live networks trained *simultaneously*.  With num_policies > 1
    # the run switches to co-evolution: every hand samples one network per seat,
    # uniformly and with replacement, from the population -- so a network faces
    # other members and fresh copies of itself -- and every seat's decisions
    # train whichever network produced them.  This is a departure from the rest
    # of the file, where a single network drives all three seats and only the
    # learner seats yield data.  Each network keeps its own optimiser and replay
    # buffer; ``replay_capacity`` is split evenly between them so the total
    # memory footprint is unchanged.  num_policies == 1 is ordinary
    # shared-network self-play and leaves every other code path untouched.
    num_policies: int = 1

    # --- replay buffer ------------------------------------------------------
    # The specification suggests 1_000_000.  That is supported, but the default
    # is smaller so that `python train.py` is comfortable on a laptop
    # (observations dominate memory: ~910 float16 per transition, x2 with
    # next-observations stored).
    replay_capacity: int = 1_000_000
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


def reward_scale(env: EnvConfig) -> float:
    """Typical magnitude of the terminal reward, used to make alpha/beta dimensionless.

    The improvement operator computes ``exp(Q / (alpha + beta))``, so ``alpha``
    and ``beta`` silently carry the units of the reward unless Q is normalised
    first.  With `bb_normalized` rewards reaching +-50, an unnormalised operator
    exponentiates +-50 and collapses instantly to a deterministic policy.
    """
    if env.reward_mode == "bb_normalized":
        return max(1.0, env.starting_stack / max(1, env.big_blind))
    if env.reward_mode == "chip_return":
        return max(1.0, float(env.starting_stack))
    return 1.0  # binary and normalized_chip_return are already O(1)


def resolve_q_head(cfg: "Config") -> tuple:
    """(bounded, scale) for the Q head, honouring explicit overrides."""
    bound = reward_bound(cfg.env)
    bounded = cfg.model.bounded_q if cfg.model.bounded_q is not None else bound is not None
    if not bounded:
        return False, 1.0
    scale = cfg.model.q_scale if cfg.model.q_scale is not None else (bound or 1.0)
    return True, float(scale)


#: Named trunk sizes for controlled capacity comparisons (research plan §1).
MODEL_PRESETS = {
    "tiny": dict(
        hidden_dim=128, num_residual_blocks=4, head_hidden=128,
        embed_cards=64, embed_board=64, embed_players=64,
        embed_pot_history=128, embed_position=32,
    ),
    "medium": dict(
        hidden_dim=256, num_residual_blocks=8, head_hidden=256,
        embed_cards=128, embed_board=128, embed_players=128,
        embed_pot_history=256, embed_position=64,
    ),
    "large": dict(
        hidden_dim=512, num_residual_blocks=10, head_hidden=512,
        embed_cards=256, embed_board=256, embed_players=256,
        embed_pot_history=512, embed_position=128,
    ),
}


def model_config(preset: str, **overrides) -> ModelConfig:
    """A :class:`ModelConfig` for a named size preset (tiny/medium/large)."""
    if preset not in MODEL_PRESETS:
        raise ValueError(f"unknown model preset {preset!r}; choose from {list(MODEL_PRESETS)}")
    return ModelConfig(**{**MODEL_PRESETS[preset], **overrides})


def updates_for_transitions(train_cfg, new_transitions: int) -> int:
    """Number of gradient updates for a batch of freshly collected transitions.

    Derived from ``transitions_per_update`` (the replay-ratio knob); falls back
    to the fixed ``updates_per_iteration`` when that is unset.  At least one
    update whenever any data was collected.
    """
    if train_cfg.transitions_per_update is None:
        return train_cfg.updates_per_iteration
    if new_transitions <= 0:
        return 0
    return max(1, round(new_transitions / train_cfg.transitions_per_update))


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
