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
    # Width of the shared card embedding (``common.nets.CardEmbedding``, the
    # same module paradigm B's networks use).  Card slots are pulled out of the
    # cards/board groups and embedded as bags — hole, flop, turn, river — rather
    # than passed through a flat linear layer as raw one-hots.  0 restores the
    # older flat encoding; checkpoints predating this default to 0 on load.
    card_embedding_dim: int = 64
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
    # Written as softmax logits this is
    #   score(a) = Q(a)/(alpha+beta) + [beta/(alpha+beta)] * log pi_ref(a),
    # so there are two independent knobs: the weight on Q is 1/(alpha+beta)
    # (greediness) and the weight on the reference is beta/(alpha+beta) (trust
    # region).  A reference weight of 1 reproduces pi_ref exactly; below 1 the
    # step flattens pi_ref toward uniform *every update*.
    #
    # Default is alpha = 0.05, beta = 0.5 (reference weight ~0.91, Q weight
    # ~1.8).  The earlier alpha = beta = 0.5 fails on the default
    # `normalized_chip_return` reward: early in training the Q-spread between a
    # state's actions is tiny (~0.01-0.02), so the Q term is negligible and the
    # reference weight of 0.5 flattens the policy toward uniform faster than the
    # signal can sharpen it -- a self-reinforcing cold start where play stays
    # uniform (entropy pinned near log(#legal), kl_target_vs_policy ~ 0).  With
    # beta/(alpha+beta) ~ 0.91 the reference is preserved rather than flattened,
    # so even small advantages accumulate and the policy escapes.  This is the
    # measured break-even setting from the README alpha sweep and matches the
    # benchmark's `full` config.  Binary rewards keep Q in [-1, 1] (order-1
    # spreads), so raise alpha there -- alpha + beta ~ 1 is fine.
    alpha: float = 0.05  # entropy regularisation coefficient
    beta: float = 0.5    # reverse-KL (trust region) coefficient
    # Reference policy for the improvement operator.  "current" anchors to the
    # live network (detached) -- i.e. the policy immediately before the update.
    # "behavior" anchors to the policy stored with the transition, which grows
    # stale as the buffer ages.  "snapshot" anchors to a *fixed* network handed
    # to the trainer; only "current" and "behavior" are meaningful as a global
    # default, since "snapshot" needs a per-slot reference network (it is what
    # anchored veteran champions use -- see ``champion_mode``).
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
    # Exploration temperature applied to the *improved* policy when acting:
    # ``sampling(a) proportional to improved(a) ** (1/T)``.  T<1 sharpens the
    # behaviour toward the top action (more exploitation); T>1 flattens it toward
    # uniform (more exploration); T=1 leaves the improved policy unchanged.  It
    # also tilts the bootstrap value V(s)=sum_a sampling(a) Q(a) that the
    # lambda-returns use: lower T pulls V toward max-Q (a greedier, more
    # best-response-like target).  A single-network sweep at alpha=0.05 (0.1..1.0,
    # 3 seeds, 300 iters) put the best strength vs the tight-aggressive heuristic
    # in a shallow 0.3-0.5 band and a sharp penalty on either side (0.1 -> -284,
    # 0.6 -> -198 bb/100 vs ~ -40 at 0.3-0.5); 0.4 sits in that optimum.  Raising
    # it back toward 1.0 trades chip EV for a looser, higher-variance policy.
    sampling_temperature: float = 0.4
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

    # --- role-structured league population ---------------------------------
    # Instead of N interchangeable learners, split the population by role:
    #   * ``league_learners``   updated continuously (the working edge);
    #   * ``league_champions``  frozen historical snapshots, never updated --
    #     they stop the league drifting off strategies that already worked,
    #     which is what lets naive self-play cycle;
    #   * ``league_explorers``  learners wiped and reinitialised *from scratch*
    #     every ``league_explorer_reset_hands`` hands.  From scratch, not
    #     cloned or mutated: a fresh random policy is genuinely
    #     off-distribution, an offspring of an incumbent is not.
    # Defaults are the 16/8/8 = 32 split.
    league_population: bool = False
    league_learners: int = 16
    league_champions: int = 8
    league_explorers: int = 8
    # Explorers are wiped every this many *total league* hands (256/iteration at
    # the defaults, so 512k in a 2000-iteration run).  Same failure mode as
    # ``league_promotion_hands``: set above what a run generates and the explorer
    # role is inert -- the slots are never reinitialised and behave as ordinary
    # learners.  20k = every ~78 iterations, so it fires twice in a default
    # 200-iteration run and ~25 times over 2000.
    league_explorer_reset_hands: int = 20_000
    # Iterations between management passes (recompute metrics, promote, cull).
    league_manage_every: int = 50
    # Metric sample sizes.  100k-1M hands and ~100k states are the sizes that
    # make these estimates tight; the defaults here are laptop-sized so a run
    # starts immediately -- raise them for a real experiment.
    league_strength_hands: int = 20_000
    league_diversity_states: int = 4_000
    # score = strength_weight * norm(strength) + diversity_weight * norm(diversity).
    # Ranking on strength alone is what collapses a league to N copies of one
    # policy; the diversity term keeps a weaker-but-different member's slot.
    league_strength_weight: float = 0.6
    league_diversity_weight: float = 0.4

    # --- matchmaking --------------------------------------------------------
    # How the three seats are filled.  Learners must keep meeting history, not
    # just each other, or they overfit to the newest policies:
    #   * learner_vs_learner -- all three seats learners (the working edge);
    #   * learner_vs_champion -- one learner against two frozen champions;
    #   * mixed -- learner + champion + explorer.
    # Normalised if they do not sum to 1.
    league_match_learner_vs_learner: float = 0.50
    league_match_learner_vs_champion: float = 0.30
    league_match_mixed: float = 0.20

    # --- champion promotion and retirement ----------------------------------
    # Champions are promoted on *performance*, never on a schedule: a learner
    # must sustain ``league_promotion_mbb_per_100`` over at least
    # ``league_promotion_hands`` hands of real league play, and beat the *median*
    # champion generation it has faced (``league_promotion_min_generations``,
    # 0.5 = median).  A relative median gate, not "beat 6 of all 8": as champions
    # strengthen the absolute version becomes unmeetable and the ladder freezes
    # (observed -- promotions stopped for eight passes at the strong end).
    league_promotion_mbb_per_100: float = 35.0
    # *Per member*, counted from this slot's own record (cleared when the slot is
    # reset), NOT total league hands.  A member accrues roughly
    #   hands_per_iteration * num_players / league_size
    # hands per iteration -- ~32/iteration at the 256-hand, 32-net defaults, so
    # ~2.4k by iteration 100 and ~48k over a 2000-iteration run.  Anything above
    # that ceiling silently freezes the whole ladder: both promotion paths gate
    # on it, so no learner is ever promoted, the champion pool stays at its
    # random initialisation and every champion metric reads flat forever
    # (`train_league` now warns when the gate is out of reach).  2k is reached at
    # iteration ~83, so the ladder starts rolling on the second management pass
    # of even a default 200-iteration run.  It is a *small* sample for a
    # 35 mbb/100 estimate -- the median-generation, over-specialisation and
    # external-panel gates are what actually filter promotions; this one only
    # keeps a brand-new slot out.  Raise it for long runs, but never above
    # `hands_per_iteration * num_players / league_size * iterations`.
    league_promotion_hands: int = 2_000
    league_promotion_min_generations: float = 0.5
    # A generation counts as "faced" only past this many shared hands, so the
    # median gate is judged on real samples rather than one-hand noise.
    league_gauntlet_min_hands: int = 100
    # A learner that beats the newest champion but loses badly to the oldest is
    # over-specialised -- it has learned the current meta, not the game.  Such a
    # learner is not promoted (the anti-over-specialisation gate).
    league_promotion_reject_over_specialised: bool = True
    # Each promotion cycle freezes two *kinds* of member, so the champion pool
    # accumulates strong strategies and unusual ones rather than eight variants
    # of the same idea.  The "most different" pick draws from the top
    # ``league_most_different_top_k`` learners by score, *not* the strict
    # promotion gate -- otherwise diversity injection stops exactly when the
    # ladder is strong and only one learner can clear the gate.
    league_promote_elite: bool = True
    league_promote_most_different: bool = True
    league_most_different_top_k: int = 4
    # Strength floor for the "most different" pool: only learners at least this
    # strong in-league are eligible, so diversity injection never freezes a
    # near-random (freshly reset) learner as a champion.  0 = must be net-positive
    # in-league; in a zero-sum league that is the upper ~half, so the pool stays
    # populated and most_different keeps firing.
    league_most_different_min_mbb_per_100: float = 0.0
    # When a promotion must evict a champion, remove the most *redundant* one
    # (lowest behavioural diversity to the rest of the pool) rather than the
    # oldest.  This preserves genuinely distinct strategies -- including old ones
    # -- and keeps the champion ladder from collapsing toward one style.  Stale
    # champions are still evicted first.
    league_evict_most_redundant: bool = True
    # External competence panel for the "most different" gate.  The diversity
    # slot seeks *unusual* learners, and unusual often means "loses to standard
    # opponents"; the in-league floor cannot catch that (such a learner is
    # in-league strong).  So a most_different candidate must also be no worse
    # than ``league_most_different_min_panel_bb_per_100`` against the *worst* of a
    # small fixed panel -- kept distinct from the held-out eval opponent
    # (tight_aggressive) so evaluation stays independent.
    league_external_panel: tuple = ("heuristic", "loose_passive")
    league_most_different_min_panel_bb_per_100: float = -150.0
    league_panel_hands: int = 300

    # --- exploitability -----------------------------------------------------
    # Approximate exploitability: freeze an agent in two seats, train a fresh
    # best-responder in the third, and report how much it wins (bb/100).  0 would
    # mean no adversary can gain -- a Nash equilibrium; a large value means the
    # agent is far from unexploitable however strong it looks against a fixed
    # heuristic.  A learned best response is a *lower bound* on true
    # exploitability.  Measured at the end of a league run, and every
    # ``league_exploitability_every`` management passes when that is > 0.
    league_exploitability_every: int = 0
    league_exploitability_iters: int = 300
    # A champion is stale (evicted first, so the pool keeps rolling) if it loses
    # this badly *in-league* OR falls below ``league_champion_retire_vs_heuristic``
    # against the held-out heuristic.  The external gate matters because a
    # champion catastrophic against a competent outside opponent can still look
    # fine in-league if the league has not learned to punish its particular
    # weirdness -- in-league strength alone is relative, not absolute.
    league_champion_retire_mbb_per_100: float = -50_000.0
    league_champion_retire_vs_heuristic: float = -150.0  # bb/100; NaN eval = skip
    league_champion_min_hands: int = 5_000  # don't judge a champion on noise

    # --- champion mode: frozen ladder, or anchored veterans -----------------
    # "frozen" (the default, and the classic league) makes every champion a hard
    # snapshot.  That is load-bearing for two things and only two: the gauntlet
    # needs a measuring stick that does not move (if champions chase the current
    # learner meta, the generational spread collapses and the
    # over-specialisation test silently becomes a no-op), and "beat champion N"
    # must mean the same thing at iteration 500 and at iteration 5000.
    #
    # "anchored" buys back what frozen costs.  Promotion can only ever freeze a
    # strategy the learner pool *currently holds*; a champion encoding a line the
    # learners have since abandoned is the one thing in the league that cannot be
    # recreated, and today it just rots until it is evicted.  So the pool splits:
    #   * the first ``league_champion_anchors`` champion slots stay hard-frozen
    #     -- the gauntlet spine, and the only slots the gauntlet reads;
    #   * the rest become *veterans*: they keep an optimiser and a replay buffer,
    #     train on their own hands, and develop their abandoned branch.
    # This is a diversity mechanism, not a strength one.  Veterans mostly face
    # learners, so left alone they would chase the current meta and become
    # learners with extra steps -- which is what the anchor below prevents.
    champion_mode: str = "frozen"     # "frozen" | "anchored"
    # Champion slots that stay hard-frozen under "anchored".  Needs >= 2 for the
    # over-specialisation test (it compares the oldest and newest generation),
    # and >= 4 keeps the median-generation promotion gate meaningful.  Ignored
    # when champion_mode is "frozen" (there, every champion is an anchor).
    league_champion_anchors: int = 4
    # A veteran's improvement target is anchored to its *own* frozen snapshot
    # rather than to its current policy (``reference_policy="snapshot"``), so
    # ``beta`` bounds how far the target can sit from the snapshot instead of
    # bounding one step from wherever it drifted to last update.  Drift is then
    # an observable, not a hope: KL(veteran || snapshot) is measured over the
    # validation bank every management pass, and a veteran past
    # ``league_veteran_max_kl`` is reset to its snapshot (optimiser included).
    # 0 or below disables the budget.  Note the anchor applies to the training
    # *target* only -- the veteran acts with its own live network, as it must,
    # since it is a real opponent.
    league_veteran_max_kl: float = 0.05
    # Veterans train slower than learners: they are developing an existing line,
    # not searching for one, and a full-rate optimiser blows the drift budget in
    # a handful of passes.
    league_veteran_lr_scale: float = 0.1
    # A freshly reset/culled learner is protected from being culled again for
    # this many management passes, so it gets time to develop instead of being
    # re-culled every pass while it is still the youngest (observed churn: three
    # slots absorbed 19 of 20 culls).
    league_cull_grace_passes: int = 2
    # Fixed seed for the heuristic eval and the diversity validation bank, so the
    # per-pass series is *paired* (same deals / same states).  Without it a frozen
    # ladder alone swung +7..+158 bb/100 across passes from card variance.
    league_eval_seed: int = 20_240

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
        card_embedding_dim=32,
        embed_cards=64, embed_board=64, embed_players=64,
        embed_pot_history=128, embed_position=32,
    ),
    "medium": dict(
        hidden_dim=256, num_residual_blocks=8, head_hidden=256,
        card_embedding_dim=64,
        embed_cards=128, embed_board=128, embed_players=128,
        embed_pot_history=256, embed_position=64,
    ),
    "large": dict(
        hidden_dim=512, num_residual_blocks=10, head_hidden=512,
        card_embedding_dim=128,
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
