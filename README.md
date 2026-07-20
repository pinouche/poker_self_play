# Three-Player Poker Self-Play

A KLENT-inspired reinforcement learning system for 3-handed No-Limit Texas
Hold'em. One shared ResNet-style network plays every seat, trained purely by
self-play against itself.

No MCTS. No Transformer. No language model. No opponent-specific networks.

```
canonicalised observation
        |
   feature-group encoders (cards / board / players / pot+history / position)
        |
   concat -> Linear(128) -> 6 residual MLP blocks
        |
        +-- policy head -> pi(a|s)
        +-- Q head      -> Q(s,a)
```

---

## Quickstart

```bash
python train.py                      # self-play training with defaults
python evaluate.py --checkpoint checkpoints/latest.pt --hands 3000
python infer.py --checkpoint checkpoints/latest.pt   # or no checkpoint for a random net
python -m pytest tests/ -q           # 166 tests
```

The default objective is stack-normalized chip return with a scaled-tanh
critic. The two alternatives worth running are a linear critic on the same
objective, and the conventional BB-denominated objective:

```bash
python train.py --reward-mode normalized_chip_return                 # A (default)
python train.py --reward-mode normalized_chip_return --unbounded-q   # B, linear critic
python train.py --reward-mode bb_normalized                          # C, BB units
```

Requires PyTorch and NumPy. Nothing else.

`train.py`, `evaluate.py` and `infer.py --show-spec` all print the exact
observation layout and dimensionality at startup.

---

## Design decisions

These are the points the specification left open. Each one is a deliberate
choice, and each is configurable.

### Poker rules

Standard 3-handed No-Limit Hold'em. Button = dealer, SB = dealer+1,
BB = dealer+2. **Three handed, the button acts first preflop** and the small
blind acts first on every later street — a detail that is wrong in many toy
implementations.

Fully implemented: minimum raises, the betting-reopen rule (an all-in that
raises by less than a full raise does *not* reopen the action for players who
already matched the last full-raise level), multi-way side pots, uncalled-bet
returns, split pots, and odd chips going to the first winner clockwise from the
button.

### Reward: stack-normalized chip return (default)

**Money is the reward.** For seat *i*:

```
r_i = net_chip_delta_i / stack_before_hand_i
```

Normalising by *that seat's own stack at the start of the hand* — stored per
seat at reset, not read from a global constant — makes the reward a
risk-adjusted return: the same 100-chip win is worth more to a short stack than
to a deep one, which is the correct scaling when stacks are uneven.

Rewards are **terminal-only**. There is no shaping for winning pots, betting,
surviving streets or reaching showdown; that would invite the agent to farm the
reward function instead of learning profitable play.

| mode | reward | Q head |
|---|---|---|
| `normalized_chip_return` *(default)* | `Δchips / stack_before_hand` | `2·tanh` |
| `bb_normalized` | `Δchips / big_blind` | linear |
| `chip_return` | `Δchips` | linear |
| `binary` | `sign(Δchips)` | `tanh` |

One consequence worth stating for every mode: **a player who folds the button
preflop, having invested nothing, scores 0.** They risked nothing and lost
nothing, so a correct fold is never punished.

`binary` is retained for comparison only. It maximises win *rate* rather than
chip EV and the resulting policy learns to almost never fold — measured under
[Results](#results-reward-design).

### Clipping, and why not at ±1

The reward is clipped symmetrically as a guard against rule or side-pot bugs.
The limit defaults to the **tightest bound the rules actually allow**, which
three-handed is `num_players - 1 = 2`, not 1:

> A seat can lose at most its own stack (−1), but can win *every* opponent's
> at-risk chips (+2). The bound is asymmetric because the game is.

Clipping at ±1 is actively harmful here, and it is measurable. Over 12,000
random-play seat outcomes:

* **48% of all winning outcomes exceed +1** and would be truncated;
* no loss ever falls below −1, so clipping only ever removes upside;
* the mean reward moves from **exactly 0.0 to −0.057**.

That last number is the problem: clipping at ±1 turns a zero-sum game into a
negative-sum one. Every seat then has a systematic incentive not to play — a
fold bias that is precisely the mirror image of the binary reward's never-fold
bias. `tests/test_environment.py` pins both behaviours.

Set `reward_clip` explicitly to override, or `0` to disable clipping entirely.

### Bounded Q head, scaled to the return range

A tanh Q head is only coherent if the *return target* stays inside the squashed
range. Rewards are terminal-only, so the return is bounded by the reward and
the question reduces to whether the reward is bounded.

When it is, the head is `scale · tanh(·)` with `scale` set to the reward bound —
**2.0** for three-handed normalized chip returns. A plain `tanh` capped at 1
would make the largest legitimate wins literally unrepresentable by the critic.
Unbounded modes (`bb_normalized`, `chip_return`) get a linear head
automatically. `--unbounded-q` forces a linear head regardless, which is
experiment B below.

### Objective vs. metric

What the network optimises and how its poker strength is measured are kept
separate on purpose. Training uses the configured `reward_mode`; evaluation
always reports **bb/100** computed from raw chip deltas, independent of the
reward mode. A model can improve its normalised reward without improving the
metric you actually care about, and that has to be visible.

### Folding when you can check

The specification's §5 example lists FOLD as legal with no bet outstanding.
It *is* legal in casino poker, but it is strictly dominated by checking and
only injects noise into self-play. Default: **disabled**. Set
`EnvConfig.allow_fold_when_check_available = True` to follow the specification
literally.

### Action abstraction

Fixed 10-action space; the environment masks what is illegal.

```
0 FOLD   1 CHECK   2 CALL
3 BET_SMALL (0.33x pot)   4 BET_MEDIUM (0.66x pot)   5 BET_LARGE (1.00x pot)
6 RAISE_SMALL (2x bet)    7 RAISE_MEDIUM (3x bet)    8 RAISE_LARGE (4x bet)
9 ALL_IN
```

Two rules keep the abstraction collision-free, so every legal action is a
distinct chip amount:

* a bet/raise bucket that would meet or exceed the stack is masked out —
  `ALL_IN` covers it;
* when the stack is at or below the call amount, `ALL_IN` is masked out —
  `CALL` already means "call for what I have";
* buckets clamped by the minimum raise into the same amount are de-duplicated.

The sizing family is chosen by whether a bet *exists*, not by whether the actor
*faces* one, so the big blind's preflop option is correctly a raise.

### Three-player perspective

Observations are canonicalised to relative seats:

```
0 = SELF   1 = OPPONENT_LEFT   2 = OPPONENT_RIGHT
```

`OPPONENT_LEFT` is the next seat clockwise. Villain-left acting therefore sees
`SELF = villain_left, LEFT = villain_right, RIGHT = hero`. The network receives
no absolute seat identity; position that genuinely matters is supplied
explicitly as distance from the button (dealer / SB / BB one-hot).

### Terminal rewards and per-seat returns

This is the part most likely to be silently wrong, so it is worth spelling out.

A transition belongs to the seat that acted, and **that seat's "next state" is
its own next decision point** — not the state that immediately follows, which
usually belongs to a different player. Self-play therefore builds one decision
chain per seat and runs the λ-return recursion along each chain independently:

```
G_t = gamma * [ (1 - lam) * V(s_{t+1}) + lam * G_{t+1} ]      (t < T)
G_T = terminal reward for that seat
```

A seat that folds on the flop terminates its chain there — its outcome is
already determined — while the other seats keep acting. Every transition
stores its `player_perspective`, and `tests/test_self_play.py` asserts that
with `gamma = lam = 1` every target equals *that seat's* own outcome, which is
the test that catches training all three seats from hero's point of view.

### Observation dimensionality

**982 features** with the default config. Printed at startup by every entry
point:

| group | dim | contents |
|---|---|---|
| `cards` | 146 | own hole cards (2x18), masked opponent slots (72), derived features (38) |
| `board` | 94 | five board slots (5x18), street one-hot (4) |
| `players` | 42 | 3 x 14 per-player features, ordered SELF / LEFT / RIGHT |
| `pot_history` | 694 | pot features (12), action history (32x21), legal mask (10) |
| `position` | 6 | position relative to the button |

Every card is rank one-hot (13) + suit one-hot (4) + present mask (1). Unknown
cards are zero-filled and masked. Monetary values are normalised by the big
blind and by the effective stack, with epsilon-guarded division.

The 72 opponent-card features exist and are **always zero** during play. They
are kept so the leakage test has an explicit region to assert on;
`ObsConfig.include_opponent_card_slots = False` removes them.

### Equity feature

Disabled by default, per the specification. When enabled
(`ObsConfig.use_equity_feature`), equity is estimated by Monte-Carlo rollout
sampling opponent holdings from *the full deck minus the acting player's own
cards and the board* — never from the cards actually dealt. The leakage test
covers this path specifically.

---

## Information leakage

Priority #2 in the specification, and tested by construction rather than by
inspecting feature indices. `tests/test_information_leakage.py` mutates the
hidden world and requires the encoded observation to come out bit-for-bit
identical:

* replace both opponents' hole cards with different cards -> identical tensor;
* reorder the undealt deck, changing every future board card -> identical
  tensor;
* two hands where hero's aces win versus lose -> identical tensor at the same
  decision point.

Any leak, however indirect, breaks that equality.

---

## Policy improvement

The KL/entropy-regularised improvement operator:

```
pi_new(a|s)  proportional to  exp( (Q(s,a) + beta * log pi_ref(a|s)) / (alpha + beta) )
```

restricted to legal actions and renormalised over them. Equivalently:

```
pi_new  proportional to  pi_ref^(beta/(alpha+beta)) * exp(Q/(alpha+beta))
```

* smaller `alpha + beta` -> greedier step;
* larger `beta/alpha` -> stays closer to the reference policy;
* `alpha, beta -> 0` -> greedy `argmax Q`.

Defaults are `alpha = beta = 0.5`. With binary rewards Q lives in `[-1, 1]`, so
`alpha + beta = 1` spreads the best and worst action by up to ~7x while still
regularising. (`alpha = beta = 1.0` is a valid but very conservative choice —
it moves the policy so little that learning is slow.)

The reference policy defaults to the **current network's policy, detached** —
i.e. the policy immediately before the update. `reference_policy = "behavior"`
anchors to the behaviour policy stored with each transition instead, which is a
truer trust region but grows stale as the buffer ages.

The mask is applied *before* normalisation everywhere, so an illegal action can
never receive probability mass.

## Losses

```
L_total = q_weight * L_Q + policy_weight * L_policy + entropy_weight * L_entropy
```

* `L_Q` — Huber regression of `Q(s, a_taken)` onto the λ-return target. Only
  the taken action is trained; counterfactual targets are not available
  model-free.
* `L_policy` — cross-entropy of the current policy against the improved policy,
  over legal actions only.
* `L_entropy` — *negative* mean entropy, so a positive coefficient rewards
  exploration.

Defaults: `1.0 / 1.0 / 0.01`, all configurable.

---

## Layout

```
environment/     rules: cards, hand evaluation, betting, state, the env itself
representation/  canonicalisation, observation encoding, action encoding
model/           residual blocks, heads, the shared network
training/        self-play, replay buffer, returns, policy improvement, trainer
evaluation/      baseline agents and the match-play harness
inference/       the public suggest_action API
tests/           149 tests
config.py        every tunable, serialised into checkpoints
train.py  evaluate.py  infer.py
```

The `Agent` interface and `NetworkAgent` live in `training/self_play.py` so the
dependency graph stays acyclic (`evaluation` imports `training`, never the
reverse).

Checkpoints carry their full `Config`, so a trained network can never be loaded
against an observation layout it was not trained on — `load_checkpoint` checks
and refuses.

---

## Inference API

```python
from inference.suggest_action import SuggestionEngine

engine = SuggestionEngine.from_checkpoint("checkpoints/latest.pt")
engine.suggest(table_state, temperature=1.0, mode="argmax")
```

```json
{
  "action": "CHECK",
  "probabilities": {"FOLD": 0.0, "CHECK": 0.63, "...": "..."},
  "q_values": {"FOLD": null, "CHECK": 0.18, "...": "..."},
  "legal_actions": ["CHECK", "BET_SMALL", "BET_MEDIUM", "BET_LARGE", "ALL_IN"],
  "acting_player": "hero",
  "bet_to": 60,
  "chips_to_put_in": 60,
  "state_value": 0.18
}
```

Probabilities sum to 1 over legal actions; illegal actions report probability
`0.0` and a `null` Q-value. `mode` is `argmax` or `sample`; `temperature = 0`
collapses onto the best action.

Three things the raw table-state format does not carry, and how they are
handled:

* **whose turn it is** — defaults to `hero`; pass `"to_act"` to override;
* **action history** — absent, so the history block is zero-padded and masked.
  A suggestion from a bare snapshot is weaker than one made inside a live hand;
  pass `"action_history"` to fill it in;
* **who contributed what on earlier streets** — only the total pot is given, so
  the dead portion is split evenly between live players. This affects
  normalised features only, never legality.

Malformed states raise `TableStateError` — duplicate cards, a board that does
not match the street, a pot smaller than the posted bets, an acting player with
no hole cards, and so on.

---

## Evaluation

Three-handed poker is not zero-sum between any *pair* of players, and the
button has a real edge, so a single seat assignment is not a fair comparison.
Every matchup is played in **all three seat rotations** and reported both per
seat and per agent.

Baselines: random (uniform over legal actions), calling station, always-fold,
and an equity-versus-pot-odds heuristic bot that reads only its own cards and
the public board. Also supported: current network vs. an older checkpoint, and
the self-play population against itself.

Tracked per seat and per agent: win / loss / tie rate, average reward, average
chip return, and bb/100.

A useful sanity check that falls out of this: three copies of the same agent
must show a win rate near 1/3 and average chips of exactly 0.

### bb/100 is extremely noisy, and independent hands are not enough

With 50bb stacks the same checkpoint measured −500 bb/100 against a calling
station at 300 hands and +69 bb/100 at 3000. Even at **900 hands the 95%
interval is ±230 to ±350 bb/100** — wider than most effects worth detecting.
Any comparison reported without an interval is likely to be noise.

Two variance reductions are built in, both in `evaluation/evaluate.py`:

* **Duplicate deals** (`duplicate_deal_scores`). Each deal is replayed three
  times from the *same* deck order with the hero rotated through every seat,
  and the three results averaged. This cancels most of the "who was dealt
  aces" term and roughly halves the interval at equal hand count.
* **Paired comparison** (`compare_agents_paired`). Because the deck order is
  fixed by the deal seed, two agents can be scored on *identical* cards and
  compared deal-by-deal, so the interval is computed on the per-deal
  difference rather than on two independent samples.

`bb_per_100_interval` reports the mean with a 95% interval and a
`significant` flag. Use it.

---

## Results: reward design

### Binary reward optimises the wrong objective

Two 250-iteration runs (~16k self-play hands each, identical apart from
`reward_mode`), evaluated over 3000 hands with seat rotation. "chip-return"
here is the earlier `Δchips / starting_stack` variant:

| matchup | bb/100 | win rate |
|---|---:|---:|
| binary vs 2x calling station | +69 | 29.1% |
| **chip-return** vs 2x calling station | **+209** | 25.5% |
| binary vs 2x random | +424 | 48.6% |
| chip-return vs 2x random | +301 | 30.5% |
| **chip-return vs 2x binary** | **+87** | 19.2% |
| binary vs 2x chip-return | −126 | 48.7% |

The head-to-head agrees in both directions: the chip-return network beats the
binary network. And the inversion is the whole story — **the binary network
wins 48.7% of hands against the chip-return network while losing 126 bb/100.**

The mechanism is visible in the policies. Playing greedily, the binary network
reaches showdown in **100%** of self-play hands; the chip-return network in
45%. Sampling at temperature 1, they fold 18.5% and 27.7% of the time
respectively when folding is legal.

This is not a bug — it is what the objective says. Under `sign(net chips)`,
once you have posted a blind, folding locks in −1 while calling preserves a
chance of +1, *no matter how much money the call costs*. The binary reward
cannot see the difference between losing 1 chip and losing 500, so the optimal
binary policy is to essentially never fold. It genuinely maximises win rate;
win rate is just not what wins money at poker.

`binary` is therefore no longer the default. It is retained for comparison and
as a clean, bounded, easily-learned signal, but it is not a poker objective.

### alpha and beta silently carried units

Discovered while running experiment C below. The improvement operator computes
`exp(Q / (alpha + beta))`, so the *reward scale* changes what `alpha` and
`beta` mean. Under `bb_normalized` the reward reaches ±50, and the operator
exponentiates ±50:

| \|Q\| | best/worst probability ratio at `alpha + beta = 1` |
|---:|---:|
| 0.3 | 1.8x |
| 1.0 | 7.4x |
| 50.0 | 2.7e43 |

The policy collapsed to deterministic on the first update — entropy 0.13,
3.0 decisions per hand (everyone folding immediately), showdown rate 0.00. The
Q loss compounded it: Huber with `delta = 1` on targets of ±50 is effectively
linear, so the Q term outweighed the policy term ~50x.

Both are fixed by normalising Q by the reward scale — inside the operator, and
on both sides of the Huber loss. `alpha`, `beta`, `huber_delta`, `q_weight` and
`policy_weight` now mean the same thing in every reward mode. The effect on
the same `bb_normalized` configuration:

| | q_loss | entropy | decisions/hand | showdown |
|---|---:|---:|---:|---:|
| unnormalised operator | 16.4 | 0.13 | 3.0 | 0.00 |
| scale-corrected | 0.30 | 1.40 | 5.1 | 0.55 |

This is invisible under `binary` and `normalized_chip_return`, where the scale
is 1.0 and the correction is a mathematical no-op. It only bites when the
reward magnitude changes — which is exactly what happens when you switch
objectives.

### Experiment matrix

Three runs of **1000 iterations (~64k self-play hands each)**, same seed,
differing only in objective and Q head. Scored on **1200 duplicate deals
(3600 hands) per cell**, identical cards for every agent, under one common rule
set. `*` = 95% interval excludes zero.

| | objective | Q head | vs random | vs calling station |
|---|---|---|---:|---:|
| **A** | `normalized_chip_return` | `2·tanh` | +356 ±106 * | +200 ±101 * |
| **B** | `normalized_chip_return` | linear | +399 ±104 * | +190 ±94 * |
| **C** | `bb_normalized` | linear | +324 ±101 * | +49 ±107 |

Paired differences on identical deals, and head-to-head:

| paired difference | vs random | vs station | | head to head | bb/100 |
|---|---:|---:|---|---|---:|
| A − B | −43 ±125 | +10 ±121 | | A vs 2x B | −2 ±87 |
| A − C | +32 ±117 | **+151 ±135 \*** | | B vs 2x A | +6 ±89 |
| B − C | +75 ±122 | **+141 ±126 \*** | | C vs 2x A | **+147 ±84 \*** |

All three also lose to the equity-based heuristic bot, which remains the
strongest player in the pool (700 duplicate deals = 2100 hands per cell):

| | vs tight-aggressive | vs loose-passive |
|---|---:|---:|
| **A** | −214 ±98 * | −253 ±114 * |
| **B** | −395 ±99 * | −481 ±108 * |
| **C** | −149 ±90 * | −334 ±120 * |

| paired difference | vs tight-aggressive | vs loose-passive |
|---|---:|---:|
| **A − B** | **+181 ±108 \*** | **+229 ±141 \*** |
| A − C | −65 ±110 | +81 ±147 |
| B − C | −246 ±108 * | −147 ±148 |

### What experiment B actually answers

The bounded Q head **does** help — but the effect is only visible against an
opponent strong enough to punish bad play:

| opponent | A − B | significant? |
|---|---:|---|
| random | −43 ±125 | no |
| calling station | +10 ±121 | no |
| head-to-head | −2 ±87 | no |
| tight-aggressive heuristic | **+181 ±108** | **yes** |
| loose-passive heuristic | **+229 ±141** | **yes** |

Against random and calling-station opponents A and B are indistinguishable;
against both heuristic variants A beats B by ~180–230 bb/100 with intervals
that exclude zero. So the answer to "is the tanh unnecessarily restrictive?" is
**no — the scaled-tanh critic is better**, and the reason the difference
vanishes against the weak baselines is that a bot which never folds and never
raises cannot exploit a worse-calibrated critic. The baseline has to be strong
enough for the difference to have somewhere to show up.

The ranking against the heuristics is **A ≈ C > B**, while against the calling
station it was **A ≈ B > C**. Three-player poker is not transitive, and neither
is "strength" here: which agent looks best depends on the opponent pool. If you
have to pick one number to optimise, pick results against the strongest
available opponent.

### Why it loses to the heuristic: alpha caps the policy

Diagnosing the trained default agent at real decision points against the
heuristic:

| measurement | value | reading |
|---|---:|---|
| Q spread over legal actions | 0.458 | the **critic has learned** — it separates actions |
| policy entropy | 1.581 | uniform over 5 actions is 1.609: the **policy is ~random** |
| KL(improved ‖ current) | 0.0005 | the improvement step moves almost nothing |
| greedy action mix | ALL_IN **26%** | it jams a quarter of the time |
| avg won / avg lost | 226 / 382 chips | wins small, loses big — at a 50% win rate |

The cause is algebraic, not a bug. The improvement operator is

```
pi_new  ∝  pi_ref^(β/(α+β)) · exp(Q/(α+β))
```

and its **fixed point** (where `pi = improve(pi)`) solves
`pi^(α/(α+β)) ∝ exp(Q/(α+β))`, i.e.

```
pi*  ∝  exp(Q / alpha)          ← beta cancels entirely
```

**β controls how fast you approach the fixed point; α alone decides how sharp
that fixed point is.** With the measured Q spread of 0.458:

| alpha | best/worst probability ratio | entropy at the fixed point |
|---:|---:|---:|
| 0.5 (old default) | 2.5 | 1.559 |
| 0.2 | 9.9 | 1.339 |
| 0.1 | 97 | 0.895 |
| 0.05 | 9509 | 0.365 |
| *uniform* | 1.0 | 1.609 |

Measured entropy was **1.581** against a predicted **1.559** — the policy had
converged, and the fixed point itself was nearly uniform. At α = 0.5 the agent
*cannot* play sharply no matter how good its critic gets, and near-uniform play
over this action space means shoving 26% of the time.

This also explains the plateau below: extra hands sharpen the critic, but the
policy is pinned by α, so nothing downstream improves.

### Fixing it: the alpha sweep

Each row is 300 iterations (19k hands), evaluated on 600-700 duplicate deals
against the held-out `tight_aggressive` heuristic. The old default is 1000
iterations (64k hands) for reference:

| config | vs heuristic (bb/100) | entropy | ALL_IN | FOLD |
|---|---:|---:|---:|---:|
| α = 0.5, **1000** iters (old default) | −196 ±106 | 1.57 | 20% | 14% |
| α = 0.20 | −239 ±102 | 1.52 | 32% | 18% |
| α = 0.10 | −171 ±85 | 1.46 | 16% | 25% |
| **α = 0.05** | **−42 ±78** | 1.01 | 5% | 23% |
| α = 0.02 | −244 ±85 | 0.68 | 6% | 8% |
| α = 0.01 | −53 ±67 | 0.43 | 15% | 37% |
| α = 0.10 + opponent pool | −96 ±80 | 1.36 | 17% | 26% |
| α = 0.05 + opponent pool | −62 ±58 | 1.10 | 5% | 26% |

**What is solid:** lowering α from 0.5 is worth roughly **140 bb/100**, and does
it with 3× less training. The best configurations cluster at −40 to −60 and
their intervals cover zero (break-even against the heuristic); the old default
was −196. The mechanism is verified independently of the sweep: predicted
fixed-point entropy 1.559 versus measured 1.581.

**What is not solid:** the exact optimum. The response is *not monotone* —
α = 0.02 collapses to −244 while its neighbours 0.05 and 0.01 both reach ~−50.
A 200-point swing between adjacent settings cannot be explained by the ±80
evaluation intervals.

That is the methodological trap in this table: the intervals measure
*evaluation* noise (card luck), but every row is a **single training seed**, and
seed-to-seed trajectory variance is evidently larger. α = 0.02 folds 8% of the
time and α = 0.01 folds 37% — qualitatively different strategies from adjacent
hyperparameters. Pinning the optimum needs 3-5 seeds per setting compared on
seed means. Until then α ≈ 0.05 is a working default, not a tuned value.

**The two fixes are not additive.** The opponent pool is worth ~75 bb/100 at
α = 0.10 (−171 → −96) but nothing at α = 0.05 (−42 → −62, intervals heavily
overlapping). The plausible reading is that both address the same underlying
defect — a critic calibrated against a bad opponent distribution. Lowering α
fixes it by making self-play itself sharper; the pool fixes it by importing
competent opponents. Do either and the other adds little.

Note the causal chain runs through the *critic*, not just the actor: evaluation
is greedy, so the shove rate falling from 20% to 5% means the critic itself
stopped rating ALL_IN best.

**Still not beating the heuristic.** Break-even is not the goal; see the
roadmap for what is left.

### Training plateaus early

Same runs, evaluated at intermediate checkpoints (900 duplicate deals each):

| iteration | A vs random | A vs station | C vs random | C vs station |
|---:|---:|---:|---:|---:|
| 250 | +342 ±121 | +257 ±113 | +331 ±124 | +201 ±109 |
| 500 | +398 ±123 | +186 ±119 | +434 ±114 | +349 ±127 |
| 750 | +260 ±119 | +304 ±108 | +321 ±122 | +249 ±122 |
| 1000 | +380 ±125 | +204 ±118 | +364 ±117 | +55 ±122 |

Every column is flat within its interval. **Training from 16k to 64k hands
bought no measurable improvement** against fixed baselines. The bottleneck is
not the number of hands, and it is not the critic — the plausible candidates
are network capacity, the coarse bet abstraction, and self-play against a
single shared policy providing limited strategic diversity.

### A correction, twice over

This conclusion was measured three times and moved twice; the history is worth
keeping.

1. **150 iterations, 2500 independent hands, no intervals** — claimed A beats B
   and that the bounded head mattered. Not supported by its own evidence: at
   that sample size the interval against a calling station is roughly ±200
   bb/100, wider than the effect claimed.
2. **1000 iterations, duplicate deals, intervals, weak baselines** — A and B
   indistinguishable everywhere. Read as a null result.
3. **Same runs, heuristic opponents** — A beats B by +181 ±108 and +229 ±141.

The reconciliation is that (2) was measured against opponents too weak to
discriminate, not that the effect was absent. Steps (1) and (3) agree in
direction, but (1) had no right to that conclusion at the time.

Two lessons, both now baked into `evaluation/evaluate.py`: always report an
interval, and **your choice of baseline bounds what you are able to detect.**
A calling station cannot tell a good agent from a mediocre one.

---|---|---|---:|---:|
| **A** | `normalized_chip_return` | `2·tanh` | +193 | **+128** |
| **B** | `normalized_chip_return` | linear | +221 | **+1** |
| **C** | `bb_normalized` | linear | +254 | +99 |

Head to head, one seat against two copies of the other:

| matchup | bb/100 | | matchup | bb/100 |
|---|---:|---|---|---:|
| A vs 2x B | **+121** | | B vs 2x A | −50 |
| C vs 2x B | **+93** | | B vs 2x C | −101 |
| C vs 2x A | +51 | | A vs 2x C | −6 |

**Experiment B answers its own question: the bounded Q head is not limiting
performance.** With the objective held fixed, replacing the scaled tanh with a
linear head made the agent *worse* on every measure — near-breakeven against a
calling station, and losing in both directions of the head-to-head. Four
independent measurements agree. The likely reason is conditioning rather than
capacity: the squash keeps early Q estimates inside the range the returns
actually occupy, and a linear head has to discover that scale from a very noisy
terminal-only signal.

A and C are comparable — the A-vs-C head-to-head is +51 / −6, which is inside
the noise at this sample size. Both are sound choices. A is the default because
its bounded critic is better conditioned; C is the conventional poker
parameterisation and is worth running if you prefer BB units end to end.

Note this only became a fair test after the scaled tanh: a plain `tanh` capped
at 1 cannot represent the +1..+2 band that 16% of outcomes land in, so the
earlier `normalized_chip_return` runs were handicapping their own critic.

---

## Roadmap: what actually moves the needle

Ordered by measured or expected value per unit of effort. The first two are
implemented and measured; the rest are not done yet.

### 1. Lower alpha (done — worth ~150 bb/100)

`alpha` alone sets how sharp the converged policy can be (`pi* ∝ exp(Q/alpha)`).
At the old default of 0.5 the agent was pinned near-uniform and shoved 20-32%
of the time regardless of how good its critic got. Config-only change; see the
sweep above.

### 2. Mixed-opponent and league training (done)

`--opponent-pool heuristic,loose_passive,calling_station,checkpoint`
`--opponent-mix-prob 0.5`

Pure self-play calibrates Q against exactly one distribution: its own current
policy. Seating fixed bots and frozen past checkpoints in 1-2 seats for a
fraction of hands widens that distribution. Transitions are collected **only**
from learner seats — an opponent's decisions were not drawn from the policy
being improved and must never become training data.

Frozen snapshots are taken every `league_snapshot_every` iterations, keeping
the most recent `league_size`.

### 3. Enable the equity feature (not done — likely the next big one)

`ObsConfig.use_equity_feature = True`

The heuristic's entire edge is equity versus pot odds, computed by rollout. The
network currently has to learn that relationship from scratch out of a
terminal-only, extremely noisy signal. The feature is already implemented and
provably leak-free (it samples opponent holdings from the full deck minus the
acting player's own cards and the board, never from what was actually dealt).
Cost: slower self-play, since it runs a Monte-Carlo rollout per decision.

### 4. All-in EV runouts (not done — biggest variance reduction available)

When betting is closed and players are all-in, the environment currently deals
one random runout and pays the winner. Replacing that with the **exact expected
split over all remaining boards** removes the single largest source of noise
from the learning signal, at no cost in correctness — it is the same
expectation with far lower variance. Standard practice in poker RL. This is the
training-time analogue of the duplicate-deal trick already used in evaluation.

### 5. Finer action abstraction (not done)

Three bet sizes and three raise sizes is coarse. Real solvers use many more,
especially small river sizings. This widens the policy space the agent can
express and is the most likely ceiling once the above are in.

### 6. Throughput (not done)

~35 hands/second single-threaded, bottlenecked on Python observation building
and one single-sample forward pass per decision. Batching decisions across
parallel hands should give 10-50x, which makes everything above cheaper to
iterate on.

### What does *not* help

* **More hands at fixed hyperparameters.** Measured flat from 16k to 64k hands.
* **Reward shaping** for pot size, aggression, or reaching showdown. It invites
  the agent to farm the reward function; the terminal-only signal is correct.

---

## Performance and limitations

* ~30 hands/second single-threaded on a laptop CPU. The bottleneck is Python
  observation construction plus one single-sample forward pass per decision,
  not the network. Batching decisions across parallel hands is the obvious
  next optimisation; per the specification's priorities, it was not done
  prematurely.
* Default replay capacity is 100k transitions (~400 MB reserved, observations
  stored as float16). 1M is supported — set `replay_capacity` — at ~4 GB.
* λ-return targets are computed at collection time from the collecting
  network's values, so they age slightly as the buffer turns over. With
  terminal-only rewards and short chains (a few decisions per seat per hand)
  the targets are close to Monte-Carlo returns, so the staleness is mild.
  `next_observation` is stored for every transition and is available if you
  want to recompute bootstraps at training time instead.
* The action abstraction is coarse. Real solvers use far more sizings.
* There is no exploitability computation — only the proxy of results against
  fixed baselines and older checkpoints.
* `binary` reward maximises win rate, not chip EV — see
  [Results](#binary-reward-optimises-the-wrong-objective).  It is not the
  default; it is kept for comparison.
* The reward normalises by *per-hand* starting stack, which optimises expected
  percentage return rather than long-run bankroll.  Because `reset()` restores
  stacks every hand, each hand is an independent episode and the two coincide
  in the default configuration; they diverge if you add persistent stacks.
  Log-utility shaping (`U(C) = log C`) would make the objective explicitly
  risk-averse and is deliberately *not* implemented.
* The trained agent is not a strong poker player. A 390k-parameter network
  beats the random and calling-station baselines comfortably and still loses
  chips to the equity-based heuristic bot. More hands is *not* the fix: results
  are flat from 16k to 64k hands (see
  [Training plateaus early](#training-plateaus-early)). The open candidates are
  network capacity, the coarse bet abstraction, and the limited strategic
  diversity of self-play against a single shared policy.
