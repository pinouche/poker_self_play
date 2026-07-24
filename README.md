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

Defaults are `alpha = 0.05`, `beta = 0.5`. Writing the operator as softmax
logits, `score(a) = Q(a)/(alpha+beta) + [beta/(alpha+beta)]·log π_ref(a)`, so the
weight on the reference is `beta/(alpha+beta)`: at `1.0` it reproduces `π_ref`
exactly, below `1.0` it flattens `π_ref` toward uniform on every update. The old
`alpha = beta = 0.5` gives a reference weight of `0.5`, which — on the default
`normalized_chip_return` reward, where early Q-spreads between actions are tiny
(~0.01–0.02) — flattens the policy toward uniform faster than the signal can
sharpen it. The result is a self-reinforcing cold start: play stays uniform,
entropy pins near `log(#legal)`, and `kl_target_vs_policy ≈ 0`. Lowering `alpha`
to `0.05` raises the reference weight to ~`0.91` (preserve, don't flatten) and
the Q weight to ~`1.8`, so small advantages accumulate and the policy escapes.
This is the measured break-even setting from the alpha sweep below and matches
the benchmark's `full` config. (With binary rewards Q lives in `[-1, 1]`, so
`alpha + beta ≈ 1` is fine there — raise `alpha` back up for that mode.)

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

## Roadmap: implemented

All six items below are implemented and tested. The first two are measured;
the rest are mechanisms whose effect still needs a full training comparison.

### 1. Lower alpha (measured: ~140 bb/100)

`alpha` alone sets how sharp the converged policy can be (`pi* ∝ exp(Q/alpha)`).
At the old default of 0.5 the agent was pinned near-uniform and shoved 20-32% of
the time regardless of how good its critic got. See the sweep above.

### 2. Mixed-opponent and league training (measured: ~75 bb/100 at alpha=0.10)

```
--opponent-pool heuristic,loose_passive,calling_station,checkpoint
--opponent-mix-prob 0.5 --league-recency-decay 0.5
```

Pure self-play calibrates Q against exactly one distribution: its own current
policy. Fixed bots and frozen snapshots occupy 1-2 seats for a fraction of
hands. Transitions are collected **only** from learner seats -- an opponent's
decisions were not drawn from the policy being improved.

**Snapshots are sampled by recency**: the newest gets weight 1, the one before
it `league_recency_decay`, then `decay²`, and so on, so the learner mostly
faces recent (stronger) versions of itself while older ones stay in the mix to
avoid cycling against a single opponent. At the default 0.5 with a league of 4
the shares are 54 / 26 / 13 / 7 percent.

### 3. Equity feature, memoised (now on by default)

`ObsConfig.use_equity_feature = True`. Naively this costs 40x in observation
throughput. It is made affordable by memoising on a **suit-isomorphic** key:
equity is invariant under relabelling suits, so taking the lexicographically
smallest image over all 24 suit permutations collapses 1326 preflop hole
combinations to 169 and merges every board with the same suit pattern. Measured
~70% cache hit rate and a 5.4x speedup over the uncached version.

### 4. All-in EV runouts (measured: reward std 0.79 -> 0.40)

`EnvConfig.all_in_ev_runout = True`. When betting is closed, the pot is settled
on the **expected split over every remaining board** rather than one random
runout -- enumerated exactly when there are few completions, sampled otherwise.
Identical in expectation (mean reward stays 0.0, chips conserve to 1e-13) at
**half the standard deviation**, which is worth as much as quadrupling the
number of hands. A concrete board is still dealt so observations and history
stay meaningful; only the chips are settled on the expectation. Preset boards
always take the concrete path so scripted hands stay reproducible.

### 5. Configurable bet abstraction

The action space is still *fixed* for a given configuration -- the network needs
a constant output width -- but its **size is derived** from `bet_fractions` and
`raise_multipliers`:

```
0 FOLD, 1 CHECK, 2 CALL, <bet sizings>, <raise sizings>, ALL_IN last
```

The default three-and-three reproduces the documented ten-action space exactly,
ids and names included. A finer setting widens the mask, every action-history
slot, and the network heads automatically:

| sizings | actions | observation dim |
|---|---:|---:|
| 3 bets, 3 raises (default) | 10 | 984 |
| 5 bets, 5 raises | 14 | 1116 |
| 8 bets, 6 raises | 18 | 1248 |

Names stay `BET_SMALL/MEDIUM/LARGE` for the classic three and become
size-derived (`BET_25`, `RAISE_2_5X`) otherwise. The heuristic bot expresses its
preferences as "largest available raise" rather than fixed ids, so it keeps
working in any space.

### 6. Batched self-play (measured: 2.3x)

`--self-play-envs 64` advances that many hands in lockstep so their decisions
batch into a single forward pass. Semantics are identical -- at temperature 0
the batched and sequential workers agree transition for transition on the same
deal, which is asserted in the tests.

The speedup is **2.3x, not the 10-50x estimated earlier**: batching removes the
per-decision forward pass, which shifts the bottleneck onto Python-side
observation construction. Getting further means vectorising the encoder, not
larger batches.

### What does *not* help

* **More hands at fixed hyperparameters.** Measured flat from 16k to 64k hands.
* **Reward shaping** for pot size, aggression, or reaching showdown.

---

## Benchmark

`benchmark.py` trains a matrix of configurations and scores each against the
three fixed baselines. It is an **ablation from one reference configuration**
rather than a grid, so every difference is attributable to a single change.

```bash
python benchmark.py train    --config full --seed 0 --root runs/   # per cell
python benchmark.py evaluate --config full --seed 0 --root runs/
python benchmark.py report   --root runs/ --out runs/report        # tables + plots
python benchmark.py replot   --report runs/report                  # redraw figures
```

`report` writes `<prefix>_histories.json` alongside the tables, holding every
cell's per-iteration metrics for every seed. `replot` redraws both figures from
that file and `<prefix>.json` alone, so the plots can be restyled after the run
directories — which are large, and hold the checkpoints — have been deleted.
Without it, restyling a curve means re-training the cell.

Results below: 8 configurations x 2 seeds, 250 iterations (16k hands) each,
scored on 600 duplicate deals (1800 hands) per cell.

| config | vs random | vs calling station | vs heuristic |
|---|---:|---:|---:|
| `full` (α=0.05, equity, EV, pool) | +523 | +926 | −77 |
| `no_equity` | +544 | +703 | −52 |
| `no_ev_runout` | +264 | +689 | −25 |
| `no_pool` | +504 | +379 | −15 |
| `fine_abstraction` (5×5 sizings) | +628 | +939 | −39 |
| `alpha_0.10` | +640 | +818 | −166 |
| **`alpha_0.02`** | +504 | +914 | **+11** |
| `legacy` (the original config) | +313 | +211 | **−413** |

![results](benchmark_results/report_results.png)

### The noise floor

Seed-to-seed spread over just two seeds reaches **490 bb/100** against the
calling station, ~200 against random and ~190 against the heuristic. Any
difference smaller than that is not a result. The evaluation intervals
(±40-90) are much tighter than the seed spread, which is exactly why the plot
shows the **seed range** as whiskers rather than the evaluation interval.

### What survives that filter

* **The overall improvement is large and unambiguous.** Against the heuristic,
  `legacy` −413 → roughly break-even. Against the calling station, +211 → +900.
  Against random, +313 → +500-640.
* **All-in EV runouts are the clearest single win.** Removing them costs
  −259 bb/100 against random with a seed spread of only ~16-53, the tightest
  measurement in the table. The training curves show why directly: `q_loss`
  sits ~3x higher without them.
* **Lower alpha keeps helping against the strongest opponent**: −166 (α=0.10)
  → −77 (α=0.05) → +11 (α=0.02). Note α=0.02 was *unstable* in the earlier
  sweep and is now the best cell — the instability was caused by the noisy Q
  signal that EV runouts removed. The two changes interact.
* **`alpha_0.02` is the first configuration that is not losing to the
  heuristic** (+11, interval covering zero).

### What does not survive it

* **Equity is not demonstrated.** +22 against random (noise), and it costs
  3-4x self-play throughput. On this evidence I would leave it off.
* **The finer abstraction is promising but unproven**: +105/+13/+38, all inside
  the seed noise. It is never worse, which is mildly encouraging.

### A contamination caveat, and it matters

The opponent pool contains `calling_station`, `heuristic` and `loose_passive`
— and two of those are also *evaluation baselines*. For any pool-using
configuration the "vs calling station" and "vs heuristic" columns are partly
**training on the test opponent**.

The evidence that this matters: dropping the pool costs 548 bb/100 against the
calling station but only **18 against random** — and random is the one baseline
*not* in the pool. So the pool's apparent benefit is largely contamination, and
`vs random` is the only clean generalisation measure in the table.

Fixing this properly means holding a family of opponents out of training
entirely. The `--opponent-pool` flag makes that easy; it was not done here.

### Training curves

![curves](benchmark_results/report_curves.png)

Reading them:

* **entropy** — `legacy` (purple) sits flat at 1.30 against a uniform bound of
  1.61: the original failure, a policy ignoring its critic. `alpha_0.02` (blue)
  is lowest at ~0.65 and still visibly oscillates.
* **q_loss** — `no_ev_runout` (pink, 0.14-0.47) and `legacy` (purple, ~0.17)
  are several times higher than every EV-runout configuration (~0.05). This is
  the variance reduction visible directly in the learning signal.
* **kl_target_vs_policy** — collapses toward zero for every configuration by
  iteration ~150: the policy has reached its fixed point, and after that only a
  better critic (or a smaller alpha) moves it.

---

## Alpha sweep on a clean setup

The first benchmark's pool contained `calling_station` and heuristic-family
bots that were *also* evaluation baselines. This sweep removes them: the pool
is **self-snapshots only**, so all three baselines are genuinely held out.
Equity is off (never demonstrated, 3-4x throughput cost) and the finer 5x5
abstraction is kept. Five alpha values, **3 seeds each**, 250 iterations,
700 duplicate deals (2100 hands) per cell.

| alpha | vs random | vs calling station | vs heuristic |
|---:|---:|---:|---:|
| 0.01 | +593 | +713 | −66 |
| 0.02 | +644 | +523 | −66 |
| 0.03 | +602 | +388 | −108 |
| 0.04 | +600 | +584 | −74 |
| 0.05 | +559 | +405 | −50 |

![alpha sweep](benchmark_results/alpha_sweep/report_results.png)

### Alpha has no detectable effect below 0.05

Within-alpha seed spread reaches **471 bb/100** against the calling station,
305 against random and 235 against the heuristic. The between-alpha variation
(85, 325 and 58 respectively) is *smaller than the noise*. This is a flat
response, not a curve with an optimum.

That is consistent with the mechanism rather than contradicting it. Alpha still
does exactly what the algebra says -- the entropy curves separate cleanly and
monotonically by alpha (0.63, 0.85, 0.98, 1.12, 1.22 at convergence, against a
uniform bound of 1.69). The earlier 0.5 → 0.05 change was worth ~140 bb/100
because it moved the policy off the near-uniform ceiling. Once alpha is small
enough that the policy acts on its critic at all, **making it smaller does not
help further**.

The cost of going too low is visible in the training curves: alpha = 0.01 has
a clearly higher `q_loss` throughout (~0.065 versus ~0.050-0.060), because a
sharper policy explores less and the critic learns less about untaken actions.
That is the exploration/exploitation trade-off appearing directly in the
learning signal.

![alpha sweep curves](benchmark_results/alpha_sweep/report_curves.png)

### The break-even against the heuristic did not reproduce

The earlier benchmark reported `alpha_0.02` at **+11 bb/100** against the
heuristic -- the first configuration not losing to it. On the clean setup the
same alpha gives **−66**, and every alpha in the sweep loses by 50-108.

The difference is exactly the contamination: that cell trained against
`heuristic` and `loose_passive`, from the same family as the evaluation bot.
**The apparent break-even was training on the test opponent, not strength.**

The same correction shows against the calling station: `full` measured +926
with the station in its training pool, while these clean runs measure +388 to
+713 without it.

### Where this leaves the agent

Against genuinely held-out opponents, after 16k hands of self-play:

* **beats random by ~600 bb/100** and **the calling station by ~400-700**, both
  decisively and at every alpha;
* **still loses to the equity-based heuristic by 50-110 bb/100.**

Alpha is no longer the lever. The remaining candidates are the ones the
ablation could not settle: a larger network, materially more hands now that
batching makes them cheaper, and a held-out-opponent league that is broader
than self-snapshots alone.

---

## Capacity sweep (alpha fixed at 0.02)

Alpha is settled, so it is fixed at 0.02 here and the trunk size is varied
instead. Four sizes, 3 seeds each, 250 iterations, plus a deeper self-snapshot
league at the base size. Same clean setup as the alpha sweep (self-snapshots
only, no equity, finer abstraction).

| config | params | vs random | vs calling station | vs heuristic |
|---|---:|---:|---:|---:|
| `cap_small` | 119k | +479 | +378 | −48 |
| `cap_base` | 409k | +644 | +523 | −66 |
| `cap_large` | 1.6M | +596 | +329 | −55 |
| `cap_xlarge` | 4.1M | +378 | +84 | −55 |
| `league_broad` | 409k | +474 | +362 | −92 |

![capacity](benchmark_results/capacity/report_results.png)

### Capacity is not the lever either — and this one is a clean negative

**Against the heuristic every size lands between −48 and −92, and the
within-size seed spread (8 to 270 bb/100) covers that entire range.** There is
no capacity effect that survives the noise. If anything the two biggest changes
(`cap_xlarge`, `league_broad`) are *worse*, not better.

The interesting part is *why*, and the training curves answer it directly.
**Bigger networks fit the critic strictly better**: final `q_loss` is ~0.058
for `cap_base`, ~0.047 for `cap_large` and `league_broad`. The regression the
Q head is asked to solve is easier for a larger trunk, exactly as expected. But
that better fit **does not become better play**:

| config | final q_loss | Q spread over legal actions | vs heuristic |
|---|---:|---:|---:|
| `cap_small` | 0.052 | 0.151 | −48 |
| `cap_base` | 0.058 | 0.223 | −66 |
| `cap_large` | 0.049 | 0.185 | −55 |
| `cap_xlarge` | 0.055 | 0.185 | −36..−55 |

Lower q_loss, same strength; and the Q *spread* (which is what alpha turns into
a policy) does not widen with capacity at all. So the network is already able
to fit the value targets it is given — **the ceiling is not the model's
capacity to represent the value function.** It is upstream: the quality of the
targets (a terminal-only signal, model-free, taken action only) and the
strategic variety of the self-play opponent, which a bigger network cannot
manufacture.

This is a more useful result than a win would have been: it rules out the
cheapest hypothesis and points at the two expensive ones.

### Was xlarge just undertrained? And does more data help? (corrected)

The obvious objection to the capacity result is that a bigger network on 16k
hands is starved of data. `long_base` / `long_large` (1000 iterations, 64k
hands) were meant to test it, and their endpoints looked alarming:

| | 250 iters | 1000 iters (vs heuristic) |
|---|---|---|
| `base` | −72, −64, −63 | −214, −127 |

I first read that as "4x more hands makes it monotonically worse." **That was
wrong, and it is worth showing exactly how it was wrong**, because it is the
easiest mistake to make in this domain.

Two endpoints cannot distinguish degradation from a noisy tail. So I checkpointed
the base config every 100 iterations to 1000 (2 seeds) and scored each against
all three baselines:

![trajectory](benchmark_results/trajectory/trajectory.png)

The vs-heuristic panel **oscillates with no trend at all**: the correlation
between iteration and bb/100 is **0.04**, the fitted slope is +16 per 1000
iterations, and the value swings between −27 and −456. Seed 0 was −27 (nearly
break-even) at iteration 500 and −157 at 1000; the two seeds dip out of phase.
Adjacent checkpoints swing by ~100-170 bb/100, far more than the ±65 evaluation
interval, so the oscillation is real trajectory dynamics, not measurement noise.

**The endpoint comparison was an artifact.** The 1000-iteration runs happened to
land in a down-swing; iteration 500 would have said "more data is *better*."
Comparing two samples of a directionless oscillation is not a measurement.

### What is actually going on

This is **non-stationarity**, the textbook failure mode of naive self-play and
the reason serious systems (AlphaStar, OpenAI Five) use populations rather than
pure self-play. The agent's opponent is its own moving policy, and poker has
rock-paper-scissors structure (tight beats loose-aggressive beats calling beats
tight), so as the self-play policy drifts around that cycle, its matchup against
any *fixed* style swings by hundreds of bb/100 without converging.

The one genuine, signed trend is the opposite of what I claimed, and against a
different opponent: the agent slowly gets **worse at exploiting random and the
calling station** (correlation −0.67 and −0.48, roughly −300 bb/100 over 1000
iterations) while staying massively positive against them. It is un-learning
crude exploitation as it specialises against itself -- but strength against the
disciplined heuristic just oscillates.

### What still holds

* **Capacity is not the bottleneck.** The 250-iteration capacity sweep is a fair
  fixed-budget comparison, and it holds: bigger networks fit the critic strictly
  better (lower q_loss) and play the same or worse. That result does not depend
  on the endpoint mistake.
* **A single self-play checkpoint is not a stable measure of strength.** Its
  matchup against a fixed opponent is a snapshot of an oscillation. Any claim
  about "the agent's strength" should average over checkpoints, or the
  measurement is dominated by *when* you happened to stop.
* **Opponent diversity is still the indicated lever** -- and now for a sharper
  reason. A diverse, held-out league is exactly what damps this oscillation: it
  stops the policy from chasing its own tail around the strategy cycle. Pure
  self-play and self-snapshot leagues cannot, because every opponent shares the
  current style.

### A note on method

I reported the endpoint comparison as a finding before I had the curve, and it
was wrong. The lesson is specific: **against a non-stationary training process,
never compare two endpoints -- measure the trajectory.** The trajectory is
cheap (checkpoint during training, evaluate each), and `trajectory.py` /
`plot_trajectory.py` in `benchmark_results/trajectory/` reproduce it.

---

## Population self-play

The trajectory result -- strength against a fixed opponent oscillating with no
convergence -- is the standard failure of naive self-play, and the standard fix
is a *population*.  `--population` switches the training distribution to:

**A library of frozen checkpoints as opponents.** Every hand seats the learner
once and fills the other two seats with policies sampled from a capacity-bounded
library of past snapshots.  Sampling is **recency-weighted, linearly**: of the
`K` retained snapshots the newest is drawn with weight `K`, the next `K-1`, down
to `1` for the oldest -- a straight-line decline, not the geometric decay used
by the older league (`--league-weighting` selects either).  Facing a
distribution of past selves, rather than only the current self, is what damps
the strategy-cycling.

**Randomised full games.** Each game is dealt from a random button, a random
blind level (`population_big_blind_choices`) and independent per-seat stack
depths (`population_stack_min_bb`..`population_stack_max_bb`), then played from
the button's first action to showdown.  The agent sees the whole range of stack
depths instead of one fixed 50bb setup.  Blinds are now a per-hand property of
the game state, so nothing downstream assumes a fixed big blind.

**A scenario mixture over entry streets.** Most hands start preflop, but a
configurable minority have the learner *enter* on the flop, turn or river
(`entry_street_probs`, default 0.75 / 0.13 / 0.08 / 0.04).  The earlier streets
are still played out -- by a population stand-in in the learner's seat and by
the villains in theirs -- so the state the learner inherits has a realistic
pot, board and opponent range.  It is generated by policies, not sampled
uniformly.  Only the learner's own decisions, from its entry street onward,
become training transitions.

The motivation is coverage: if only ~10% of complete hands reach the river,
complete-hand self-play spends almost all of its gradient on preflop and flop
decisions.  Starting a slice of hands directly on the turn or river
concentrates experience where it is otherwise rare, without resorting to
synthetic states.

```bash
python train.py --population --league-weighting linear
python benchmark.py train --config population --seed 0 --root runs/
```

Because the three seats hold heterogeneous policies, population play does not
batch the way single-network self-play does -- each seat acts with its own
forward pass -- so it runs at the sequential rate rather than the batched one.
Design and rationale live in `training/population.py`.

## Co-evolving population (`--num-policies n`)

`--population` above trains **one** network against *frozen* copies of its past
self.  `--num-policies n` is a different regime: `n` networks are all **live**
and **all optimised at once**.

```bash
python train.py --num-policies 5 --checkpoint-dir checkpoints/pop5
```

`--num-policies` is the only flag that changes; everything else keeps its
default, so this is the standard 200-iteration run with the medium (~1.6M
parameter) network, batched across the default `--self-play-envs 64` -- just
with five networks instead of one.  To match a non-default baseline, pass the
same flags you used before (e.g. `--iterations`, `--self-play-envs`).

Each hand samples one network per seat -- **uniformly, with replacement** --
from the population of `n`.  With replacement means a network can be dealt into
several seats and so play against copies of itself; two seats can also draw two
different members that then play each other.  Every seat's decisions become
training data for the network that produced them (not just one "hero" seat), so
all three seated networks improve from the same hand.  The seat perspective and
lambda-return bookkeeping are exactly the ordinary per-seat machinery; only the
*owner* of each chain differs.

Each network is otherwise independent: its own random initialisation (so the
population starts diverse), its own AdamW optimiser, and its own replay buffer.
The configured `replay_capacity` is split evenly across the `n` buffers, so the
total memory footprint matches a single-network run, and because each network
sees ~`1/n` of the transitions and does ~`1/n` of the updates, its replay ratio
still matches the single-network case.  `--num-policies 1` is exactly ordinary
shared-network self-play and leaves every other code path untouched.

Unlike frozen-library population play, the heterogeneous seats **do** batch:
within each lockstep step the pending decisions are grouped by network and each
network runs a single forward pass over the rows it owns, so `--self-play-envs`
still buys throughput.  Checkpoints are written per network as
`latest_net<k>.pt` (and `iter_<it>_net<k>.pt`); evaluation reports a headline
line per network.  `--num-policies` and `--population` are mutually exclusive.
The worker is `CoevolutionSelfPlayWorker` in `training/self_play.py`.

Two practical notes.  The replay budget is split, so peak memory matches a
single-network run, but wall-clock per iteration is somewhat higher -- five
optimisers step each iteration and the batched forward pass is split into
per-network groups.  And `--resume` is not supported for a population (each
network has its own checkpoint); start co-evolution runs fresh.

### Heterogeneous populations (`--heterogeneous-population`)

By default the five members are **clones** -- same architecture, same objective,
different random inits -- so they co-adapt into near-identical policies and the
population's diversity is mostly wasted.  `--heterogeneous-population` instead
gives each member a distinct *archetype* from `POPULATION_ARCHETYPES`
(`training/population.py`), varying four axes:

* **capacity** -- a mix of `tiny` / `medium` / `large` trunks;
* **improvement sharpness** -- different `alpha`;
* **exploration** -- different `sampling_temperature`;
* **objective** -- `reward_mode` per member: `normalized_chip_return` (chip-EV,
  aggressive value-betting) vs `binary` (win-rate, tight/nitty).

Mechanically it stays trivial: a member is just its own `Config`, and
`build_population_configs` returns the base config `n` times (clones) or `n`
archetypes (heterogeneous).  Every member still shares the observation and action
layout, so they interoperate at one table; each acts with its own
`(alpha, beta, temperature, q_scale)` and its transitions are scored under its
own `reward_mode` (the pure `config.seat_reward` lets one hand be scored
differently per seat).  The default is off, so clone behaviour is unchanged.

```bash
python train.py --num-policies 5 --heterogeneous-population --checkpoint-dir checkpoints/het5
```

All members share the same network **recipe** (`model/`): per-group feature
embeddings (`Linear → LayerNorm → ReLU`) for cards / board / players / pot-history
/ position, a residual-MLP trunk with `LayerNorm` + `ReLU` throughout, and
**separate** policy and Q heads off the shared trunk.

### Evaluating each network

`benchmark.py evaluate` only knows the named benchmark configs, so a co-evolution
run's per-network checkpoints are scored with `eval_population.py`.  It loads
every `latest_net<k>.pt` in a directory and runs the standard battery from
`evaluation.evaluate` -- self-play, then the network in one seat against the
random, calling-station and tight-aggressive heuristic baselines in the other
two:

```bash
python eval_population.py --checkpoint-dir checkpoints/pop5 --hands 2000
```

It prints the full per-agent / per-seat table for each network, then a compact
`bb/100` comparison across the population:

```
=== bb/100 vs baselines (higher is better) ===
  net              vs_random    vs_calling_station          vs_heuristic
  0                    588.1                -364.7               -1167.7
  1                    978.7                 444.4                -285.2
  ...
```

`--which iter_000200` scores a specific snapshot instead of `latest`; `--hands`
sets the sample size (default 1000 -- raise it, since `bb/100` on independent
deals is high-variance), and `--seed` / `--device` behave as elsewhere.  Each
network is scored solo, so treat small net-to-net gaps with caution.

## Does population self-play beat the heuristic? (measured: no)

The population feature was built because the trajectory result -- strength vs a
fixed opponent oscillating with no convergence -- is the textbook motivation for
a population.  So the same trajectory experiment was run for the population
config (2 seeds, checkpoints every 100 iterations to 1000) and overlaid on plain
self-play.

![plain vs population](benchmark_results/population/plain_vs_population.png)

Mean strength across checkpoints (iteration >= 200, dropping the ragged startup):

| opponent | plain self-play | population |
|---|---:|---:|
| heuristic | −127 ±87 | **−232 ±89** |
| random | +560 ±145 | **+713 ±135** |
| calling station | +483 ±166 | +318 ±94 |

**Population did not solve the heuristic problem.** It is still below the
heuristic at every checkpoint of both seeds, and on average *worse* than plain
self-play (−232 vs −127).  Crucially, it also **did not damp the oscillation**:
the vs-heuristic standard deviation is unchanged (89 vs 87).  The whole premise
-- that facing a distribution of past selves would stabilise strength against a
fixed style -- is not borne out here.

### What population play *did* do

It is a **generalist**, and the trade is legible:

* **It resists the "un-learning" drift.**  Plain self-play slowly gets worse at
  beating random over training (the −0.67 correlation from the trajectory
  section); population stays stronger and steadier against random (+713 vs +560).
* **It is more stable but less exploitative against specific styles.**  Against
  the calling station its variance is far lower (std 94 vs 166) but its mean is
  lower too (+318 vs +483): it does not specialise as hard on punishing that
  bot's passivity.

So population play buys robustness against weak/varied opponents at the cost of
peak exploitation of specific ones -- but it does **not** close the gap to the
disciplined heuristic.

### Why not -- and what this does not yet isolate

The likely reason is that a **library of past selves is a style monoculture.**
Every snapshot descends from the same self-play lineage and shares the same
blind spot against the heuristic's equity-disciplined, fold-heavy style.
Diversity of *strength* (weak early snapshots to strong recent ones) is not
diversity of *style*.  To beat the heuristic the population would need a policy
that plays like it -- which only the heuristic itself provides, and seating it
would be training on the evaluation opponent.

Two honesty caveats on this experiment:

* **It is a bundled change.**  The population config alters three things at once
  -- library opponents, randomised stacks/blinds (15-200bb), and the
  later-street entry mixture -- so the −232 cannot be attributed to any one of
  them.  Isolating the library effect needs an ablation (population opponents at
  fixed 50bb, preflop-only), which has not been run.
* **The evaluation is out of the population's training distribution.**  It trains
  across stack depths and entry streets but is scored only at fixed 50bb
  complete hands.  A generalist judged at one depth may look worse than it is;
  the vs-random gain suggests the broader training is not purely a handicap.

The blunt summary: population self-play as specified made the agent a more
robust generalist but did **not** beat the heuristic, and did not stabilise the
matchup against it. The evidence continues to point at *style* diversity held
out from evaluation as the missing ingredient -- not more self-derived opponents.

## Research program: neural self-play vs. search (staged)

The system is being evolved through a sequence of independently-evaluable
experiments toward the question: *how far does a small model-free self-play
system get in 3-player poker, and how much does explicit search / game-theoretic
structure add?*  Stages: (1) baseline size comparison, (2) blueprint policies,
(3) depth-limited search, (4) belief-aware search, (5) local subgame solving,
(6) hybrid agent.  Each stage is switchable and measured before the next begins.

### Current architecture and training loop (baseline being evolved)

* **Environment** (`environment/`): 3-handed no-limit hold'em with per-hand
  blinds and stacks, EV all-in runouts, a configurable bet abstraction, and
  strict information hiding (leakage tests in `tests/test_information_leakage.py`).
* **Representation** (`representation/`): structured features -> a fixed vector,
  canonicalised to the acting seat (SELF / OPPONENT_LEFT / OPPONENT_RIGHT).
* **Model** (`model/`): per-group feature encoders -> a residual-MLP trunk ->
  a policy head and a per-action Q head.  Default is now the **medium** preset
  (256-wide, 8 blocks, ~1.6M params); `tiny` and `large` presets exist for
  controlled comparison.
* **Training** (`training/`): batched self-play fills a replay buffer; the
  trainer regresses Q(s, a_taken) onto a lambda-return and moves the policy
  toward the KL/entropy-regularised improvement of the current Q.  Opponent
  diversity via checkpoint leagues and full population self-play already exist.

### The biggest measured bottlenecks (not model capacity)

Prior experiments in this file establish, with confidence intervals, that the
ceiling is **not** representational capacity:

* **Capacity does not help.** A 250-iteration size sweep showed bigger trunks
  fit the critic better (lower q_loss) but play the same or worse.
* **More hands do not monotonically help.** Strength against a fixed opponent
  *oscillates* over training (correlation with iteration ~0.04) -- classic
  self-play non-stationarity, not a data ceiling.
* **Population self-play helps generalisation but not the heuristic gap.** It
  resists un-learning against weak opponents but stays ~230 bb/100 below the
  equity heuristic, because a library of past selves is a *style* monoculture.

So the likely bottlenecks are sample efficiency, imperfect-information
reasoning, and the absence of look-ahead -- which is why the program adds
*search*, not parameters.

### Stage 1: baseline size comparison (implemented, running)

Two things the research plan flagged were fixed first:

* **Replay ratio is now a first-class knob.** Training was doing ~12 new
  transitions per gradient update (each transition reused ~21x) -- a very high
  replay ratio that amplifies overfitting to stale self-play data.
  `transitions_per_update` (default 512) now drives the number of updates, so
  the ratio is ~1 and configurable.  Every run logs cumulative hands,
  transitions, updates and the realised replay ratio.
* **AdamW** replaces Adam, and **medium (256/8)** is the default baseline.

Stage 1 trains `tiny` / `medium` / `large` on the *same number of self-play
hands* (not equal gradient updates), self-play only, and compares strength over
hands against the three fixed baselines.  Configs: `stage1_tiny`,
`stage1_medium`, `stage1_large` in `benchmark.py`.

### Stage 1 results

3 sizes x 2 seeds, ~1.02M self-play hands each, self-play only, replay ratio ~1,
scored on 500 duplicate deals per checkpoint against the three baselines.

![size comparison](benchmark_results/stage1/size_comparison.png)

Final strength (mean over last 2 checkpoints x 2 seeds):

| size | params | vs heuristic | vs random | vs calling station |
|---|---:|---:|---:|---:|
| tiny | 0.34M | **−6 ±17** | +445 | +375 |
| medium | 1.6M | −110 ±159 | +533 | +777 |
| large | 7.0M | −18 ±60 | +572 | +373 |

Two findings, one of them a correction.

**1. Size does not matter, confirmed at 1M hands.** All three trunks converge to
roughly break-even against the heuristic; `tiny` is the tightest (−6 ±17) and
`large` is not better than `tiny`.  With hands and replay ratio held fixed, a
20x parameter range makes no difference to final strength.  This is the
controlled comparison the earlier capacity sweep only approximated, and it holds.

**2. More hands *do* help -- which revises an earlier claim.** Against the
heuristic every size climbs steadily over training:

| size | early (0.13-0.26M hands) | late (0.9-1.02M hands) |
|---|---:|---:|
| tiny | −393 | −6 |
| medium | −458 | −110 |
| large | −124 | −18 |

That is a **+350-390 bb/100 improvement over ~1M hands**, and the curve rises
across the whole range rather than plateauing.  Earlier in this file I concluded
"more hands do not help" from flat results between 16k and 64k hands.  **That
conclusion was confounded by the replay ratio.**  At ~21x reuse the network
overfit stale self-play data and could not benefit from more hands; at ~1x reuse
(fresh data) the same additional hands drive a large, monotone improvement to
break-even -- the first time anything in this project reached break-even against
a held-out opponent.

**Caveat -- it is a bundled change.** Versus the earlier baseline, three things
changed together: replay ratio (21 -> 1), hand count (64k -> 1M), and the
optimizer (Adam -> AdamW).  The learning curve (improvement accruing with hands,
not appearing instantly) points at the *data* as the driver, unlocked by the low
replay ratio, but a clean attribution needs an ablation -- low replay ratio at
64k hands, or high replay ratio at 1M -- which has not been run.  Per-seed
oscillation is also still visible (e.g. medium seed0 swings from −781 to +1 to
−383); the low replay ratio raised the *level* to break-even without eliminating
the non-stationarity.

**Where this leaves the program.** Stage 1 establishes a real, reproducible
break-even baseline against the heuristic with a 0.34M-parameter model -- and
confirms, at 1M hands, that scaling the network is not the lever.  That is
exactly the setup the plan wants before adding search: a small, well-trained
model-free agent whose remaining gap (still losing outright to the heuristic on
its bad-variance seeds, still oscillating) is plausibly a *reasoning* gap rather
than a capacity or data gap.  Stage 2 (blueprint policies) and Stage 3
(depth-limited search) follow.

## Plotting results

Three entry points, depending on what you have.

### 1. Replot a finished experiment (no run directories needed)

Both experiments are archived under `benchmark_results/`, and the archive is
self-sufficient: `report.json` holds the scores and `report_histories.json`
holds every seed's training curve. Run directories are large and routinely
deleted, so this is the path that still works months later.

```bash
python benchmark.py replot --report benchmark_results/alpha_sweep/report
python benchmark.py replot --report benchmark_results/ablation/report --out /tmp/ablation
```

Writes `<out>_results.png` (bar chart, whiskers = seed range) and
`<out>_curves.png` (training curves).

### 2. Regenerate tables and figures from run directories

```bash
python benchmark.py report --root benchmark_runs/ --out benchmark_runs/report
```

Produces `report.md`, `report.json`, `report_histories.json` and both figures.
Do this once while the run directories still exist; afterwards use `replot`.

### 3. Ad-hoc comparison of arbitrary runs

Any directory containing a `history.json` works -- including ordinary
`train.py` checkpoint directories, not just benchmark cells.

```bash
python -m training.plotting checkpoints/runA checkpoints/runB \
    --metrics entropy kl_target_vs_policy q_loss --window 9 --out compare.png

python -m training.plotting benchmark_runs/sweep_a0.0*__seed0 --no-plot   # table only
```

`--no-plot` prints just the table of final smoothed values, which is often all
you need:

```
run                            entropy        q_loss
----------------------------------------------------
sweep_a0.01__seed0              0.6119        0.0669
sweep_a0.05__seed0              1.2194        0.0572
```

### Reading the figures

* **Whiskers on the bar chart are the seed range, not the evaluation
  interval.** Seed variation is the larger of the two here (up to ~470 bb/100
  against the calling station), so it is the honest error bar.
* `q_loss` and `kl_target_vs_policy` are drawn on a **log axis**: both fall by
  one to two orders of magnitude, and a linear axis squashes the informative
  tail onto zero.
* The dashed line on the entropy panel is `log(mean legal actions)` -- the
  uniform-policy bound. It depends on the bet abstraction (~4.5 legal actions
  by default, ~5.4 with the finer grid), so `benchmark.py` sets it from the
  configuration rather than hardcoding it.

Policy **entropy** is the metric to watch, and it is informative in both
directions: near the uniform bound means the policy is ignoring its critic;
near zero means it has stopped exploring, and the `q_loss` panel will show the
critic suffering for it.

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
