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
python evaluate.py --checkpoint checkpoints/latest.pt --hands 600
python infer.py --checkpoint checkpoints/latest.pt   # or no checkpoint for a random net
python -m pytest tests/ -q           # 149 tests
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

### Reward: binary by default

`reward_mode = "binary"`, defined as **`sign(net chips for the hand)`**:

| outcome | reward |
|---|---|
| finished the hand up chips | +1 |
| finished the hand down chips | −1 |
| finished exactly even | 0 |

This gives the specification's win/lose/tie semantics while handling folds,
side pots and split pots without special cases. One consequence worth stating:
**a player who folds the button preflop, having invested nothing, scores 0, not
−1.** That is the economically correct signal — they risked nothing and lost
nothing — and it avoids punishing a correct fold.

`chip_return` and `normalized_chip_return` are also implemented. Selecting
`chip_return` automatically disables the tanh on the Q head, since raw chip
returns are unbounded.

Binary reward has a real cost: it maximises win *rate* rather than chip EV, and
the resulting policy learns to almost never fold. This is measured and
explained under [Results](#results-binary-reward-optimises-the-wrong-objective).

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

**bb/100 is extremely noisy.** With 50bb stacks, a 300-hand evaluation can
easily swing several hundred bb/100 — in one run the same checkpoint measured
−500 bb/100 against a calling station at 300 hands and +69 bb/100 at 3000.
Use thousands of hands before believing a difference.

---

## Results: binary reward optimises the wrong objective

Two 250-iteration runs (~16k self-play hands each, identical apart from
`reward_mode`), evaluated over 3000 hands with seat rotation:

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

The default remains `binary` as specified, and it works — it beats every
baseline and is the right choice if you want a clean, bounded, easily-learned
signal. **For actual poker strength, use `--reward-mode normalized_chip_return`.**

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
* The default binary reward maximises win rate, not chip EV — see
  [Results](#results-binary-reward-optimises-the-wrong-objective).
* The trained agent is not a strong poker player. ~16k hands of self-play with
  a 390k-parameter network beats random and calling-station baselines
  comfortably and still loses chips to the equity-based heuristic bot. Getting
  further needs orders of magnitude more hands, batched self-play, and a finer
  bet abstraction — not a different architecture.
