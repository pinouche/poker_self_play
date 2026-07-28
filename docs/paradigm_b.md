# Paradigm B: game-theoretic solving (CFR, belief states, search)

[Paradigm A](paradigm_a.md) is model-free self-play RL. Its ceiling was
measured — the league's best agent beats a tight-aggressive heuristic by
+57 bb/100, but a best-responder beats *it* by ~+160 bb/100. Strong and
**exploitable**. Paradigm B goes after the other thing: a strategy a
knowledgeable adversary cannot beat, i.e. an approximate Nash equilibrium.
The north-star metric changes from "bb/100 against a bot" to
**exploitability** — how much the best possible counter-strategy wins — which
is a property of the strategy alone and is *computed exactly*, not estimated
by playing matches.

Following `docs/paradigm_b_roadmap.md`, this is built and validated on **Leduc hold'em**
before anything is pointed at hold'em, because Leduc is small enough that
exploitability can be computed exactly and its equilibrium value is known
(-0.085606 chips to the first player). Stages 0-3 of the roadmap are
implemented; stages 4 (heads-up no-limit) and 5 (three-player) are not.

## The result

Every row is the same agent-quality question — *what does the best possible
counter-strategy win against this?* — answered by exact best response over the
full 9,457-node Leduc tree. This is the output of `python solve.py benchmark`:

```
method                                     exploitability       value  seconds
full-depth CFR, 1000 iterations                   0.00138    -0.08560        4
full-depth range CFR, 1000 iterations             0.00048    -0.08560        3
full lookahead + safe re-solving                  0.00502    -0.08595        9
depth-limited, no value function                  0.32312    -0.07398        3
depth-limited, ReBeL value network                0.04656    -0.08741        7
exact equilibrium (literature)                    0.00000    -0.08561
```

Two results sit in there. **Continual re-solving works**: an agent that keeps no
strategy at all, only a belief state, and re-solves from scratch every time the
board lands — safely, honouring the counterfactual values its previous solve
promised — is 0.005 exploitable against 0.0005 for solving the whole game
offline. **And the depth limit is what costs**, not the network: cutting the
lookahead to one of Leduc's two betting rounds takes the same machinery from
0.005 to 0.047.

Search quality is bought with iterations, and the curve keeps going a long way
past where it looks flat. The same network, evaluated at different search
budgets:

| search iterations per decision | ReBeL network | exact leaf solving |
|---|---|---|
| 300 | 0.0585 | 0.0571 |
| 1,000 | 0.0466 | 0.0497 |
| 3,000 | 0.0269 | 0.0392 |
| 10,000 | **0.0164** | 0.0293 |

That second column is the striking one. The **learned value function beats
exactly solving the leaf subgames** — by 1.8x at 10,000 iterations, while being
about a thousand times cheaper to query. This is the ReBeL claim, and it is not
a compute artefact: throwing ten times more work at the exact solver (leaf
iterations 100 -> 1,000) moves it 0.0571 -> 0.0523, an 8% gain for 10x the time.
The network wins because it is *smooth*. Solving a subgame from cold returns
some equilibrium of it, and the per-hand split of value between equilibria is
not unique, so as the trunk's ranges shift the leaf values jump between splits
and the trunk is regret-matching against a jittery target. A network cannot
jump; it interpolates.

Solved-game numbers as a sanity check: full-depth CFR converges to a game value
of **-0.0856031** against the literature's **-0.0856064**, and the exact
best-response routine returns exploitability of **0.0** (to 1e-16) on the
analytically-known Kuhn equilibrium family.

## What is actually new here (versus paradigm A)

Three things paradigm A structurally could not do:

**Counterfactual regret.** Not "did this action lead to reward" but "what would
this action have been worth had I reached here, weighted by how often the
opponent lets me". Regret matching on that quantity provably converges to Nash
in two-player zero-sum games — the guarantee no amount of policy-gradient
self-play provides.

**Public belief states.** The state being reasoned over is not "my cards" but
the common-knowledge probability distribution over *both* players' hands given
the public betting. A raise multiplies your range by the raising frequency of
each hand and renormalises; the board card zeroes the hands it consumes. This is
the representation that lets an agent reason about *what its own bet revealed* —
paradigm A's observation encoder has no place to put such a thing.

**Search at decision time.** The policy is not a network output. It is the
result of running CFR, at play time, in the subgame the agent can currently see.

## Layout

```
paradigm_b/
  core/                  game-independent solver machinery
    game/                  explicit game trees: Kuhn, Leduc, the tree builder
    cfr/                   regret matching, CFR / CFR+ / linear / discounted,
                           exact best response and exploitability
    belief/                public tree, ranges, PBS, Bayesian propagation
    search/                range-form (vector) CFR, depth-limited subgames, the
                           re-solving gadget, continual re-solving, range BR
  leduc/                 stages 0-3, where every claim is proved exactly
    value_net/             PBS -> per-hand values: features, net, targets
    rebel/                 self play through belief space, buffer, training loop
  holdem/                stage 4: real cards, 1,326-combo ranges
    engine/                combos, strength, showdown, betting, public tree,
                           translation, suit isomorphism
    net/                   features, value net, policy net, leaf evaluators
    data/                  situation sampling, exact river labels, bootstrapping
    selfplay.py            ReBeL trajectory collection — used by both arms
    arm1_fixed/            frozen-label regime          -> arms.md
    arm2_iterative/        ReBeL Algorithm 1            -> arms.md
    arms_common/           what the two share
    compare/               the head-to-head and the drift probe
  cli/                   leduc.py, holdem.py, compare.py
solve.py                 a shim gathering all of `paradigm_b/cli`
```

```bash
python solve.py cfr --game leduc --iterations 1000     # solve it outright
python solve.py rebel --iterations 60 --checkpoint checkpoints/rebel.pt
python solve.py benchmark --checkpoint checkpoints/rebel.pt
python solve.py holdem --iterations 400                # real hold'em endgame
```

## Stage 0 — game tree, CFR, exact exploitability

`paradigm_b/core/game/` materialises a game into an explicit tree of chance / decision /
terminal nodes with integer information-set ids; Leduc comes out at the
canonical **288 information sets** and 9,457 nodes. Information sets are keyed
by card *rank*, which is a lossless abstraction (suits are strategically
irrelevant), while the tree still deals physical cards so card removal stays
exact.

`paradigm_b/core/cfr/` implements four variants over one traversal. Measured on Leduc after
1000 iterations:

| variant | exploitability |
|---|---|
| vanilla CFR (Zinkevich 2007) | 0.297 |
| CFR+ (Tammelin 2014) | 0.0131 |
| linear CFR | 0.0627 |
| discounted CFR (Brown & Sandholm 2019) | **0.0014** |

DCFR is the default everywhere downstream.

> **The bug that costs an order of magnitude.** The strategy profile must be
> frozen for the duration of a traversal. An information set is reached from
> many histories; recomputing regret matching after each one — which is what the
> obvious implementation does, and what most tutorial code does — means those
> histories are evaluated against *different* strategies and the accumulated
> quantity stops being a counterfactual regret. It still converges, at
> O(1/sqrt(T)) instead of O(1/T). On Kuhn that is 6.8e-3 versus 2.4e-4 after
> 1000 iterations, a 28x difference, from three lines of code.

## Stage 1 — public belief states and range propagation

The same equilibrium, computed a completely different way: one traversal of the
**public** tree (465 nodes instead of 9,457) carrying a 6-vector per player
instead of one hand at a time. Terminal values become a single matrix product
against the board's win/lose/tie matrix.

Card removal is exact and is where the arithmetic is easy to get wrong: the two
players hold *distinct* cards, so the joint distribution is not the product of
the marginals — the product spreads 1/36 over 36 ordered pairs where the truth
spreads 1/30 over the 30 legal ones. Hence a 6/5 correction wherever opponent
mass is summed, and a zero diagonal on every hand-vs-hand matrix.

The tests do not check this against itself. They check that:

* per-hand counterfactual values from one range traversal equal the per-hand
  expected values computed on the world tree, **to 1e-12**;
* Bayes-propagated beliefs equal posteriors obtained by brute-force enumeration
  of all 30 deals, through both action updates and the board deal, **to 1e-12**;
* the extracted policy is suit-symmetric to **exactly 0.0**;
* its exploitability, by exact best response, is < 0.006.

## Stage 2 — value network and depth-limited search

`paradigm_b/leduc/value_net/` predicts, for a public belief state, what every hand is worth to
both players. Two structural constraints are built into the module rather than
hoped for: values of impossible hands (the board card) are masked to exactly
zero, and the reach-weighted values of the two players are projected to sum to
zero (DeepStack's zero-sum layer). The head works in pot-sized units and is
scaled to chips inside `forward`, so a big pot does not dominate the loss.

Supervised against the exact solver on randomly sampled belief states: **R² =
0.974, MAE 0.25 chips** (6,000 samples, 130k parameters).

This stage produced the two findings that took the longest to track down.

### Finding 1: leaf values must come from per-iteration values, not the average strategy

Search fed the *value of the average strategy* at its depth limit was
**0.78 exploitable** — worse than using no value function at all. The same
search fed the *average of the per-iteration values* is **0.065**. A 12x
difference, and it is not a tuning artefact:

A reach-weighted average strategy is undefined for a hand with zero reach, so
it falls back to uniform — that is, to nonsense. Those are exactly the hands
whose counterfactual values the parent's regrets depend on: *"what would folding
this hand have cost me"* is a question about a hand you are currently never
continuing with. Per-iteration values come from regret-matched strategies, which
are well defined for every hand.

This is why ReBeL trains on `(sum_t v^t)/T` rather than on the value of the
average policy. The paper states it; the reason only becomes obvious when the
other choice is measured.

### Finding 2: re-solving from ranges alone is unsafe, and the gadget fixes it

Take the exact equilibrium (exploitability 0.0019) and change one thing: play
the second betting round by re-solving it from the ranges the equilibrium itself
produces, rather than from the equilibrium's own strategy. Exploitability
becomes **0.117** — 60x worse, from re-solving to an equilibrium at every step.

The reason is that the *total* value of a subgame is pinned down by the arriving
ranges, but the split of that total across individual hands is not. Different
equilibria of the same subgame divide it differently, and the strategy that led
there was chosen assuming one particular split. Deliver another and the opponent
profits by steering in with the hands you shortchanged.

`paradigm_b/core/search/subgame.py` implements the DeepStack / CFR-D **re-solving gadget**: the
opponent chooses, hand by hand, whether to play the subgame or to *opt out* and
collect the counterfactual value the previous solve promised that hand. Any
strategy we adopt must make opting out unattractive, which restores the missing
guarantee. Same experiment with the gadget: **0.013**, a 9x recovery, and a test
asserts directly that no hand ends up more than 0.05 chips better than promised.

## Stage 3 — the ReBeL self-play loop

`paradigm_b/leduc/rebel/` implements Algorithm 1 of Brown et al. 2020: at a belief state, run
depth-limited CFR with the current network at the leaves; record
`(belief state, root values)` as a training example; sample a **random CFR
iteration** and descend into a leaf reached by *that* iterate's policy; repeat.
Train the network on the buffer. The random-iteration detail matters — the
average policy is what converges, but the network has to be accurate at the
belief states every *iterate* visits, because those are what the next search
will query.

The fixed point is the point: a network that predicts the values of the policy
its own search produces. Feed search a value function describing some *other*
continuation — even an exactly-solved one — and the trunk optimises against
assumptions the continuation will not honour. That is why the trained network
(0.046) edges out exactly-solved leaves (0.051) despite being far less accurate:
it is consistent with what actually happens next.

Exploitability of the searching agent, measured by walking every reachable
belief state, re-solving at each, projecting onto the world tree and running
exact best response:

```
iteration  5   exploitability 0.088     iteration 35   exploitability 0.049
iteration 10   exploitability 0.046     iteration 40   exploitability 0.052
iteration 20   exploitability 0.063     iteration 50   exploitability 0.068
iteration 30   exploitability 0.071     iteration 60   exploitability 0.058
```

Most of the gain arrives in the first ten iterations (~14 minutes on a laptop
CPU), after which the curve looks flat.

**It is not flat — the evaluation was under-iterated.** These numbers all use
300 CFR iterations per decision. The same final network measures 0.047 at 1,000
iterations and **0.0164 at 10,000**, and at that budget it beats exactly solving
the leaf subgames (0.0293). The apparent plateau was a property of the
measurement, not of the agent; the default evaluation budget is now 1,000. The
lesson generalises: for a search-based agent, "how good is it" is not a single
number, and reporting one without the search budget attached is meaningless.

What the training curve does show is that the *network* saturates quickly on a
game this small — 15,000 belief states is plenty to learn six hands' worth of
values — while the *search* keeps converting extra iterations into strength.

## Stage 4 — real hold'em: the turn endgame

Leduc has six possible hands. Hold'em has 1,326, and the machinery had to earn
the jump rather than be told it. **Turn Endgame Hold'em** is ReBeL's own
published benchmark and the smallest thing that is genuinely hold'em: a real
52-card deck, both players holding two cards, betting on the turn, a river card,
betting again, showdown. Exploitability is still *exactly* computable, so
nothing here is asserted on faith.

What is real: the cards, all 1,326 combinations per player with no bucketing or
clustering, exact card removal, and the actual hand evaluator from paradigm A.
What is abstracted: the bet sizes (half pot, pot, all-in, one raise per round).
That abstraction is the honest gap between this and a bot you could sit down at
a table — a human can bet sizes it cannot represent — and closing it needs
action translation, not more search.

### The endgame solves

`python solve.py holdem --iterations 400`, on one CPU:

```
turn endgame on As Ks 7h 3h: 1928 public decision nodes, 1128 hands each, pot 20, stack 100
  iter    25  exploitability   3.4159 chips  best responses +1.1764 / +2.2396  (33s)
  iter    50  exploitability   1.0033 chips  best responses -0.1368 / +1.1401  (66s)
  iter   100  exploitability   0.3364 chips  best responses -0.4362 / +0.7726  (127s)
  iter   200  exploitability   0.1122 chips  best responses -0.5399 / +0.6521  (251s)
  iter   400  exploitability   0.0392 chips  best responses -0.5749 / +0.6142  (495s)
```

0.039 chips of exploitability in a 20-chip pot — two tenths of one percent. Note
the best-response values either side of it: even playing perfectly, the player
who acts first loses about 0.57 chips. That is **position**, priced by the
solver rather than assumed, and it is a useful sign the numbers mean what they
claim to.

### What "solvable" means here, and what it does not

The rows below include *solving the endgame outright*, which is possible because
an endgame is small: one fixed turn board, one fixed pot, one fixed stack, 1,928
public nodes.  That is the reason to use it as a benchmark — ground truth exists,
so every other method can be checked against it rather than believed.

It is emphatically **not** a claim that hold'em is solvable.  An endgame is
tractable *given* its board, its pot and its arriving ranges, and the full game
hands you the whole family: 270,725 turn boards, times every reachable pot and
stack, times every arriving range pair — and ranges are continuous, so that last
factor is not a count.  At eight minutes each, the boards alone are about four
single-core *years*, and that is still only for the uniform ranges used here; a
real agent reaches the turn with whatever preflop and flop betting shaped, which
is a different endgame every hand.

Generalising across that family is exactly, and only, what the value network is
for.  Which is why the interesting number below is not the reference solve — it
is whether the network can stand in for one.

### Search on the endgame, scored the same way

Every configuration below plays the endgame and is scored by the same exact best
response.  All four use **60 CFR iterations per decision**, so they are directly
comparable to each other — and *not* to the 400-iteration reference above, which
is why the plain solve is listed twice.

| method | exploitability (chips, pot 20) | seconds |
|---|---|---|
| solve the endgame outright, 400 iterations | **0.0392** | 495 |
| solve the endgame outright, 50 iterations | 1.0033 | 66 |
| full lookahead + safe re-solving, 60 iterations | 0.8618 | 212 |
| depth-limited + exact river solving, 60 iterations | 1.0389 | 358 |
| depth-limited + **ReBeL value network**, 60 iterations | 7.5975 | 146 |
| depth-limited + no value function, 60 iterations | 35.8233 | 147 |

Two things read straight off this.

**Depth-limited search with good river values is free.** Restricting the
lookahead to the turn and asking something else for the river costs nothing
measurable against searching both streets (1.04 against 0.86, with the plain
solve at the same budget sitting at 1.00).  That is the whole premise of
depth-limited solving, on real cards.

**The value network is the weak link, and it is a data problem.** It is worth
4.7x over having no value function — the qualitative claim survives the jump from
Leduc — but it is 7x behind exactly solving the river, which *inverts* the Leduc
result where the learned function beat exact leaf solving.  The reason is not
subtle: on Leduc the network fitted a twelve-dimensional output from ~15,000
belief states; here it has a **2,652-dimensional output** and self play on one
CPU produced a few thousand.  Trained loss corresponds to an RMSE of several
chips on values that reach a hundred.  This is the point in the roadmap where
the compute in the ReBeL paper — 90 machines, 720 GPUs — stops being a footnote
about scale and becomes the reason their network worked and this one is thin.

The honest summary of stage 4: the *machinery* transfers to hold'em intact and
is verified there; the *network* does not, for want of data, and no amount of
cleverness in the solver substitutes for that.

### What had to be built, and what did not

The solver did not change. `paradigm_b/core/search/` — vector CFR, the re-solving gadget,
continual re-solving, the range best response — runs on hold'em unmodified,
because everything game-specific was factored into a `HandSpace`: how many hands
exist, which the board blocks, what a terminal is worth, and the constant that
turns two independent ranges into a legal joint deal. Adding a game means
writing one of those.

Two pieces of hold'em-specific work were unavoidable:

**Blocking became the dominant cost.** In Leduc a hand blocked one card; here
every hand blocks 51 others, so "the opponent mass I can actually face" is an
inclusion-exclusion over both my cards rather than a subtraction.

**Showdowns had to stop being a matrix.** Written directly, a showdown is a
1,326 x 1,326 comparison — 1.7M operations at every showdown node of every
iteration. Sorting hands by strength once per board turns it into prefix sums,
with a per-card prefix table for the blocked part: **0.27 ms per node instead of
660 ms, a 2,400x speedup**, verified against the brute-force definition to 1e-11.
Without it this experiment would not run at all.

### Making it fast enough: profile before rewriting

The obvious next move was a C++ solver. Profiling said otherwise:

```
ncalls  tottime  function
 21312    2.003  numpy cumsum                  <- 55%
 10656    0.957  showdown_values               <- 81% cumulative
  5784    0.085  _decision_values (tree walk)  <- 2%
```

The tree walk — the part a C++ rewrite replaces — was **2%** of runtime. Four
fifths sat inside one numpy routine that was doing dense work on structurally
sparse data:

| routine | was | now | why |
|---|---|---|---|
| `showdown_values` | dense `(52, n)` prefix table, 56,212 entries | compacted to `2n` = 2,162 | a hand belongs to exactly 2 of the 52 rows; the other 50 are structurally zero |
| `card_masses` | `(52, 1326)` matrix product | two `bincount` scatter-adds | same reason — 26x the arithmetic for the same sums |

**1.10s → 0.270s per iteration, 4.1x, still pure numpy**, with the showdown
values still matching the brute-force definition to 1e-9. C++ would have bought
a fraction of that, because the hot path was already compiled. The lever ordering
worth remembering: *algorithm (4x) > batching (~2x) > language (~2-3x)*, and only
GPU batching changes the picture, at a scale where many subgames are solved at
once.

### Stage A — randomised situations, and testing on boards never seen

The first hold'em run had a defect worth stating plainly: every trajectory began
from the same root, so its first training example was **byte-identical every
time**. Of 9,600 "belief states", ~4,800 were one input repeated and ~4,800 were
distinct river states — all on a single board, which was also the board it was
tested on. No generalisation was being measured at all.

`paradigm_b/holdem/data/sampling.py` now draws every trajectory from a fresh situation the way
DeepStack does: random board, random pot, random stack, and ranges from a
mixture of shapes real betting produces —

| shape | what it imitates |
|---|---|
| uniform | nobody has shown anything |
| dirichlet | arbitrary noise, concentration randomised |
| tilted | a range that has bet (or, negative slope, one that has been capped) |
| capped | a player who has only called |
| sparse | a very narrow line |

and evaluation runs on **held-out boards excluded from training**, which is the
only question worth asking of a value network: an endgame solver answers one
board; the network's entire purpose is the 270,725 it cannot afford to solve.

### Stage C — bet abstraction and translation

`bet_fractions` is now a property of the betting state, so the abstraction is a
configuration rather than a constant: `(0.5, 1.0)` or
`(0.33, 0.5, 0.75, 1.0, 1.5)`, all-in always last.

More importantly, `paradigm_b/holdem/engine/translation.py` implements the **pseudo-harmonic
mapping** (Ganzfried & Sandholm 2013), without which the agent simply cannot be
handed a real opponent's action. An opponent who bets 63% of the pot into a tree
containing 50% and 100% has to be mapped somewhere:

```
abstraction sizes (pot fractions): 0.5, 1.0, all-in
  opponent raises to  10  ->  {half: 1.0}
  opponent raises to  15  ->  {half: 0.429, pot: 0.571}
  opponent raises to  20  ->  {pot: 1.0}
  opponent raises to  60  ->  {pot: 0.389, all-in: 0.611}
```

Two properties are load-bearing. The mapping is **randomised**, so an opponent
cannot learn which side of a boundary a size lands on. And it leans toward the
*larger* size rather than interpolating linearly — which is what removes the
classic attack of betting fractionally more than an abstraction size all game
and having it treated as that size.

Translation is lossy by construction: the agent answers a slightly different
question from the one it was asked, and that loss is a floor on exploitability
that no amount of search removes. The only cure is a finer abstraction.

### Stage B — three streets, and suit isomorphism

The betting state now carries `num_rounds`, so a tree can be rooted at the flop
and chain flop -> turn -> river, one depth-limited solve per street:

```
flop tree  : decisions  8  leaves 343   (turn belief states)
turn tree  : decisions  8  leaves 336   (river belief states)
river tree : decisions  8  leaves   0   (real showdowns)
board grows: As Ks 7h -> As Ks 7h 2c -> As Ks 7h 2c 2d
chance weight flop->turn 1/45   turn->river 1/44
```

Only the depth limit makes this affordable — expanded in full, a flop-rooted
tree branches 49 ways at the turn and 48 again at the river.

`paradigm_b/holdem/engine/isomorphism.py` collapses boards that differ only by renaming suits.
A♠K♠7♥ and A♥K♥7♦ are the same board, and there are **1,755 distinct flops
rather than 22,100** — verified by enumeration, with hand strengths and ranges
shown to permute consistently. That is a 12x reduction in everything downstream:
boards to solve, boards to cache, and positions the network must learn
separately.

**Where exactness runs out.** Exploitability is still exact for turn endgames,
but a flop-rooted agent would need re-solving at all 49 turn states and all
49x48 river states to be scored the same way — thousands of solves per
measurement. This is precisely the transition predicted earlier: past the
endgame scale, exact best response stops being affordable and LBR lower bounds
take over. The range-form traversal LBR needs is already in
`paradigm_b/core/search/best_response.py`; what stage B does not yet have is an exactly-scored
flop agent, and it cannot have one.

## What is not done

* **Stage 4 (heads-up no-limit hold'em).** The real lift. Needs an action
  abstraction with translation, a range representation over 1,326 hole-card
  combinations per player (bucketing or an embedding — the core modelling
  decision), and LBR for exploitability estimates once exact best response
  becomes intractable. `paradigm_b/core/search/best_response.py` already has the range-form
  traversal LBR is built from.
* **Stage 5 (three-player).** No algorithm has clean guarantees; the honest
  target is Pluribus-style "empirically beats strong opponents". An MCCFR
  blueprint is the missing piece.
* **Beating the paradigm-A agents.** Different game (Leduc vs 3-player no-limit
  hold'em), so the two paradigms are not yet comparable. That comparison arrives
  with stage 4, on heads-up.
* **Closing the last factor of ten under a depth limit.** With the whole game in
  view the agent reaches 0.005; restricted to one betting round of lookahead it
  reaches 0.016 at a large search budget. On Leduc that restriction throws away
  half the game, so the gap is expected — but whether it stays this size when the
  depth limit is a small fraction of the game (as in hold'em, where it is far
  less severe) is exactly what stage 4 measures.

## References

* **ReBeL** — Brown, Bakhtin, Lerer, Gong, NeurIPS 2020 — the template.
* **DeepStack** — Moravčík et al., *Science* 2017 — continual re-solving.
* **CFR-D / decomposition** — Burch, Johanson, Bowling, AAAI 2014 — the gadget.
* **CFR+** — Tammelin 2014. **Discounted CFR** — Brown & Sandholm 2019.
* **Leduc** — Southey et al., UAI 2005; game value -0.085606424078.
