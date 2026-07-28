# Frozen labels vs refreshed labels

Two ways to feed a value network, run head to head at equal budgets.

## The question

Both arms bootstrap — search with a learned network at the leaves *is*
bootstrapping, so "bootstrapped data vs ReBeL" is not the distinction. The
difference is:

> **When were the labels made, and by which network?**

**Arm 1 (`fixed/`)** labels the world once, in layers, lowest street first.
The river is solved exactly to real showdowns. A teacher is fitted on it. The
turn is then a depth-limited solve whose river leaves are priced by *that
teacher*, and the flop the same trick one street higher. Those labels are
written to disk and frozen. A **fresh** student is then trained on the file
alone — no solver, no teacher. Arm 1's turn and flop labels are permanently the
opinion of one early network.

**Arm 2 (`iterative/`)** is ReBeL Algorithm 1. Sample a situation, search with
the **current** network at the leaves, record what search concluded, take
gradient steps, repeat. Labels improve as the network does; stale ones age out
of the replay buffer. Nothing is ever frozen.

## Files

```
common/         only what both arms share — sharing it is what makes it fair
  fitting.py      the one gradient loop both arms use
  budget.py       SpendRecord + the leaf-evaluation counter
  evaluation.py   held-out situations, exploitability, the resolver/tree bound
  situations.py   StreetMix: drawing flop/turn/river in fixed proportions
  storage.py      the run directory layout, atomic writes
fixed/          arm 1
  store.py        the on-disk artifact: sharded, memory-mapped, source-separated
  build.py        generating it street by street, saving each teacher
  student.py      training a fresh student from it, solver never invoked
iterative/      arm 2
  journal.py      append-only record of every label the loop produced
  student.py      Algorithm 1, spending its budgets exactly
experiment.py   the runner: holds seeds, boards and budgets equal
relabel.py      the staleness probe
```

## Running it

```bash
python solve.py compare --run runs/labels-01 \
    --label-budget 12000 --update-budget 6000 \
    --river-examples 8000 --turn-examples 3000 --flop-examples 1000 \
    --teacher-updates 3000 \
    --hidden-dim 512 --residual-blocks 3 --card-embedding-dim 64 \
    --trajectories-per-iteration 24 --updates-per-iteration 40 \
    --eval-boards 2 --eval-iterations 40 --flop-depth-limit 1 \
    --workers 8 --seed 0

python solve.py label-drift --run runs/labels-01 --sample-size 64
```

The artifact is reused across runs, so a second `compare` with a different seed
or budget costs nothing extra to prepare; pass `--rebuild-dataset` to force a
regenerate.

### `compare` flags

| flag | default | what it controls |
|---|---|---|
| `--run` | *required* | run directory; everything is written under it |
| `--label-budget` | 20000 | labelled examples **each** arm may consume |
| `--update-budget` | 4000 | gradient steps **each** arm may take |
| `--river-examples` | 20000 | exact river labels in the artifact |
| `--turn-examples` | 6000 | bootstrapped turn labels |
| `--flop-examples` | 2000 | bootstrapped flop labels |
| `--self-play-examples` | 0 | frozen on-policy control source (see confound below) |
| `--teacher-updates` | 4000 | gradient steps per teacher fit; sets label quality above the river |
| `--trajectories-per-iteration` | 16 | arm 2's generation rate |
| `--updates-per-iteration` | 40 | arm 2's training rate |
| `--batch-size` | 128 | both arms and the teachers |
| `--learning-rate` | 1e-3 | both arms and the teachers |
| `--hidden-dim` / `--residual-blocks` / `--card-embedding-dim` | 1536 / 6 / 128 | value-net size, shared by both arms |
| `--eval-boards` | 2 | held-out boards **per street** |
| `--eval-iterations` | 40 | CFR iterations in the scoring agent's search |
| `--flop-depth-limit` | 1 | flop lookahead when scoring; 2 is ~170x the work |
| `--eval-every` | off | mid-run scoring cadence, for a learning curve |
| `--checkpoint-every` | 10 | arm 2 student snapshots |
| `--rebuild-dataset` | off | regenerate the artifact instead of reusing it |
| `--workers` | 8 | parallelism for river generation only |
| `--seed` / `--device` | 0 / cpu | |

**Sizing note.** `label-budget / update-budget` should roughly match
`trajectories-per-iteration × labels-per-trajectory / updates-per-iteration`, or
arm 2 exhausts one budget long before the other and spends the tail either
generating labels it never trains on, or training on a frozen buffer. A
trajectory yields one label per street it passes through (~2–3 from the turn).

### `label-drift` flags

| flag | default | what it controls |
|---|---|---|
| `--run` | *required* | an existing run directory |
| `--checkpoint` | arm 2's student | the evaluator to re-solve with; its size is read from the weights |
| `--sample-size` | 64 | belief states re-solved per source |
| `--cfr-iterations` | per-source, from the manifest | override; see the warning below |

## What is held equal

| | |
|---|---|
| starting weights | one seeded net, cloned into both students; `weights_identical` proves it |
| held-out boards | built once, excluded from both arms' data, used for both scores |
| labels | both consume exactly `label_budget` |
| gradient updates | both take exactly `update_budget` |
| optimiser loop | literally the same function, so only the data differs |

Board exclusion is **subset-aware**: a held-out flop `(2,7,30)` also excludes
the turn `(2,7,30,45)`. Exact-tuple matching would leak the test board's
texture into training. Note this is a set relation, not a prefix one — sorting
does not make a flop a prefix of its own turn.

## How many teachers?

**Two**, by default — not one per street. Each teacher exists to price the
leaves of the street above it:

| teacher | fitted on | prices the leaves of |
|---|---|---|
| `teacher-after-river.pt` | river | the **turn** solve |
| `teacher-after-turn.pt` | river + turn | the **flop** solve |

The flop is the top street, so nothing is bootstrapped from it and no teacher
is fitted after it. The exception is `--self-play-examples > 0`: that source is
generated last and needs an evaluator, which adds `teacher-after-flop.pt`.

Note the teacher is **one network trained cumulatively**, not a fresh network
per street — `teacher-after-turn` is `teacher-after-river` trained further on
river *and* turn data. And note that the teachers are kept rather than
discarded: they are the only record of *what produced each label*, without
which a frozen turn label is an unattributable number.

## Reading the result

The metric is **exploitability on held-out boards**, not loss. Loss is measured
against each arm's own labels, so it is not comparable across arms — and Arm 1
can post the *lower* loss precisely because its frozen labels are stale and
self-consistent, which is the failure mode being hunted.

Per-street shape is the finding, not the aggregate:

| street | if labels go stale | if staleness doesn't matter |
|---|---|---|
| river | tie — both solve it exactly | tie |
| turn | small gap to iterative | tie |
| flop | largest gap (two teacher layers) | tie |

A flat difference across all three streets means something *other* than label
staleness is driving it — most likely the confound below.

### Neither arm is a floor for the other

It is tempting to assume the frozen arm can at best tie — that refreshing
labels is strictly more information, so Arm 1 is a lower bound on Arm 2. That
is **not** true, and the experiment is only worth running because it isn't.

Where Arm 1 can genuinely win:

* **Front-loaded label quality.** Arm 1's turn labels are written by a teacher
  already fitted on the *entire* river dataset. Arm 2's first labels are
  produced by a near-random network, and at equal label budget a real fraction
  of everything it ever sees is that early garbage.
* **Coverage beats relevance on this metric.** Arm 1 sweeps a designed spread
  of boards and range shapes. Arm 2 sees only what its own search reaches,
  which early on is a narrow, self-reinforcing slice chosen by a bad network.
* **Label reuse and noise averaging.** At equal update budget Arm 1 does many
  epochs over a clean fixed set; Arm 2 sees noisier labels fewer times each.
* **A stationary target.** Arm 2 regresses onto a target that moves underneath
  it. Arm 1's does not.

Where Arm 2 wins: only it can approach the fixed point (network ≈ value of
search using that network), and Arm 1 permanently inherits its teacher's bias,
compounding upward river → turn → flop.

So expect a **crossover, not an ordering**: Arm 1 ahead at small budgets, Arm 2
ahead at large ones. Locating that crossover is arguably the most useful thing
these runs produce.

This also bounds what the drift probe can tell you. Drift ≈ 0 means the frozen
labels are already what a stronger evaluator would write, so refreshing buys
nothing and the arms should **tie on label quality** — at which point Arm 1's
coverage and reuse advantages could win it the run outright. Large drift means
headroom exists for Arm 2, but it still has to convert that headroom within
budget while paying the coverage cost. The probe answers the *label-staleness*
question, not the *which-arm-wins* question.

## Three things that will bite you

**The confound.** River/turn/flop are sampled from a distribution somebody
designed; Arm 2's data comes from belief states its own search reached. So the
arms differ in *two* ways at once — stale labels **and** input distribution —
and a win for either cannot be attributed. Set `self_play_examples > 0` to
freeze a copy of the on-policy distribution into the artifact; that third
configuration holds the distribution fixed and varies only label freshness.

**The held-out set is drawn from the designed distribution.** Evaluation
situations come from `SituationConfig` — the same sampler Arm 1's data comes
from. Boards are held out, but the *distribution* is Arm 1's home turf, and
Arm 2 is partly being scored on spots its own search would rarely visit. This
is a defensible choice (it asks "what is a board you have not seen worth?",
which is the value net's actual job) but it is a thumb on the scale, and a
result close to a tie should be read with it in mind. Neither arm is
structurally guaranteed to win: expect Arm 1 ahead at small budgets, where its
labels are made by an already-fitted teacher while Arm 2's first labels come
from a near-random network, and Arm 2 ahead at large ones, where only it can
approach the fixed point.

**Cost is not equal just because labels are.** An exact river label comes out
of one batched solve shared across 64 situations; a self-play label costs a
solve of its own plus a network call per leaf. Hence `leaf_evaluations` and
`solver_calls` alongside the budgets. Wall clock is reported but is not the
headline — it is machine-dependent, and the artifact's build cost is meant to
be amortised (`amortisation_break_even` says over how many students).

## The probe (`relabel.py`)

The training comparison says which student won, not why. The probe asks the
question directly: take the artifact's **inputs**, re-solve them with a
stronger network, and measure how far the labels moved.

**Re-solve at the count the labels were generated with.** "Exact" means exact
terminal values, not a converged strategy — DCFR at 3 iterations and at 6 gives
measurably different root values on the same river spot. The probe reads the
count from the manifest per source by default. Relabelling the river at a
different budget reports solver convergence as drift, and the river is supposed
to be the control: it has no leaves, so the network is never consulted and the
stored label must reproduce. A non-zero river drift means the pipeline is
inconsistent and no other number can be trusted.

**It is a lower bound on staleness.** Drift is measured against whatever
evaluator you hand it. Hand it a weak network and the drift is understated; the
number is only meaningful if that network really is stronger than the teacher
that wrote the labels. Passing the winning arm's finished student is the
default for exactly that reason, but it is an assumption, not a guarantee.

## Evaluation cost

Unbounded continual re-solving from a flop root is ~11,000 solves, because each
flop belief state fans out to ~49 turns and each of those to ~48 rivers. The
turn is 241 solves and is left exact; the river is one. So the flop is bounded,
and **the agent and its scorer are bounded to the same depth** — bound only the
tree and the agent wastes effort on nodes nobody scores; bound only the
resolver and the scorer finds decision nodes filled with uniform play and
reports the resulting disaster as the network's exploitability.

| flop `depth_limit` | tree | leaf network evaluations |
|---|---|---|
| 1 (default) | 364 nodes | 343 |
| 2 | 92,288 nodes | 58,800 |
