# Poker: self-play RL, and search over belief states

Two ways to build a poker agent, in one repository, kept apart on purpose.

**Paradigm A** learns a policy that *beats* opponents: one shared network plays
every seat of three-handed no-limit hold'em, trained purely by self-play and
scaled up through a role-structured league. The metric is bb/100. It works, and
its ceiling has been measured — the best agent beats a tight-aggressive
heuristic by +57 bb/100 while a best-responder beats *it* by ~+160. Strong, and
decisively **exploitable**.

**Paradigm B** goes after the thing A cannot reach: a strategy a knowledgeable
adversary cannot beat. Counterfactual regret, public belief states, and
depth-limited search with a learned value network — the ReBeL recipe. The
metric changes to **exploitability**, and on Leduc hold'em it is computed
exactly rather than estimated. Stages 0–3 are built and validated; stage 4
(real hold'em endgames) is under way.

Inside paradigm B, one open question has its own experiment: **is a labelled
dataset a reusable asset, or do labels go stale?** Two arms run head to head at
equal budgets — *arm 1* labels the world once and freezes it, *arm 2* is ReBeL
Algorithm 2 and never freezes anything.

## Layout

```
paradigm_a/     (1) pure self-play RL              -> docs/paradigm_a.md
paradigm_b/
  core/             game trees, CFR, belief states, search
  leduc/            stages 0-3: the machine, proved correct on a toy game
  holdem/           stage 4: real cards, 1,326-combo ranges
    arm1_fixed/     (2) frozen dataset             -> docs/arms.md
    arm2_iterative/ (3) ReBeL Algorithm 2          -> docs/arms.md
    data/store.py   replay buffer, in memory or sharded onto disk
    data/augmentation.py  the suit and chip symmetries, applied on read
  cli/              leduc.py, holdem.py, compare.py
common/         cards, hand evaluation, and the neural primitives both share
docs/           the write-ups
tests/          mirrors the tree: paradigm_a/, paradigm_b/{core,leduc,holdem}/
train.py        shim -> paradigm_a/cli/train.py
solve.py        shim -> every paradigm_b/cli command
```

## Running it

```bash
# paradigm A
python train.py --iterations 2000 --league --network-size large
python -m paradigm_a.cli.evaluate --checkpoint checkpoints/latest.pt --hands 3000

# paradigm B: stages 0-3 on Leduc
python solve.py cfr --game leduc --iterations 1000
python solve.py rebel --iterations 60 --checkpoint checkpoints/rebel.pt
python solve.py benchmark --checkpoint checkpoints/rebel.pt

# paradigm B: stage 4, a real hold'em endgame
python solve.py holdem --iterations 400

# arm 1 vs arm 2, at equal label and update budgets
python solve.py compare --run runs/labels-01 --label-budget 20000
python solve.py label-drift --run runs/labels-01
```

```bash
pytest -m "not slow"        # the fast suite
pytest tests/paradigm_b     # one paradigm at a time
```

## The full-budget run

Arm 2 on its own, at ReBeL's own training budget and its 5x replay reuse. This
is the training run rather than the experiment; `solve.py compare` is the
experiment.

```bash
mkdir -p runs/rebel-full && nohup python run_arm2.py --run runs/rebel-full \
  --label-budget 896000000 --update-budget 4375000 --labels-per-update 204.8 \
  --trajectories-per-iteration 146 --updates-per-iteration 5 \
  --batch-size 1024 --learning-rate 3e-4 \
  --buffer-size 1000000 --purge-after-iterations 250000 --state-every 60000 \
  --actors 8 --learner-threads 1 --device mps \
  --hidden-dim 1536 --residual-blocks 6 --hours 168 \
  --progress-every-hours 5 --progress-slumbot-hands 1000 \
  --progress-slumbot-workers 4 --progress-boards 3 \
  > runs/rebel-full/log.txt 2>&1 &
```

**Where the budgets come from.** Appendix D trains for 1,750 epochs of 2,560,000
examples at batch 1024, so the paper presents 4.48e9 examples and takes
4,480,000,000 / 1024 = **4,375,000** gradient steps. Reuse is not stated; the
replay buffer (12M) and the 720 V100s devoted to generation both point at a few
presentations per label rather than tens, so this run targets **5x**, which
fixes the unique labels at 4.48e9 / 5 = **896,000,000** and
`--labels-per-update` at 1024 / 5 = **204.8**. That last flag is what actually
enforces the ratio: on the actor path the learner is throttled to the steps its
accepted labels entitle it to, so reuse cannot drift with process scheduling.
`--trajectories-per-iteration 146` against `--updates-per-iteration 5` matches
the same ratio (146 x 7 / 5 = 204.4 labels per update) so the budget-mismatch
warning stays quiet. `--dry-run` prints the plan and exits; it should read
`rebel_epochs 1750.0` and `reuse_per_label 5.0`.

**It does not finish. It takes 168 hours and 10 minutes**, because `--hours`
binds and the budgets are nowhere near reachable:

| | |
|---|---|
| generation rate (8 actors, mps, this net) | **~29-31 labels/s** |
| labels in 168 h | **~18M** of 896,000,000 → **2%** |
| updates (labels / 204.8) | **~87,000** of 4,375,000 → **2%** |
| the label budget alone would need | **~8,600 h**, i.e. about a year |
| after the deadline | final state write, then `evaluate_agent` with LBR — 6-10 min |

Whatever fraction the run completes keeps the ratios exactly, so `--hours` is
the only knob needed for a smaller version of the same run.

**Where the rate comes from, and one figure not to reuse.** The 12-hour run of
2026-08-01 delivered 1,269,597 labels in 12.01 h of wall clock: **29.36
labels/s**, which is what `run_arm2.py` prints when it finishes. An earlier
version of this section quoted ~87 labels/s, from
`labels / (generation_seconds + training_seconds)`. That denominator is two
*learner-side* timers — the drain wait and the gradient steps — and the actors
are separate processes that keep generating through anything the learner is
doing. The 66% of that run's wall clock lost to the quadratic journal manifest
(fixed since, and `--journal` is off by default) was not actor idle time: the
queue never filled and 37% of drains returned nothing. So removing it frees a
core rather than tripling generation. Two independent measurements agree with
29-36 and none with 87: steady-state mid-run windows at ~36 labels/s, and 15.2
labels/s at 8 actors under a much heavier update load.

**The one lever that is worth ~2x, and it is not a constant factor.** At 8
actors on MPS the run is GPU-bound, and a cheaper forward converts almost 1:1
into labels/s (the fused-forward change measured 1.13x on the forward and 1.14x
end to end). Forward throughput at batch 1024, measured paired and interleaved
in one process — the only protocol this machine's thermal drift permits:

| net | forward |
|---|---|
| `--hidden-dim 1536 --residual-blocks 6` (this run) | 1.00x |
| `--hidden-dim 1024 --residual-blocks 4` | **2.08x** |
| `--hidden-dim 768 --residual-blocks 3` | **2.66x** |

The trade is capacity, and at this data scale it is probably not a trade at all:
the 12-hour run's loss was still falling at 57,000 steps, so the network is
data-starved rather than capacity-starved, and a smaller one both generates and
trains about twice as fast. Everything else has been measured and is not a
lever — a batching server is 1.1x, bf16 is 1.07x (and fp16 overflows: values are
chips), batching trajectories is 0.62x, and the CFR regret arithmetic is ~2% of
a label. The one unexplored piece is the all-in river subtree, ~96 five-card
terminals re-walked every CFR iteration below a node with no decisions under it;
fusing its 48 rivers into a numba kernel attacks most of the 12% in
`terminal_values`, but CPU savings convert poorly at a GPU-bound operating point.

**The honest framing of the budget.** 896M labels is ReBeL's number divided by
its reuse, and ReBeL put 720 V100s behind generation. One M4 Max is not within
three orders of magnitude of that, and no constant-factor work closes it. The
question this run can answer is what ~18M labels are worth when they are spent
well — which is what `--labels-per-update`, read-time augmentation and the
network size are for, and none of those are throughput problems.

**The buffer, which is now reachable.** A row costs **11,037 bytes** — fp16
features, fp16 targets, and a 5-byte board that the 1,326-wide mask is rebuilt
from on the way out — against 27,368 for the float32-with-mask form it replaces.
So ReBeL's 12M-example buffer is **132 GB**, not 313: past this machine's 137 GB
of RAM once eight actors and torch are also resident, but nothing at all for
`--buffer-dir`, which puts the rows in append-only shards on disk and makes a
state write a manifest instead of a copy.

Whether 12M is *wanted* is a different question from whether it fits. A 168-hour
run generates ~18M labels, so the choice is really what fraction of the run stays
resident:

| `--buffer-size` | resident | turnovers in an 18M-label run |
|---|---|---|
| 1M | 11.0 GB | 17.8 |
| 2M | 22.1 GB | 8.9 |
| 4M | 44.1 GB | 4.5 |
| 12M | 132.4 GB (disk) | **1.5** |

At 12M almost nothing is ever evicted and the run trains to the end on labels a
much weaker network wrote; 2-4M keeps a staleness profile close to the 12-hour
run's and still fits in memory. Reuse per label does not depend on buffer size at
all — that is `--labels-per-update` — so a bigger buffer buys diversity within a
batch, not more training.

What still cannot be matched is *dispersion*: a label's 5 presentations fall
inside ~4,900 gradient steps here against the paper's ~58,600. Read-time
augmentation softens this — every presentation is a fresh suit relabelling and
chip scale rather than the same row five times — but the batches remain more
correlated than the paper's.

**Following it.** Two lines land in `progress.jsonl` and in the log every 5
hours of wall clock (`--progress-every-hours`, which exists because an iteration
is not a fixed amount of work — its rate moves several-fold with
`--labels-per-update`, so an iteration count is a poor proxy for a duration):

```
progress iter 91,000  labels 1,240,912  updates 6,059  expl 23.66+/-5.5 (f16.5 t30.8 r0.7)  lbr 45.3+/-9.9  n=3/street  slumbot -412+/-398 mbb/g over 1000 hands  [13.8 min]
```

(the `expl` and `lbr` halves are a real measurement, of an *untrained* network
at 3 boards/street — not a result)

`expl` is exploitability against an exact best responder on held-out boards,
`lbr` the off-abstraction responder from `arms_common/lbr.py`, and
`--progress-slumbot-hands` adds mbb/g against the real Slumbot server. Both
halves run on a background thread against a weight snapshot, so the learner
never waits for them. `python watch_arm2.py --run runs/rebel-full --follow`
shows the same line under the live spend.

**The two ± are not the same quantity, and neither is ReBeL's.** Table 1 of the
paper reports LBR as 881 ± 94 mbb/g: LBR *played hands* there, so the ± is
sampling error over the hands dealt. Nothing is dealt on this side —
`lbr_values` walks the whole tree against full 1,326-combo ranges and enumerates
every runout, so a situation's number is exact and repeating the evaluation
returns it bit for bit. What is uncertain here is *which held-out situations
were drawn*, and that spread is large, so the ± on `expl` and `lbr` is the
standard error over situations (`ddof=1`, streets added in quadrature for the
aggregate). It needs `--progress-boards` above 1 to exist and is printed only
when it does — a single board has a mean and no spread, and `+/-0.0` would read
as precision rather than as one sample. The Slumbot ± *is* the paper's kind:
Monte Carlo over dealt hands, and at 1,000 hands it is still ~±400 mbb/g.

Because the held-out boards are fixed by `--seed` for the whole run, successive
progress points attack the identical situations — the series is paired, so a
change between two points is a change in the network even where the ± is wide.
The ± says how far the *level* might be from the situation distribution's mean;
it does not blur the trend. Costs, measured on one thread at the full net size:
**0.8 min** per evaluation at 1 board/street, **3.6 min** at 3, plus ~10 min for
1,000 Slumbot hands over 4 workers. `--progress-boards 3` is the recommended
setting — it is what buys the ± at all, and ~14 min every 5 hours is under 5% of
four of the sixteen cores. Those same two measurements are themselves the
argument for it: the identical untrained network scored `expl 21.98` on one set
of held-out boards and `23.66 ± 5.5` on another.

**Restarting.** State (both nets, both optimisers, the buffer, the spend, the
RNG) is written every `--state-every` iterations and again on the way out, so a
run stopped by `--hours`, a crash or a reboot continues with:

```bash
python run_arm2.py --run runs/rebel-full --resume   # ... plus the same flags
```

Budgets are **totals**, so a resumed run finishes the remainder rather than
starting another one; `run_state` holds only the latest snapshot, swapped in by
rename. Repeat the flags: the network shape, the action abstraction and
`--buffer-size` are fingerprinted and a resume that disagrees is refused rather
than silently retrained, and the rest of the flags are simply not stored.
Checkpoints are weights only and cannot continue a run. `--state-every` counts
*iterations*, and an iteration is one drain of the actor queue rather than a
fixed amount of work — 217,174 of them in the 12-hour run, so ~5-10 a second and
`--state-every 60000` is a snapshot every two or three hours.

State is versioned, and the fp16 buffer rows bumped it to 3: a `run_state`
written before that holds float32-with-mask arrays and is refused rather than
misread. `runs/arm2-12h/run_state` is one of those and can no longer be resumed
from — its `student.pt` weights are still loadable.

## Documentation

| | |
|---|---|
| [`docs/paradigm_a.md`](docs/paradigm_a.md) | self-play RL: design, rewards, the league, every measured result, and the limitations |
| [`docs/paradigm_b.md`](docs/paradigm_b.md) | CFR, belief states and search: what was built, and the numbers stages 0–4 hit |
| [`docs/paradigm_b_roadmap.md`](docs/paradigm_b_roadmap.md) | the staged plan, the honest target, and the three-player problem |
| [`docs/arms.md`](docs/arms.md) | frozen labels vs refreshed labels: what is held equal, and how to read the result |

## What is shared, and what is not

The two paradigms share exactly one package, `common/`: card primitives, hand
evaluation, and the neural primitives (`CardEmbedding`, `ReBeLMLP`) that both
networks are built from. That list is short on purpose — anything that drifts
into being used by one side alone belongs back in that side's tree.
