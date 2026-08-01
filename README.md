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
  --progress-slumbot-workers 4 --progress-boards 1 \
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

**It does not finish.** Measured generation is ~87 labels/s (the 12-hour run of
2026-08-01, counting its gen+train seconds; its wall clock was inflated by the
journal, which is off by default now), and this config trains 9x less per label
so generation gets more of the machine — call it 87-120 labels/s. 896M labels is
**2,000-3,100 hours**, i.e. 85-130 days. The learner is not the constraint:
4.375M steps at 15.2/s is 80 hours. Generation is ~30x the bottleneck, which is
what the paper's 720 V100s were buying. Whatever fraction the run completes
keeps the ratios exactly, so shortening `--hours` is the only knob needed for a
smaller version of the same run.

**The one thing that cannot be matched.** ReBeL's 12M-example buffer would be
313 GB here (26.7 KB per example, measured). At 1M the buffer is 27.4 GB
resident — with a transient ~11 GB more at each state write, since
`_save_circular` re-orders each field before writing — and a label's 5
presentations fall inside ~4,900 gradient steps instead of the paper's ~58,600.
Same reuse rate, ~12x more correlated batches.

**Following it.** Two lines land in `progress.jsonl` and in the log every 5
hours of wall clock (`--progress-every-hours`, which exists because an iteration
is not a fixed amount of work — its rate moves several-fold with
`--labels-per-update`, so an iteration count is a poor proxy for a duration):

```
progress iter 1,626  labels 6,809  updates 33  expl 17.39 (f14.4 t20.4 r14.6)  lbr 48.6  [0.6 min]
```

`expl` is exploitability against a best responder on held-out boards, `lbr` the
off-abstraction responder, and with `--progress-slumbot-hands` the line carries
mbb/g against Slumbot with its standard error — which at 1,000 hands is still
~+/-400 mbb/g, so read the series and not any single point. Both halves run on a
background thread against a snapshot, so the learner never waits for them; each
evaluation costs ~15-20 min of four threads. `python watch_arm2.py --run
runs/rebel-full --follow` shows the same line under the live spend.

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
Checkpoints are weights only and cannot continue a run.

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
