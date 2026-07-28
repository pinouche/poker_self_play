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
Algorithm 1 and never freezes anything.

## Layout

```
paradigm_a/     (1) pure self-play RL              -> docs/paradigm_a.md
paradigm_b/
  core/             game trees, CFR, belief states, search
  leduc/            stages 0-3: the machine, proved correct on a toy game
  holdem/           stage 4: real cards, 1,326-combo ranges
    arm1_fixed/     (2) frozen dataset             -> docs/arms.md
    arm2_iterative/ (3) ReBeL Algorithm 1          -> docs/arms.md
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
