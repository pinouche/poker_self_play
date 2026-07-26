# Paradigm B: game-theoretic solving (search over belief states)

Paradigm A (self-play RL + league) is done. Its ceiling was measured: the
strongest agent beats a fixed tight-aggressive heuristic by +57 bb/100 but a
best-responder beats *it* by ~+160 bb/100 — a strong **exploitative** bot,
decisively far from unexploitable. The best-responder-vs-champion-vs-heuristic
result even formed a clean rock-paper-scissors cycle: the strategy space is
non-transitive, so no single policy dominates and optimising against one
opponent just moves you around the cycle.

Paradigm B targets the other thing: a strategy a knowledgeable adversary
**cannot** beat — an approximate **Nash equilibrium**. This needs three things
paradigm A never had: **counterfactual/regret reasoning**, **belief states
(ranges)**, and **search at decision time**. The north-star metric changes from
"bb/100 vs a bot" to **exploitability (distance to Nash)**.

The template is **ReBeL** (Brown et al., 2020) — the imperfect-information
analogue of AlphaZero: depth-limited search over *public belief states* with a
learned value network, trained by self-play. **DeepStack** (2017) is the
precursor; **Libratus** (2017) the CFR+subgame-solving cousin; **Pluribus**
(2019) the multiplayer (6-max) empirical result.

---

## The honest target, and the 3-player problem

Everything with a convergence guarantee — CFR → Nash, ReBeL → Nash, "Nash =
unexploitable" — holds **only in two-player zero-sum (2p0s)**. Our game is
**3-player**, where:

- Nash is not unique and not even the right target (a 3-player Nash can still be
  jointly exploited; deviations by one player are not zero-sum against another).
- No algorithm has clean guarantees. The state of the art is **Pluribus**:
  an MCCFR blueprint + depth-limited search, aiming for *"empirically beats
  strong opponents,"* not *"solved."*

**Recommendation: build and validate the whole machine on 2p0s first** (where
exploitability is well-defined and drives toward 0), then extend to 3-player
empirically. Do not start on the 3-player no-limit game — you will have no metric
that tells you whether you are making progress.

---

## The three pillars (what to build)

1. **Counterfactual regret (CFR).** Track, per information set and action, the
   *counterfactual regret*; the time-average strategy provably converges to Nash
   in 2p0s. Variants: vanilla CFR, CFR+ (faster), MCCFR (sampled, scalable),
   Deep CFR (regrets via a network, no card abstraction).

2. **Public belief states (PBS) and ranges.** The "state" you reason over is not
   your cards — it is the **common-knowledge probability distribution over every
   player's private hand**, given the public betting/board. Actions update ranges
   by Bayes' rule (betting narrows the range; a check-heavy line widens it). This
   is the representation paradigm A completely lacks and the reason it can't
   reason about *"what my bet reveals about my hand."*

3. **Depth-limited search + a value network.** At decision time, run CFR on the
   current subgame down to a depth limit, and evaluate leaf PBSs with a learned
   **counterfactual value network** V(PBS) → per-hand values. This is
   "continual re-solving" (DeepStack) / the search step of ReBeL. It is where
   most of the strength lives — a raw policy net without search is far weaker.

---

## Reuse vs build

**Reuse from this repo:**

- `environment/` — the rules engine (betting, side pots, showdown), the hand
  evaluator, and especially the **all-in EV runout** (already computes expected
  value over remaining boards — exactly the low-variance leaf evaluation CFR
  wants). Keep it as the game engine *under* the tree.
- `model/residual_blocks.py`, `model/heads.py` — the `FeatureGroupEncoder` +
  `ResidualTrunk` building blocks transfer to the value net (different I/O).
- The Monte-Carlo equity code — useful for range/value computations.
- Config, checkpointing, the run/monitor harness.
- **Test suite:** the saved paradigm-A league agent + `exploiter.pt` are a ready
  held-out battery — "does B beat A, and is B less exploitable than A?"

**Build new (roughly one module each):**

- `game/` — explicit **game-tree / information-set** representation, starting
  with a toy game (Leduc). Enumerable infosets, chance nodes, terminal payoffs.
- `cfr/` — regret matching, CFR / CFR+ / MCCFR, average-strategy tracking, and an
  **exact best-response / exploitability** routine.
- `belief/` — PBS representation and **Bayesian range propagation** through
  actions and chance.
- `search/` — depth-limited re-solving over a subgame given a PBS.
- `value_net/` — V(PBS) → counterfactual values per hand (reuse the trunk;
  inputs = pot, board, both range vectors; outputs = per-hand values / a value
  for each bucket).
- `rebel/` — the ReBeL self-play+search training loop.
- `exploitability/` (real one) — **exact** BR on small games; **Local Best
  Response (LBR)** as a cheap lower bound on large games. This *replaces* the
  learned-best-responder hack used to probe paradigm A.

---

## Staged roadmap (each stage has a metric)

| # | Milestone | Validation metric |
|---|---|---|
| **0** | **Toy game + tabular CFR.** Implement **Kuhn**, then **Leduc** hold'em as an enumerable tree. Vanilla CFR / CFR+. Exact best-response. | CFR **exploitability → 0** on Leduc, matching known values. This proves the regret machinery before any nets or hold'em complexity. |
| **1** | **Public belief states + range propagation** on Leduc. | Range updates are Bayes-consistent (sum to 1, match tree-computed posteriors). |
| **2** | **Counterfactual value network** on Leduc PBSs; depth-limited re-solving that calls the net at the leaf. | The net regresses CFR values; **depth-limited re-solve stays low-exploitability** vs full CFR. (This is DeepStack, in miniature.) |
| **3** | **ReBeL self-play loop** on Leduc: at each PBS run depth-limited CFR with the value net, sample a leaf PBS, recurse; train the value net on the CFR values. | **Exploitability → near 0**; reproduces the ReBeL/DeepStack Leduc numbers. Now the full machine is validated on a game where you can *prove* it works. |
| **4** | **Scale to heads-up no-limit hold'em (HUNL).** Neural value net over ranges (ReBeL avoids explicit card abstraction), finer **action abstraction** + translation, bigger nets. | **Low LBR exploitability**, and it **decisively beats the paradigm-A league agent** and `exploiter.pt`. This is the DeepStack/Libratus/ReBeL tier — a large lift. |
| **5** | **Extend to 3-player** (the original game). Pluribus recipe: **MCCFR blueprint** (abstracted, offline) + **depth-limited search** at play. | Empirically beats the league agents (and humans if available); approximate exploitability (LBR) is low. **No guarantees** — this is the research frontier. |

Stages 0–3 are the tractable, validate-the-idea core and are worth doing even if
you stop there — they are where the concepts become real. Stage 4 is a serious
engineering project. Stage 5 is open-ended research.

---

## Concrete first step

Build **Stage 0**: a Leduc hold'em tree + tabular CFR+ + an exact-exploitability
routine, with a test that asserts exploitability drops below a threshold after N
iterations (Leduc's game value and equilibrium are known). It is self-contained
(~a few hundred lines), needs none of the neural machinery, and is the
foundation every later stage builds on. Ship it, watch exploitability → 0, then
add belief states (Stage 1).

A useful intermediate before full ReBeL, if you want an earlier hold'em win:
**river-subgame re-solving** on top of the paradigm-A blueprint (the river
subgame is small enough to solve near-exactly given ranges). Libratus-style
nested solving at the last street alone measurably cuts exploitability, and it
exercises the belief/range and search code you'll need anyway.

---

## Metric discipline (the thing paradigm A lacked)

- **Small games:** compute **exact** best-response value → exact exploitability.
  This is the ground truth that tells you the implementation is correct.
- **Hold'em:** exact BR is intractable; use **Local Best Response (LBR)** — a
  shallow, cheap best-responder that gives a *lower bound* on exploitability.
  (This is how DeepStack was shown hard to exploit; it is the right tool, unlike
  the from-scratch/warm-start learned BR hacked in for paradigm A, which was
  either too weak or unstable.)
- Keep reporting bb/100 vs the league agents and the heuristic **as secondary**
  sanity checks, but exploitability is the north star.

---

## Key references

- **ReBeL** — Brown, Bakhtin, Lerer, Gong, *Combining Deep RL and Search for
  Imperfect-Information Games*, NeurIPS 2020. (The template; PBS + value net +
  search + self-play; Nash-convergent in 2p0s.)
- **DeepStack** — Moravčík et al., *Science* 2017. (Continual re-solving + deep
  counterfactual value networks; the precursor.)
- **Libratus** — Brown & Sandholm, *Science* 2017. (MCCFR blueprint + nested safe
  subgame solving; beat HUNL pros.)
- **Pluribus** — Brown & Sandholm, *Science* 2019. (6-max; blueprint + depth-
  limited search; the multiplayer recipe for Stage 5.)
- **Deep CFR** — Brown et al., ICML 2019. (CFR with function approximation, no
  abstraction; a stepping stone for the value/regret nets.)
- **CFR / CFR+** — Zinkevich et al. 2007; Tammelin 2014.
- **Local Best Response** — Lisý & Bowling 2017. (Cheap exploitability lower
  bound for large games.)

---

## Risks / open questions

- **3-player has no clean theory.** Budget for Stage 5 being empirical and
  fiddly; the honest ceiling there is Pluribus-style "beats strong opponents,"
  not "solved."
- **Search is the expensive part** at play time (real-time re-solving). Depth
  limit, iteration count, and abstraction granularity are the main knobs.
- **Value-net input design** (how to encode a range over 1326 hole-card combos,
  bucketed or embedded) is the core modelling decision at Stage 4.
- This is a substantially larger effort than paradigm A — DeepStack/ReBeL were
  multi-person research projects. Validating on Leduc (Stages 0–3) first is what
  keeps it tractable and honest.
