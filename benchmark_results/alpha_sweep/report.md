# Benchmark results

Duplicate-deal scoring: every deal is replayed with the hero in all three
seats on the same deck order. `*` marks a 95% interval that excludes zero,
but note it covers evaluation noise only -- the seed spread is wider and is
shown as whiskers in the results plot.
Every row differs only in the swept parameter.

| config | seeds | vs random | vs calling station | vs heuristic |
|---|---:|---:|---:|---:|
| `sweep_a0.01` | 3 | +593 ±38* | +713 ±56* | -66 ±29* |
| `sweep_a0.02` | 3 | +644 ±42* | +523 ±61* | -66 ±29* |
| `sweep_a0.03` | 3 | +602 ±40* | +388 ±54* | -108 ±29* |
| `sweep_a0.04` | 3 | +600 ±38* | +584 ±66* | -74 ±30* |
| `sweep_a0.05` | 3 | +559 ±39* | +405 ±58* | -50 ±27* |
