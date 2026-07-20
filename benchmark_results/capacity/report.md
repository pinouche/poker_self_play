# Benchmark results

Duplicate-deal scoring: every deal is replayed with the hero in all three
seats on the same deck order. `*` marks a 95% interval that excludes zero,
but note it covers evaluation noise only -- the seed spread is wider and is
shown as whiskers in the results plot.
Every row differs only in the swept parameter.

| config | seeds | vs random | vs calling station | vs heuristic |
|---|---:|---:|---:|---:|
| `cap_small` | 3 | +479 ±35* | +378 ±59* | -48 ±28* |
| `cap_base` | 3 | +644 ±42* | +523 ±61* | -66 ±29* |
| `cap_large` | 3 | +596 ±37* | +329 ±60* | -55 ±29* |
| `cap_xlarge` | 3 | +378 ±35* | +84 ±48* | -36 ±24* |
| `league_broad` | 3 | +474 ±37* | +362 ±59* | -92 ±28* |
