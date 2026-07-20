# Benchmark results

Duplicate-deal scoring: every deal is replayed with the hero in all three
seats on the same deck order. `*` marks a 95% interval that excludes zero,
but note it covers evaluation noise only -- the seed spread is wider and is
shown as whiskers in the results plot.
Rows below `full` are single-change ablations from it.

| config | seeds | vs random | vs calling station | vs heuristic |
|---|---:|---:|---:|---:|
| `full` | 2 | +523 ±51* | +926 ±91* | -77 ±43* |
| `no_equity` | 2 | +544 ±58* | +703 ±82* | -52 ±44* |
| `no_ev_runout` | 2 | +264 ±39* | +689 ±77* | -25 ±31 |
| `no_pool` | 2 | +504 ±45* | +379 ±72* | -15 ±34 |
| `fine_abstraction` | 2 | +628 ±55* | +939 ±84* | -39 ±39 |
| `alpha_0.10` | 2 | +640 ±53* | +818 ±89* | -166 ±46* |
| `alpha_0.02` | 2 | +504 ±51* | +914 ±86* | +11 ±40 |
| `legacy` | 2 | +313 ±54* | +211 ±91* | -413 ±55* |

## Ablation deltas (config minus `full`)

| config | vs random | vs calling station | vs heuristic |
|---|---:|---:|---:|
| `no_equity` | +22 | -224 | +25 |
| `no_ev_runout` | -259 | -238 | +51 |
| `no_pool` | -18 | -548 | +62 |
| `fine_abstraction` | +105 | +13 | +38 |
| `alpha_0.10` | +117 | -108 | -89 |
| `alpha_0.02` | -19 | -12 | +88 |
| `legacy` | -210 | -715 | -336 |
