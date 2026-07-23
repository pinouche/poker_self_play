"""Stage-1 size comparison: learning curves and final strength by network size."""
import json, glob, sys, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else f"{root}/stage1.png"
SIZES = ["tiny","medium","large"]; COL = {"tiny":"#55A868","medium":"#4C72B0","large":"#C44E52"}
PARAMS = {"tiny":"0.34M","medium":"1.6M","large":"7.0M"}

# curves[size][seed] = list of {iteration, hands?, opp:{bb_per_100}}
curves = {s: {} for s in SIZES}
for f in sorted(glob.glob(f"{root}/stage1_*__seed*/trajectory.json")):
    m = re.search(r"stage1_(\w+)__seed(\d+)", f)
    size, seed = m.group(1), int(m.group(2))
    if size in curves: curves[size][seed] = json.load(open(f))

opps = ["heuristic","random","calling_station"]
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), squeeze=False)
HANDS_PER_ITER = 256
for j, opp in enumerate(opps):
    ax = axes[0][j]
    for size in SIZES:
        seeds = curves[size]
        if not seeds: continue
        by_it = {}
        for c in seeds.values():
            for p in c:
                h = p["iteration"]*HANDS_PER_ITER/1e6
                by_it.setdefault(round(h,3), []).append(p[opp]["bb_per_100"])
        xs = sorted(by_it); mean=[np.mean(by_it[x]) for x in xs]
        for c in seeds.values():
            hs=[p["iteration"]*HANDS_PER_ITER/1e6 for p in c]; ys=[p[opp]["bb_per_100"] for p in c]
            ax.plot(hs, ys, color=COL[size], alpha=0.25, lw=1.0)
        ax.plot(xs, mean, color=COL[size], lw=2.4, label=f"{size} ({PARAMS[size]})")
    ax.axhline(0, color="0.3", lw=0.8, ls="--")
    ax.set_title(f"vs {opp}"); ax.set_xlabel("self-play hands (millions)")
    if j==0: ax.set_ylabel("bb/100"); ax.legend(fontsize=9, title="network size")
    ax.grid(alpha=0.25)
fig.suptitle("Stage 1: strength vs network size over self-play hands (self-play only, replay ratio ~1)")
fig.tight_layout(rect=(0,0,1,0.95)); fig.savefig(out, dpi=140); print("wrote", out)
print()
# final table (mean over last 2 checkpoints x seeds)
print(f"{'size':<10}{'params':>8}   " + "".join(f"{o:>18}" for o in opps))
for size in SIZES:
    seeds=curves[size]
    if not seeds: continue
    cells=[]
    for opp in opps:
        vals=[p[opp]["bb_per_100"] for c in seeds.values() for p in c[-2:]]
        cells.append(f"{np.mean(vals):+6.0f}±{np.std(vals):<4.0f}")
    print(f"{size:<10}{PARAMS[size]:>8}   " + "".join(f"{c:>18}" for c in cells))
