"""Plot evaluation-vs-opponent over training iterations, across seeds."""
import json, glob, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "trajectory.png")
files = sorted(glob.glob(f"{root}/*/trajectory.json"))
curves = {f.split("/")[-2]: json.load(open(f)) for f in files}

opps = ["heuristic", "random", "calling_station"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), squeeze=False)
colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for j, opp in enumerate(opps):
    ax = axes[0][j]
    # per-seed lines
    all_by_iter = {}
    for k,(name, curve) in enumerate(curves.items()):
        xs = [p["iteration"] for p in curve]
        ys = [p[opp]["bb_per_100"] for p in curve]
        hi = [p[opp]["ci_half_width"] for p in curve]
        ax.errorbar(xs, ys, yerr=hi, color=colours[k], alpha=0.5, lw=1.2,
                    marker="o", ms=3, capsize=2, label=name)
        for x,y in zip(xs,ys): all_by_iter.setdefault(x,[]).append(y)
    # seed-mean line in black
    if len(curves) > 1:
        xs = sorted(all_by_iter)
        mean = [np.mean(all_by_iter[x]) for x in xs]
        ax.plot(xs, mean, color="black", lw=2.2, label="seed mean")
    ax.axhline(0, color="0.3", lw=0.8, ls="--")
    ax.set_title(f"vs {opp}"); ax.set_xlabel("iteration")
    if j==0: ax.set_ylabel("bb/100"); ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
fig.suptitle("Strength against fixed opponents over self-play training (base config)")
fig.tight_layout(rect=(0,0,1,0.95))
fig.savefig(out, dpi=140); print("wrote", out)
