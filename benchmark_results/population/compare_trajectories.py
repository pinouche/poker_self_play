"""Overlay plain self-play vs population self-play trajectories."""
import json, glob, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plain_root = sys.argv[1]   # dir with long_base__seed*/trajectory.json
pop_root   = sys.argv[2]   # dir with long_population__seed*/trajectory.json
out        = sys.argv[3] if len(sys.argv) > 3 else "compare_traj.png"

def load(root):
    curves = {}
    for f in sorted(glob.glob(f"{root}/*/trajectory.json")):
        curves[f.split("/")[-2]] = json.load(open(f))
    # also accept flat seed*.json (archived form)
    for f in sorted(glob.glob(f"{root}/seed*.json")):
        curves[os.path.basename(f)] = json.load(open(f))
    return curves

plain, pop = load(plain_root), load(pop_root)
opps = ["heuristic", "random", "calling_station"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), squeeze=False)

def plot_group(ax, curves, colour, label):
    all_by_iter = {}
    for c in curves.values():
        xs = [p["iteration"] for p in c]; ys = [p[opp]["bb_per_100"] for p in c]
        ax.plot(xs, ys, color=colour, alpha=0.30, lw=1.0)
        for x,y in zip(xs,ys): all_by_iter.setdefault(x,[]).append(y)
    xs = sorted(all_by_iter); mean=[np.mean(all_by_iter[x]) for x in xs]
    ax.plot(xs, mean, color=colour, lw=2.4, label=label)
    vals = np.array([v for vs in all_by_iter.values() for v in vs])
    return vals

stats = {}
for j, opp in enumerate(opps):
    ax = axes[0][j]
    vp = plot_group(ax, plain, "#C44E52", "plain self-play")
    vq = plot_group(ax, pop,   "#4C72B0", "population")
    stats[opp] = (vp, vq)
    ax.axhline(0, color="0.3", lw=0.8, ls="--")
    ax.set_title(f"vs {opp}"); ax.set_xlabel("iteration")
    if j==0: ax.set_ylabel("bb/100"); ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
fig.suptitle("Plain vs population self-play: strength over training (lines = seed mean)")
fig.tight_layout(rect=(0,0,1,0.95)); fig.savefig(out, dpi=140)
print("wrote", out); print()
print(f"{'opponent':<16}{'plain mean±std':>22}{'population mean±std':>24}{'osc. amp (std)':>18}")
print("-"*80)
for opp,(vp,vq) in stats.items():
    print(f"{opp:<16}{vp.mean():>10.0f} ±{vp.std():<9.0f}{vq.mean():>13.0f} ±{vq.std():<9.0f}"
          f"   plain {vp.std():.0f} -> pop {vq.std():.0f}")
