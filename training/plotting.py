"""Plot training metrics from one or more runs.

Every run writes ``history.json`` into its checkpoint directory (one record per
iteration).  This module overlays those records so hyperparameter settings can
be compared directly.

Policy **entropy** is the headline metric and the default panel.  It is a
diagnostic rather than an objective, and it is informative in both directions:

* entropy near ``log(num legal actions)`` means the policy is ignoring its
  critic -- the failure that ``alpha = 0.5`` produced;
* entropy near zero means the policy has stopped exploring, so the critic never
  learns the value of untaken actions, and the strategy is readable.

Usage::

    python -m training.plotting checkpoints/runA checkpoints/runB
    python -m training.plotting run* --metrics entropy q_loss --out compare.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

# Headless by default so this works over SSH and in CI.
import matplotlib

if not os.environ.get("DISPLAY") and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

DEFAULT_METRICS = ("entropy", "loss", "q_loss", "policy_loss")

#: Metrics where a reference line makes the plot readable.
REFERENCE_LINES = {
    "entropy": ("uniform over 5 legal actions", 1.6094379),
}


def load_history(path: str) -> List[dict]:
    """Read a run's ``history.json``; ``path`` may be the file or its directory."""
    if os.path.isdir(path):
        path = os.path.join(path, "history.json")
    with open(path) as fh:
        return json.load(fh)


def run_label(path: str) -> str:
    """A short name for the legend, preferring the checkpoint directory name."""
    path = path.rstrip("/")
    if path.endswith("history.json"):
        path = os.path.dirname(path)
    return os.path.basename(path) or path


def series(history: Sequence[dict], metric: str):
    """(iterations, values) for ``metric``, skipping iterations that lack it.

    Early iterations have no training metrics at all -- the buffer is still
    filling -- so the series legitimately starts partway in.
    """
    xs, ys = [], []
    for record in history:
        if metric in record and record[metric] is not None:
            xs.append(record.get("iteration", len(xs) + 1))
            ys.append(record[metric])
    return xs, ys


def smooth(values: Sequence[float], window: int) -> List[float]:
    """Centred moving average; self-play metrics are noisy per iteration."""
    if window <= 1 or len(values) < window:
        return list(values)
    out = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def plot_runs(
    runs: Dict[str, Sequence[dict]],
    metrics: Sequence[str] = DEFAULT_METRICS,
    window: int = 1,
    out: Optional[str] = None,
    title: Optional[str] = None,
    show_raw: bool = True,
):
    """Overlay ``metrics`` for every run; returns the matplotlib figure."""
    metrics = [m for m in metrics if any(series(h, m)[0] for h in runs.values())]
    if not metrics:
        raise ValueError("none of the requested metrics appear in any run")

    columns = min(2, len(metrics))
    rows = (len(metrics) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows, columns, figsize=(7.0 * columns, 4.0 * rows), squeeze=False
    )
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, metric in enumerate(metrics):
        axis = axes[index // columns][index % columns]
        for run_index, (label, history) in enumerate(runs.items()):
            xs, ys = series(history, metric)
            if not xs:
                continue
            colour = colours[run_index % len(colours)]
            if window > 1 and show_raw:
                axis.plot(xs, ys, color=colour, alpha=0.18, linewidth=0.8)
            axis.plot(xs, smooth(ys, window), color=colour, label=label, linewidth=1.6)

        if metric in REFERENCE_LINES:
            note, value = REFERENCE_LINES[metric]
            axis.axhline(value, color="0.4", linestyle="--", linewidth=1.0)
            axis.annotate(
                note,
                xy=(0.99, value),
                xycoords=("axes fraction", "data"),
                ha="right",
                va="bottom",
                fontsize=8,
                color="0.35",
            )

        axis.set_title(metric)
        axis.set_xlabel("iteration")
        axis.grid(alpha=0.25, linewidth=0.6)
        if index == 0:
            axis.legend(fontsize=8)

    for spare in range(len(metrics), rows * columns):
        axes[spare // columns][spare % columns].axis("off")

    if title:
        figure.suptitle(title)
        figure.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        figure.tight_layout()

    if out:
        figure.savefig(out, dpi=140)
        print(f"wrote {out}")
    return figure


def plot_paths(
    paths: Sequence[str],
    metrics: Sequence[str] = DEFAULT_METRICS,
    window: int = 1,
    out: Optional[str] = None,
    title: Optional[str] = None,
):
    runs = {}
    for path in paths:
        try:
            runs[run_label(path)] = load_history(path)
        except FileNotFoundError:
            print(f"skipping {path}: no history.json")
    if not runs:
        raise SystemExit("no runs could be loaded")
    return plot_runs(runs, metrics=metrics, window=window, out=out, title=title)


def summarize(paths: Sequence[str], metrics: Sequence[str] = DEFAULT_METRICS) -> str:
    """Text table of each run's final smoothed value, for quick comparison."""
    lines = [f"{'run':<24}" + "".join(f"{m:>14}" for m in metrics)]
    lines.append("-" * len(lines[0]))
    for path in paths:
        try:
            history = load_history(path)
        except FileNotFoundError:
            continue
        cells = []
        for metric in metrics:
            _, ys = series(history, metric)
            tail = ys[-20:]
            cells.append(f"{sum(tail) / len(tail):>14.4f}" if tail else f"{'-':>14}")
        lines.append(f"{run_label(path):<24}" + "".join(cells))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot self-play training metrics")
    parser.add_argument("runs", nargs="+", help="checkpoint directories or history.json files")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--window", type=int, default=9, help="moving-average window")
    parser.add_argument("--out", type=str, default="training_metrics.png")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true", help="print the table only")
    args = parser.parse_args()

    print(summarize(args.runs, args.metrics))
    if not args.no_plot:
        plot_paths(args.runs, args.metrics, window=args.window, out=args.out, title=args.title)


if __name__ == "__main__":
    main()
