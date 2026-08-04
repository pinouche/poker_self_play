#!/usr/bin/env python
"""Live progress for a running ``run_arm2.py``, read-only.

The training loop prints at the start and at the end and nothing in between, so
a long run looks frozen from the outside.  Everything needed to follow it is
already on disk, though, and this reads it without touching the run:

* ``run_state/meta.json`` — the full spend record and iteration counter,
  rewritten atomically on every state snapshot;
* ``journal/manifest.json`` — one entry per flushed iteration, so the label
  count is exact rather than inferred from bytes.

Safe against a snapshot happening mid-read: the state directory is swapped into
place by rename, so a read either sees the whole previous version or the whole
new one, and a torn JSON is retried rather than crashing the monitor.

Usage::

    python watch_arm2.py --run runs/arm2-12h              # one shot
    python watch_arm2.py --run runs/arm2-12h --follow     # every 60s, with deltas
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional


def read_json(path: Path, attempts: int = 5) -> Optional[dict]:
    """Read JSON that another process may be replacing underneath us."""
    for _ in range(attempts):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.2)
    return None


def disk_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return int(subprocess.check_output(["du", "-sk", str(path)]).split()[0]) * 1024


def journal_labels(run: Path) -> Optional[int]:
    manifest = read_json(run / "journal" / "manifest.json")
    if manifest is None:
        return None
    return sum(s.get("count", s.get("rows", 0)) for s in manifest.get("shards", []))


def last_progress(run: Path) -> Optional[dict]:
    """The most recent progress evaluation, or None if there has been one.

    Appended a line at a time by a thread inside the run, so the last line can
    be half-written when this reads it; a torn line is simply not shown yet.
    """
    path = run / "progress.jsonl"
    if not path.exists():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def snapshot(run: Path) -> dict:
    meta = read_json(run / "run_state" / "meta.json") or {}
    spend = meta.get("spend", {})
    config = read_json(run / "config.json") or {}
    args = config.get("args", {})
    started = os.path.getctime(run / "config.json") if (run / "config.json").exists() else None
    return {
        "labels": spend.get("labels"),
        "updates": spend.get("updates"),
        "leaf_evaluations": spend.get("leaf_evaluations"),
        "generation_seconds": spend.get("generation_seconds"),
        "training_seconds": spend.get("training_seconds"),
        "journal_seconds": spend.get("journal_seconds"),
        "iteration": meta.get("iteration"),
        "journal_labels": journal_labels(run),
        "elapsed": (time.time() - started) if started else None,
        "hours": args.get("hours"),
        "labels_per_update": args.get("labels_per_update"),
        "batch_size": args.get("batch_size"),
        "buffer_size": args.get("buffer_size"),
        "buffer_held": (meta.get("buffer") or {}).get("size"),
        "progress": last_progress(run),
        "state_bytes": disk_bytes(run / "run_state"),
        "journal_bytes": disk_bytes(run / "journal"),
        "alive": bool(
            subprocess.run(
                ["pgrep", "-f", f"run_arm2.py --run {run}"], capture_output=True
            ).stdout.strip()
        ),
    }


def render(now: dict, previous: Optional[dict]) -> str:
    lines = []
    alive = "RUNNING" if now["alive"] else "NOT RUNNING"
    elapsed = now["elapsed"] or 0.0
    hours = now["hours"] or 0.0
    lines.append(
        f"[{time.strftime('%H:%M:%S')}] {alive}   this session {elapsed/3600:5.2f} h"
        + (f" of {hours:g} h ({100*elapsed/3600/hours:.1f}%)" if hours else "")
    )

    # The journal is the exact label count; the state's is as of the last
    # snapshot, so it lags by up to state_every iterations.
    labels = now["journal_labels"] or now["labels"] or 0
    # Both are *cumulative across resumes*, because the spend record is restored
    # with the rest of the state — so the rate must not be divided by this
    # session's wall clock, which resets every time the run is picked up again.
    worked = (now["generation_seconds"] or 0.0) + (now["training_seconds"] or 0.0)
    if labels and worked:
        overall = labels / worked
        remaining = max(hours * 3600 - elapsed, 0) if hours else 0
        lines.append(
            f"  labels {labels:>10,}   {overall:6.1f}/s over {worked/3600:.2f} h worked"
            + (f"   projected {labels + overall*remaining:>11,.0f}" if remaining else "")
        )
    if now["updates"]:
        ratio = (now["labels"] or 0) / now["updates"]
        reuse = (now["batch_size"] or 0) / max(now["labels_per_update"] or 1, 1e-9)
        lines.append(
            f"  updates {now['updates']:>9,}   ratio {ratio:5.2f} labels/update"
            f"   (target {now['labels_per_update']}, reuse {reuse:.1f}x)"
        )
    if now["generation_seconds"] is not None and now["training_seconds"] is not None:
        total = now["generation_seconds"] + now["training_seconds"]
        share = 100 * now["training_seconds"] / total if total else 0
        lines.append(
            f"  learner: waiting {now['generation_seconds']/60:7.1f} min, "
            f"training {now['training_seconds']/60:6.1f} min ({share:.0f}% training)"
        )
        # Anything the three timers do not cover is time the loop cannot
        # explain.  It was 66% once; show it rather than wait to find it in a
        # post-mortem.  Only meaningful within one session: the timers are
        # cumulative and ``elapsed`` is not, so a resumed run shows this
        # negative until it has worked longer than it was resumed for.
        journal = now["journal_seconds"] or 0.0
        unexplained = elapsed - total - journal
        lines.append(
            f"  journal {journal/60:7.1f} min"
            f"   unexplained {unexplained/60:7.1f} min"
            + (f" ({100*unexplained/elapsed:.0f}% of wall)" if elapsed else "")
        )
    if now["buffer_held"] is not None:
        lines.append(
            f"  buffer {now['buffer_held']:>10,} / {now['buffer_size']:,}"
            f"   iteration {now['iteration']:,}"
            f"   disk {(now['state_bytes']+now['journal_bytes'])/1e9:5.1f} GB"
        )

    # The only two numbers that say whether it is getting *better*; the rest of
    # this panel says only that it is getting *bigger*.
    if now["progress"] is not None:
        from paradigm_b.holdem.arms_common.progress import summarise

        lines.append("  " + summarise(now["progress"]))

    if previous is not None:
        dt = (now["elapsed"] or 0) - (previous["elapsed"] or 0)
        dl = (now["journal_labels"] or 0) - (previous["journal_labels"] or 0)
        du = (now["updates"] or 0) - (previous["updates"] or 0)
        if dt > 0:
            lines.append(
                f"  since last: {dl:>8,} labels ({dl/dt:5.1f}/s), {du:>6,} updates"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args()

    run = Path(args.run)
    previous = None
    while True:
        now = snapshot(run)
        print(render(now, previous), flush=True)
        if not args.follow:
            return
        if not now["alive"]:
            print("  process gone; stopping monitor", flush=True)
            return
        previous = now
        print(flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
