"""Fixed labels vs refreshed labels: the two ways to feed a value network.

See ``README.md`` in this directory for what the experiment tests and how to
read its output.  The short version:

``fixed/``       arm 1 — label the world once, freeze it, train on the file.
``iterative/``   arm 2 — ReBeL Algorithm 1, labels remade by the current net.
``common/``      only what both arms share; sharing it is what makes it fair.
``experiment.py``the runner that holds seeds, held-out boards and budgets equal.
``relabel.py``   the staleness probe: how far the frozen labels actually drifted.
"""

from holdem.compare.experiment import (
    ArmResult,
    ComparisonConfig,
    ComparisonResult,
    format_report,
    run_comparison,
)
from holdem.compare.relabel import measure_label_drift

__all__ = [
    "ArmResult",
    "ComparisonConfig",
    "ComparisonResult",
    "format_report",
    "run_comparison",
    "measure_label_drift",
]
