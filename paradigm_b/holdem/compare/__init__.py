"""Fixed labels vs refreshed labels: the two ways to feed a value network.

This package is the *comparison*; the arms themselves are its siblings.  See
``docs/arms.md`` for what the experiment tests and how to read its output.  The
short version:

``../arm1_fixed/``      label the world once, freeze it, train on the file.
``../arm2_iterative/``  ReBeL Algorithm 1, labels remade by the current net.
``../arms_common/``     only what both arms share; sharing it is what makes it fair.
``experiment.py``       the runner that holds seeds, held-out boards and budgets equal.
``relabel.py``          the staleness probe: how far the frozen labels actually drifted.
"""

from paradigm_b.holdem.compare.experiment import (
    ArmResult,
    ComparisonConfig,
    ComparisonResult,
    format_report,
    run_comparison,
)
from paradigm_b.holdem.compare.relabel import measure_label_drift

__all__ = [
    "ArmResult",
    "ComparisonConfig",
    "ComparisonResult",
    "format_report",
    "run_comparison",
    "measure_label_drift",
]
