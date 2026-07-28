"""Arm 2 — the iterative regime: ReBeL Algorithm 1, labels refreshed forever.

``journal.py``  the append-only record of every label the loop produced.
``student.py``  the loop itself, spending exactly the budgets it is given.
"""

from holdem.compare.iterative.journal import JournalEntry, TrajectoryJournal
from holdem.compare.iterative.student import OnlineStudentConfig, fit_online_student

__all__ = [
    "JournalEntry",
    "TrajectoryJournal",
    "OnlineStudentConfig",
    "fit_online_student",
]
