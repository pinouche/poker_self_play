"""Arm 1 — the fixed regime: label once, freeze, train forever on the result.

``store.py``    the on-disk artifact: sharded, memory-mapped, source-separated.
``build.py``    generating it, street by street, saving each teacher.
``student.py``  training a fresh student from it, with no solver in the loop.
"""

from holdem.compare.fixed.build import DatasetBuildConfig, build_layered_dataset
from holdem.compare.fixed.store import (
    KNOWN_SOURCES,
    MANIFEST_FILENAME,
    STREET_SOURCES,
    DatasetManifest,
    DatasetStore,
)
from holdem.compare.fixed.student import FixedStudentConfig, fit_fixed_student

__all__ = [
    "DatasetBuildConfig",
    "build_layered_dataset",
    "KNOWN_SOURCES",
    "MANIFEST_FILENAME",
    "STREET_SOURCES",
    "DatasetManifest",
    "DatasetStore",
    "FixedStudentConfig",
    "fit_fixed_student",
]
