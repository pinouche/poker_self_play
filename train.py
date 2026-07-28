#!/usr/bin/env python
"""Paradigm A entry point — a shim so ``python train.py`` keeps working.

The code lives in :mod:`paradigm_a.cli.train`; this file only exists because
that is a long thing to type for the command you run most.  Equivalent::

    python -m paradigm_a.cli.train --iterations 2000
"""

from paradigm_a.cli.train import main

if __name__ == "__main__":
    main()
