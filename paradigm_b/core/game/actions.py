"""The limit-betting action alphabet shared by the toy poker games.

Three actions suffice for Kuhn and Leduc: fold, call (which is a check when
there is nothing to call), and raise (which is a bet when nobody has bet).
Keeping one alphabet across games means the solvers, the public tree and the
networks never need per-game action bookkeeping.
"""

from __future__ import annotations

FOLD = 0
CALL = 1
RAISE = 2

NUM_ACTIONS = 3
ACTION_NAMES = ("f", "c", "r")


def action_name(action: int) -> str:
    return ACTION_NAMES[action]


def history_str(actions) -> str:
    """Compact betting history, e.g. ``"crc"``."""
    return "".join(ACTION_NAMES[a] for a in actions)
