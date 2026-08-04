"""Hold'em endgames: real cards, 1,326-combination ranges, no card abstraction.

Stage 4 — the same machine ``paradigm_b/leduc`` proves correct, at a scale
where exact exploitability is out of reach and every number is an estimate.

``engine``    the rules: combos, strength, showdown, betting, the public tree.
``net``       features, the value network, the policy network, leaf evaluators.
``data``      sampling situations and labelling them (exact on the river,
              bootstrapped above it).
``selfplay``  ReBeL trajectory collection — search, record, descend, repeat.
              **Both arms use this**; it is the loop, not a regime.

The two label regimes built on top of it, and the experiment that scores them
against each other:

``arm1_fixed``      label once into a frozen on-disk artifact, then train a
                    fresh student on the file alone — no solver in the loop.
``arm2_iterative``  ReBeL Algorithm 1 — labels remade by the current network,
                    forever; nothing is frozen.
``arms_common``     what the two share, which is what makes the match fair.
``compare``         the runner and the staleness probe.  See ``docs/arms.md``.
"""

from paradigm_b.holdem.engine.betting import Betting
from paradigm_b.holdem.engine.combos import NUM_COMBOS, board_mask, combo_index, pair_correction
from paradigm_b.holdem.engine.public_tree import PublicState, build_turn_tree
from paradigm_b.holdem.engine.showdown import showdown_values
from paradigm_b.holdem.engine.space import TurnEndgameSpace
from paradigm_b.holdem.engine.strength import hand_ranks

__all__ = [
    "Betting",
    "NUM_COMBOS",
    "board_mask",
    "combo_index",
    "pair_correction",
    "PublicState",
    "build_turn_tree",
    "showdown_values",
    "TurnEndgameSpace",
    "hand_ranks",
]
