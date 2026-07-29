"""Scoring a value network as an agent, on boards it never trained on.

The metric is **exploitability**, not loss.  Loss is measured against each arm's
own labels, so it is not comparable across arms — and the offline arm can post
the *lower* loss precisely because its frozen labels are stale and
self-consistent, which is the failure mode this experiment exists to detect.

Exploitability asks something the labels cannot influence: turn the network into
an agent (continual re-solving, network at the leaves), then compute exactly how
much a best responder beats it.  Lower is better.

**The agent and its scorer must stop in the same place.**  A best-response tree
bounded to *n* betting rounds has decision nodes only within those rounds, so the
resolver is bounded to *n* rounds too (``max_resolve_rounds``).  Get this wrong
in either direction and the number is meaningless: bound only the tree and the
agent wastes enormous effort solving nodes nobody scores; bound only the
resolver and the scorer finds decision nodes the agent never produced behaviour
for, filled with uniform play, and reports the resulting disaster as the
network's exploitability.

**Why the flop is bounded at all.**  Unbounded continual re-solving from a flop
root is roughly 11,000 solves, because every flop belief state fans out to ~49
turns and each of those to ~48 rivers.  The turn costs 241 solves and is left
exact; the river costs one.  The flop default of one round is what makes a
three-street score affordable at all:

===================  ==============  ==========================
flop ``depth_limit``  tree size       leaf network evaluations
===================  ==============  ==========================
1 (default)           364 nodes       343
2                     92,288 nodes    58,800
===================  ==============  ==========================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from paradigm_b.holdem.engine.combos import board_mask
from paradigm_b.holdem.arms_common.situations import STREET_NAMES
from paradigm_b.holdem.net.value_net import HoldemValueNet
from paradigm_b.holdem.engine.public_tree import PublicState, build_endgame_tree
from paradigm_b.holdem.data.sampling import SituationConfig, held_out_boards, sample_situation
from paradigm_b.holdem.engine.space import TurnEndgameSpace
from paradigm_b.holdem.net.leaf_values import NetLeafValues
from paradigm_b.core.search import ContinualResolver, ResolveConfig
from paradigm_b.core.search.best_response import subgame_exploitability


@dataclass
class TestSituation:
    """One frozen held-out situation: board, betting, and both starting ranges."""

    space: TurnEndgameSpace
    root: PublicState
    reach: np.ndarray

    @property
    def board_cards(self) -> int:
        return len(self.root.board)

    @property
    def street(self) -> str:
        return STREET_NAMES.get(self.board_cards, str(self.board_cards))

    def tree(self, depth_limit: Optional[int] = None):
        return build_endgame_tree(self.root, depth_limit=depth_limit)


@dataclass
class EvaluationConfig:
    """The held-out situations, and how strictly each street is scored."""

    situations: SituationConfig = field(default_factory=SituationConfig)
    streets: Tuple[int, ...] = (3, 4, 5)
    boards_per_street: int = 2
    search_iterations: int = 40
    safe_resolving: bool = False
    # See the module docstring: the flop is bounded because it is otherwise
    # ~11,000 solves.  ``None`` for any street asks for the exact tree.
    flop_tree_depth_limit: Optional[int] = 1
    device: str = "cpu"

    def tree_depth_limit(self, board_cards: int) -> Optional[int]:
        return self.flop_tree_depth_limit if board_cards == 3 else None


def make_held_out_situations(
    rng: np.random.Generator,
    config: SituationConfig,
    streets: Tuple[int, ...] = (3, 4, 5),
    boards_per_street: int = 1,
) -> Dict[int, Tuple[TestSituation, ...]]:
    """Fixed evaluation situations, keyed by how many board cards they show.

    Boards are drawn so no two conflict at any street — see
    :func:`~holdem.sampling.conflicts_with_held_out` — because a held-out flop
    that reappears as three of a held-out turn's four cards is one board tested
    twice, and, worse, is a board the training data was never told to avoid.
    """
    situations: Dict[int, Tuple[TestSituation, ...]] = {}
    claimed: list = []
    for board_cards in streets:
        boards = held_out_boards(rng, boards_per_street, board_cards, excluded=claimed)
        claimed.extend(boards)
        street_situations = []
        for board in boards:
            # Reuse the sampler for pot/stack/ranges, then pin the board.
            unconstrained = SituationConfig(
                **{**config.__dict__, "board_cards": board_cards, "excluded_boards": ()}
            )
            _, root, reach = sample_situation(rng, unconstrained)
            mask = board_mask(board)
            reach = np.stack([r * mask for r in reach])
            reach /= reach.sum(axis=1, keepdims=True)
            street_situations.append(
                TestSituation(
                    space=TurnEndgameSpace(board),
                    root=PublicState(betting=root.betting, board=board),
                    reach=reach,
                )
            )
        situations[board_cards] = tuple(street_situations)
    return situations


def all_held_out_boards(
    situations: Dict[int, Tuple[TestSituation, ...]]
) -> Tuple[Tuple[int, ...], ...]:
    """Every held-out board, flattened — what training data must exclude."""
    return tuple(
        situation.root.board
        for street in situations.values()
        for situation in street
    )


def exploitability_on(
    net: HoldemValueNet,
    situation: TestSituation,
    search_iterations: int,
    safe_resolving: bool,
    device: str = "cpu",
    tree_depth_limit: Optional[int] = None,
) -> float:
    """Exploitability of ``net`` played as a continual-resolving agent.

    ``tree_depth_limit`` bounds both the scoring tree and the agent's own
    re-solving, and the same network prices the resulting leaves on both sides,
    so the agent is never charged for a horizon it was not given.
    """
    net.eval()
    leaf_values = NetLeafValues(net, situation.space, device=device)
    resolver = ContinualResolver(
        leaf_values,
        ResolveConfig(
            iterations=search_iterations,
            depth_limit=1,
            safe_resolving=safe_resolving,
            max_resolve_rounds=tree_depth_limit,
        ),
        space=situation.space,
        tree_builder=build_endgame_tree,
    )
    strategies, _ = resolver.run(root=situation.root, reach=situation.reach)

    tree = situation.tree(depth_limit=tree_depth_limit)
    filled = {
        node.public: strategies.get(
            node.public,
            np.full(
                (situation.space.num_hands, node.num_actions),
                1.0 / node.num_actions,
            ),
        )
        for node in tree.decision_nodes()
    }
    total, _ = subgame_exploitability(
        tree,
        filled,
        situation.reach,
        leaf_value_fn=leaf_values if tree.leaves() else None,
        space=situation.space,
    )
    return float(total)


def evaluate_agent(
    net: HoldemValueNet,
    situations: Dict[int, Tuple[TestSituation, ...]],
    config: EvaluationConfig,
) -> Dict[str, float]:
    """Mean per-street exploitability, plus the aggregate over streets.

    The aggregate is the mean over *streets*, not over situations, so a street
    that happens to carry more held-out boards does not dominate the headline.

    **The river is excluded from the headline aggregate, on purpose.**  Scoring
    a river-rooted situation builds a depth-limited tree that has no leaves —
    the hand runs straight to showdown — so the value network is never
    consulted and the score measures only the CFR search.  Two different
    networks produce *byte-identical* river numbers, and averaging that
    constant into the headline just dilutes the comparison.  It is still
    reported per-street, and still worth reading: it is a free check that the
    solver is behaving, and if it ever differs between two agents something is
    wrong.  ``aggregate_all_streets`` keeps the undiluted mean for continuity.
    """
    scores: Dict[str, float] = {}
    for board_cards, street_situations in sorted(situations.items()):
        name = STREET_NAMES.get(board_cards, str(board_cards))
        per_board = [
            exploitability_on(
                net,
                situation,
                config.search_iterations,
                config.safe_resolving,
                config.device,
                tree_depth_limit=config.tree_depth_limit(board_cards),
            )
            for situation in street_situations
        ]
        scores[name] = float(np.mean(per_board))
        scores[f"{name}_worst"] = float(np.max(per_board))
    street_means = [v for k, v in scores.items() if not k.endswith("_worst")]
    scores["aggregate_all_streets"] = (
        float(np.mean(street_means)) if street_means else float("nan")
    )
    # Streets where the agent's own re-solve is depth-limited, and therefore
    # where the network actually decides anything.  The river is not one.
    sensitive = [
        scores[STREET_NAMES[cards]]
        for cards in sorted(situations)
        if cards < 5 and STREET_NAMES[cards] in scores
    ]
    scores["aggregate"] = float(np.mean(sensitive)) if sensitive else scores[
        "aggregate_all_streets"
    ]
    return scores
