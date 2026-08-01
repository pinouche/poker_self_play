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
from typing import Dict, Optional, Sequence, Tuple

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
from paradigm_b.holdem.arms_common.lbr import LBRConfig, lbr_values


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
    # Also score with a responder that is allowed *off* the agent's abstraction
    # (see :mod:`.lbr`).  Off by default: it is a second pair of traversals per
    # situation, and — more to the point — it answers a different question from
    # the exact best response above, which stays the headline number because it
    # is the one that is exact.
    local_best_response: bool = False
    lbr: LBRConfig = field(default_factory=LBRConfig)

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


def scores_on(
    net: HoldemValueNet,
    situation: TestSituation,
    search_iterations: int,
    safe_resolving: bool,
    device: str = "cpu",
    tree_depth_limit: Optional[int] = None,
    lbr: Optional[LBRConfig] = None,
) -> Dict[str, float]:
    """Every score for ``net`` played as a continual-resolving agent, here.

    ``tree_depth_limit`` bounds both the scoring tree and the agent's own
    re-solving, and the same network prices the resulting leaves on both sides,
    so the agent is never charged for a horizon it was not given.

    Passing ``lbr`` adds the off-abstraction responder's numbers alongside the
    exact one, computed against the identical strategy.
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
    scores = {"exploitability": float(total)}
    if lbr is not None:
        # Same strategy, same situation, a responder with different actions.
        # Re-solving to get the strategy a second time would double the cost of
        # the expensive half for nothing.
        scores.update(
            lbr_values(situation.space, filled, situation.root, situation.reach, lbr)
        )
    return scores


def exploitability_on(
    net: HoldemValueNet,
    situation: TestSituation,
    search_iterations: int,
    safe_resolving: bool,
    device: str = "cpu",
    tree_depth_limit: Optional[int] = None,
) -> float:
    """Just the exact best-response number; see :func:`scores_on`."""
    return scores_on(
        net, situation, search_iterations, safe_resolving, device, tree_depth_limit
    )["exploitability"]


def _stderr(values: Sequence[float]) -> float:
    """Standard error of the mean over *situations*, not over hands.

    Worth being exact about what this is and is not, because the ReBeL paper
    reports a ``±`` on its LBR column and it is a different quantity.  There,
    LBR is an opponent that *plays hands*: 881 ± 94 mbb/g is a sample mean over
    dealt hands and the ± is the sampling error of a match that could have gone
    otherwise.  Here, nothing is dealt — :mod:`.lbr` walks the whole tree
    against full 1,326-combo ranges and enumerates every runout, so one
    situation's number is exact and repeating the measurement returns it bit for
    bit.

    What *is* uncertain here is which situations were drawn.  The held-out
    boards, pots, stacks and starting ranges come from a sampler, and the spread
    across them is large.  So this is the error bar of "what would this agent
    score on an average situation from this distribution", which is the honest
    analogue of the paper's ±, and it needs ``boards_per_street`` above 1 to
    exist at all.

    ``ddof=1`` because these are a sample of situations, not the population.
    """
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _aggregate_stderr(stderrs: Sequence[float]) -> float:
    """The stderr of a mean of per-street means, streets taken as independent.

    Each street is scored on its own held-out boards, drawn without conflict
    with the others', so the errors do not share a situation and add in
    quadrature: ``se = sqrt(sum(se_i^2)) / k``.
    """
    if not stderrs:
        return 0.0
    return float(np.sqrt(sum(s * s for s in stderrs)) / len(stderrs))


def evaluate_agent(
    net: HoldemValueNet,
    situations: Dict[int, Tuple[TestSituation, ...]],
    config: EvaluationConfig,
) -> Dict[str, float]:
    """Mean per-street exploitability, plus the aggregate over streets.

    Every mean is reported with a ``_stderr`` beside it, over the held-out
    situations that went into it — see :func:`_stderr` for why that is dispersion
    over *situations* and not the paper's dispersion over *hands*.  With one
    board per street it is zero, which is the truth: a single situation has a
    mean and no spread.

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
            scores_on(
                net,
                situation,
                config.search_iterations,
                config.safe_resolving,
                config.device,
                tree_depth_limit=config.tree_depth_limit(board_cards),
                lbr=config.lbr if config.local_best_response else None,
            )
            for situation in street_situations
        ]
        exploitabilities = [board["exploitability"] for board in per_board]
        scores[name] = float(np.mean(exploitabilities))
        scores[f"{name}_worst"] = float(np.max(exploitabilities))
        scores[f"{name}_stderr"] = _stderr(exploitabilities)
        scores[f"{name}_boards"] = float(len(per_board))
        if config.local_best_response:
            for key in ("lbr_classic", "lbr_full"):
                values = [board[key] for board in per_board]
                scores[f"{name}_{key}"] = float(np.mean(values))
                scores[f"{name}_{key}_stderr"] = _stderr(values)
    # Averaged over the street keys *by name*, not by pattern-matching the
    # score dict.  The old "everything that is not ``_worst``" rule silently
    # swallowed any new per-street entry — the LBR ones would have been
    # averaged into the headline exploitability without a word.
    street_names = [
        STREET_NAMES.get(cards, str(cards)) for cards in sorted(situations)
    ]
    street_means = [scores[name] for name in street_names if name in scores]
    scores["aggregate_all_streets"] = (
        float(np.mean(street_means)) if street_means else float("nan")
    )
    scores["aggregate_all_streets_stderr"] = _aggregate_stderr(
        [scores[f"{name}_stderr"] for name in street_names if f"{name}_stderr" in scores]
    )
    if config.local_best_response:
        for key in ("lbr_classic", "lbr_full"):
            per_street = [
                scores[f"{name}_{key}"]
                for name in street_names
                if f"{name}_{key}" in scores
            ]
            if per_street:
                scores[f"aggregate_{key}"] = float(np.mean(per_street))
                scores[f"aggregate_{key}_stderr"] = _aggregate_stderr(
                    [
                        scores[f"{name}_{key}_stderr"]
                        for name in street_names
                        if f"{name}_{key}_stderr" in scores
                    ]
                )
    # Streets where the agent's own re-solve is depth-limited, and therefore
    # where the network actually decides anything.  The river is not one.
    sensitive = [
        scores[STREET_NAMES[cards]]
        for cards in sorted(situations)
        if cards < 5 and STREET_NAMES[cards] in scores
    ]
    scores["aggregate_stderr"] = _aggregate_stderr(
        [
            scores[f"{STREET_NAMES[cards]}_stderr"]
            for cards in sorted(situations)
            if cards < 5 and f"{STREET_NAMES[cards]}_stderr" in scores
        ]
    ) if sensitive else scores["aggregate_all_streets_stderr"]
    scores["aggregate"] = float(np.mean(sensitive)) if sensitive else scores[
        "aggregate_all_streets"
    ]
    return scores
