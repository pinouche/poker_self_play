"""Playing a real hand, one decision at a time, by continual re-solving.

Everything else in this project scores the agent over *belief states*: the
exploitability routines walk whole endgame trees with range vectors, and LBR
walks the same trees with a smarter responder.  Neither ever holds two cards
and picks an action.  That is fine for the numbers they produce and useless the
moment the opponent is a server on the internet that deals you ``Ac Td`` and
waits.

:class:`ResolvingAgent` is the missing half.  It keeps the belief state the
solver needs — a ``(2, 1326)`` reach — alongside the one card pair it was
actually dealt, and exposes the three things a real hand asks of a player:

``act``             solve the current round, then draw an action from the
                    resolved strategy *for the hand we hold*.
``observe``         the opponent acted; fold that action's probabilities into
                    their range, which is how the agent reasons about what
                    their betting revealed.
``observe_board``   a chance outcome arrived; mask both ranges by it and
                    invalidate the solve, so the next decision re-solves from
                    the new belief state.

**The re-solve boundary is a betting round.**  A solve is rooted at the current
public state with ``depth_limit=1``, so it covers this round's betting and
stops; every decision inside the round reads the same strategy map, and the
first decision of the next round finds itself outside it and re-solves.  That
is one solve per round rather than one per decision, and it is also what makes
the agent's own strategy self-consistent within a round.

**Ranges are renormalised per player after every update.**  Multiplying a reach
by action probabilities at every decision drives it toward zero over a hand,
and float32 underflow would eventually empty a range.  Rescaling one player's
reach by a constant leaves the resolved strategy unchanged — it scales that
player's counterfactual values uniformly, and CFR's regrets with them — so this
costs nothing and keeps the arithmetic in range.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from paradigm_b.core.cfr.tabular_cfr import CFRConfig
from paradigm_b.core.search.policy import StrategyMap, strategy_map
from paradigm_b.core.search.subgame import SubgameSolver
from paradigm_b.holdem.engine.betting import Betting, preflop_betting
from paradigm_b.holdem.engine.combos import board_mask, combo_index
from paradigm_b.holdem.engine.public_tree import PublicState, build_turn_tree
from paradigm_b.holdem.engine.space import TurnEndgameSpace
from paradigm_b.holdem.net.leaf_values import NetLeafValues


@dataclass
class PlayConfig:
    """How hard the agent thinks at each decision.

    The defaults match :class:`~holdem.selfplay.HoldemSelfPlayConfig` so that an
    agent plays at the strength its training loop assumed.  ``exploration`` has
    no counterpart here on purpose: it exists to widen the *training*
    distribution, and mixing uniform noise into a strategy at test time would
    only make the agent worse.
    """

    search_iterations: int = 40
    river_iterations: int = 60
    depth_limit: int = 1
    cfr: CFRConfig = field(default_factory=CFRConfig.linear_cfr_d)
    device: str = "cpu"


class ResolvingAgent:
    """One seat, one hand, playing by re-solving each betting round."""

    def __init__(
        self,
        net,
        seat: int,
        hole_cards: Tuple[int, int],
        betting: Optional[Betting] = None,
        board: Sequence[int] = (),
        config: Optional[PlayConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.config = config or PlayConfig()
        self.seat = seat
        self.rng = rng if rng is not None else np.random.default_rng()
        # The space is pinned to the *starting* board because that is what
        # fixes ``pair_correction`` — it describes how the hole cards were
        # dealt, not what the board later becomes.  A hand played from the
        # blinds therefore has an empty one, exactly as ``initial_situation``
        # builds it for self play.
        self.space = TurnEndgameSpace(tuple(board))
        net.eval()
        self.leaf_values = NetLeafValues(net, self.space, device=self.config.device)
        self.public = PublicState(
            betting=betting if betting is not None else preflop_betting(),
            board=tuple(board),
        )
        self.reach = self.space.initial_reach()
        self.hand = combo_index(*hole_cards)
        self._strategies: Optional[StrategyMap] = None
        self.solves = 0

    # --- the belief state --------------------------------------------------
    @property
    def opponent(self) -> int:
        return 1 - self.seat

    def _renormalise(self, player: int) -> None:
        total = self.reach[player].sum()
        if total > 0.0:
            self.reach[player] /= total

    def _strategy_at(self, public: PublicState) -> np.ndarray:
        """The resolved strategy here, re-solving if we have left the subgame.

        The cache miss *is* the round boundary: a solve covers one betting
        round, so the first decision of the next round is the first public
        state the map does not contain.
        """
        if self._strategies is None or public not in self._strategies:
            self._solve(public)
        actions = public.legal_actions()
        strategy = self._strategies.get(public) if self._strategies else None
        if strategy is None:
            # A decision the solve did not produce behaviour for.  Uniform is
            # the same default the scoring path fills in, so an agent measured
            # here and an agent measured there play the same way.
            return np.full((self.space.num_hands, len(actions)), 1.0 / len(actions))
        return strategy

    def _solve(self, public: PublicState) -> None:
        tree = build_turn_tree(public, depth_limit=self.config.depth_limit)
        has_leaves = bool(tree.leaves())
        solver = SubgameSolver(
            tree,
            leaf_value_fn=self.leaf_values if has_leaves else None,
            config=self.config.cfr,
            space=self.space,
        )
        # The river has no depth limit — the hand runs to real showdowns — so
        # it gets the longer solve the self-play loop gives it rather than the
        # shorter one that exists to keep a network-priced search affordable.
        iterations = (
            self.config.search_iterations if has_leaves else self.config.river_iterations
        )
        solver.solve(reach=self.reach, iterations=iterations, seed_root_value=True)
        self._strategies = strategy_map(solver)
        self.solves += 1

    # --- the three things a hand asks -------------------------------------
    def act(self) -> Tuple[int, int]:
        """Choose an action for the hand we hold.  Returns ``(action, total)``.

        ``total`` is the chip total the action commits *in the agent's own
        accounting*, which is what a caller needs to translate the decision
        into whatever sizes the real game accepts.
        """
        if self.public.to_move() != self.seat:
            raise ValueError("not this agent's turn")
        strategy = self._strategy_at(self.public)
        actions = self.public.legal_actions()
        probabilities = np.asarray(strategy[self.hand], dtype=float)
        total = probabilities.sum()
        # A hand the solve gave no mass to still has to do something.
        probabilities = (
            probabilities / total
            if total > 0.0
            else np.full(len(actions), 1.0 / len(actions))
        )
        index = int(self.rng.choice(len(actions), p=probabilities))
        action = actions[index]
        self._advance(strategy, index, self.seat)
        return action, self.public.betting.contributions[self.seat]

    def observe(self, action: int, total: Optional[int] = None) -> None:
        """The opponent took ``action``, committing ``total`` chips if given.

        ``total`` is what makes an off-abstraction bet expressible: the action
        id says which column of the strategy to condition their range on, and
        the chips say what actually went into the pot.  They come apart exactly
        when the opponent bets a size the abstraction does not hold, which is
        the case a translating caller has to handle.
        """
        if self.public.to_move() != self.opponent:
            raise ValueError("not the opponent's turn")
        strategy = self._strategy_at(self.public)
        actions = self.public.legal_actions()
        if action not in actions:
            raise ValueError(f"action {action} is not legal here: {actions}")
        self._advance(strategy, actions.index(action), self.opponent, total)

    def _advance(
        self,
        strategy: np.ndarray,
        index: int,
        player: int,
        total: Optional[int] = None,
    ) -> None:
        """Condition ``player``'s range on the action, then step the state."""
        actions = self.public.legal_actions()
        self.reach[player] = self.reach[player] * strategy[:, index]
        self._renormalise(player)
        betting = (
            self.public.betting.apply(actions[index])
            if total is None
            else self.public.betting.apply_amount(actions[index], total)
        )
        self.public = PublicState(betting=betting, board=self.public.board)

    def observe_board(self, cards: Sequence[int]) -> None:
        """A chance outcome arrived: mask both ranges and force a re-solve."""
        self.public = self.public.with_cards(tuple(cards))
        mask = board_mask(tuple(self.public.board))
        self.reach = self.reach * mask
        for player in range(2):
            self._renormalise(player)
        # The belief state moved; whatever was solved is about the old one.
        self._strategies = None

    @property
    def is_terminal(self) -> bool:
        return self.public.is_terminal

    @property
    def awaiting_board(self) -> bool:
        return self.public.awaiting_board
