"""Solver machinery that knows nothing about any particular poker game.

``game``    extensive-form games and their materialised trees (Kuhn, Leduc).
``cfr``     regret matching, CFR/CFR+/DCFR, exact best response.
``belief``  public belief states, ranges, Bayesian propagation.
``search``  depth-limited re-solving over a public tree.

Everything here is parameterised by a :class:`~paradigm_b.core.game.base.Game`
or a :class:`~paradigm_b.core.search.space.HandSpace`, which is what lets the
same code drive both Leduc and full hold'em.
"""
