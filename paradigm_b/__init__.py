"""Paradigm B: game-theoretic solving — search over public belief states.

Where paradigm A learns a policy that *beats* opponents, paradigm B searches
for one a knowledgeable adversary cannot beat.  The metric changes from bb/100
to **exploitability**, the distance to Nash.  The template is ReBeL (Brown et
al., 2020): depth-limited CFR over public belief states, leaves priced by a
learned counterfactual value network, trained by self-play.

``core``    the game-independent machinery: game trees, CFR and exact best
            response, public belief states and Bayesian range propagation,
            depth-limited re-solving.
``leduc``   stages 0-3 on Leduc hold'em — the reference implementation, small
            enough that exploitability can be computed exactly and the whole
            machine *proved* to work.
``holdem``  stage 4: real cards, 1,326-combination ranges, no card
            abstraction — and the arm 1 / arm 2 label-regime experiment.
``cli``     entry points: ``leduc``, ``holdem``, ``compare``.

Written up in ``docs/paradigm_b.md``.
"""
