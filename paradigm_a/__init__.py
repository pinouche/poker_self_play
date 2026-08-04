"""Paradigm A: pure self-play reinforcement learning.

Three-player no-limit hold'em played by a single shared policy/Q network
trained against copies of itself, scaling up through population self-play to a
role-structured league.  The metric is bb/100 against held-out opponents.

``config``          every knob, in one dataclass tree.
``environment``     the three-player no-limit rules engine.
``representation``  raw state -> canonicalised, fixed-size observation.
``model``           the shared policy/Q network.
``training``        self-play, replay, policy improvement, the league.
``evaluation``      baseline opponents and match play.
``inference``       the public "table state in, action out" API.
``cli``             entry points: train, benchmark, evaluate, infer.

Its ceiling was measured and is documented in ``docs/paradigm_a.md``: a strong
*exploitative* bot, decisively far from unexploitable.  Paradigm B
(``paradigm_b/``) targets the other thing.
"""
