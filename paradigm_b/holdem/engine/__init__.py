"""The hold'em endgame rules engine: cards, betting, and the public tree.

No learning here and no solver — this is the geometry every other part of
``paradigm_b/holdem`` reasons over.

``combos``       the 1,326 two-card hands and the blocking arithmetic they force.
``space``        the hand space: which combos are live given a board.
``strength``     hand strength for every combo on a given board.
``showdown``     showdown counterfactual values, in linear time.
``betting``      no-limit betting for an endgame.
``public_tree``  the tree a spectator who knows no cards can see.
``translation``  mapping a real bet onto the action abstraction.
``isomorphism``  boards that are the same board with the suits renamed.
"""
