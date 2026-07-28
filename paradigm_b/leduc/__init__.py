"""Stages 0-3: the whole ReBeL machine, validated on Leduc hold'em.

Small enough to compute exact exploitability, so every claim here is proved
rather than estimated.  ``value_net`` is the counterfactual value network over
Leduc belief states; ``rebel`` is the self-play loop that trains it.
"""
