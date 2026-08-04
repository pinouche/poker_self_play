"""Making training situations, and labelling them.

``sampling``    random belief states, in the style DeepStack and ReBeL draw them.
``batched``     solve many river situations at once on one shared tree.
``generation``  fast bulk generation of *exactly* solved river labels.
``bootstrap``   labels for the streets above the river, where "exact" is out of
                reach and a leaf is priced by whatever network you hand it.

The river/above-the-river split is the fault line the arm 1 vs arm 2 experiment
runs along: river labels are ground truth and never go stale, everything above
them is one network's opinion and might.
"""
