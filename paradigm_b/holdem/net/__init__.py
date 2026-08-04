"""The neural side of hold'em: what a belief state looks like, and its value.

``features``     encoding a public belief state as network input.
``value_net``    the counterfactual value network, V(PBS) -> value per combo.
``policy``       the policy network used to warm-start search.
``leaf_values``  leaf evaluators — exact solving, or a call to the value net.

``leaf_values`` is the seam between this package and the solver: it is how a
depth-limited search asks "what is this leaf worth?", and swapping the exact
evaluator for the learned one is the whole of what ReBeL does.
"""
