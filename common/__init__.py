"""Code shared by both paradigms.

Everything here is used by paradigm A *and* paradigm B, which is the only
reason it is here — this package is deliberately small, and a module that
drifts into being used by one side alone belongs back in that side's tree.

``cards``           card primitives: ranks, suits, the deck.
``hand_evaluator``  best-five-of-seven evaluation and hand categories.
``nets``            neural primitives: the Deep CFR card embedding and the
                    ReBeL LayerNorm/GeLU stack.
"""
