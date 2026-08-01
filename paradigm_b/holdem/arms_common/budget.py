"""What each arm spent, counted the same way for both.

Labelled examples and gradient updates are held *equal* by construction — that
is the experiment's fairness condition — so on their own they say nothing about
cost.  They also hide a large real difference: an exact river label comes out of
one batched solve shared across 64 situations, while a self-play label costs a
depth-limited solve of its own plus a network call for every leaf.  Reporting
only "10,000 labels each" would make two very different amounts of compute look
identical.

So each arm also reports the two quantities that actually track the work:

``leaf_evaluations``  belief states pushed through the value network.  This is
                      the dominant cost in every regime that has a network in
                      the loop, and it is the one thing a GPU would accelerate.
``solver_calls``      batched leaf-evaluation calls, i.e. how many separate
                      depth-limited solves asked the network for values.

Wall clock is recorded too, split into generation and training, but it is
deliberately *not* the headline: it is machine- and thread-dependent, and the
offline arm's generation happens once and is meant to be amortised over many
students.  The counts above are reproducible; seconds are context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

import numpy as np


@dataclass
class SpendRecord:
    """One arm's accounting.  Every field is a total, not a rate."""

    labels: int = 0
    updates: int = 0
    leaf_evaluations: int = 0
    solver_calls: int = 0
    generation_seconds: float = 0.0
    training_seconds: float = 0.0
    # Time in the journal, which is a debugging artifact rather than part of
    # either algorithm.  Kept *out* of ``total_seconds`` so that turning the
    # journal on does not charge an arm for compute the algorithm never needed
    # — but recorded, because an untimed section of the loop is how the schema-1
    # manifest silently ate 66% of a 12-hour run.  Any wall clock unaccounted
    # for by these three is a bug in the accounting, not a rounding error.
    journal_seconds: float = 0.0
    # Time the learner spent scoring itself *on its own thread*.  Off-thread
    # progress evaluations do not appear here — they cost the run contention,
    # not wall clock — so a large number here means a foreground evaluator is
    # eating the run.  Excluded from ``total_seconds`` for the same reason as
    # ``journal_seconds``: measuring an arm is not work its algorithm does.
    eval_seconds: float = 0.0

    @property
    def total_seconds(self) -> float:
        return self.generation_seconds + self.training_seconds

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["total_seconds"] = self.total_seconds
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpendRecord":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in fields})


class CountingLeafValues:
    """A leaf evaluator that tallies the work it is asked to do.

    Wraps any callable with the ``states -> (n, 2, 1326)`` leaf-value contract
    — :class:`~holdem.values.NetLeafValues` in practice — and forwards every
    call untouched.  Counting here rather than inside ``NetLeafValues`` keeps
    the tally per-*arm*: two students sharing one network object would
    otherwise pool their counts and neither number would mean anything.
    """

    def __init__(self, inner, spend: SpendRecord) -> None:
        self.inner = inner
        self.spend = spend

    def __call__(self, states: Sequence) -> np.ndarray:
        self.spend.solver_calls += 1
        self.spend.leaf_evaluations += len(states)
        return self.inner(states)

    def __getattr__(self, name: str):
        # Delegate ``space``, ``net``, ``device`` and friends to the evaluator
        # being wrapped, so this is a drop-in wherever one was expected.
        return getattr(self.inner, name)
