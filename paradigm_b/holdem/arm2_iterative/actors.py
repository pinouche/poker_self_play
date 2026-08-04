"""Many actors -> replay buffer -> one learner, the way ReBeL is actually run.

The synchronous loop in ``student.py`` generates every trajectory, *then*
trains, then repeats.  That is easy to reason about and exactly reproducible,
and it wastes almost all the wall clock: generating one label runs a full
depth-limited CFR solve — hundreds of tree traversals, each evaluating every
leaf through the network — while training on it is one forward and one backward
pass.  Measured on this repo, arm 2 spends **1629s generating against 25s
training**.  One core solves; nothing else happens.

Generation is embarrassingly parallel, though: trajectories are independent and
never talk to each other.  So the shape that fits is the standard Ape-X /
AlphaZero one, which is what ReBeL used across 90 DGX-1 machines:

    actors (N processes)          learner (this process)
    ------------------            ----------------------
    pull latest weights   <-----  publish weights every K updates
    sample a situation            drain the queue into the replay buffer
    run Algorithm 1               take gradient steps
    push examples         ----->  repeat

The buffer decouples them.  Actors never wait for the learner and the learner
never waits for data — it trains on whatever has accumulated. Actors therefore
run a network that is a little behind the learner's, by design: the staleness
costs less than the synchronisation would.

**This is not bit-reproducible, and that is a real trade-off.**  How many
trajectories land between two gradient steps depends on process scheduling, so
two runs with the same seed will not match. The fixed-vs-iterative comparison
rests on determinism (identical weights, identical budgets), so it should keep
using the synchronous path; use this one for large runs where wall clock is the
binding constraint. Budgets are still honoured exactly — the learner counts what
it accepts and stops the actors on the label that hits the cap.

Weights move through shared memory rather than pickled queues: the learner
copies parameters into a shared mirror and bumps a version counter, and actors
reload when the version they hold is stale. For an 18M-parameter network that is
~75MB copied per sync instead of per message.

.. warning::

   Actors are started with the ``spawn`` start method, so every child
   re-imports the ``__main__`` module.  A script that calls this from module
   level therefore re-runs itself in each child and spawns recursively until
   the machine dies.  **Guard the call site**::

       if __name__ == "__main__":
           fit_online_student(config, ...)

   ``solve.py`` and ``paradigm_b/cli/compare.py`` already do; anything under
   ``scratchpad/`` or a notebook must too.  :class:`ActorPool` raises rather
   than spawning if it detects it is already running inside an actor.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from paradigm_b.holdem.arms_common.situations import StreetMix, sample_trajectory_start
from paradigm_b.holdem.data.sampling import SituationConfig
from paradigm_b.holdem.net.leaf_values import NetLeafValues
from paradigm_b.holdem.net.policy import (
    HoldemPolicyNet,
    HoldemPolicyNetConfig,
    PolicyExample,
)
from paradigm_b.holdem.net.value_net import HoldemValueNet, HoldemValueNetConfig
from paradigm_b.holdem.data.store import encode_board
from paradigm_b.holdem.selfplay import Example, HoldemSelfPlayConfig, collect_trajectory


@dataclass
class ActorBatch:
    """One trajectory's worth of labels, plus what producing it cost."""

    features: np.ndarray
    masks: np.ndarray
    targets: np.ndarray
    # The board each label was solved on, as card ids padded with ``-1``, not
    # just how many cards it had.  The replay buffer stores the board *instead
    # of* the 1,326-wide mask and rebuilds the mask from it, so the cards have
    # to survive the trip from the actor process; the street the journal wants
    # is recoverable from them and the reverse is not.
    boards: np.ndarray
    leaf_evaluations: int
    solver_calls: int
    weight_version: int
    # D_pi, grouped per label so that clipping a batch against the label budget
    # drops the right policy targets with it.  Empty unless the run trains
    # theta_pi; the targets are quantised before they are sent, because a
    # trajectory carries one per decision node and they are four times the size
    # of a value label each.
    policies: List[Tuple[PolicyExample, ...]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.features)


class _CountingLeaves:
    """Per-actor tally; the learner sums them as batches arrive."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.leaf_evaluations = 0
        self.solver_calls = 0

    def __call__(self, states):
        self.solver_calls += 1
        self.leaf_evaluations += len(states)
        return self.inner(states)

    def reset(self) -> Tuple[int, int]:
        counts = (self.leaf_evaluations, self.solver_calls)
        self.leaf_evaluations = self.solver_calls = 0
        return counts


class SharedWeights:
    """A shared-memory mirror of the learner's parameters, plus a version.

    Actors poll ``version``; when it moves they copy the tensors into their own
    network.  A lock is held on both sides only for the copy itself, so an actor
    can never observe a half-written parameter set — cheap, because syncing is
    rare relative to solving.
    """

    def __init__(self, net: torch.nn.Module) -> None:
        self.tensors: Dict[str, torch.Tensor] = {
            name: value.detach().cpu().clone().share_memory_()
            for name, value in net.state_dict().items()
        }
        self.version = mp.Value("i", 0)
        self.lock = mp.Lock()

    def publish(self, net: torch.nn.Module) -> int:
        # The device-to-host transfer happens *outside* the lock.  With the
        # learner on ``mps`` this is ~75MB coming back across the bus, and
        # holding the lock through it stalls every actor that happens to check
        # for new weights meanwhile -- for the whole transfer, not for the
        # memcpy that actually needs exclusion.  On a CPU learner ``.cpu()`` is
        # a no-op view and this costs nothing.
        state = {name: value.detach().cpu() for name, value in net.state_dict().items()}
        with self.lock:
            for name, tensor in self.tensors.items():
                tensor.copy_(state[name])
            self.version.value += 1
            return self.version.value

    def load_into(self, net: torch.nn.Module) -> int:
        with self.lock:
            version = self.version.value
            # ``load_state_dict`` copies into the module's parameters, so the
            # shared tensors can be handed to it directly.  Cloning them first
            # made every sync copy the whole network twice -- once into the
            # throwaway dict, once out of it.
            net.load_state_dict(self.tensors)
        return version


def actor_loop(
    worker_index: int,
    shared: SharedWeights,
    outbox: mp.Queue,
    stop: mp.Event,
    net_config: HoldemValueNetConfig,
    self_play: HoldemSelfPlayConfig,
    situations: SituationConfig,
    street_mix: StreetMix,
    seed: int,
    device: str,
    shared_policy: Optional[SharedWeights] = None,
    policy_config: Optional[HoldemPolicyNetConfig] = None,
    preflop_start: bool = True,
) -> None:
    """One actor: pull weights, run Algorithm 2, push labels, repeat.

    When the run trains theta_pi the actor holds a second network, synced the
    same way, because ``INITIALIZE_POLICY`` needs it *while generating* — a
    warm start read from a policy the learner published ten thousand steps ago
    would warm-start search from a stale profile.
    """
    torch.set_num_threads(1)  # actors are parallel; don't fight over cores
    rng = np.random.default_rng(seed + worker_index)
    net = HoldemValueNet(net_config)
    version = shared.load_into(net)
    net.eval()
    leaves = _CountingLeaves(NetLeafValues(net, device=device))

    policy_net: Optional[HoldemPolicyNet] = None
    policy_version = -1
    if shared_policy is not None and policy_config is not None:
        policy_net = HoldemPolicyNet(policy_config)
        policy_version = shared_policy.load_into(policy_net)
        policy_net.eval()

    while not stop.is_set():
        if shared.version.value != version:
            version = shared.load_into(net)
            net.eval()
        if policy_net is not None and shared_policy.version.value != policy_version:
            policy_version = shared_policy.load_into(policy_net)
            policy_net.eval()

        space, root, reach, _ = sample_trajectory_start(
            rng, situations, street_mix, preflop_start
        )
        examples: Sequence[Example] = collect_trajectory(
            leaves,
            space,
            root,
            self_play,
            rng,
            reach=reach,
            policy_net=policy_net,
            device=device,
        )
        if not examples:
            continue
        leaf_evaluations, solver_calls = leaves.reset()
        batch = ActorBatch(
            features=np.stack([e.features for e in examples]).astype(np.float32),
            masks=np.stack([e.mask for e in examples]).astype(np.float32),
            targets=np.stack([e.values for e in examples]).astype(np.float32),
            boards=np.stack([encode_board(e.board) for e in examples]),
            leaf_evaluations=leaf_evaluations,
            solver_calls=solver_calls,
            weight_version=version,
            policies=[
                tuple(policy.quantised() for policy in e.policies) for e in examples
            ],
        )
        while not stop.is_set():
            try:
                outbox.put(batch, timeout=0.5)
                break
            except queue.Full:  # learner is behind; check the stop flag and retry
                continue


class ActorPool:
    """Owns the actor processes and the queue they publish into."""

    def __init__(
        self,
        workers: int,
        net: HoldemValueNet,
        net_config: HoldemValueNetConfig,
        self_play: HoldemSelfPlayConfig,
        situations: SituationConfig,
        street_mix: StreetMix,
        seed: int,
        device: str = "cpu",
        queue_size: int = 64,
        policy_net: Optional[HoldemPolicyNet] = None,
        policy_config: Optional[HoldemPolicyNetConfig] = None,
        preflop_start: bool = True,
    ) -> None:
        if mp.parent_process() is not None:
            raise RuntimeError(
                "ActorPool was constructed inside a child process.  With the "
                "'spawn' start method each actor re-imports __main__, so a "
                "call at module level spawns recursively until the machine "
                "dies.  Wrap the call site in `if __name__ == \"__main__\":`."
            )
        context = mp.get_context("spawn")
        self.workers = workers
        self.shared = SharedWeights(net)
        self.shared_policy = None if policy_net is None else SharedWeights(policy_net)
        self.queue: mp.Queue = context.Queue(maxsize=queue_size)
        self.stop = context.Event()
        self.processes = [
            context.Process(
                target=actor_loop,
                args=(
                    index,
                    self.shared,
                    self.queue,
                    self.stop,
                    net_config,
                    self_play,
                    situations,
                    street_mix,
                    seed,
                    device,
                    self.shared_policy,
                    policy_config,
                    preflop_start,
                ),
                daemon=True,
            )
            for index in range(workers)
        ]

    def start(self) -> "ActorPool":
        for process in self.processes:
            process.start()
        return self

    def publish(self, net: HoldemValueNet, policy_net: Optional[HoldemPolicyNet] = None) -> int:
        version = self.shared.publish(net)
        if policy_net is not None and self.shared_policy is not None:
            self.shared_policy.publish(policy_net)
        return version

    def drain(self, timeout: float = 0.1, limit: int = 256) -> List[ActorBatch]:
        """Whatever the actors have produced; blocks briefly if nothing is ready."""
        batches: List[ActorBatch] = []
        try:
            batches.append(self.queue.get(timeout=timeout))
        except queue.Empty:
            return batches
        while len(batches) < limit:
            try:
                batches.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return batches

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the actors and drain the queue so no process blocks on a put."""
        self.stop.set()
        deadline_drains = 64
        for _ in range(deadline_drains):
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        for process in self.processes:
            process.join(timeout=timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=timeout)
        self.queue.close()

    def __enter__(self) -> "ActorPool":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.shutdown()
