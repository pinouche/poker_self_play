"""Suit-and-chip-isomorphic labels: same position, and free of extra search.

The load-bearing test here is the first one.  Re-indexing a feature vector
consistently is easy to get right by accident — every permutation is
self-consistent — so the thing worth asserting is that a relabelled belief state
is one the *solver* agrees about: solve a subgame, solve its relabelling from
scratch, and require the two to produce the same counterfactual values in
permuted order.  Nothing about the second solve knows the first one happened.

Three CFR iterations is enough for that, and not because three is close to
converged — it is not.  The symmetry holds at *every* iteration rather than only
in the limit, so a run that has barely started tests it exactly as well as a
converged one does, for a fraction of the time.  The sweep over all 23
permutations goes through the showdown values directly, which needs no solve at
all.
"""

from __future__ import annotations

import numpy as np
import pytest

from paradigm_b.core.cfr import CFRConfig
from paradigm_b.core.search import SubgameSolver
from paradigm_b.holdem.data.augmentation import (
    ALL_RELABELLINGS,
    MAX_AUGMENTATIONS,
    RELABELLINGS,
    relabel_example,
    relabel_features,
    scale_example,
    transform_batch,
)
from paradigm_b.holdem.data.sampling import SituationConfig
from paradigm_b.holdem.engine.betting import Betting
from paradigm_b.holdem.engine.combos import NUM_COMBOS, board_mask
from paradigm_b.holdem.engine.isomorphism import card_permutation, combo_permutation
from paradigm_b.holdem.engine.public_tree import PublicState, build_turn_tree
from paradigm_b.holdem.engine.showdown import showdown_values
from paradigm_b.holdem.engine.space import TurnEndgameSpace
from paradigm_b.holdem.net.features import (
    BOARD_DIM,
    BOARD_OFFSET,
    POT_CHIPS_INDEX,
    RANGE_DIM,
    STACK_CHIPS_INDEX,
    encode_pbs,
)
from paradigm_b.holdem.net.value_net import HoldemValueNet, HoldemValueNetConfig
from paradigm_b.holdem.selfplay import Example, HoldemSelfPlayConfig
from paradigm_b.holdem.arm2_iterative.student import OnlineStudentConfig, fit_online_student

TINY_NET = HoldemValueNetConfig(hidden_dim=64, num_residual_blocks=1, card_embedding_dim=16)
RIVER_BOARD = (51, 47, 22, 6, 34)  # As Ks 7h 3h 9d
RELABELLING = (2, 0, 3, 1)
STAKE = 20.0


def river_root(board):
    return PublicState(
        betting=Betting(starting_pot=20, stack=100, max_raises=1), board=board
    )


def rename(board, relabelling):
    cards = card_permutation(relabelling)
    return tuple(sorted(int(cards[c]) for c in board))


def move(array, relabelling):
    """Scatter along the combo axis, the way ``canonicalise_range`` does."""
    combos = combo_permutation(relabelling)
    out = np.zeros_like(array)
    out[..., combos] = array
    return out


def random_reach(board, seed):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.random((2, NUM_COMBOS)) * board_mask(board))


def solve(board, reach, iterations=3):
    """Root counterfactual values for a river subgame — no leaves, so exact."""
    solver = SubgameSolver(
        build_turn_tree(river_root(board), depth_limit=None),
        config=CFRConfig.dcfr(),
        space=TurnEndgameSpace(board[:4]),
    )
    solver.solve(reach=reach, iterations=iterations)
    return solver.root_values()


# --- the symmetry itself ----------------------------------------------------
def test_a_relabelled_subgame_solves_to_the_relabelled_values():
    """Rename the suits, re-solve from scratch, get the same numbers moved."""
    reach = random_reach(RIVER_BOARD, 0)
    values = solve(RIVER_BOARD, reach)
    renamed = solve(rename(RIVER_BOARD, RELABELLING), move(reach, RELABELLING))

    assert np.abs(renamed - move(values, RELABELLING)).max() < 1e-9
    assert np.abs(values).max() > 0.0, "a solve that returned nothing proves nothing"


def test_every_suit_permutation_is_a_symmetry():
    """All 24 are usable, not just the one the other tests exercise.

    Through the showdown rather than a solve: it is the only place the board
    and the private hands actually meet, so it is where a relabelling would
    break if it broke anywhere.
    """
    # The draw pool is the whole group, because K copies replace the original
    # rather than joining it, so the identity is a legitimate draw.
    assert MAX_AUGMENTATIONS == 24
    assert len(RELABELLINGS) == 23, "the identity-free tuple is still the other 23"
    reach = random_reach(RIVER_BOARD, 1)[0]
    values = showdown_values(RIVER_BOARD, reach, STAKE)

    for relabelling in ALL_RELABELLINGS:
        renamed = showdown_values(
            rename(RIVER_BOARD, relabelling),
            np.ascontiguousarray(move(reach, relabelling)),
            STAKE,
        )
        assert np.abs(renamed - move(values, relabelling)).max() < 1e-9


# --- what the encoding does and does not move -------------------------------
def make_example(board=RIVER_BOARD, seed=2):
    """A labelled belief state, with real showdown values and no CFR."""
    reach = random_reach(board, seed)
    space = TurnEndgameSpace(board[:4])
    values = np.stack(
        [
            showdown_values(board, reach[1], STAKE),
            showdown_values(board, reach[0], STAKE),
        ]
    )
    return Example(
        features=encode_pbs(space.pbs(river_root(board), reach)),
        mask=board_mask(board).copy(),
        values=values,
        board=board,
    )


def test_relabelling_moves_the_cards_and_leaves_the_betting_alone():
    example = make_example()
    moved = relabel_features(example.features, RELABELLING)
    cards = card_permutation(RELABELLING)

    # Pot, stack, street, betting history: a suit cannot reach any of them.
    after_board = BOARD_OFFSET + BOARD_DIM
    assert np.array_equal(moved[after_board:], example.features[after_board:])

    # The board indicators are the same cards under their new names.
    before = example.features[BOARD_OFFSET:after_board].reshape(3, 52)
    after = moved[BOARD_OFFSET:after_board].reshape(3, 52)
    assert np.array_equal(
        np.flatnonzero(after[0]), np.sort(cards[list(RIVER_BOARD[:3])])
    )
    assert before.sum() == after.sum() == 5

    # Ranges are permuted, not rescaled.
    assert moved[:RANGE_DIM].sum() == pytest.approx(example.features[:RANGE_DIM].sum())


def test_a_relabelled_example_is_legal_on_its_new_board():
    """Mask, values and ranges all land on combos the new board leaves live."""
    example = make_example()
    moved = relabel_example(example, RELABELLING)
    renamed_board = rename(RIVER_BOARD, RELABELLING)

    assert moved.board == renamed_board
    assert np.array_equal(moved.mask, board_mask(renamed_board))
    dead = 1.0 - board_mask(renamed_board)
    assert (moved.values * dead).sum() == pytest.approx(0.0)
    ranges = moved.features[:RANGE_DIM].reshape(2, NUM_COMBOS)
    assert (ranges * dead).sum() == pytest.approx(0.0)
    # Nothing was invented or lost.
    assert moved.values.sum() == pytest.approx(example.values.sum())


def test_relabelling_is_not_the_identity_it_could_be_mistaken_for():
    """Guard against a permutation that quietly does nothing."""
    example = make_example()
    moved = relabel_example(example, RELABELLING)
    assert not np.array_equal(moved.features, example.features)
    assert not np.array_equal(moved.values, example.values)
    assert moved.board != example.board


def test_policy_targets_are_dropped_rather_than_permuted():
    assert relabel_example(make_example(), RELABELLING).policies == ()


def test_chip_isomorphism_scales_only_absolute_chips_and_values():
    example = make_example()
    scaled = scale_example(example, 1.2)
    unchanged = np.ones_like(example.features, dtype=bool)
    unchanged[[POT_CHIPS_INDEX, STACK_CHIPS_INDEX]] = False

    np.testing.assert_array_equal(scaled.features[unchanged], example.features[unchanged])
    np.testing.assert_allclose(
        scaled.features[[POT_CHIPS_INDEX, STACK_CHIPS_INDEX]],
        example.features[[POT_CHIPS_INDEX, STACK_CHIPS_INDEX]] * 1.2,
    )
    np.testing.assert_allclose(scaled.values, example.values * 1.2)
    np.testing.assert_array_equal(scaled.mask, example.mask)
    assert scaled.board == example.board


def test_chip_scale_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        scale_example(make_example(), 0.0)




# --- read-time transforms ---------------------------------------------------
#
# Both isomorphisms are applied when a row is drawn, not when it is stored.
# The tests above pin the symmetry itself on one example at a time; these pin
# that ``transform_batch`` is the same thing in bulk, and that the properties
# the storage layout depends on hold — nothing is written back, no scale
# compounds, and one solve costs one row however many times it is trained on.


def stored(examples):
    """``examples`` as the buffer keeps them: canonical fp16, board not mask."""
    from paradigm_b.holdem.data.store import encode_board

    return (
        np.stack([e.features for e in examples]).astype(np.float16),
        np.stack([e.values for e in examples]).astype(np.float16),
        np.stack([encode_board(e.board) for e in examples]),
    )


class OneTransform:
    """A generator that always draws relabelling ``k`` and chip scale ``s``."""

    def __init__(self, k: int, s: float = 1.0) -> None:
        self.k, self.s = k, s

    def integers(self, low, high, size=None):
        return np.full(size, self.k)

    def uniform(self, low, high, size=None):
        return np.full(size, self.s)


def test_transform_batch_is_the_batched_form_of_the_single_example_helpers():
    """The load-bearing equivalence.

    ``relabel_example`` and ``scale_example`` are checked above against a
    *solver* that re-solved the relabelled subgame from scratch.  Pinning the
    batched path to them carries all of that over rather than restating it, for
    every one of the 24 permutations, including the identity.
    """
    example = make_example()
    features, targets, boards = stored([example])

    for k, relabelling in enumerate(ALL_RELABELLINGS):
        got_f, got_m, got_t = transform_batch(
            features, targets, boards, OneTransform(k)
        )
        want = relabel_example(example, relabelling)
        # fp16 storage, so the comparison is to its precision -- which is
        # relative, and these fixtures carry unnormalised showdown values in
        # the thousands.  What a real label carries is pot-normalised and two
        # orders of magnitude smaller; see
        # ``test_a_realistic_label_survives_the_fp16_round_trip``.
        np.testing.assert_allclose(got_f[0], want.features, rtol=2e-3, atol=1e-3)
        np.testing.assert_allclose(got_t[0], want.values, rtol=2e-3)
        np.testing.assert_array_equal(got_m[0], want.mask)

    got_f, _, got_t = transform_batch(
        features, targets, boards, OneTransform(0, s=1.17)
    )
    want = scale_example(example, 1.17)
    np.testing.assert_allclose(got_f[0], want.features, rtol=3e-3, atol=1e-3)
    np.testing.assert_allclose(got_t[0], want.values, rtol=3e-3)


def test_the_mask_is_rebuilt_from_the_board_rather_than_stored():
    """5 bytes of card ids in place of 5,304 bytes of float32, or 19% of a row.

    The mask has to come back as the mask of the *relabelled* board, not of the
    stored one — a permuted mask over an unpermuted board would be legal-looking
    and silently wrong, since both have the same weight.
    """
    example = make_example()
    features, targets, boards = stored([example])

    for k, relabelling in enumerate(ALL_RELABELLINGS):
        _, mask, _ = transform_batch(features, targets, boards, OneTransform(k))
        cards = card_permutation(relabelling)
        moved = tuple(sorted(int(cards[c]) for c in example.board))
        np.testing.assert_array_equal(mask[0], board_mask(moved))


def test_two_boards_of_different_streets_share_one_fixed_width_field():
    """A flop row and a river row are the same size; ``-1`` marks the rest."""
    flop = make_example(board=RIVER_BOARD[:3], seed=4)
    river = make_example(seed=5)
    features, targets, boards = stored([flop, river])

    _, masks, _ = transform_batch(features, targets, boards, OneTransform(0))
    np.testing.assert_array_equal(masks[0], board_mask(RIVER_BOARD[:3]))
    np.testing.assert_array_equal(masks[1], board_mask(RIVER_BOARD))
    assert masks[0].sum() > masks[1].sum()  # fewer cards block fewer combos


def test_every_draw_reaches_a_different_point_of_the_orbit():
    """What the periodic in-place renewal pass was approximating.

    A stored row used to be pinned to the one transformation it was written
    under, so re-drawing it returned a byte-identical batch member until a
    renewal pass moved it.  Drawing the transformation here instead means the
    orbit is resampled every time, without bound and without a pass.
    """
    features, targets, boards = stored([make_example()])
    rng = np.random.default_rng(0)

    seen = {
        transform_batch(features, targets, boards, rng)[0].tobytes()
        for _ in range(60)
    }
    assert len(seen) > 20, "re-drawing one row must not keep returning one orientation"


def test_sampling_writes_nothing_back_to_the_stored_rows():
    """The property that makes a buffer larger than memory affordable.

    Renewal was an in-place rewrite: at 60M rows, re-transforming 5% of the
    buffer each iteration is ~33 GB of dirty pages per iteration, which on disk
    is more traffic than training itself. A transform that reads and returns a
    copy costs nothing to store.
    """
    features, targets, boards = stored([make_example(seed=s) for s in (2, 3, 4)])
    before = (features.copy(), targets.copy(), boards.copy())

    rng = np.random.default_rng(1)
    for _ in range(20):
        transform_batch(features, targets, boards, rng)

    for after, original in zip((features, targets, boards), before):
        np.testing.assert_array_equal(after, original)


def test_chip_scales_cannot_compound():
    """Why canonical storage removes the bookkeeping renewal needed.

    Applying a fresh draw from [0.75, 1.25] *on top of* a scale a row already
    carries is a random product, not a draw — ``E[log s] < 0``, so repeated
    application walks the pot toward zero.  The in-place version tracked each
    row's accumulated factor and divided it back out; a canonical row has
    nothing to divide out, and every draw is one draw from the interval.
    """
    example = make_example()
    features, targets, boards = stored([example])
    pot = float(example.features[POT_CHIPS_INDEX])

    rng = np.random.default_rng(2)
    for _ in range(200):
        got, _, _ = transform_batch(features, targets, boards, rng)
        ratio = float(got[0, POT_CHIPS_INDEX]) / pot
        assert 0.75 - 1e-2 <= ratio <= 1.25 + 1e-2


def test_the_ablation_serves_rows_exactly_as_stored():
    example = make_example()
    features, targets, boards = stored([example])

    got_f, got_m, got_t = transform_batch(
        features, targets, boards, np.random.default_rng(3), relabel=False, rescale=False
    )
    np.testing.assert_allclose(got_f[0], example.features, rtol=2e-3, atol=1e-3)
    np.testing.assert_allclose(got_t[0], example.values, rtol=2e-3)
    np.testing.assert_array_equal(got_m[0], example.mask)


# --- wired into the loop ----------------------------------------------------
def student(augmentations: int, **kwargs):
    return fit_online_student(
        OnlineStudentConfig(
            # Preflop-rooted (the default), so 7 labels a trajectory: 2 against
            # 5 updates is 2.8 labels per update against the budgets' 2.43,
            # inside the mismatch warning's tolerance.  Tripping it here would
            # be noise about the test rather than about augmentation.
            trajectories_per_iteration=2,
            updates_per_iteration=5,
            batch_size=8,
            self_play=HoldemSelfPlayConfig(search_iterations=3, river_iterations=3),
            situations=SituationConfig(board_cards=4),
            value_net=TINY_NET,
            suit_augmentations=augmentations,
        ),
        label_budget=17,
        update_budget=7,
        net=HoldemValueNet(TINY_NET),
        rng=np.random.default_rng(0),
        **kwargs,
    )


def test_transforms_are_not_charged_to_the_budget():
    """The point of the whole exercise: more data, identical search."""
    plain = student(0)
    augmented = student(2)

    assert augmented.spend.labels == plain.spend.labels == 17
    assert augmented.spend.updates == plain.spend.updates == 7
    # Same budget, same solving — the transforms cost no search at all.
    assert augmented.spend.leaf_evaluations == plain.spend.leaf_evaluations
    assert augmented.spend.solver_calls == plain.spend.solver_calls


def test_one_solve_costs_one_row_whatever_k_says():
    """K is no longer a storage multiplier, which is the 2x in the row budget.

    It used to be the paper's "multiplying distinct training examples by a
    factor of K", implemented as K rows per solve.  Transforming on read gets
    the same multiplication of *distinct* examples out of one row, so the
    buffer holds the label count and nothing else — and the same bytes now
    carry K times as many solved belief states.
    """
    for k in (0, 1, 2, 5):
        assert student(k).history[-1]["buffer"] == 17


def test_the_transform_stream_does_not_perturb_generation():
    """Turning augmentation on must not change which subgames get searched.

    The synchronous loop feeds one generator to both generation and training,
    so a transform drawn from it would advance the stream that picks the next
    situation — and the ablation would silently stop being a comparison.  The
    buffer draws its transforms from a stream of its own to prevent exactly
    this.
    """
    plain = student(0)
    augmented = student(2)

    for a, b in zip(plain.history, augmented.history):
        assert a["generated"] == b["generated"]
        assert a["labels"] == b["labels"]


def test_augmentation_leaves_the_journal_to_real_labels_only(tmp_path):
    from paradigm_b.holdem.arm2_iterative.journal import TrajectoryJournal

    journal_path = tmp_path / "journal"
    result = student(2, journal_path=journal_path)

    assert result.spend.labels == 17
    examples, _ = TrajectoryJournal.open(journal_path).read()
    assert len(examples) == 17
