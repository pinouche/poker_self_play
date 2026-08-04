"""Card primitives and hand evaluation."""

import random

import pytest

from common.cards import NUM_CARDS, Card, Deck, parse_cards
from common.hand_evaluator import (
    FLUSH,
    FOUR_OF_A_KIND,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    STRAIGHT,
    STRAIGHT_FLUSH,
    THREE_OF_A_KIND,
    TWO_PAIR,
    evaluate_hand,
)


def hand(*texts):
    return [Card.from_str(t) for t in texts]


# --- cards -----------------------------------------------------------------
def test_card_roundtrip_through_id_and_string():
    for card_id in range(NUM_CARDS):
        card = Card.from_id(card_id)
        assert card.id == card_id
        assert Card.from_str(str(card)) == card
        assert Card.from_dict(card.to_dict()) == card


def test_parse_cards_accepts_dicts_and_strings():
    parsed = parse_cards([{"rank": "4", "suit": "h"}, "Ks", {"rank": "10", "suit": "c"}])
    assert [str(c) for c in parsed] == ["4h", "Ks", "Tc"]


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        Card.from_str("Zx")
    with pytest.raises(ValueError):
        Card(rank=1, suit=0)
    with pytest.raises(ValueError):
        Card(rank=5, suit=9)


def test_deck_deals_without_duplicates():
    deck = Deck(random.Random(0))
    dealt = deck.deal(NUM_CARDS)
    assert len(set(dealt)) == NUM_CARDS
    assert deck.remaining == 0


def test_deck_excludes_preset_cards():
    excluded = hand("As", "Kd")
    deck = Deck(random.Random(0))
    deck.reset(exclude=excluded)
    assert deck.remaining == NUM_CARDS - 2
    assert all(card not in excluded for card in deck.remaining_cards())


# --- hand categories -------------------------------------------------------
@pytest.mark.parametrize(
    "cards,category",
    [
        (hand("As", "Ks", "Qs", "Js", "Ts"), STRAIGHT_FLUSH),
        (hand("5s", "4s", "3s", "2s", "As"), STRAIGHT_FLUSH),  # steel wheel
        (hand("9c", "9d", "9h", "9s", "2c"), FOUR_OF_A_KIND),
        (hand("9c", "9d", "9h", "2s", "2c"), FULL_HOUSE),
        (hand("Ac", "Jc", "9c", "5c", "2c"), FLUSH),
        (hand("9c", "8d", "7h", "6s", "5c"), STRAIGHT),
        (hand("Ac", "2d", "3h", "4s", "5c"), STRAIGHT),  # wheel
        (hand("9c", "9d", "9h", "5s", "2c"), THREE_OF_A_KIND),
        (hand("9c", "9d", "5h", "5s", "2c"), TWO_PAIR),
        (hand("9c", "9d", "7h", "5s", "2c"), PAIR),
        (hand("Ac", "Jd", "9h", "5s", "2c"), HIGH_CARD),
    ],
)
def test_five_card_categories(cards, category):
    assert evaluate_hand(cards)[0] == category


def test_seven_cards_pick_the_best_five():
    # A flush is available, but so are quads; quads must win.
    cards = hand("9c", "9d", "9h", "9s", "2c", "5c", "Kc")
    assert evaluate_hand(cards)[0] == FOUR_OF_A_KIND


def test_full_house_from_two_trips_uses_the_higher_set():
    rank = evaluate_hand(hand("9c", "9d", "9h", "5s", "5c", "5d", "2c"))
    assert rank[0] == FULL_HOUSE
    assert rank[1] == 9 and rank[2] == 5


def test_wheel_straight_ranks_below_six_high_straight():
    wheel = evaluate_hand(hand("Ac", "2d", "3h", "4s", "5c", "Kd", "Qh"))
    six_high = evaluate_hand(hand("2d", "3h", "4s", "5c", "6d", "Kd", "Qh"))
    assert wheel[0] == six_high[0] == STRAIGHT
    assert six_high > wheel


def test_straight_flush_beats_quads_and_quads_beat_full_house():
    straight_flush = evaluate_hand(hand("9s", "8s", "7s", "6s", "5s", "2c", "3d"))
    quads = evaluate_hand(hand("9c", "9d", "9h", "9s", "5s", "2c", "3d"))
    boat = evaluate_hand(hand("9c", "9d", "9h", "5s", "5c", "2c", "3d"))
    assert straight_flush > quads > boat


def test_kickers_break_ties():
    better = evaluate_hand(hand("Ac", "Ad", "Kh", "9s", "5c"))
    worse = evaluate_hand(hand("Ac", "Ad", "Qh", "9s", "5c"))
    assert better > worse


def test_identical_hand_strength_compares_equal():
    # Same ranks, different suits, no flush possible: an exact tie.
    a = evaluate_hand(hand("Ac", "Kd", "9h", "5s", "2c"))
    b = evaluate_hand(hand("Ad", "Kh", "9s", "5c", "2d"))
    assert a == b


def test_evaluator_requires_five_cards():
    with pytest.raises(ValueError):
        evaluate_hand(hand("Ac", "Kd"))
