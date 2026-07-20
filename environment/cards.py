"""Card primitives.

A card carries an integer ``rank`` in [2, 14] (2..9, T=11? no: T=10, J=11,
Q=12, K=13, A=14) and an integer ``suit`` in [0, 4) indexing ``"cdhs"``.
A dense id in [0, 52) is available for array indexing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"

NUM_RANKS = 13
NUM_SUITS = 4
NUM_CARDS = 52

MIN_RANK = 2
MAX_RANK = 14

_RANK_FROM_CHAR = {c: i + 2 for i, c in enumerate(RANK_CHARS)}
_RANK_FROM_CHAR["10"] = 10  # tolerate the two-character spelling
_SUIT_FROM_CHAR = {c: i for i, c in enumerate(SUIT_CHARS)}


@dataclass(frozen=True, order=True)
class Card:
    rank: int  # 2..14
    suit: int  # 0..3

    def __post_init__(self) -> None:
        if not (MIN_RANK <= self.rank <= MAX_RANK):
            raise ValueError(f"rank out of range: {self.rank}")
        if not (0 <= self.suit < NUM_SUITS):
            raise ValueError(f"suit out of range: {self.suit}")

    @property
    def rank_index(self) -> int:
        """Zero-based rank index, suitable for one-hot encoding."""
        return self.rank - MIN_RANK

    @property
    def id(self) -> int:
        """Dense id in [0, 52)."""
        return self.rank_index * NUM_SUITS + self.suit

    def __str__(self) -> str:
        return f"{RANK_CHARS[self.rank_index]}{SUIT_CHARS[self.suit]}"

    __repr__ = __str__

    # --- construction ------------------------------------------------------
    @staticmethod
    def from_id(card_id: int) -> "Card":
        if not (0 <= card_id < NUM_CARDS):
            raise ValueError(f"card id out of range: {card_id}")
        return Card(rank=card_id // NUM_SUITS + MIN_RANK, suit=card_id % NUM_SUITS)

    @staticmethod
    def from_str(text: str) -> "Card":
        text = text.strip()
        if len(text) < 2:
            raise ValueError(f"cannot parse card: {text!r}")
        rank_part, suit_part = text[:-1], text[-1]
        return Card(rank=parse_rank(rank_part), suit=parse_suit(suit_part))

    @staticmethod
    def from_dict(d: dict) -> "Card":
        """Parse ``{"rank": "4", "suit": "h"}`` as used by the public API."""
        return Card(rank=parse_rank(d["rank"]), suit=parse_suit(d["suit"]))

    def to_dict(self) -> dict:
        return {"rank": RANK_CHARS[self.rank_index], "suit": SUIT_CHARS[self.suit]}


def parse_rank(value) -> int:
    if isinstance(value, int):
        if MIN_RANK <= value <= MAX_RANK:
            return value
        raise ValueError(f"unknown rank: {value!r}")
    key = str(value).strip().upper()
    if key in _RANK_FROM_CHAR:
        return _RANK_FROM_CHAR[key]
    raise ValueError(f"unknown rank: {value!r}")


def parse_suit(value) -> int:
    if isinstance(value, int):
        if 0 <= value < NUM_SUITS:
            return value
        raise ValueError(f"unknown suit: {value!r}")
    key = str(value).strip().lower()
    if key in _SUIT_FROM_CHAR:
        return _SUIT_FROM_CHAR[key]
    raise ValueError(f"unknown suit: {value!r}")


def parse_cards(items: Iterable) -> List[Card]:
    """Parse a list of dicts or strings into cards."""
    out: List[Card] = []
    for item in items:
        if isinstance(item, Card):
            out.append(item)
        elif isinstance(item, dict):
            out.append(Card.from_dict(item))
        else:
            out.append(Card.from_str(str(item)))
    return out


def cards_to_str(cards: Sequence[Card]) -> str:
    return " ".join(str(c) for c in cards)


class Deck:
    """A shuffled 52-card deck with an explicit RNG for reproducibility."""

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self.rng = rng if rng is not None else random.Random()
        self._cards: List[Card] = []
        self.reset()

    def reset(self, exclude: Optional[Iterable[Card]] = None) -> None:
        excluded = set(exclude or ())
        self._cards = [Card.from_id(i) for i in range(NUM_CARDS) if Card.from_id(i) not in excluded]
        self.rng.shuffle(self._cards)

    def deal(self, n: int = 1) -> List[Card]:
        if n > len(self._cards):
            raise RuntimeError("deck exhausted")
        dealt, self._cards = self._cards[:n], self._cards[n:]
        return dealt

    def deal_one(self) -> Card:
        return self.deal(1)[0]

    @property
    def remaining(self) -> int:
        return len(self._cards)

    def remaining_cards(self) -> List[Card]:
        return list(self._cards)
