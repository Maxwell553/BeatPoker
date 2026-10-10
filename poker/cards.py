"""Card representation, parsing, and deck.

Cards are integers 0..51:
    rank = card % 13   (0=2, ..., 12=A)
    suit = card // 13  (0=c, 1=d, 2=h, 3=s)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Union

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"
SUIT_NAMES = ("clubs", "diamonds", "hearts", "spades")
RANK_NAMES = (
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Jack",
    "Queen",
    "King",
    "Ace",
)

Card = int


def rank_of(card: int) -> int:
    return card % 13


def suit_of(card: int) -> int:
    return card // 13


def make_card(rank: int, suit: int) -> int:
    return suit * 13 + rank


def card_to_str(card: int) -> str:
    return RANK_CHARS[rank_of(card)] + SUIT_CHARS[suit_of(card)]


def parse_card(s: str) -> int:
    s = s.strip()
    if len(s) != 2:
        raise ValueError(f"Invalid card: {s!r}")
    r = RANK_CHARS.find(s[0].upper())
    su = SUIT_CHARS.find(s[1].lower())
    if r < 0 or su < 0:
        raise ValueError(f"Invalid card: {s!r}")
    return make_card(r, su)


def parse_cards(s: str) -> List[int]:
    """Parse 'AhKd' or 'Ah Kd' or 'Ah,Kd' into a list of cards."""
    s = s.replace(",", " ").replace(" ", "")
    if len(s) % 2 != 0:
        raise ValueError(f"Invalid card string: {s!r}")
    return [parse_card(s[i : i + 2]) for i in range(0, len(s), 2)]


def cards_to_str(cards: Sequence[int]) -> str:
    return " ".join(card_to_str(c) for c in cards)


class Deck:
    """Standard 52-card deck with Fisher-Yates shuffle via a supplied RNG."""

    __slots__ = ("_cards", "_i")

    def __init__(self, rng=None, dead: Iterable[int] = ()):
        import random

        self._cards = [c for c in range(52) if c not in set(dead)]
        self._i = 0
        (rng or random).shuffle(self._cards)

    def remaining(self) -> int:
        return len(self._cards) - self._i

    def draw(self, n: int = 1) -> List[int]:
        if self._i + n > len(self._cards):
            raise RuntimeError("Deck exhausted")
        out = self._cards[self._i : self._i + n]
        self._i += n
        return out

    def draw_one(self) -> int:
        return self.draw(1)[0]


def pretty(cards: Union[int, Sequence[int]]) -> str:
    if isinstance(cards, int):
        return card_to_str(cards)
    return cards_to_str(cards)
