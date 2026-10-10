"""Board texture classification for postflop sizing and frequencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from poker.cards import rank_of, suit_of


@dataclass
class Texture:
    paired: bool
    trips: bool
    monotone: bool
    two_tone: bool
    rainbow: bool
    flush_possible: bool
    flush_draw: bool  # exactly 2 of a suit on flop
    straight_possible: bool
    connectedness: float  # 0..1
    highness: float  # 0..1 mean rank
    wetness: float  # 0..1
    ace_high: bool
    king_high: bool
    broadway_count: int
    low_connected: bool


def analyze(board: Sequence[int]) -> Texture:
    ranks = sorted(rank_of(c) for c in board)
    suits = [suit_of(c) for c in board]
    suit_counts = [suits.count(s) for s in range(4)]
    max_suit = max(suit_counts)
    rank_counts = [ranks.count(r) for r in range(13)]
    paired = max(rank_counts) >= 2
    trips = max(rank_counts) >= 3
    monotone = max_suit >= 3 and len(board) >= 3 and (max_suit >= len(board) or max_suit >= 3)
    if len(board) == 3:
        monotone = max_suit == 3
        two_tone = max_suit == 2
        rainbow = max_suit == 1
    else:
        two_tone = max_suit == 2
        rainbow = max_suit <= 1
        monotone = max_suit >= 4 or (len(board) == 3 and max_suit == 3)

    flush_possible = max_suit >= 3
    flush_draw = max_suit == 2

    uniq = sorted(set(ranks))
    # Connectedness: gaps among unique ranks
    connectedness = 0.0
    straight_possible = False
    if len(uniq) >= 2:
        span = uniq[-1] - uniq[0]
        if span <= 4:
            connectedness = 1.0 - (span - (len(uniq) - 1)) / 5.0
        # wheel
        if 12 in uniq and any(r <= 4 for r in uniq):
            connectedness = max(connectedness, 0.55)
        # two-card windows
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                if uniq[j] - uniq[i] <= 4:
                    connectedness = max(connectedness, 0.4)
        # 3 unique within a 5-rank window => straight possible on flop with one pair of holes
        if len(board) >= 3:
            bits = 0
            for r in uniq:
                bits |= 1 << r
            if 12 in uniq:
                bits |= 1 << -1 if False else bits
            # A2345 wheel bits
            wheel_ranks = {12, 0, 1, 2, 3}
            if len(set(uniq) & wheel_ranks) >= 3:
                straight_possible = True
            for start in range(0, 9):
                window = set(range(start, start + 5))
                if len(set(uniq) & window) >= 3:
                    straight_possible = True
                    break

    highness = (sum(ranks) / len(ranks)) / 12.0
    broadway_count = sum(1 for r in ranks if r >= 9)
    ace_high = max(ranks) == 12
    king_high = max(ranks) == 11
    low_connected = max(ranks) <= 8 and connectedness >= 0.5

    wet = 0.0
    if flush_possible:
        wet += 0.45
    elif flush_draw:
        wet += 0.18
    if straight_possible:
        wet += 0.35
    wet += 0.2 * connectedness
    if paired:
        wet += 0.05
    if monotone:
        wet += 0.15
    wetness = max(0.0, min(1.0, wet))

    return Texture(
        paired=paired,
        trips=trips,
        monotone=monotone and max_suit >= 3,
        two_tone=two_tone,
        rainbow=rainbow,
        flush_possible=flush_possible,
        flush_draw=flush_draw,
        straight_possible=straight_possible,
        connectedness=max(0.0, min(1.0, connectedness)),
        highness=highness,
        wetness=wetness,
        ace_high=ace_high,
        king_high=king_high,
        broadway_count=broadway_count,
        low_connected=low_connected,
    )
