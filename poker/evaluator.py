"""Fast, exact 5/6/7-card Hold'em evaluator.

Higher integer = stronger hand. Category occupies the top nibble so hands
of different ranks never collide on kickers.

The 7-card path evaluates all C(7,5)=21 five-card subsets — this is the
definition of Texas Hold'em and is therefore exact.
"""

from __future__ import annotations

from typing import Sequence, Tuple

# Categories, high to low. Multiplied by CAT_BASE so kickers never overflow.
HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
TRIPS = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
QUADS = 7
STRAIGHT_FLUSH = 8

CAT_BASE = 1 << 24  # 16,777,216 > 13^5 = 371,293

RANK_CLASS = {
    HIGH_CARD: "High Card",
    PAIR: "Pair",
    TWO_PAIR: "Two Pair",
    TRIPS: "Three of a Kind",
    STRAIGHT: "Straight",
    FLUSH: "Flush",
    FULL_HOUSE: "Full House",
    QUADS: "Four of a Kind",
    STRAIGHT_FLUSH: "Straight Flush",
}

# Precomputed: for a 13-bit rank mask, (is_straight, high_rank).
# high_rank is 3 for the wheel (5-high), 12 for broadway.
_STRAIGHT_TABLE = [None] * 8192


def _build_straight_table() -> None:
    # 5-card windows, Ace-high down to 6-high (ranks 12..4 as the top).
    windows = []
    for high in range(12, 3, -1):  # 12 (AKQJT) ... 4 (65432)
        mask = 0
        for r in range(high - 4, high + 1):
            mask |= 1 << r
        windows.append((mask, high))
    wheel = (1 << 12) | 0b1111  # A,2,3,4,5
    for bits in range(8192):
        found = False
        for mask, high in windows:
            if (bits & mask) == mask:
                _STRAIGHT_TABLE[bits] = (True, high)
                found = True
                break
        if not found:
            if (bits & wheel) == wheel:
                _STRAIGHT_TABLE[bits] = (True, 3)  # 5-high
            else:
                _STRAIGHT_TABLE[bits] = (False, 0)


_build_straight_table()


def _kickers(*ranks: int) -> int:
    """Pack ranks (high first) as base-13 digits."""
    v = 0
    for r in ranks:
        v = v * 13 + r
    return v


def evaluate5(cards: Sequence[int]) -> int:
    """Evaluate exactly five cards. Returns a comparable integer (higher wins)."""
    return _eval_n(cards)


def _eval_n(cards: Sequence[int]) -> int:
    """Exact best-5 evaluation for 5–7 cards via rank counts + suit bits."""
    rc = [0] * 13
    sc = [0, 0, 0, 0]
    sb = [0, 0, 0, 0]
    rb = 0
    for c in cards:
        r = c % 13
        s = c // 13
        rc[r] += 1
        sc[s] += 1
        sb[s] |= 1 << r
        rb |= 1 << r

    is_st, st_high = _STRAIGHT_TABLE[rb]

    flush_val = -1
    for s in range(4):
        if sc[s] >= 5:
            fs, fh = _STRAIGHT_TABLE[sb[s]]
            if fs:
                return STRAIGHT_FLUSH * CAT_BASE + fh
            k0 = k1 = k2 = k3 = k4 = -1
            nkick = 0
            bits = sb[s]
            for r in range(12, -1, -1):
                if bits & (1 << r):
                    if nkick == 0:
                        k0 = r
                    elif nkick == 1:
                        k1 = r
                    elif nkick == 2:
                        k2 = r
                    elif nkick == 3:
                        k3 = r
                    else:
                        k4 = r
                        nkick += 1
                        break
                    nkick += 1
            flush_val = FLUSH * CAT_BASE + _kickers(k0, k1, k2, k3, k4)
            break

    quads = trips1 = trips2 = -1
    pair1 = pair2 = pair3 = -1
    n_pairs = 0
    for r in range(12, -1, -1):
        c = rc[r]
        if c == 4:
            quads = r
        elif c == 3:
            if trips1 < 0:
                trips1 = r
            else:
                trips2 = r
        elif c == 2:
            if n_pairs == 0:
                pair1 = r
            elif n_pairs == 1:
                pair2 = r
            else:
                pair3 = r
            n_pairs += 1

    if quads >= 0:
        k = -1
        for r in range(12, -1, -1):
            if r != quads and rc[r]:
                k = r
                break
        v = QUADS * CAT_BASE + _kickers(quads, k)
        return v if v > flush_val else flush_val

    # Full house: trips + pair, or two trips, or trips + two pair (use best pair)
    if trips1 >= 0 and (n_pairs or trips2 >= 0):
        pr = trips2 if trips2 >= 0 else pair1
        v = FULL_HOUSE * CAT_BASE + _kickers(trips1, pr)
        return v if v > flush_val else flush_val

    if flush_val >= 0:
        return flush_val
    if is_st:
        return STRAIGHT * CAT_BASE + st_high
    if trips1 >= 0:
        k1 = k2 = -1
        nk = 0
        for r in range(12, -1, -1):
            if r != trips1 and rc[r]:
                if nk == 0:
                    k1 = r
                    nk = 1
                else:
                    k2 = r
                    break
        return TRIPS * CAT_BASE + _kickers(trips1, k1, k2)
    if n_pairs >= 2:
        k = -1
        skip1, skip2 = pair1, pair2
        for r in range(12, -1, -1):
            if r != skip1 and r != skip2 and rc[r]:
                k = r
                break
        return TWO_PAIR * CAT_BASE + _kickers(pair1, pair2, k)
    if n_pairs == 1:
        k1 = k2 = k3 = -1
        nk = 0
        for r in range(12, -1, -1):
            if r != pair1 and rc[r]:
                if nk == 0:
                    k1 = r
                elif nk == 1:
                    k2 = r
                else:
                    k3 = r
                    break
                nk += 1
        return PAIR * CAT_BASE + _kickers(pair1, k1, k2, k3)
    k0 = k1 = k2 = k3 = k4 = 0
    nk = 0
    for r in range(12, -1, -1):
        if rc[r]:
            if nk == 0:
                k0 = r
            elif nk == 1:
                k1 = r
            elif nk == 2:
                k2 = r
            elif nk == 3:
                k3 = r
            else:
                k4 = r
                break
            nk += 1
    return HIGH_CARD * CAT_BASE + _kickers(k0, k1, k2, k3, k4)


def evaluate(cards: Sequence[int]) -> int:
    """Best 5-card Hold'em hand from 5, 6, or 7 cards."""
    n = len(cards)
    if n < 5:
        raise ValueError("Need at least 5 cards")
    return _eval_n(cards)


def category(value: int) -> int:
    return value // CAT_BASE


def category_name(value: int) -> str:
    return RANK_CLASS[category(value)]


def compare_hands(a: Sequence[int], b: Sequence[int]) -> int:
    """1 if a wins, -1 if b wins, 0 if tie."""
    va, vb = evaluate(a), evaluate(b)
    if va > vb:
        return 1
    if va < vb:
        return -1
    return 0


def describe(cards: Sequence[int]) -> Tuple[int, str]:
    v = evaluate(cards)
    return v, category_name(v)
