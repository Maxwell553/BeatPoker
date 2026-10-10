"""169 strategically distinct Hold'em hands and the 13x13 grid."""

from __future__ import annotations

from typing import List, Tuple

RANKS = "23456789TJQKA"  # index 0=2 ... 12=A


def hand169_code(hi: int, lo: int, suited: bool) -> str:
    if hi < lo:
        hi, lo = lo, hi
    if hi == lo:
        return RANKS[hi] + RANKS[lo]
    return RANKS[hi] + RANKS[lo] + ("s" if suited else "o")


def parse_hand169(s: str) -> Tuple[int, int, bool, bool]:
    """Return (hi_rank, lo_rank, suited, is_pair)."""
    s = s.strip()
    if len(s) < 2:
        raise ValueError(s)
    r1 = RANKS.index(s[0].upper())
    r2 = RANKS.index(s[1].upper())
    if r1 == r2:
        return r1, r2, False, True
    suited = True
    if len(s) > 2:
        if s[2] in "oO":
            suited = False
        elif s[2] in "sS":
            suited = True
        else:
            raise ValueError(s)
    else:
        suited = True
    hi, lo = max(r1, r2), min(r1, r2)
    return hi, lo, suited, False


def all_hand169() -> List[str]:
    out = []
    for i in range(12, -1, -1):
        for j in range(12, -1, -1):
            if i == j:
                out.append(RANKS[i] + RANKS[j])
            elif i > j:
                out.append(RANKS[i] + RANKS[j] + "s")
            else:
                out.append(RANKS[j] + RANKS[i] + "o")
    # The nested loop above is messy. Build cleanly:
    out = []
    for hi in range(12, -1, -1):
        out.append(RANKS[hi] + RANKS[hi])
        for lo in range(hi - 1, -1, -1):
            out.append(RANKS[hi] + RANKS[lo] + "s")
            out.append(RANKS[hi] + RANKS[lo] + "o")
    return out


HANDS_169 = all_hand169()
HAND_INDEX = {h: i for i, h in enumerate(HANDS_169)}


def hole_to_169(c1: int, c2: int) -> str:
    r1, r2 = c1 % 13, c2 % 13
    s1, s2 = c1 // 13, c2 // 13
    hi, lo = max(r1, r2), min(r1, r2)
    if r1 == r2:
        return RANKS[r1] + RANKS[r2]
    return RANKS[hi] + RANKS[lo] + ("s" if s1 == s2 else "o")


def combos_of_169(code: str) -> List[Tuple[int, int]]:
    """All 2-card combos (c1 < c2) realizing this 169-hand."""
    hi, lo, suited, is_pair = parse_hand169(code)
    out = []
    if is_pair:
        cards = [s * 13 + hi for s in range(4)]
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = cards[i], cards[j]
                out.append((a, b) if a < b else (b, a))
        return out
    for s1 in range(4):
        for s2 in range(4):
            if suited and s1 != s2:
                continue
            if not suited and s1 == s2:
                continue
            a, b = s1 * 13 + hi, s2 * 13 + lo
            out.append((a, b) if a < b else (b, a))
    return out


# Combo counts: pair=6, suited=4, offsuit=12
def combo_count_169(code: str) -> int:
    hi, lo, suited, is_pair = parse_hand169(code)
    if is_pair:
        return 6
    return 4 if suited else 12


def chen_score(code: str) -> float:
    """Sklansky-Chubukov-adjacent Chen formula (higher is stronger)."""
    hi, lo, suited, is_pair = parse_hand169(code)
    # ranks: 0=2 ... 12=A. Chen values:
    chen_val = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0]
    if is_pair:
        return max(5.0, 2.0 * chen_val[hi])
    score = chen_val[hi]
    if suited:
        score += 2.0
    gap = hi - lo - 1
    if gap == 1:
        score -= 1.0
    elif gap == 2:
        score -= 2.0
    elif gap == 3:
        score -= 4.0
    elif gap >= 4:
        score -= 5.0
    if gap <= 1 and hi >= 9 and lo >= 8:  # T+ connectors / one-gappers
        score += 1.0
    elif gap == 0:
        score += 1.0
    return score


def playability(code: str) -> float:
    """Deep-stack speculative value: suited, connected, pairs, wheel aces."""
    hi, lo, suited, is_pair = parse_hand169(code)
    s = 0.0
    if is_pair:
        s += 1.4 if hi <= 6 else 0.6  # small pairs mine sets
        return s
    if suited:
        s += 0.8
        if hi == 12 and lo <= 4:
            s += 0.5  # A5s-A2s wheel / blocker
        gap = hi - lo - 1
        if gap == 0:
            s += 0.9
        elif gap == 1:
            s += 0.45
        elif gap == 2:
            s += 0.15
    else:
        if hi >= 9 and lo >= 9:
            s += 0.25  # broadway offsuit
    return s


def preflop_score(code: str, stack_bb: float) -> float:
    """Rank hands for opening: high-card strength, plus speculative weight when deep."""
    ch = chen_score(code)
    pl = playability(code)
    hi, lo, suited, is_pair = parse_hand169(code)
    highness = (hi + lo) / 24.0
    # Deep stacks (200–1000bb) upweight implied-odds hands; short stacks downweight them.
    if stack_bb <= 60:
        spec = 0.05
        high_w = 1.15
    elif stack_bb <= 120:
        spec = 0.22
        high_w = 1.0
    elif stack_bb <= 250:
        spec = 0.45
        high_w = 0.95
    elif stack_bb <= 600:
        spec = 0.7
        high_w = 0.9
    else:
        spec = 0.9
        high_w = 0.88
    return high_w * (ch + 3.0 * highness) + spec * pl * 4.0
