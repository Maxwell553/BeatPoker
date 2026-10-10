"""1326-combo range representation and standard range-string parser.

Range strings follow solver convention:
    '22+,ATs+,AJo+,KQs,KQo,76s+'
    'A2s-A5s:0.5,JJ:0.4'
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from poker.equity import N_COMBOS, all_combos, combo_index
from strategy.hands import (
    HANDS_169,
    RANKS,
    combo_count_169,
    combos_of_169,
    hole_to_169,
    parse_hand169,
)

COMBO_LIST = all_combos()


class Range:
    """Weights over the 1326 hole-card combos. Weight 0 = not in range."""

    __slots__ = ("w",)

    def __init__(self, weights: Optional[np.ndarray] = None):
        if weights is None:
            self.w = np.zeros(N_COMBOS, dtype=np.float64)
        else:
            self.w = np.asarray(weights, dtype=np.float64).copy()
            if self.w.shape != (N_COMBOS,):
                raise ValueError("weights must have shape (1326,)")

    @classmethod
    def uniform(cls) -> "Range":
        r = cls()
        r.w[:] = 1.0
        return r

    @classmethod
    def from_string(cls, spec: str) -> "Range":
        r = cls()
        r.add_string(spec)
        return r

    @classmethod
    def from_combos(cls, combos: Iterable[Tuple[int, int]], weight: float = 1.0) -> "Range":
        r = cls()
        for a, b in combos:
            r.w[combo_index(a, b)] = weight
        return r

    @classmethod
    def from_hole(cls, c1: int, c2: int) -> "Range":
        r = cls()
        r.w[combo_index(c1, c2)] = 1.0
        return r

    def copy(self) -> "Range":
        r = Range()
        r.w = self.w.copy()
        return r

    def add_string(self, spec: str) -> None:
        spec = spec.strip()
        if not spec:
            return
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            weight = 1.0
            if ":" in part:
                part, w = part.split(":", 1)
                weight = float(w)
                part = part.strip()
            for code in expand_token(part):
                for a, b in combos_of_169(code):
                    self.w[combo_index(a, b)] = weight

    def mask_dead(self, dead: Sequence[int]) -> "Range":
        r = self.copy()
        if not dead:
            return r
        dead_set = set(dead)
        for i, (a, b) in enumerate(COMBO_LIST):
            if r.w[i] > 0 and (a in dead_set or b in dead_set):
                r.w[i] = 0.0
        return r

    def scale(self, factor: float) -> "Range":
        r = self.copy()
        r.w *= factor
        return r

    def union(self, other: "Range") -> "Range":
        r = self.copy()
        r.w = np.maximum(r.w, other.w)
        return r

    def plus(self, other: "Range") -> "Range":
        r = self.copy()
        r.w = r.w + other.w
        return r

    def intersect_weights(self, other: "Range") -> "Range":
        r = self.copy()
        r.w = r.w * (other.w > 0)
        return r

    def mul_range(self, other: "Range") -> "Range":
        r = self.copy()
        r.w = r.w * other.w
        return r

    def normalize(self) -> "Range":
        r = self.copy()
        s = r.w.sum()
        if s > 0:
            r.w /= s
        return r

    def total(self) -> float:
        return float(self.w.sum())

    def n_combos(self) -> float:
        return float((self.w > 1e-12).sum())

    def weighted_combos(self) -> float:
        return float(self.w.sum())

    def combos(self, min_w: float = 1e-12) -> List[Tuple[int, int]]:
        return [COMBO_LIST[i] for i, w in enumerate(self.w) if w > min_w]

    def combo_weights(self, min_w: float = 1e-12) -> Tuple[List[Tuple[int, int]], List[float]]:
        cs, ws = [], []
        for i, w in enumerate(self.w):
            if w > min_w:
                cs.append(COMBO_LIST[i])
                ws.append(float(w))
        return cs, ws

    def contains_hole(self, c1: int, c2: int) -> float:
        return float(self.w[combo_index(c1, c2)])

    def percentile_by_169(self) -> Dict[str, float]:
        """Average weight of each 169-hand (0-1)."""
        out = {}
        for code in HANDS_169:
            idxs = [combo_index(a, b) for a, b in combos_of_169(code)]
            out[code] = float(np.mean(self.w[idxs]))
        return out

    def frac_of_full(self) -> float:
        return float(self.w.sum() / N_COMBOS)


def expand_token(tok: str) -> List[str]:
    """Expand 'ATs+', '22+', 'A2s-A5s', 'KQo', '76s+' into a list of 169-codes."""
    tok = tok.strip()
    if not tok:
        return []
    plus = tok.endswith("+")
    if plus:
        tok = tok[:-1]
    if "-" in tok:
        a, b = tok.split("-", 1)
        return _expand_dash(a, b)

    hi, lo, suited, is_pair = parse_hand169(tok)
    if not plus:
        return [_code(hi, lo, suited, is_pair)]
    out = []
    if is_pair:
        for r in range(lo, 13):
            out.append(RANKS[r] + RANKS[r])
        return out
    # suited/offsuit plus: raise the kicker toward the high card
    # ATs+ => ATs, AJs, AQs, AKs. 76s+ => 76s, 87s, 98s, ... KQs (same gap).
    gap = hi - lo
    if gap == 0:
        return [_code(hi, lo, suited, True)]
    # Non-connectors retain the high card: K8o+ includes K9o ... KQo.
    # Only connector chains move both ranks (76s+ includes 87s ... AKs).
    # This matches the native solver's documented range notation.
    if gap > 1:
        for k in range(lo, hi):
            out.append(_code(hi, k, suited, False))
        return out
    # Connector / gappers: keep gap, raise the pair of ranks
    for h in range(hi, 13):
        l = h - gap
        if l < 0:
            continue
        if h == l:
            continue
        out.append(_code(h, l, suited, False))
    return out


def _expand_dash(a: str, b: str) -> List[str]:
    h1, l1, s1, p1 = parse_hand169(a)
    h2, l2, s2, p2 = parse_hand169(b)
    if p1 and p2:
        lo, hi = min(l1, l2), max(l1, l2)
        return [RANKS[r] + RANKS[r] for r in range(lo, hi + 1)]
    if s1 != s2 or p1 or p2:
        raise ValueError(f"Cannot expand {a}-{b}")
    # A2s-A5s: same high, low from 2 to 5
    if h1 == h2:
        lo, hi = min(l1, l2), max(l1, l2)
        return [_code(h1, k, s1, False) for k in range(lo, hi + 1)]
    # 76s-T9s: same gap
    gap = h1 - l1
    if h2 - l2 != gap:
        raise ValueError(f"Cannot expand {a}-{b}")
    start, end = min(h1, h2), max(h1, h2)
    return [_code(h, h - gap, s1, False) for h in range(start, end + 1)]


def _code(hi: int, lo: int, suited: bool, is_pair: bool) -> str:
    if is_pair or hi == lo:
        return RANKS[hi] + RANKS[lo]
    return RANKS[hi] + RANKS[lo] + ("s" if suited else "o")


def top_percent_range(codes_sorted: Sequence[str], pct: float, stack_bb: float = 100.0) -> Range:
    """Take leading `pct` of combos from a 169 ranking (already sorted best-first)."""
    target = pct * N_COMBOS
    r = Range()
    acc = 0.0
    for code in codes_sorted:
        n = combo_count_169(code)
        if acc >= target:
            break
        remain = target - acc
        w = 1.0 if n <= remain else max(0.0, remain / n)
        for a, b in combos_of_169(code):
            r.w[combo_index(a, b)] = w
        acc += n * w
    return r


def ranked_hands(stack_bb: float) -> List[str]:
    from strategy.hands import preflop_score

    return sorted(HANDS_169, key=lambda c: preflop_score(c, stack_bb), reverse=True)
