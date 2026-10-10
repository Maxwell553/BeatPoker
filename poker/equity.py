"""Equity calculation: exact on later streets, Monte Carlo preflop/flop vs wide ranges."""

from __future__ import annotations

import random
from itertools import combinations
from typing import Iterable, List, Optional, Sequence, Tuple

from poker.evaluator import evaluate

# Combo index for 1326 hole-card combinations. c1 < c2 in 0..51.
N_COMBOS = 1326


def combo_index(c1: int, c2: int) -> int:
    if c1 > c2:
        c1, c2 = c2, c1
    return c1 * (103 - c1) // 2 + (c2 - c1 - 1)


def combo_from_index(idx: int) -> Tuple[int, int]:
    # Inverse of combo_index. Solve a*(103-a)/2 <= idx.
    # Linear scan is fine (52 iterations).
    for a in range(52):
        n = 51 - a
        if idx < n:
            return a, a + 1 + idx
        idx -= n
    raise IndexError("combo index out of range")


_COMBO_TABLE: Optional[List[Tuple[int, int]]] = None


def all_combos() -> List[Tuple[int, int]]:
    global _COMBO_TABLE
    if _COMBO_TABLE is None:
        _COMBO_TABLE = [combo_from_index(i) for i in range(N_COMBOS)]
    return _COMBO_TABLE


def blocked(combo: Tuple[int, int], dead: Sequence[int]) -> bool:
    a, b = combo
    return a in dead or b in dead


def evaluate_showdown(hole: Sequence[int], board: Sequence[int]) -> int:
    return evaluate(list(hole) + list(board))


def _winner(h1: Sequence[int], h2: Sequence[int], board: Sequence[int]) -> int:
    v1 = evaluate(list(h1) + list(board))
    v2 = evaluate(list(h2) + list(board))
    if v1 > v2:
        return 1
    if v1 < v2:
        return -1
    return 0


def enumerate_equity(
    hero: Sequence[int],
    villain_combos: Sequence[Tuple[int, int]],
    board: Sequence[int],
    dead: Sequence[int] = (),
    villain_weights: Optional[Sequence[float]] = None,
) -> Tuple[float, float, float]:
    """Exact equity of `hero` vs a weighted villain range given a board (3–5 cards).

    Returns (win, tie, equity) where equity = win + tie/2, each in [0,1].
    Remaining board cards are fully enumerated.
    """
    dead_set = set(dead) | set(hero) | set(board)
    remaining = [c for c in range(52) if c not in dead_set]
    need = 5 - len(board)
    w_win = 0.0
    w_tie = 0.0
    w_tot = 0.0

    if villain_weights is None:
        villain_weights = [1.0] * len(villain_combos)

    if need == 0:
        hv = evaluate(list(hero) + list(board))
        for comb, w in zip(villain_combos, villain_weights):
            if w <= 0 or comb[0] in dead_set or comb[1] in dead_set:
                continue
            vv = evaluate([comb[0], comb[1]] + list(board))
            if hv > vv:
                w_win += w
            elif hv == vv:
                w_tie += w
            w_tot += w
    elif need == 1:
        hv_cache_hero = None  # computed per river
        for comb, w in zip(villain_combos, villain_weights):
            if w <= 0 or comb[0] in dead_set or comb[1] in dead_set:
                continue
            blocked_v = {comb[0], comb[1]}
            win = tie = tot = 0
            for r in remaining:
                if r in blocked_v:
                    continue
                b = list(board) + [r]
                hv = evaluate(list(hero) + b)
                vv = evaluate([comb[0], comb[1]] + b)
                if hv > vv:
                    win += 1
                elif hv == vv:
                    tie += 1
                tot += 1
            if tot:
                w_win += w * (win / tot)
                w_tie += w * (tie / tot)
                w_tot += w
    else:
        # Flop: C(45,2) ~ 990 runouts per villain combo. Exact is feasible for
        # modest ranges; callers should cap range size or use MC.
        for comb, w in zip(villain_combos, villain_weights):
            if w <= 0 or comb[0] in dead_set or comb[1] in dead_set:
                continue
            blocked_v = {comb[0], comb[1]}
            avail = [c for c in remaining if c not in blocked_v]
            win = tie = tot = 0
            for t, r in combinations(avail, 2):
                b = list(board) + [t, r]
                hv = evaluate(list(hero) + b)
                vv = evaluate([comb[0], comb[1]] + b)
                if hv > vv:
                    win += 1
                elif hv == vv:
                    tie += 1
                tot += 1
            if tot:
                w_win += w * (win / tot)
                w_tie += w * (tie / tot)
                w_tot += w

    if w_tot <= 0:
        return 0.5, 0.0, 0.5
    return w_win / w_tot, w_tie / w_tot, (w_win + 0.5 * w_tie) / w_tot


def _sample_from_range(
    combos: Sequence[Tuple[int, int]],
    weights: Sequence[float],
    dead: set,
    rng,
) -> Optional[Tuple[int, int]]:
    pool = [(c, w) for c, w in zip(combos, weights) if w > 0 and c[0] not in dead and c[1] not in dead]
    if not pool:
        return None
    tw = sum(w for _, w in pool)
    u = rng.random() * tw
    acc = 0.0
    for c, w in pool:
        acc += w
        if acc >= u:
            return c
    return pool[-1][0]


def monte_carlo_equity(
    hero: Sequence[int],
    villain_combos: Sequence[Tuple[int, int]],
    board: Sequence[int],
    n_samples: int = 800,
    rng: Optional[random.Random] = None,
    villain_weights: Optional[Sequence[float]] = None,
    extra_villains: int = 0,
    extra_villain_ranges: Optional[Sequence[Tuple[Sequence[Tuple[int, int]], Sequence[float]]]] = None,
) -> float:
    """Monte Carlo equity of hero vs one (or more) villain hands from range.

    Extra opponents sample from `extra_villain_ranges` when provided; otherwise
    random remaining cards (legacy fallback).
    """
    rng = rng or random.Random()
    dead0 = set(hero) | set(board)
    if villain_weights is None:
        villain_weights = [1.0] * len(villain_combos)
    # Filter valid combos
    pool: List[Tuple[Tuple[int, int], float]] = []
    for comb, w in zip(villain_combos, villain_weights):
        if w > 0 and comb[0] not in dead0 and comb[1] not in dead0:
            pool.append((comb, w))
    if not pool:
        return 0.5
    tw = sum(w for _, w in pool)
    cdf = []
    acc = 0.0
    for comb, w in pool:
        acc += w / tw
        cdf.append((acc, comb))

    def sample_villain(dead: set) -> Optional[Tuple[int, int]]:
        # A few attempts to respect card removal
        for _ in range(40):
            u = rng.random()
            lo, hi = 0, len(cdf) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if cdf[mid][0] < u:
                    lo = mid + 1
                else:
                    hi = mid
            comb = cdf[lo][1]
            if comb[0] not in dead and comb[1] not in dead:
                return comb
        for comb, _ in pool:
            if comb[0] not in dead and comb[1] not in dead:
                return comb
        return None

    need = 5 - len(board)
    wins = ties = 0
    used = 0
    for _ in range(n_samples):
        dead = set(dead0)
        v = sample_villain(dead)
        if v is None:
            continue
        dead.add(v[0])
        dead.add(v[1])
        villains = [v]
        ok = True
        extras = list(extra_villain_ranges or ())
        for k in range(extra_villains):
            sampled = None
            if k < len(extras):
                cs, ws = extras[k]
                sampled = _sample_from_range(cs, ws, dead, rng)
            if sampled is None:
                avail = [c for c in range(52) if c not in dead]
                if len(avail) < 2:
                    ok = False
                    break
                a, b = rng.sample(avail, 2)
                sampled = (a, b)
            villains.append(sampled)
            dead.add(sampled[0])
            dead.add(sampled[1])
        if not ok:
            continue
        avail = [c for c in range(52) if c not in dead]
        if len(avail) < need:
            continue
        extra_board = rng.sample(avail, need) if need else []
        full = list(board) + extra_board
        hv = evaluate(list(hero) + full)
        best_v = -1
        n_best = 0
        for vh in villains:
            vv = evaluate([vh[0], vh[1]] + full)
            if vv > best_v:
                best_v = vv
                n_best = 1
            elif vv == best_v:
                n_best += 1
        if hv > best_v:
            wins += 1
        elif hv == best_v:
            ties += 1.0 / (n_best + 1)  # split among tying players
            wins += 0.0
        used += 1
    if used == 0:
        return 0.5
    return (wins + ties) / used


def equity_vs_range(
    hero: Sequence[int],
    villain_combos: Sequence[Tuple[int, int]],
    board: Sequence[int],
    n_samples: int = 600,
    rng: Optional[random.Random] = None,
    villain_weights: Optional[Sequence[float]] = None,
    extra_villains: int = 0,
    extra_villain_ranges: Optional[Sequence[Tuple[Sequence[Tuple[int, int]], Sequence[float]]]] = None,
) -> float:
    """Dispatch to exact or MC depending on street / range size / multiway."""
    n_v = sum(1 for i, c in enumerate(villain_combos) if (villain_weights is None or villain_weights[i] > 0) and c[0] not in hero and c[1] not in hero and c[0] not in board and c[1] not in board)
    need = 5 - len(board)
    if extra_villains == 0 and need == 0:
        _, _, eq = enumerate_equity(hero, villain_combos, board, villain_weights=villain_weights)
        return eq
    if extra_villains == 0 and need == 1 and n_v <= 80:
        _, _, eq = enumerate_equity(hero, villain_combos, board, villain_weights=villain_weights)
        return eq
    return monte_carlo_equity(
        hero,
        villain_combos,
        board,
        n_samples=n_samples,
        rng=rng,
        villain_weights=villain_weights,
        extra_villains=extra_villains,
        extra_villain_ranges=extra_villain_ranges,
    )


def hand_strength_percentile(
    hero: Sequence[int],
    hero_range_combos: Sequence[Tuple[int, int]],
    board: Sequence[int],
    hero_weights: Optional[Sequence[float]] = None,
) -> float:
    """Fraction of hero's own range that our hand beats on this board (made-hand percentile)."""
    hv = evaluate(list(hero) + list(board))
    dead = set(hero) | set(board)
    better = worse = 0.0
    if hero_weights is None:
        hero_weights = [1.0] * len(hero_range_combos)
    for comb, w in zip(hero_range_combos, hero_weights):
        if w <= 0 or comb[0] in dead or comb[1] in dead:
            continue
        if set(comb) == set(hero):
            continue
        vv = evaluate([comb[0], comb[1]] + list(board))
        if hv > vv:
            worse += w  # we beat this combo of our range
        elif hv < vv:
            better += w
        else:
            worse += 0.5 * w
            better += 0.5 * w
    tot = better + worse
    if tot <= 0:
        return 0.5
    return worse / tot


def range_vs_range_equity(
    range_a: Sequence[Tuple[int, int]],
    range_b: Sequence[Tuple[int, int]],
    board: Sequence[int],
    n_samples: int = 400,
    rng: Optional[random.Random] = None,
    wa: Optional[Sequence[float]] = None,
    wb: Optional[Sequence[float]] = None,
) -> float:
    """Approximate equity of range A vs range B (A's share of the pot)."""
    rng = rng or random.Random()
    if wa is None:
        wa = [1.0] * len(range_a)
    if wb is None:
        wb = [1.0] * len(range_b)
    dead_board = set(board)
    pa = [(c, w) for c, w in zip(range_a, wa) if w > 0 and c[0] not in dead_board and c[1] not in dead_board]
    pb = [(c, w) for c, w in zip(range_b, wb) if w > 0 and c[0] not in dead_board and c[1] not in dead_board]
    if not pa or not pb:
        return 0.5
    sa, sb = sum(w for _, w in pa), sum(w for _, w in pb)

    def pick(pool, s, dead):
        for _ in range(30):
            u = rng.random() * s
            acc = 0.0
            for c, w in pool:
                acc += w
                if acc >= u and c[0] not in dead and c[1] not in dead:
                    return c
        for c, w in pool:
            if c[0] not in dead and c[1] not in dead:
                return c
        return None

    need = 5 - len(board)
    wins = 0.0
    used = 0
    for _ in range(n_samples):
        ca = pick(pa, sa, dead_board)
        if ca is None:
            continue
        dead = set(dead_board) | set(ca)
        cb = pick(pb, sb, dead)
        if cb is None:
            continue
        dead |= set(cb)
        avail = [c for c in range(52) if c not in dead]
        if len(avail) < need:
            continue
        extra = rng.sample(avail, need) if need else []
        full = list(board) + extra
        va = evaluate([ca[0], ca[1]] + full)
        vb = evaluate([cb[0], cb[1]] + full)
        if va > vb:
            wins += 1
        elif va == vb:
            wins += 0.5
        used += 1
    return wins / used if used else 0.5
