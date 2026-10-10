"""Compiled / vectorized inner loop for live-search payoffs.

The recursive Python CFR in live_solve stays the policy walker. This module
speeds the bottleneck: pairwise (and 3-way) showdown shares over sampled
runouts. Numba is used when installed; otherwise a vectorized numpy path.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from poker.evaluator import CAT_BASE, evaluate

Combo = Tuple[int, int]

_STRAIGHT = np.zeros(8192, dtype=np.int32)
_ST_HIGH = np.zeros(8192, dtype=np.int32)


def _build_straight() -> None:
    windows = []
    for high in range(12, 3, -1):
        mask = 0
        for r in range(high - 4, high + 1):
            mask |= 1 << r
        windows.append((mask, high))
    wheel = (1 << 12) | 0b1111
    for bits in range(8192):
        found = False
        for mask, high in windows:
            if (bits & mask) == mask:
                _STRAIGHT[bits] = 1
                _ST_HIGH[bits] = high
                found = True
                break
        if not found and (bits & wheel) == wheel:
            _STRAIGHT[bits] = 1
            _ST_HIGH[bits] = 3


_build_straight()

try:
    import numba as _nb

    @_nb.njit(cache=True)
    def _eval_nb(c0, c1, b0, b1, b2, b3, b4, n_board):
        rc = np.zeros(13, dtype=np.int32)
        sc = np.zeros(4, dtype=np.int32)
        sb = np.zeros(4, dtype=np.int32)
        rb = 0
        cards = np.empty(7, dtype=np.int32)
        cards[0] = c0
        cards[1] = c1
        cards[2] = b0
        cards[3] = b1
        cards[4] = b2
        n = 5
        if n_board >= 4:
            cards[5] = b3
            n = 6
        if n_board >= 5:
            cards[6] = b4
            n = 7
        for i in range(n):
            c = cards[i]
            r = c % 13
            s = c // 13
            rc[r] += 1
            sc[s] += 1
            sb[s] |= 1 << r
            rb |= 1 << r
        is_st = _STRAIGHT[rb]
        st_high = _ST_HIGH[rb]
        flush_val = -1
        for s in range(4):
            if sc[s] >= 5:
                fs = _STRAIGHT[sb[s]]
                fh = _ST_HIGH[sb[s]]
                if fs:
                    return 8 * CAT_BASE + fh
                k = np.empty(5, dtype=np.int32)
                nk = 0
                bits = sb[s]
                for r in range(12, -1, -1):
                    if bits & (1 << r):
                        k[nk] = r
                        nk += 1
                        if nk == 5:
                            break
                v = 0
                for t in range(5):
                    v = v * 13 + k[t]
                flush_val = 5 * CAT_BASE + v
                break
        quads = trips1 = trips2 = -1
        pair1 = pair2 = -1
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
                n_pairs += 1
        if quads >= 0:
            k = 0
            for r in range(12, -1, -1):
                if r != quads and rc[r]:
                    k = r
                    break
            v = 7 * CAT_BASE + quads * 13 + k
            return v if v > flush_val else flush_val
        if trips1 >= 0 and (n_pairs or trips2 >= 0):
            pr = trips2 if trips2 >= 0 else pair1
            v = 6 * CAT_BASE + trips1 * 13 + pr
            return v if v > flush_val else flush_val
        if flush_val >= 0:
            return flush_val
        if is_st:
            return 4 * CAT_BASE + st_high
        if trips1 >= 0:
            k1 = k2 = 0
            nk = 0
            for r in range(12, -1, -1):
                if r != trips1 and rc[r]:
                    if nk == 0:
                        k1 = r
                        nk = 1
                    else:
                        k2 = r
                        break
            return 3 * CAT_BASE + (k1 * 13 + k2) + trips1 * 169
        if n_pairs >= 2:
            k = 0
            for r in range(12, -1, -1):
                if r != pair1 and r != pair2 and rc[r]:
                    k = r
                    break
            return 2 * CAT_BASE + (pair1 * 169 + pair2 * 13 + k)
        if n_pairs == 1:
            ks = np.zeros(3, dtype=np.int32)
            nk = 0
            for r in range(12, -1, -1):
                if r != pair1 and rc[r]:
                    ks[nk] = r
                    nk += 1
                    if nk == 3:
                        break
            return 1 * CAT_BASE + pair1 * 2197 + ks[0] * 169 + ks[1] * 13 + ks[2]
        ks = np.zeros(5, dtype=np.int32)
        nk = 0
        for r in range(12, -1, -1):
            if rc[r]:
                ks[nk] = r
                nk += 1
                if nk == 5:
                    break
        v = 0
        for t in range(5):
            v = v * 13 + ks[t]
        return v

    @_nb.njit(cache=True)
    def _fill_vals_nb(combos, boards, out):
        for i in range(combos.shape[0]):
            c0 = combos[i, 0]
            c1 = combos[i, 1]
            for r in range(boards.shape[0]):
                blocked = False
                for k in range(5):
                    x = boards[r, k]
                    if x == c0 or x == c1:
                        blocked = True
                        break
                if blocked:
                    out[i, r] = -1
                else:
                    out[i, r] = _eval_nb(
                        c0, c1,
                        boards[r, 0], boards[r, 1], boards[r, 2],
                        boards[r, 3], boards[r, 4], 5,
                    )

    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False
    _eval_nb = None  # type: ignore
    _fill_vals_nb = None  # type: ignore


def has_numba() -> bool:
    return bool(_HAS_NUMBA)


def _blocks(a: Combo, b: Combo) -> bool:
    return a[0] == b[0] or a[0] == b[1] or a[1] == b[0] or a[1] == b[1]


def _combo_arr(combos: Sequence[Combo]) -> np.ndarray:
    return np.asarray([[c[0], c[1]] for c in combos], dtype=np.int32)


def _values_on_runouts(
    combos: Sequence[Combo],
    board: Sequence[int],
    runouts: Sequence[Sequence[int]],
) -> np.ndarray:
    """(C, R) hand values; -1 if the combo collides with the runout."""
    C = len(combos)
    R = len(runouts)
    out = np.empty((C, R), dtype=np.int64)
    b = list(board)
    if _HAS_NUMBA and R > 0:
        try:
            boards = np.zeros((R, 5), dtype=np.int32)
            for r, ru in enumerate(runouts):
                row = b + list(ru)
                for k, x in enumerate(row[:5]):
                    boards[r, k] = x
            tmp = np.empty((C, R), dtype=np.int64)
            _fill_vals_nb(_combo_arr(combos), boards, tmp)
            return tmp
        except Exception:
            pass
    for i, c in enumerate(combos):
        hole = {c[0], c[1]}
        for r, ru in enumerate(runouts):
            if hole & set(ru):
                out[i, r] = -1
            else:
                out[i, r] = evaluate([c[0], c[1]] + b + list(ru))
    return out


def payoff_matrix_fast(
    hero: Sequence[Combo],
    vill: Sequence[Combo],
    board: Sequence[int],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """P[i,j] = hero pot share in [0,1]. Blocked hole pairs are NaN.

    Evaluates (H+V)*R hands instead of H*V*R.
    """
    H, V = len(hero), len(vill)
    P = np.full((H, V), np.nan, dtype=np.float64)
    dead = set(board)
    need = 5 - len(board)
    block = np.zeros((H, V), dtype=np.bool_)
    for i, hc in enumerate(hero):
        for j, vc in enumerate(vill):
            if _blocks(hc, vc):
                block[i, j] = True

    if need == 0:
        hv = np.array([evaluate([c[0], c[1]] + list(board)) for c in hero], dtype=np.int64)
        vv = np.array([evaluate([c[0], c[1]] + list(board)) for c in vill], dtype=np.int64)
        P = np.where(hv[:, None] > vv[None, :], 1.0, np.where(hv[:, None] < vv[None, :], 0.0, 0.5))
        P = P.astype(np.float64)
        P[block] = np.nan
        return P

    remaining = [c for c in range(52) if c not in dead]
    n_ru = max(4, n_samples)
    runouts: List[List[int]] = []
    for _ in range(n_ru):
        pick = rng.choice(len(remaining), size=need, replace=False)
        runouts.append([remaining[int(x)] for x in pick])

    h_vals = _values_on_runouts(hero, board, runouts)
    v_vals = _values_on_runouts(vill, board, runouts)
    # Compare per runout; ignore blocked samples (-1).
    wins = np.zeros((H, V), dtype=np.float64)
    used = np.zeros((H, V), dtype=np.float64)
    for r in range(len(runouts)):
        hv = h_vals[:, r][:, None]
        vv = v_vals[None, :, r]
        ok = (hv >= 0) & (vv >= 0)
        wins += ok * (hv > vv).astype(np.float64) + 0.5 * ok * (hv == vv).astype(np.float64)
        used += ok.astype(np.float64)
    P = np.divide(wins, used, out=np.full((H, V), np.nan), where=used > 0)
    P[block] = np.nan
    return P


def payoff_tensor3(
    hero: Sequence[Combo],
    v1: Sequence[Combo],
    v2: Sequence[Combo],
    board: Sequence[int],
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pot-share tensors (H, V1, V2) for three-way showdown. Blocked triples are NaN."""
    H, A, B = len(hero), len(v1), len(v2)
    need = 5 - len(board)
    dead = set(board)
    if need == 0:
        hv = np.array([evaluate([c[0], c[1]] + list(board)) for c in hero], dtype=np.int64)
        a = np.array([evaluate([c[0], c[1]] + list(board)) for c in v1], dtype=np.int64)
        b = np.array([evaluate([c[0], c[1]] + list(board)) for c in v2], dtype=np.int64)
        mx = np.maximum(np.maximum(hv[:, None, None], a[None, :, None]), b[None, None, :])
        nwin = (hv[:, None, None] == mx).astype(np.float64)
        nwin += (a[None, :, None] == mx).astype(np.float64)
        nwin += (b[None, None, :] == mx).astype(np.float64)
        s0 = np.where(hv[:, None, None] == mx, 1.0 / nwin, 0.0)
        s1 = np.where(a[None, :, None] == mx, 1.0 / nwin, 0.0)
        s2 = np.where(b[None, None, :] == mx, 1.0 / nwin, 0.0)
    else:
        remaining = [c for c in range(52) if c not in dead]
        n_ru = max(4, n_samples)
        runouts = []
        for _ in range(n_ru):
            pick = rng.choice(len(remaining), size=need, replace=False)
            runouts.append([remaining[int(x)] for x in pick])
        hv = _values_on_runouts(hero, board, runouts)
        av = _values_on_runouts(v1, board, runouts)
        bv = _values_on_runouts(v2, board, runouts)
        acc0 = np.zeros((H, A, B), dtype=np.float64)
        acc1 = np.zeros((H, A, B), dtype=np.float64)
        acc2 = np.zeros((H, A, B), dtype=np.float64)
        used = np.zeros((H, A, B), dtype=np.float64)
        for r in range(len(runouts)):
            h = hv[:, r][:, None, None]
            x = av[None, :, r][:, :, None]
            y = bv[None, None, :, r]
            ok = (h >= 0) & (x >= 0) & (y >= 0)
            mx = np.maximum(np.maximum(h, x), y)
            nwin = (h == mx).astype(np.float64) + (x == mx).astype(np.float64) + (y == mx).astype(np.float64)
            nwin = np.maximum(nwin, 1.0)
            acc0 += ok * np.where(h == mx, 1.0 / nwin, 0.0)
            acc1 += ok * np.where(x == mx, 1.0 / nwin, 0.0)
            acc2 += ok * np.where(y == mx, 1.0 / nwin, 0.0)
            used += ok.astype(np.float64)
        s0 = np.divide(acc0, used, out=np.full_like(acc0, np.nan), where=used > 0)
        s1 = np.divide(acc1, used, out=np.full_like(acc1, np.nan), where=used > 0)
        s2 = np.divide(acc2, used, out=np.full_like(acc2, np.nan), where=used > 0)

    for i, hc in enumerate(hero):
        for j, c1 in enumerate(v1):
            if _blocks(hc, c1):
                s0[i, j, :] = np.nan
                s1[i, j, :] = np.nan
                s2[i, j, :] = np.nan
        for k, c2 in enumerate(v2):
            if _blocks(hc, c2):
                s0[i, :, k] = np.nan
                s1[i, :, k] = np.nan
                s2[i, :, k] = np.nan
    for j, c1 in enumerate(v1):
        for k, c2 in enumerate(v2):
            if _blocks(c1, c2):
                s0[:, j, k] = np.nan
                s1[:, j, k] = np.nan
                s2[:, j, k] = np.nan
    return s0, s1, s2
