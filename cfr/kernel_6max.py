"""Numba 6-max MCCFR iterate: compiled eval + full hand walk.

Replaces the Python sixmax_v2 walk so this laptop can train millions of
iterations instead of tens of thousands.
"""

from __future__ import annotations

import numpy as np

from poker.evaluator import CAT_BASE, _STRAIGHT_TABLE
from strategy.hands import HAND_INDEX, hole_to_169

N_PF, N_HANDS, PF_ACTS = 18, 169, 4
N_FL_SPOT, N_FLOP, FL_ACTS = 12, 50, 4

_ST_IS = np.zeros(8192, dtype=np.int8)
_ST_HI = np.zeros(8192, dtype=np.int8)
for _bits in range(8192):
    _ok, _hi = _STRAIGHT_TABLE[_bits]
    _ST_IS[_bits] = 1 if _ok else 0
    _ST_HI[_bits] = int(_hi)

_HID = np.zeros((52, 52), dtype=np.int32)
for _a in range(52):
    for _b in range(52):
        if _a != _b:
            _HID[_a, _b] = HAND_INDEX[hole_to_169(_a, _b)]

try:
    import numba as _nb

    @_nb.njit(cache=True)
    def eval_n(cards, n, st_is, st_hi):
        rc = np.zeros(13, dtype=np.int32)
        sc = np.zeros(4, dtype=np.int32)
        sb = np.zeros(4, dtype=np.int32)
        rb = 0
        for i in range(n):
            c = cards[i]
            r = c % 13
            s = c // 13
            rc[r] += 1
            sc[s] += 1
            sb[s] |= 1 << r
            rb |= 1 << r
        is_st = st_is[rb]
        st_high = st_hi[rb]
        flush_val = -1
        cat_base = 1 << 24
        for s in range(4):
            if sc[s] >= 5:
                fs = st_is[sb[s]]
                fh = st_hi[sb[s]]
                if fs:
                    return 8 * cat_base + fh
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
                v = 0
                for kk in (k0, k1, k2, k3, k4):
                    v = v * 13 + max(kk, 0)
                flush_val = 5 * cat_base + v
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
            v = 7 * cat_base + quads * 13 + k
            if v > flush_val:
                return v
            return flush_val
        if trips1 >= 0 and (n_pairs or trips2 >= 0):
            pr = trips2 if trips2 >= 0 else pair1
            v = 6 * cat_base + trips1 * 13 + pr
            if v > flush_val:
                return v
            return flush_val
        if flush_val >= 0:
            return flush_val
        if is_st:
            return 4 * cat_base + st_high
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
            return 3 * cat_base + (trips1 * 13 + k1) * 13 + k2
        if n_pairs >= 2:
            k = 0
            for r in range(12, -1, -1):
                if r != pair1 and r != pair2 and rc[r]:
                    k = r
                    break
            return 2 * cat_base + (pair1 * 13 + pair2) * 13 + k
        if n_pairs == 1:
            k1 = k2 = k3 = 0
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
            return 1 * cat_base + ((pair1 * 13 + k1) * 13 + k2) * 13 + k3
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
        v = 0
        for kk in (k0, k1, k2, k3, k4):
            v = v * 13 + kk
        return v

    @_nb.njit(cache=True)
    def _rm(regrets):
        n = regrets.shape[0]
        out = np.empty(n, dtype=np.float64)
        s = 0.0
        for i in range(n):
            v = regrets[i]
            if v < 0.0:
                v = 0.0
            out[i] = v
            s += v
        if s <= 1e-15:
            inv = 1.0 / n
            for i in range(n):
                out[i] = inv
            return out
        for i in range(n):
            out[i] /= s
        return out

    @_nb.njit(cache=True)
    def _upd(reg, ssum, sig, ua, u, t):
        n = reg.shape[0]
        for i in range(n):
            ssum[i] += t * sig[i]
            reg[i] += ua[i] - u

    @_nb.njit(cache=True)
    def _bin_right(edges, value):
        lo = 0
        hi = edges.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if edges[mid] <= value:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @_nb.njit(cache=True)
    def _draw_flags(h0, h1, b0, b1, b2):
        hs0, hs1 = h0 // 13, h1 // 13
        bs0, bs1, bs2 = b0 // 13, b1 // 13, b2 // 13
        fd = 0
        for s in range(4):
            hc = (1 if hs0 == s else 0) + (1 if hs1 == s else 0)
            bc = (1 if bs0 == s else 0) + (1 if bs1 == s else 0) + (1 if bs2 == s else 0)
            if hc >= 1 and hc + bc == 4:
                fd = 1
                break
        has = np.zeros(13, dtype=np.int8)
        has[h0 % 13] = 1
        has[h1 % 13] = 1
        has[b0 % 13] = 1
        has[b1 % 13] = 1
        has[b2 % 13] = 1
        tmp = np.empty(14, dtype=np.int32)
        n = 0
        if has[12]:
            tmp[n] = -1
            n += 1
        for r in range(13):
            if has[r]:
                tmp[n] = r
                n += 1
        for i in range(n):
            for j in range(i + 1, n):
                if tmp[j] < tmp[i]:
                    x = tmp[i]
                    tmp[i] = tmp[j]
                    tmp[j] = x
        oesd = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i + 1) == 4 and (tmp[j] - tmp[i]) == 3:
                    oesd = 1
        return fd, oesd

    @_nb.njit(cache=True)
    def flop_bkt(h0, h1, f0, f1, f2, edges, st_is, st_hi):
        cards = np.empty(5, dtype=np.int32)
        cards[0] = h0
        cards[1] = h1
        cards[2] = f0
        cards[3] = f1
        cards[4] = f2
        v = eval_n(cards, 5, st_is, st_hi)
        made = v / ((1 << 24) * 9.0)
        b = _bin_right(edges, made)
        fd, oesd = _draw_flags(h0, h1, f0, f1, f2)
        extra = 1 if (fd == 1 or oesd == 1) else 0
        return int(min(49, (b % 25) * 2 + extra))

    @_nb.njit(cache=True)
    def iterate_n(
        reg_pf, sum_pf, reg_fl, sum_fl, cv_num, cv_den,
        hid_tab, st_is, st_hi, flop_edges, stack, n_iters, t0, seed,
    ):
        np.random.seed(seed)
        deck = np.empty(52, dtype=np.int32)
        holes = np.empty((6, 2), dtype=np.int32)
        cards7a = np.empty(7, dtype=np.int32)
        cards7b = np.empty(7, dtype=np.int32)
        live = np.empty(6, dtype=np.int8)
        inv = np.empty(6, dtype=np.float64)
        ua = np.empty(4, dtype=np.float64)
        for it in range(n_iters):
            t = t0 + it + 1.0
            for i in range(52):
                deck[i] = i
            for i in range(51, 34, -1):
                j = np.random.randint(0, i + 1)
                tmp = deck[i]
                deck[i] = deck[j]
                deck[j] = tmp
            for s in range(6):
                holes[s, 0] = deck[2 * s]
                holes[s, 1] = deck[2 * s + 1]
            f0, f1, f2 = deck[12], deck[13], deck[14]
            tcard, rcard = deck[15], deck[16]
            for upd in range(6):
                for s in range(6):
                    live[s] = 1
                    inv[s] = 0.0
                inv[4] = 0.5
                inv[5] = 1.0
                current = 1.0
                n_raises = 0
                for s in range(6):
                    if live[s] == 0:
                        continue
                    if n_raises == 0 and inv[s] >= current - 1e-9 and s == 5:
                        continue
                    seat = s
                    if n_raises <= 0:
                        spot = seat
                    elif n_raises == 1:
                        spot = 6 + seat
                    else:
                        spot = 12 + seat
                    h = hid_tab[holes[s, 0], holes[s, 1]]
                    sig = _rm(reg_pf[spot, h])
                    eq = 0.50
                    pot = inv[0] + inv[1] + inv[2] + inv[3] + inv[4] + inv[5] + 1.0
                    extra = stack - current
                    if extra > pot * 0.70:
                        extra = pot * 0.70
                    ua[0] = -inv[s]
                    call_amt = current
                    ua[1] = eq * (pot + max(0.0, current - inv[s]) + extra) - call_amt - 0.5 * extra
                    raise_to = 2.5 if n_raises == 0 else current * 2.7
                    if raise_to > stack:
                        raise_to = stack
                    ua[2] = eq * (pot + raise_to) - raise_to
                    ua[3] = eq * (2.0 * stack) - stack
                    u = 0.0
                    for i in range(4):
                        u += sig[i] * ua[i]
                    if s == upd:
                        _upd(reg_pf[spot, h], sum_pf[spot, h], sig, ua, u, t)
                    csum = 0.0
                    roll = np.random.random()
                    ai = 3
                    for i in range(4):
                        csum += sig[i]
                        if roll <= csum:
                            ai = i
                            break
                    if ai == 0:
                        live[s] = 0
                        continue
                    if ai == 3:
                        inv[s] = stack
                        current = stack
                        n_raises += 1
                        continue
                    if ai == 2:
                        inv[s] = raise_to
                        current = raise_to
                        n_raises += 1
                        continue
                    inv[s] = current
                n_live = 0
                a = 0
                b = 0
                for s in range(6):
                    if live[s]:
                        if n_live == 0:
                            a = s
                        elif n_live == 1:
                            b = s
                        n_live += 1
                if n_live < 2:
                    continue
                pot_type = 0 if n_raises <= 1 else (1 if n_raises == 2 else 2)
                pot = inv[0] + inv[1] + inv[2] + inv[3] + inv[4] + inv[5]
                cards7a[0] = holes[a, 0]
                cards7a[1] = holes[a, 1]
                cards7a[2] = f0
                cards7a[3] = f1
                cards7a[4] = f2
                cards7a[5] = tcard
                cards7a[6] = rcard
                cards7b[0] = holes[b, 0]
                cards7b[1] = holes[b, 1]
                cards7b[2] = f0
                cards7b[3] = f1
                cards7b[4] = f2
                cards7b[5] = tcard
                cards7b[6] = rcard
                va = eval_n(cards7a, 7, st_is, st_hi)
                vb = eval_n(cards7b, 7, st_is, st_hi)
                eq_ab = 1.0 if va > vb else (0.0 if va < vb else 0.5)
                extra = stack - inv[a]
                if extra > pot * 0.70:
                    extra = pot * 0.70
                u_leaf_a = eq_ab * (pot + extra) - inv[a] - 0.5 * extra
                ha = flop_bkt(holes[a, 0], holes[a, 1], f0, f1, f2, flop_edges, st_is, st_hi)
                hb = flop_bkt(holes[b, 0], holes[b, 1], f0, f1, f2, flop_edges, st_is, st_hi)
                nodes_p = np.empty(4, dtype=np.int32)
                nodes_p[0] = b
                nodes_p[1] = a
                nodes_p[2] = a
                nodes_p[3] = b
                for node in range(4):
                    player = nodes_p[node]
                    priv = ha if player == a else hb
                    sp = pot_type * 4 + node
                    if sp > 11:
                        sp = 11
                    sig = _rm(reg_fl[sp, priv])
                    stk = stack - inv[player]
                    u_leaf = u_leaf_a if player == a else -u_leaf_a
                    ua[0] = 0.0 if node < 2 else -inv[player]
                    ex = stk
                    if ex > pot * 0.55:
                        ex = pot * 0.55
                    ua[1] = u_leaf + 0.15 * ex
                    ua[2] = u_leaf + 0.35 * ex
                    jam_x = stk
                    if jam_x > pot:
                        jam_x = pot
                    ua[3] = u_leaf + 0.45 * jam_x
                    u = 0.0
                    for i in range(4):
                        u += sig[i] * ua[i]
                    if player == upd:
                        _upd(reg_fl[sp, priv], sum_fl[sp, priv], sig, ua, u, t)
                pt = pot_type
                if pt > 2:
                    pt = 2
                cv_num[ha, hb, pt] += u_leaf_a / max(pot, 1.0)
                cv_den[ha, hb, pt] += 1.0
            if int(t) % 400 == 0:
                a = t ** 1.5 / (t ** 1.5 + 1.0)
                for i in range(reg_pf.shape[0]):
                    for j in range(reg_pf.shape[1]):
                        for k in range(reg_pf.shape[2]):
                            reg_pf[i, j, k] *= a
                for i in range(reg_fl.shape[0]):
                    for j in range(reg_fl.shape[1]):
                        for k in range(reg_fl.shape[2]):
                            reg_fl[i, j, k] *= a
        return t0 + n_iters

    COMPILED = True
except Exception:  # pragma: no cover
    COMPILED = False
    iterate_n = None
    flop_bkt = None


def hid_table() -> np.ndarray:
    return _HID


def straight_tables():
    return _ST_IS, _ST_HI


def flop_edges() -> np.ndarray:
    from strategy.card_abs import load_clusters

    cl = load_clusters()
    return np.ascontiguousarray(cl["flop_edges"], dtype=np.float64)


def warmup() -> bool:
    """Compile kernels in this process so worker caches hit."""
    if not COMPILED or iterate_n is None:
        return False
    edges = flop_edges()
    st_is, st_hi = straight_tables()
    iterate_n(
        np.zeros((N_PF, N_HANDS, PF_ACTS), dtype=np.float64),
        np.zeros((N_PF, N_HANDS, PF_ACTS), dtype=np.float64),
        np.zeros((N_FL_SPOT, N_FLOP, FL_ACTS), dtype=np.float64),
        np.zeros((N_FL_SPOT, N_FLOP, FL_ACTS), dtype=np.float64),
        np.zeros((N_FLOP, N_FLOP, 3), dtype=np.float64),
        np.zeros((N_FLOP, N_FLOP, 3), dtype=np.float64),
        hid_table(),
        st_is,
        st_hi,
        edges,
        100.0,
        2,
        0.0,
        1,
    )
    return True
