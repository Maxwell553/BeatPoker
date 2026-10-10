"""Numba MCCFR v3: 6-max preflop + real HU flop/turn/river among remaining.

External-sampling linear MCCFR. One traverser per hand. Postflop is a 2-player
public tree (check / 33 / 75 / jam, one raise) with exact 7-card showdown.
Turn and river are played — not a 0.70-pot continuation leaf.

Preflop is a closed betting round: action wraps so the opener faces 3-bets,
and uncalled bets cannot see a flop.

Tree v30: v29 facing raise/jam vs a bet, plus first-in EV mixes
opponent fold/call/raise. Vs a raise, THIS hand's fold/call mix is
used (AA calls, 72o folds) — not "everyone folds to a check-raise"
(that made PFR check and BB flat 72o). First-in jam still banned
at SPR>2.4. HU still passes allow_jam=1 (tree 21). Unvisited
BB vs SB/BTN 40/35/25 fold/call/3bet.
"""

from __future__ import annotations

import numpy as np

from poker.evaluator import _STRAIGHT_TABLE
from strategy.hands import HAND_INDEX, hole_to_169

N_PF, N_HANDS, PF_ACTS = 150, 169, 4
N_FL_SPOT, N_FLOP, FL_ACTS = 14, 108, 4
N_TN_SPOT, N_TURN, TN_ACTS = 14, 72, 4
N_RV_SPOT, N_RIVER, RV_ACTS = 14, 36, 4
CAT_BASE = 1 << 24
KICK_SPAN = 13 ** 5
HU_PF_NODES = 8

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
    def _sample(sig):
        roll = np.random.random()
        csum = 0.0
        n = sig.shape[0]
        for i in range(n):
            csum += sig[i]
            if roll <= csum:
                return i
        return n - 1

    @_nb.njit(cache=True)
    def _sample_legal(sig, legal):
        n = sig.shape[0]
        n_ok = 0
        for i in range(n):
            if legal[i] != 0:
                n_ok += 1
        if n_ok > 1 and np.random.random() < 0.05:
            pick = int(np.random.random() * n_ok)
            seen = 0
            for i in range(n):
                if legal[i] != 0:
                    if seen == pick:
                        return i
                    seen += 1
        return _sample(sig)

    @_nb.njit(cache=True)
    def _sample_trav(sig, legal, is_trav):
        if is_trav != 0:
            return _sample_legal(sig, legal)
        return _sample(sig)

    @_nb.njit(cache=True)
    def _rm_legal(regrets, legal, default_i=0, default_p=1.0):
        n = regrets.shape[0]
        out = np.empty(n, dtype=np.float64)
        s = 0.0
        n_ok = 0
        for i in range(n):
            if legal[i] == 0:
                out[i] = 0.0
                continue
            n_ok += 1
            v = regrets[i]
            if v < 0.0:
                v = 0.0
            out[i] = v
            s += v
        if n_ok == 0:
            inv = 1.0 / n
            for i in range(n):
                out[i] = inv
            return out
        if s <= 1e-15:
            # Fold if legal — uniform-over-legal 25% jams 72o vs a 3-bet
            # and poisons RFI raise CFV so UTG never steals. default_i=1
            # is HU IP first-in: unvisited cbets instead of checking.
            # default_p<1 with fold+call legal is HU facing a cbet: MDF
            # mix so OOP does not overfold and kill flatting.
# default_p<1 with fold+3bet (no call) is 6-max BB vs EP (tree 22):
# unvisited 3bets 50% so 72o RFI is -EV.
            if default_p < 0.999 and legal[0] != 0:
                for i in range(n):
                    out[i] = 0.0
                if legal[1] != 0:
                    out[0] = default_p
                    out[1] = 1.0 - default_p
                    return out
                if legal[2] != 0:
                    out[0] = default_p
                    out[2] = 1.0 - default_p
                    return out
            di = default_i
            if di < 0 or di >= n or legal[di] == 0:
                di = 0
            if legal[di] != 0:
                for i in range(n):
                    out[i] = 1.0 if i == di else 0.0
                return out
            inv = 1.0 / n_ok
            for i in range(n):
                out[i] = inv if legal[i] != 0 else 0.0
            return out
        for i in range(n):
            out[i] /= s
        return out

    @_nb.njit(cache=True)
    def _pf_sig(reg, legal, seat, n_raises):
        # BB vs an open. Tree 24's unvisited 50/50 fold/3bet (call weight 0)
        # never sampled flats, so SB's steal CFV only saw fold-or-3bet.
        # 30% 3bet is not enough FE; 72o steal stayed +EV. Put call mass on
        # the prior when call is legal so SB prices in playing 72o OOP vs a
        # flatting BB. vs EP (call illegal) stays 50% 3bet.
        if n_raises == 1 and seat == 5 and legal[2] != 0:
            s = 0.0
            n = reg.shape[0]
            for i in range(n):
                if legal[i] == 0:
                    continue
                v = reg[i]
                if v > 0.0:
                    s += v
            if s <= 1e-15:
                out = np.zeros(n, dtype=np.float64)
                if legal[1] != 0:
                    out[0] = 0.40
                    out[1] = 0.35
                    out[2] = 0.25
                else:
                    out[0] = 0.50
                    out[2] = 0.50
                return out
            return _rm_legal(reg, legal)
        return _rm_legal(reg, legal)

    @_nb.njit(cache=True)
    def _upd_legal(reg, ssum, sig, ua, u, t, legal):
        n = reg.shape[0]
        for i in range(n):
            if legal[i] == 0:
                continue
            ssum[i] += t * sig[i]
            reg[i] += ua[i] - u

    @_nb.njit(cache=True)
    def _n_live(live):
        n = 0
        for s in range(6):
            n += int(live[s])
        return n

    @_nb.njit(cache=True)
    def _oop_pfr_agg(oop_seat, n_raises, a, b):
        # SRP: earlier seat opened. 3bet: later seat 3bet. 4bet: opener 4bets.
        if n_raises <= 1:
            pfr = a if a < b else b
        elif n_raises == 2:
            pfr = a if a > b else b
        else:
            pfr = a if a < b else b
        return 1 if oop_seat == pfr else 0

    @_nb.njit(cache=True)
    def _foldout_u(upd, live, inv):
        pot = 0.0
        for s in range(6):
            pot += inv[s]
        if live[upd] == 1:
            return pot - inv[upd]
        return -inv[upd]

    @_nb.njit(cache=True)
    def _pf_legal(seat, n_raises, to_call, stack, current, raiser):
        legal = np.zeros(4, dtype=np.int8)
        facing_jam = current >= stack - 1e-9
        can_first_jam = 0  # no 100bb open-jam (BTN AKo shove nitted the steal)
        # 100bb jam over a 2.5bb open polarizes BB into jam-or-fold
        # (KQo shoves, 22 folds). 100bb jam over an 11bb 3bet does the
        # same to the 4bet node (22 jams, flats stay -EV). 4bet is 35bb;
        # jam only vs a 4bet (current >= 20).
        can_jam_raise = n_raises >= 1 and current >= 20.0 - 1e-9
        ep_rfi = n_raises == 0 and seat < 5
        if to_call <= 1e-9:
            legal[1] = 1
            if not facing_jam:
                legal[2] = 1
                if can_first_jam:
                    legal[3] = 1
            return legal
        legal[0] = 1
        if ep_rfi:
            if not facing_jam:
                legal[2] = 1
                if can_first_jam:
                    legal[3] = 1
            return legal
        # vs a 2.5bb open: non-BB is 3-bet or fold. BB may flat BTN/SB
        # only — v20 let BB flat EP and UTG/HJ nitted while SB stole 100%.
        bb_flat = seat == 5 and (raiser == 3 or raiser == 4)
        vs_no_flat = n_raises == 1 and not facing_jam and not bb_flat
        if vs_no_flat:
            legal[2] = 1
            return legal
        legal[1] = 1
        if not facing_jam:
            legal[2] = 1
            if can_jam_raise:
                legal[3] = 1
        return legal

    @_nb.njit(cache=True)
    def _next_pf_actor(start, live, inv, current, n_raises, stack, bb_opt):
        for k in range(6):
            s = (start + k) % 6
            if live[s] == 0:
                continue
            if inv[s] >= stack - 1e-9:
                continue
            if inv[s] < current - 1e-9:
                return s
            if n_raises == 0 and s == 5 and bb_opt == 0:
                return s
        return -1

    @_nb.njit(cache=True)
    def _draw(h0, h1, board, n_board):
        fd = 0
        hs0, hs1 = h0 // 13, h1 // 13
        for s in range(4):
            hc = (1 if hs0 == s else 0) + (1 if hs1 == s else 0)
            bc = 0
            for i in range(n_board):
                if board[i] // 13 == s:
                    bc += 1
            if hc >= 1 and hc + bc == 4:
                fd = 1
                break
        has = np.zeros(13, dtype=np.int8)
        has[h0 % 13] = 1
        has[h1 % 13] = 1
        for i in range(n_board):
            has[board[i] % 13] = 1
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
    def _str4(v):
        cat = v // CAT_BASE
        kick = v - cat * CAT_BASE
        q = kick * 4 // KICK_SPAN
        if q < 0:
            q = 0
        if q > 3:
            q = 3
        return q

    @_nb.njit(cache=True)
    def bkt_flop(h0, h1, f0, f1, f2, st_is, st_hi):
        cards = np.empty(5, dtype=np.int32)
        cards[0] = h0
        cards[1] = h1
        cards[2] = f0
        cards[3] = f1
        cards[4] = f2
        v = eval_n(cards, 5, st_is, st_hi)
        cat = v // CAT_BASE
        if cat > 8:
            cat = 8
        board = np.empty(3, dtype=np.int32)
        board[0] = f0
        board[1] = f1
        board[2] = f2
        fd, oesd = _draw(h0, h1, board, 3)
        draw = 2 if (fd == 1 and oesd == 1) else (1 if (fd == 1 or oesd == 1) else 0)
        b = cat * 12 + _str4(v) * 3 + draw
        if b > 107:
            b = 107
        return b

    @_nb.njit(cache=True)
    def bkt_turn(h0, h1, f0, f1, f2, tc, st_is, st_hi):
        cards = np.empty(6, dtype=np.int32)
        cards[0] = h0
        cards[1] = h1
        cards[2] = f0
        cards[3] = f1
        cards[4] = f2
        cards[5] = tc
        v = eval_n(cards, 6, st_is, st_hi)
        cat = v // CAT_BASE
        if cat > 8:
            cat = 8
        board = np.empty(4, dtype=np.int32)
        board[0] = f0
        board[1] = f1
        board[2] = f2
        board[3] = tc
        fd, oesd = _draw(h0, h1, board, 4)
        draw = 1 if (fd == 1 or oesd == 1) else 0
        b = cat * 8 + _str4(v) * 2 + draw
        if b > 71:
            b = 71
        return b

    @_nb.njit(cache=True)
    def bkt_river(h0, h1, f0, f1, f2, tc, rc, st_is, st_hi):
        cards = np.empty(7, dtype=np.int32)
        cards[0] = h0
        cards[1] = h1
        cards[2] = f0
        cards[3] = f1
        cards[4] = f2
        cards[5] = tc
        cards[6] = rc
        v = eval_n(cards, 7, st_is, st_hi)
        cat = v // CAT_BASE
        if cat > 8:
            cat = 8
        b = cat * 4 + _str4(v)
        if b > 35:
            b = 35
        return b

    @_nb.njit(cache=True)
    def _sd_ua(ia, pot, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd):
        if ncd < 5:
            ncd = 5
        if ncd > 7:
            ncd = 7
        cards = np.empty(7, dtype=np.int32)
        cards[0] = h0a
        cards[1] = h1a
        cards[2] = f0
        cards[3] = f1
        cards[4] = f2
        cards[5] = tc
        cards[6] = rc
        va = eval_n(cards, ncd, st_is, st_hi)
        cards[0] = h0b
        cards[1] = h1b
        vb = eval_n(cards, ncd, st_is, st_hi)
        if va > vb:
            return pot - ia
        if va < vb:
            return -ia
        return 0.5 * pot - ia

    @_nb.njit(cache=True)
    def _priv(street, who, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi):
        if who == 0:
            h0, h1 = h0a, h1a
        else:
            h0, h1 = h0b, h1b
        if street == 0:
            return bkt_flop(h0, h1, f0, f1, f2, st_is, st_hi)
        if street == 1:
            return bkt_turn(h0, h1, f0, f1, f2, tc, st_is, st_hi)
        return bkt_river(h0, h1, f0, f1, f2, tc, rc, st_is, st_hi)

    @_nb.njit(cache=True)
    def _reg_row(street, sp, priv, reg_fl, reg_tn, reg_rv):
        if street == 0:
            return reg_fl[sp, priv]
        if street == 1:
            return reg_tn[sp, priv]
        return reg_rv[sp, priv]

    @_nb.njit(cache=True)
    def _sum_row(street, sp, priv, sum_fl, sum_tn, sum_rv):
        if street == 0:
            return sum_fl[sp, priv]
        if street == 1:
            return sum_tn[sp, priv]
        return sum_rv[sp, priv]

    @_nb.njit(cache=True)
    def _sp(pt, node):
        s = pt * 4 + node
        if s < 0:
            s = 0
        if s > 11:
            s = 11
        return s

    @_nb.njit(cache=True)
    def _choose(expand_who, actor, street, sp, priv, ua_a, t, reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, legal, default_i=0, default_p=1.0):
        reg = _reg_row(street, sp, priv, reg_fl, reg_tn, reg_rv)
        ssum = _sum_row(street, sp, priv, sum_fl, sum_tn, sum_rv)
        sig = _rm_legal(reg, legal, default_i, default_p)
        if actor == expand_who:
            u = 0.0
            for i in range(4):
                u += sig[i] * ua_a[i]
            if actor == 0:
                _upd_legal(reg, ssum, sig, ua_a, u, t, legal)
            else:
                ub = np.empty(4, dtype=np.float64)
                for i in range(4):
                    ub[i] = -ua_a[i]
                _upd_legal(reg, ssum, sig, ub, -u, t, legal)
        if actor == expand_who:
            return _sample_legal(sig, legal)
        return _sample(sig)

    @_nb.njit(cache=True)
    def _raise_size(bet, stk_raiser, stk_def):
        rput = 2.5 * bet
        if rput < 2.0 * bet:
            rput = 2.0 * bet
        cap = bet + stk_def
        if cap < bet:
            cap = bet
        if rput > stk_raiser:
            rput = stk_raiser
        if rput > cap:
            rput = cap
        jput = stk_raiser
        if jput > cap:
            jput = cap
        if rput < 1e-9:
            rput = jput
        if rput < 1e-9:
            rput = stk_raiser
        if jput < 1e-9:
            jput = stk_raiser
        return rput, jput

    @_nb.njit(cache=True)
    def _fill_face_raise_ua(
        ua, facing, ia, pot, bet, stk_f, stk_d, p_fold,
        h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd,
    ):
        rput, jput = _raise_size(bet, stk_f, stk_d)
        extra_r = rput - bet
        if extra_r < 0.0:
            extra_r = 0.0
        if extra_r > stk_d:
            extra_r = stk_d
        extra_j = jput - bet
        if extra_j < 0.0:
            extra_j = 0.0
        if extra_j > stk_d:
            extra_j = stk_d
        if facing == 0:
            fold_ua = pot - ia
            cr = _sd_ua(ia + rput, pot + rput + extra_r, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
            cj = _sd_ua(ia + jput, pot + jput + extra_j, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
        else:
            fold_ua = -ia
            cr = _sd_ua(ia + extra_r, pot + rput + extra_r, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
            cj = _sd_ua(ia + extra_j, pot + jput + extra_j, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
        ua[2] = p_fold * fold_ua + (1.0 - p_fold) * cr
        ua[3] = p_fold * fold_ua + (1.0 - p_fold) * cj
        return rput, jput

    @_nb.njit(cache=True)
    def _bet_ev(p_fold, p_call, p_raise, fold_ua, call_ua, raise_fold_ua):
        return p_fold * fold_ua + p_call * call_ua + p_raise * raise_fold_ua

    @_nb.njit(cache=True)
    def _vs_raise_ev(
        bettor, ia, pot, put, stk_bettor, stk_raiser, p_hero_fold,
        h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd,
    ):
        """EV after betting `put` and facing a raise, using this hand's fold/call mix."""
        _rput, jput = _raise_size(put, stk_raiser, stk_bettor)
        extra = jput - put
        if extra < 0.0:
            extra = 0.0
        if extra > stk_bettor:
            extra = stk_bettor
        if bettor == 0:
            fold_ua = -(ia + put)
            call_ua = _sd_ua(ia + jput, pot + put + jput + extra, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
        else:
            fold_ua = pot + put - ia
            call_ua = _sd_ua(ia + extra, pot + put + jput + extra, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
        return p_hero_fold * fold_ua + (1.0 - p_hero_fold) * call_ua

    @_nb.njit(cache=True)
    def play_hu_streets(
        expand_who, oop, pot, ia, ib, sa, sb, pot_type,
        h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
        t, st_is, st_hi, start_street, check_mode, oop_agg, allow_jam,
    ):
        pt = pot_type
        if pt < 0:
            pt = 0
        if pt > 2:
            pt = 2
        ss = start_street
        if ss < 0:
            ss = 0
        if ss > 2:
            ss = 2
        ip = 1 - oop
        ua = np.empty(4, dtype=np.float64)
        legal_fa = np.zeros(4, dtype=np.int8)
        legal_fa[0] = 1
        legal_fa[1] = 1
        legal_fc = np.zeros(4, dtype=np.int8)
        legal_fc[0] = 1
        legal_fc[1] = 1
        # 6-max (allow_jam=0): facing a bet can raise/jam so sets stack.
        # First-in jam at SPR>2.4 stays banned. HU (allow_jam=1) facing
        # stays fold/call so tree 21 is unchanged.
        if allow_jam == 0:
            legal_fa[2] = 1
            legal_fa[3] = 1
        ip_def = 0
        oop_fi = 0
        oop_def = 0
        if check_mode != 0:
            # Caller OOP checks to IP (no donk). PFR OOP uses node 3 and
            # unvisited-cbets — otherwise SB steal and BB 3bets check to
            # the caller, who then prints with 72o.
            ip_def = 1
            if oop_agg != 0:
                oop_fi = 3
                oop_def = 1
        face_p = 0.25 if check_mode != 0 else 1.0
        for street in range(ss, 3):
            if sa <= 1e-9 or sb <= 1e-9:
                break
            can_jam = 1
            if allow_jam == 0:
                spr_now = min(sa, sb) / max(pot, 1.0)
                if spr_now > 2.4:
                    can_jam = 0
            legal_fi = np.ones(4, dtype=np.int8)
            legal_oop = np.ones(4, dtype=np.int8)
            if can_jam == 0:
                legal_fi[3] = 0
                legal_oop[3] = 0
            if check_mode != 0:
                legal_oop[1] = 0
                legal_oop[2] = 0
                legal_oop[3] = 0
                if oop_agg != 0:
                    legal_oop[1] = 1
                    legal_oop[2] = 1
                    legal_oop[3] = 1 if can_jam != 0 else 0
            priv_o = _priv(street, oop, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi)
            priv_i = _priv(street, ip, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi)
            stk_o = sa if oop == 0 else sb
            stk_i = sa if ip == 0 else sb
            ncd = 5 + street
            ncd_chk = ncd + 1
            if ncd_chk > 7:
                ncd_chk = 7
            ua[0] = _sd_ua(ia, pot, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
            sig_ip_face = _rm_legal(_reg_row(street, _sp(pt, 2), priv_i, reg_fl, reg_tn, reg_rv), legal_fa, 0, face_p)
            p_fold = sig_ip_face[0]
            p_call = sig_ip_face[1]
            p_raise = sig_ip_face[2] + sig_ip_face[3]
            sig_o_vs_r = _rm_legal(_reg_row(street, _sp(pt, 2), priv_o, reg_fl, reg_tn, reg_rv), legal_fc, 0, face_p)
            p_hf_o = sig_o_vs_r[0]
            for ai in range(1, 4):
                put = stk_o if ai == 3 else (0.33 * pot if ai == 1 else 0.75 * pot)
                if put > stk_o:
                    put = stk_o
                if put < 1e-9:
                    put = stk_o
                if oop == 0:
                    fold_ua = pot - ia
                    call_ua = _sd_ua(ia + put, pot + 2.0 * put, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
                else:
                    fold_ua = -ia
                    call_ua = _sd_ua(ia + put, pot + 2.0 * put, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
                raise_ua = _vs_raise_ev(
                    oop, ia, pot, put, stk_o, stk_i, p_hf_o,
                    h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd,
                )
                ua[ai] = _bet_ev(p_fold, p_call, p_raise, fold_ua, call_ua, raise_ua)
            if check_mode != 0:
                sig_ip_fi = _rm_legal(_reg_row(street, _sp(pt, 1), priv_i, reg_fl, reg_tn, reg_rv), legal_fi, ip_def)
                sig_oop_fa = _rm_legal(_reg_row(street, _sp(pt, 2), priv_o, reg_fl, reg_tn, reg_rv), legal_fa, 0, face_p)
                p_fo = sig_oop_fa[0]
                p_co = sig_oop_fa[1]
                p_ro = sig_oop_fa[2] + sig_oop_fa[3]
                sig_i_vs_r = _rm_legal(_reg_row(street, _sp(pt, 2), priv_i, reg_fl, reg_tn, reg_rv), legal_fc, 0, face_p)
                p_hf_i = sig_i_vs_r[0]
                ev = sig_ip_fi[0] * _sd_ua(ia, pot, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd_chk)
                for ai in range(1, 4):
                    put = stk_i if ai == 3 else (0.33 * pot if ai == 1 else 0.75 * pot)
                    if put > stk_i:
                        put = stk_i
                    if ip == 0:
                        fold_ua = pot - ia
                        call_ua = _sd_ua(ia + put, pot + 2.0 * put, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
                    else:
                        fold_ua = -ia
                        call_ua = _sd_ua(ia + put, pot + 2.0 * put, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
                    raise_ua = _vs_raise_ev(
                        ip, ia, pot, put, stk_i, stk_o, p_hf_i,
                        h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd,
                    )
                    ev += sig_ip_fi[ai] * _bet_ev(p_fo, p_co, p_ro, fold_ua, call_ua, raise_ua)
                ua[0] = ev
            ai_o = _choose(expand_who, oop, street, _sp(pt, oop_fi), priv_o, ua, t, reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, legal_oop, oop_def)
            if ai_o == 0:
                ua[0] = _sd_ua(ia, pot, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd_chk if check_mode != 0 else ncd)
                sig_oop_face = _rm_legal(_reg_row(street, _sp(pt, 2), priv_o, reg_fl, reg_tn, reg_rv), legal_fa, 0, face_p)
                p_fold_o = sig_oop_face[0]
                p_call_o = sig_oop_face[1]
                p_raise_o = sig_oop_face[2] + sig_oop_face[3]
                sig_i_vs_r = _rm_legal(_reg_row(street, _sp(pt, 2), priv_i, reg_fl, reg_tn, reg_rv), legal_fc, 0, face_p)
                p_hf_i = sig_i_vs_r[0]
                for ai in range(1, 4):
                    put = stk_i if ai == 3 else (0.33 * pot if ai == 1 else 0.75 * pot)
                    if put > stk_i:
                        put = stk_i
                    if ip == 0:
                        fold_ua = pot - ia
                        call_ua = _sd_ua(ia + put, pot + 2.0 * put, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
                    else:
                        fold_ua = -ia
                        call_ua = _sd_ua(ia + put, pot + 2.0 * put, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
                    raise_ua = _vs_raise_ev(
                        ip, ia, pot, put, stk_i, stk_o, p_hf_i,
                        h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd,
                    )
                    ua[ai] = _bet_ev(p_fold_o, p_call_o, p_raise_o, fold_ua, call_ua, raise_ua)
                ai_i = _choose(expand_who, ip, street, _sp(pt, 1), priv_i, ua, t, reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, legal_fi, ip_def)
                if ai_i == 0:
                    continue
                put = stk_i if ai_i == 3 else (0.33 * pot if ai_i == 1 else 0.75 * pot)
                if put > stk_i:
                    put = stk_i
                if ip == 0:
                    ia += put
                    sa -= put
                else:
                    ib += put
                    sb -= put
                pot += put
                ua[0] = -ia if oop == 0 else (pot - ia)
                ua[1] = _sd_ua(ia + (put if oop == 0 else 0.0), pot + put, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd) if oop == 0 else _sd_ua(ia, pot + put, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
                ua[2] = 0.0
                ua[3] = 0.0
                rput = put
                jput = put
                if allow_jam == 0:
                    sig_vs_r = _rm_legal(_reg_row(street, _sp(pt, 2), priv_i, reg_fl, reg_tn, reg_rv), legal_fc, 0, face_p)
                    stk_f = sa if oop == 0 else sb
                    stk_d = sa if ip == 0 else sb
                    rput, jput = _fill_face_raise_ua(
                        ua, oop, ia, pot, put, stk_f, stk_d, sig_vs_r[0],
                        h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd,
                    )
                ai_f = _choose(expand_who, oop, street, _sp(pt, 2), priv_o, ua, t, reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, legal_fa, 0, face_p)
                if ai_f == 0:
                    return -ia if oop == 0 else (pot - ia)
                if ai_f >= 2:
                    use = rput if ai_f == 2 else jput
                    if oop == 0:
                        ia += use
                        sa -= use
                    else:
                        ib += use
                        sb -= use
                    pot += use
                    extra = use - put
                    if extra < 0.0:
                        extra = 0.0
                    left_ip = sa if ip == 0 else sb
                    if extra > left_ip:
                        extra = left_ip
                    ua[0] = -ia if ip == 0 else (pot - ia)
                    if ip == 0:
                        ua[1] = _sd_ua(ia + extra, pot + extra, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
                    else:
                        ua[1] = _sd_ua(ia, pot + extra, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
                    ua[2] = 0.0
                    ua[3] = 0.0
                    ai_c = _choose(expand_who, ip, street, _sp(pt, 2), priv_i, ua, t, reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, legal_fc, 0, face_p)
                    if ai_c == 0:
                        return -ia if ip == 0 else (pot - ia)
                    if ip == 0:
                        ia += extra
                        sa -= extra
                    else:
                        ib += extra
                        sb -= extra
                    pot += extra
                    continue
                call_put = put
                if call_put > (sa if oop == 0 else sb):
                    call_put = sa if oop == 0 else sb
                if oop == 0:
                    ia += call_put
                    sa -= call_put
                else:
                    ib += call_put
                    sb -= call_put
                pot += call_put
                continue
            put = stk_o if ai_o == 3 else (0.33 * pot if ai_o == 1 else 0.75 * pot)
            if put > stk_o:
                put = stk_o
            if oop == 0:
                ia += put
                sa -= put
            else:
                ib += put
                sb -= put
            pot += put
            ua[0] = -ia if ip == 0 else (pot - ia)
            if ip == 0:
                ua[1] = _sd_ua(ia + put, pot + put, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
            else:
                ua[1] = _sd_ua(ia, pot + put, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
            ua[2] = 0.0
            ua[3] = 0.0
            rput = put
            jput = put
            if allow_jam == 0:
                sig_vs_r = _rm_legal(_reg_row(street, _sp(pt, 2), priv_o, reg_fl, reg_tn, reg_rv), legal_fc, 0, face_p)
                stk_f = sa if ip == 0 else sb
                stk_d = sa if oop == 0 else sb
                rput, jput = _fill_face_raise_ua(
                    ua, ip, ia, pot, put, stk_f, stk_d, sig_vs_r[0],
                    h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd,
                )
            ai_f = _choose(expand_who, ip, street, _sp(pt, 2), priv_i, ua, t, reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, legal_fa, 0, face_p)
            if ai_f == 0:
                return -ia if ip == 0 else (pot - ia)
            if ai_f >= 2:
                use = rput if ai_f == 2 else jput
                if ip == 0:
                    ia += use
                    sa -= use
                else:
                    ib += use
                    sb -= use
                pot += use
                extra = use - put
                if extra < 0.0:
                    extra = 0.0
                left_oop = sa if oop == 0 else sb
                if extra > left_oop:
                    extra = left_oop
                ua[0] = -ia if oop == 0 else (pot - ia)
                if oop == 0:
                    ua[1] = _sd_ua(ia + extra, pot + extra, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
                else:
                    ua[1] = _sd_ua(ia, pot + extra, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, ncd)
                ua[2] = 0.0
                ua[3] = 0.0
                ai_c = _choose(expand_who, oop, street, _sp(pt, 2), priv_o, ua, t, reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, legal_fc, 0, face_p)
                if ai_c == 0:
                    return -ia if oop == 0 else (pot - ia)
                if oop == 0:
                    ia += extra
                    sa -= extra
                else:
                    ib += extra
                    sb -= extra
                pot += extra
                continue
            call_put = put
            if call_put > (sa if ip == 0 else sb):
                call_put = sa if ip == 0 else sb
            if ip == 0:
                ia += call_put
                sa -= call_put
            else:
                ib += call_put
                sb -= call_put
            pot += call_put
        return _sd_ua(ia, pot, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, st_is, st_hi, 7)

    @_nb.njit(cache=True)
    def walk_post(
        expand_who, street, first, actor, oop, raises_left, to_call,
        pot, ia, ib, sa, sb, pot_type,
        h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
        t, st_is, st_hi, oop_agg,
    ):
        _ = (street, first, actor, raises_left, to_call)
        return play_hu_streets(
            expand_who, oop, pot, ia, ib, sa, sb, pot_type,
            h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc,
            reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
            t, st_is, st_hi, 0, 1, oop_agg, 0,
        )

    @_nb.njit(cache=True)
    def _pf_spot(seat, n_raises, current, stack, raiser):
        if n_raises <= 0:
            return seat
        if current >= stack - 1e-9:
            fb = 4
        elif current >= 20.0:
            fb = 3
        elif current >= 6.0:
            fb = 2
        else:
            fb = 1
        r = raiser
        if r < 0:
            r = 0
        if r > 5:
            r = 5
        sp = 6 + (fb - 1) * 36 + r * 6 + seat
        if sp < 6:
            sp = 6
        if sp > 149:
            sp = 149
        return sp

    @_nb.njit(cache=True)
    def apply_pf_action(s, ai, n_raises, current, stack, inv):
        if ai == 0:
            if inv[s] >= current - 1e-9:
                return current, current, n_raises, 0
            return 0.0, current, n_raises, 1
        if ai == 3:
            inv[s] = stack
            return stack, stack, n_raises + 1, 0
        if ai == 2:
            if n_raises == 0:
                raise_to = 2.5
            elif n_raises == 1:
                raise_to = 11.0
            elif n_raises == 2:
                raise_to = 35.0
            else:
                raise_to = stack
            if raise_to > stack:
                raise_to = stack
            if raise_to < current + 1e-9:
                raise_to = stack
            inv[s] = raise_to
            return raise_to, raise_to, n_raises + 1, 0
        if current > stack:
            current = stack
        inv[s] = current
        return current, current, n_raises, 0

    @_nb.njit(cache=True)
    def _pf_rollout(
        start_s, upd, live, inv, current, n_raises, stack, bb_opt, raiser,
        holes, hid_tab, reg_pf, f0, f1, f2, tc, rc,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
        cv_num, cv_den, t, st_is, st_hi,
    ):
        s0 = start_s
        steps = 0
        while steps < 24:
            steps += 1
            if _n_live(live) < 2:
                return _foldout_u(upd, live, inv)
            found = _next_pf_actor(s0, live, inv, current, n_raises, stack, bb_opt)
            if found < 0:
                break
            s = found
            to_call = current - inv[s]
            if to_call < 0.0:
                to_call = 0.0
            legal = _pf_legal(s, n_raises, to_call, stack, current, raiser)
            spot = _pf_spot(s, n_raises, current, stack, raiser)
            h = hid_tab[holes[s, 0], holes[s, 1]]
            sig = _pf_sig(reg_pf[spot, h], legal, s, n_raises)
            ai = _sample(sig)
            _cur, current, n_raises, folded = apply_pf_action(s, ai, n_raises, current, stack, inv)
            if ai == 2 or ai == 3:
                raiser = s
            if s == 5:
                bb_opt = 1
            if folded == 1:
                live[s] = 0
            s0 = (s + 1) % 6
        return post_from_preflop(
            upd, live, inv, n_raises, raiser, stack, holes, f0, f1, f2, tc, rc,
            reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
            cv_num, cv_den, t, st_is, st_hi,
        )

    @_nb.njit(cache=True)
    def finish_preflop(
        start_s, upd, live, inv, current, n_raises, stack, bb_opt, raiser,
        holes, hid_tab, reg_pf, sum_pf, f0, f1, f2, tc, rc,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
        cv_num, cv_den, t, st_is, st_hi,
    ):
        live2 = np.empty(6, dtype=np.int8)
        inv2 = np.empty(6, dtype=np.float64)
        ua = np.empty(4, dtype=np.float64)
        s0 = start_s
        steps = 0
        while steps < 24:
            steps += 1
            if _n_live(live) < 2:
                return _foldout_u(upd, live, inv)
            found = _next_pf_actor(s0, live, inv, current, n_raises, stack, bb_opt)
            if found < 0:
                break
            s = found
            to_call = current - inv[s]
            if to_call < 0.0:
                to_call = 0.0
            legal = _pf_legal(s, n_raises, to_call, stack, current, raiser)
            spot = _pf_spot(s, n_raises, current, stack, raiser)
            h = hid_tab[holes[s, 0], holes[s, 1]]
            sig = _pf_sig(reg_pf[spot, h], legal, s, n_raises)
            if s == upd:
                for ai in range(4):
                    if legal[ai] == 0:
                        ua[ai] = 0.0
                        continue
                    for k in range(6):
                        live2[k] = live[k]
                        inv2[k] = inv[k]
                    cur = current
                    nr = n_raises
                    bb2 = bb_opt
                    raiser2 = raiser
                    _c, cur, nr, folded = apply_pf_action(s, ai, nr, cur, stack, inv2)
                    if ai == 2 or ai == 3:
                        raiser2 = s
                    if s == 5:
                        bb2 = 1
                    if folded == 1:
                        live2[s] = 0
                    if _n_live(live2) < 2:
                        ua[ai] = _foldout_u(upd, live2, inv2)
                    else:
                        ua[ai] = _pf_rollout(
                            (s + 1) % 6, upd, live2, inv2, cur, nr, stack, bb2, raiser2,
                            holes, hid_tab, reg_pf, f0, f1, f2, tc, rc,
                            reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
                            cv_num, cv_den, t, st_is, st_hi,
                        )
                u = 0.0
                for i in range(4):
                    u += sig[i] * ua[i]
                _upd_legal(reg_pf[spot, h], sum_pf[spot, h], sig, ua, u, t, legal)
                sig2 = _pf_sig(reg_pf[spot, h], legal, s, n_raises)
                u2 = 0.0
                for i in range(4):
                    u2 += sig2[i] * ua[i]
                return u2
            ai = _sample(sig)
            _cur, current, n_raises, folded = apply_pf_action(s, ai, n_raises, current, stack, inv)
            if ai == 2 or ai == 3:
                raiser = s
            if s == 5:
                bb_opt = 1
            if folded == 1:
                live[s] = 0
            s0 = (s + 1) % 6
        return post_from_preflop(
            upd, live, inv, n_raises, raiser, stack, holes, f0, f1, f2, tc, rc,
            reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
            cv_num, cv_den, t, st_is, st_hi,
        )

    @_nb.njit(cache=True)
    def _hand7(h0, h1, f0, f1, f2, tc, rc, st_is, st_hi):
        cards = np.empty(7, dtype=np.int32)
        cards[0] = h0
        cards[1] = h1
        cards[2] = f0
        cards[3] = f1
        cards[4] = f2
        cards[5] = tc
        cards[6] = rc
        return eval_n(cards, 7, st_is, st_hi)

    @_nb.njit(cache=True)
    def _mw_sd_u(upd, live, inv, holes, f0, f1, f2, tc, rc, st_is, st_hi):
        """Chip EV for upd in a multiway showdown with side pots."""
        vals = np.empty(6, dtype=np.int64)
        for s in range(6):
            if live[s] == 0:
                vals[s] = -1
            else:
                vals[s] = _hand7(
                    holes[s, 0], holes[s, 1], f0, f1, f2, tc, rc, st_is, st_hi,
                )
        caps = np.empty(6, dtype=np.float64)
        n_caps = 0
        for s in range(6):
            if live[s] == 0:
                continue
            c = inv[s]
            seen = 0
            for i in range(n_caps):
                if abs(caps[i] - c) < 1e-9:
                    seen = 1
                    break
            if seen == 0:
                caps[n_caps] = c
                n_caps += 1
        for i in range(n_caps):
            for j in range(i + 1, n_caps):
                if caps[j] < caps[i]:
                    tmp = caps[i]
                    caps[i] = caps[j]
                    caps[j] = tmp
        won = 0.0
        prev = 0.0
        for k in range(n_caps):
            cap = caps[k]
            pot = 0.0
            for s in range(6):
                lo = inv[s]
                if lo > cap:
                    lo = cap
                hi = prev
                if hi > inv[s]:
                    hi = inv[s]
                contrib = lo - hi
                if contrib > 0.0:
                    pot += contrib
            best = np.int64(-1)
            n_best = 0
            for s in range(6):
                if live[s] == 0:
                    continue
                if inv[s] < cap - 1e-9:
                    continue
                v = vals[s]
                if n_best == 0 or v > best:
                    best = v
                    n_best = 1
                elif v == best:
                    n_best += 1
            if n_best > 0 and pot > 0.0:
                if live[upd] == 1 and inv[upd] >= cap - 1e-9 and vals[upd] == best:
                    won += pot / n_best
            prev = cap
        return won - inv[upd]

    @_nb.njit(cache=True)
    def _cpy6i(src, dst):
        for i in range(6):
            dst[i] = src[i]

    @_nb.njit(cache=True)
    def _cpy6f(src, dst):
        for i in range(6):
            dst[i] = src[i]

    @_nb.njit(cache=True)
    def _mw_priv(street, h0, h1, f0, f1, f2, tc, rc, st_is, st_hi):
        if street == 0:
            b = bkt_flop(h0, h1, f0, f1, f2, st_is, st_hi)
            if b < 0:
                b = 0
            if b > 107:
                b = 107
            return b
        if street == 1:
            b = bkt_turn(h0, h1, f0, f1, f2, tc, st_is, st_hi)
            if b < 0:
                b = 0
            if b > 71:
                b = 71
            return b
        b = bkt_river(h0, h1, f0, f1, f2, tc, rc, st_is, st_hi)
        if b < 0:
            b = 0
        if b > 35:
            b = 35
        return b

    @_nb.njit(cache=True)
    def _mw_to_hu(
        upd, live, inv, stack, holes, f0, f1, f2, tc, rc, n_raises, start_street,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        opp = -1
        best_inv = -1.0
        for s in range(6):
            if s != upd and live[s] == 1 and inv[s] > best_inv:
                best_inv = inv[s]
                opp = s
        if opp < 0:
            return _foldout_u(upd, live, inv)
        pot = 0.0
        for s in range(6):
            pot += inv[s]
        ia = inv[upd]
        ib = inv[opp]
        sa = stack - ia
        sb = stack - ib
        if sa < 0.0:
            sa = 0.0
        if sb < 0.0:
            sb = 0.0
        pot_type = 0 if n_raises <= 1 else (1 if n_raises == 2 else 2)
        oop_seat = -1
        for k in range(1, 7):
            s = (3 + k) % 6
            if s == upd or s == opp:
                oop_seat = s
                break
        oop = 0 if oop_seat == upd else 1
        ss = start_street
        if ss < 0:
            ss = 0
        if ss > 2:
            ss = 2
        oop_agg = _oop_pfr_agg(oop_seat, n_raises, upd, opp)
        return play_hu_streets(
            0, oop, pot, ia, ib, sa, sb, pot_type,
            holes[upd, 0], holes[upd, 1], holes[opp, 0], holes[opp, 1],
            f0, f1, f2, tc, rc,
            reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
            t, st_is, st_hi, ss, 1, oop_agg, 0,
        )

    @_nb.njit(cache=True)
    def _mw_sd_now(upd, live, inv, holes, f0, f1, f2, tc, rc, st_is, st_hi):
        if _n_live(live) < 2:
            return _foldout_u(upd, live, inv)
        return _mw_sd_u(upd, live, inv, holes, f0, f1, f2, tc, rc, st_is, st_hi)

    @_nb.njit(cache=True)
    def _mw_put(ai, pot, left):
        if ai <= 0:
            return 0.0
        if ai == 3:
            return left
        b = 0.33 * pot if ai == 1 else 0.75 * pot
        if b > left:
            b = left
        if b < 1e-9:
            b = left
        return b

    @_nb.njit(cache=True)
    def _mw_fi_legal(left, pot):
        legal = np.zeros(4, dtype=np.int8)
        legal[0] = 1
        if left > 1e-9:
            legal[1] = 1
            legal[2] = 1
            spr = 99.0
            if pot > 1e-9:
                spr = left / pot
            if spr <= 2.4:
                legal[3] = 1
        return legal

    @_nb.njit(cache=True)
    def _mw_first_actor(live, inv, stack):
        for k in range(6):
            s = (4 + k) % 6
            if live[s] == 1 and (stack - inv[s]) > 1e-9:
                return s
        return -1

    @_nb.njit(cache=True)
    def _mw_face_rv(
        k0, upd, bettor, bet, live, inv, stack, holes,
        f0, f1, f2, tc, rc, n_raises,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        live_w = np.empty(6, dtype=np.int8)
        inv_w = np.empty(6, dtype=np.float64)
        _cpy6i(live, live_w)
        _cpy6f(inv, inv_w)
        k = k0
        while k < 7:
            s = (bettor + k) % 6
            k += 1
            if live_w[s] == 0:
                continue
            left = stack - inv_w[s]
            if left <= 1e-9:
                continue
            to_put = bet
            if to_put > left:
                to_put = left
            legal = np.zeros(4, dtype=np.int8)
            legal[0] = 1
            legal[1] = 1
            priv = _mw_priv(2, holes[s, 0], holes[s, 1], f0, f1, f2, tc, rc, st_is, st_hi)
            sig = _rm_legal(reg_rv[13, priv], legal)
            if s == upd:
                ua = np.empty(4, dtype=np.float64)
                ua[0] = -inv_w[upd]
                ua[2] = 0.0
                ua[3] = 0.0
                live2 = np.empty(6, dtype=np.int8)
                inv2 = np.empty(6, dtype=np.float64)
                _cpy6i(live_w, live2)
                _cpy6f(inv_w, inv2)
                inv2[s] += to_put
                k2 = k
                folded_out = 0
                if _n_live(live2) < 2:
                    folded_out = 1
                    ua[1] = _foldout_u(upd, live2, inv2)
                while folded_out == 0 and k2 < 7:
                    s2 = (bettor + k2) % 6
                    k2 += 1
                    if live2[s2] == 0:
                        continue
                    left2 = stack - inv2[s2]
                    if left2 <= 1e-9:
                        continue
                    put2 = bet
                    if put2 > left2:
                        put2 = left2
                    legal2 = np.zeros(4, dtype=np.int8)
                    legal2[0] = 1
                    legal2[1] = 1
                    priv2 = _mw_priv(2, holes[s2, 0], holes[s2, 1], f0, f1, f2, tc, rc, st_is, st_hi)
                    sig2 = _rm_legal(reg_rv[13, priv2], legal2)
                    ai2 = _sample(sig2)
                    if ai2 == 0:
                        live2[s2] = 0
                        if _n_live(live2) < 2:
                            folded_out = 1
                            ua[1] = _foldout_u(upd, live2, inv2)
                    else:
                        inv2[s2] += put2
                if folded_out == 0:
                    if _n_live(live2) < 2:
                        ua[1] = _foldout_u(upd, live2, inv2)
                    else:
                        ua[1] = _mw_sd_now(
                        upd, live2, inv2, holes, f0, f1, f2, tc, rc, st_is, st_hi,
                    )
                u = 0.0
                for i in range(4):
                    u += sig[i] * ua[i]
                _upd_legal(reg_rv[13, priv], sum_rv[13, priv], sig, ua, u, t, legal)
                return u
            ai = _sample(sig)
            if ai == 0:
                live_w[s] = 0
                if _n_live(live_w) < 2:
                    return _foldout_u(upd, live_w, inv_w)
            else:
                inv_w[s] += to_put
        if _n_live(live_w) < 2:
            return _foldout_u(upd, live_w, inv_w)
        return _mw_sd_now(
                        upd, live_w, inv_w, holes, f0, f1, f2, tc, rc, st_is, st_hi,
                    )

    @_nb.njit(cache=True)
    def _mw_rest_rv(
        k0, upd, live, inv, pot, stack, holes,
        f0, f1, f2, tc, rc, n_raises,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        live_w = np.empty(6, dtype=np.int8)
        inv_w = np.empty(6, dtype=np.float64)
        _cpy6i(live, live_w)
        _cpy6f(inv, inv_w)
        k = k0
        while k < 6:
            s = (4 + k) % 6
            k += 1
            if live_w[s] == 0:
                continue
            left = stack - inv_w[s]
            if left <= 1e-9:
                continue
            legal = _mw_fi_legal(left, pot)
            priv = _mw_priv(2, holes[s, 0], holes[s, 1], f0, f1, f2, tc, rc, st_is, st_hi)
            sig = _rm_legal(reg_rv[12, priv], legal)
            ai = _sample(sig)
            if ai == 0:
                continue
            b = _mw_put(ai, pot, left)
            inv_w[s] += b
            return _mw_face_rv(
                1, upd, s, b, live_w, inv_w, stack, holes,
                f0, f1, f2, tc, rc, n_raises,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
        if _n_live(live_w) < 2:
            return _foldout_u(upd, live_w, inv_w)
        return _mw_sd_now(
                        upd, live_w, inv_w, holes, f0, f1, f2, tc, rc, st_is, st_hi,
                    )

    @_nb.njit(cache=True)
    def _mw_round_rv(
        k0, upd, live, inv, pot, stack, holes,
        f0, f1, f2, tc, rc, n_raises,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        live_w = np.empty(6, dtype=np.int8)
        inv_w = np.empty(6, dtype=np.float64)
        _cpy6i(live, live_w)
        _cpy6f(inv, inv_w)
        k = k0
        while k < 6:
            s = (4 + k) % 6
            k += 1
            if live_w[s] == 0:
                continue
            left = stack - inv_w[s]
            if left <= 1e-9:
                continue
            legal = _mw_fi_legal(left, pot)
            priv = _mw_priv(2, holes[s, 0], holes[s, 1], f0, f1, f2, tc, rc, st_is, st_hi)
            sig = _rm_legal(reg_rv[12, priv], legal)
            if s == upd:
                ua = np.empty(4, dtype=np.float64)
                live2 = np.empty(6, dtype=np.int8)
                inv2 = np.empty(6, dtype=np.float64)
                _cpy6i(live_w, live2)
                _cpy6f(inv_w, inv2)
                ua[0] = _mw_rest_rv(
                    k, upd, live2, inv2, pot, stack, holes,
                    f0, f1, f2, tc, rc, n_raises,
                    reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                )
                for ai in range(1, 4):
                    if legal[ai] == 0:
                        ua[ai] = 0.0
                        continue
                    _cpy6i(live_w, live2)
                    _cpy6f(inv_w, inv2)
                    b = _mw_put(ai, pot, left)
                    inv2[s] += b
                    ua[ai] = _mw_face_rv(
                        1, upd, s, b, live2, inv2, stack, holes,
                        f0, f1, f2, tc, rc, n_raises,
                        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                    )
                u = 0.0
                for i in range(4):
                    u += sig[i] * ua[i]
                _upd_legal(reg_rv[12, priv], sum_rv[12, priv], sig, ua, u, t, legal)
                return u
            ai = _sample(sig)
            if ai == 0:
                continue
            b = _mw_put(ai, pot, left)
            inv_w[s] += b
            return _mw_face_rv(
                1, upd, s, b, live_w, inv_w, stack, holes,
                f0, f1, f2, tc, rc, n_raises,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
        if _n_live(live_w) < 2:
            return _foldout_u(upd, live_w, inv_w)
        return _mw_sd_now(
                        upd, live_w, inv_w, holes, f0, f1, f2, tc, rc, st_is, st_hi,
                    )

    @_nb.njit(cache=True)
    def _mw_go_rv(
        upd, live, inv, stack, holes,
        f0, f1, f2, tc, rc, n_raises,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        nlive = _n_live(live)
        if nlive < 2:
            return _foldout_u(upd, live, inv)
        if nlive == 2:
            return _mw_to_hu(
                upd, live, inv, stack, holes, f0, f1, f2, tc, rc, n_raises, 2,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
        chips = False
        for s in range(6):
            if live[s] == 1 and (stack - inv[s]) > 1e-9:
                chips = True
                break
        if not chips:
            return _mw_sd_u(upd, live, inv, holes, f0, f1, f2, tc, rc, st_is, st_hi)
        pot = 0.0
        for s in range(6):
            pot += inv[s]
        return _mw_round_rv(
            0, upd, live, inv, pot, stack, holes,
            f0, f1, f2, tc, rc, n_raises,
            reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
        )

    @_nb.njit(cache=True)
    def _mw_face_tn(
        k0, upd, bettor, bet, live, inv, stack, holes,
        f0, f1, f2, tc, rc, n_raises,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        live_w = np.empty(6, dtype=np.int8)
        inv_w = np.empty(6, dtype=np.float64)
        _cpy6i(live, live_w)
        _cpy6f(inv, inv_w)
        k = k0
        while k < 7:
            s = (bettor + k) % 6
            k += 1
            if live_w[s] == 0:
                continue
            left = stack - inv_w[s]
            if left <= 1e-9:
                continue
            to_put = bet
            if to_put > left:
                to_put = left
            legal = np.zeros(4, dtype=np.int8)
            legal[0] = 1
            legal[1] = 1
            priv = _mw_priv(1, holes[s, 0], holes[s, 1], f0, f1, f2, tc, rc, st_is, st_hi)
            sig = _rm_legal(reg_tn[13, priv], legal)
            if s == upd:
                ua = np.empty(4, dtype=np.float64)
                ua[0] = -inv_w[upd]
                ua[2] = 0.0
                ua[3] = 0.0
                live2 = np.empty(6, dtype=np.int8)
                inv2 = np.empty(6, dtype=np.float64)
                _cpy6i(live_w, live2)
                _cpy6f(inv_w, inv2)
                inv2[s] += to_put
                k2 = k
                folded_out = 0
                if _n_live(live2) < 2:
                    folded_out = 1
                    ua[1] = _foldout_u(upd, live2, inv2)
                while folded_out == 0 and k2 < 7:
                    s2 = (bettor + k2) % 6
                    k2 += 1
                    if live2[s2] == 0:
                        continue
                    left2 = stack - inv2[s2]
                    if left2 <= 1e-9:
                        continue
                    put2 = bet
                    if put2 > left2:
                        put2 = left2
                    legal2 = np.zeros(4, dtype=np.int8)
                    legal2[0] = 1
                    legal2[1] = 1
                    priv2 = _mw_priv(1, holes[s2, 0], holes[s2, 1], f0, f1, f2, tc, rc, st_is, st_hi)
                    sig2 = _rm_legal(reg_tn[13, priv2], legal2)
                    ai2 = _sample(sig2)
                    if ai2 == 0:
                        live2[s2] = 0
                        if _n_live(live2) < 2:
                            folded_out = 1
                            ua[1] = _foldout_u(upd, live2, inv2)
                    else:
                        inv2[s2] += put2
                if folded_out == 0:
                    if _n_live(live2) < 2:
                        ua[1] = _foldout_u(upd, live2, inv2)
                    else:
                        ua[1] = _mw_go_rv(
                        upd, live2, inv2, stack, holes,
                        f0, f1, f2, tc, rc, n_raises,
                        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                    )
                u = 0.0
                for i in range(4):
                    u += sig[i] * ua[i]
                _upd_legal(reg_tn[13, priv], sum_tn[13, priv], sig, ua, u, t, legal)
                return u
            ai = _sample(sig)
            if ai == 0:
                live_w[s] = 0
                if _n_live(live_w) < 2:
                    return _foldout_u(upd, live_w, inv_w)
            else:
                inv_w[s] += to_put
        if _n_live(live_w) < 2:
            return _foldout_u(upd, live_w, inv_w)
        return _mw_go_rv(
                        upd, live_w, inv_w, stack, holes,
                        f0, f1, f2, tc, rc, n_raises,
                        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                    )

    @_nb.njit(cache=True)
    def _mw_rest_tn(
        k0, upd, live, inv, pot, stack, holes,
        f0, f1, f2, tc, rc, n_raises,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        live_w = np.empty(6, dtype=np.int8)
        inv_w = np.empty(6, dtype=np.float64)
        _cpy6i(live, live_w)
        _cpy6f(inv, inv_w)
        k = k0
        while k < 6:
            s = (4 + k) % 6
            k += 1
            if live_w[s] == 0:
                continue
            left = stack - inv_w[s]
            if left <= 1e-9:
                continue
            legal = _mw_fi_legal(left, pot)
            priv = _mw_priv(1, holes[s, 0], holes[s, 1], f0, f1, f2, tc, rc, st_is, st_hi)
            sig = _rm_legal(reg_tn[12, priv], legal)
            ai = _sample(sig)
            if ai == 0:
                continue
            b = _mw_put(ai, pot, left)
            inv_w[s] += b
            return _mw_face_tn(
                1, upd, s, b, live_w, inv_w, stack, holes,
                f0, f1, f2, tc, rc, n_raises,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
        if _n_live(live_w) < 2:
            return _foldout_u(upd, live_w, inv_w)
        return _mw_go_rv(
                        upd, live_w, inv_w, stack, holes,
                        f0, f1, f2, tc, rc, n_raises,
                        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                    )

    @_nb.njit(cache=True)
    def _mw_round_tn(
        k0, upd, live, inv, pot, stack, holes,
        f0, f1, f2, tc, rc, n_raises,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        live_w = np.empty(6, dtype=np.int8)
        inv_w = np.empty(6, dtype=np.float64)
        _cpy6i(live, live_w)
        _cpy6f(inv, inv_w)
        k = k0
        while k < 6:
            s = (4 + k) % 6
            k += 1
            if live_w[s] == 0:
                continue
            left = stack - inv_w[s]
            if left <= 1e-9:
                continue
            legal = _mw_fi_legal(left, pot)
            priv = _mw_priv(1, holes[s, 0], holes[s, 1], f0, f1, f2, tc, rc, st_is, st_hi)
            sig = _rm_legal(reg_tn[12, priv], legal)
            if s == upd:
                ua = np.empty(4, dtype=np.float64)
                live2 = np.empty(6, dtype=np.int8)
                inv2 = np.empty(6, dtype=np.float64)
                _cpy6i(live_w, live2)
                _cpy6f(inv_w, inv2)
                ua[0] = _mw_rest_tn(
                    k, upd, live2, inv2, pot, stack, holes,
                    f0, f1, f2, tc, rc, n_raises,
                    reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                )
                for ai in range(1, 4):
                    if legal[ai] == 0:
                        ua[ai] = 0.0
                        continue
                    _cpy6i(live_w, live2)
                    _cpy6f(inv_w, inv2)
                    b = _mw_put(ai, pot, left)
                    inv2[s] += b
                    ua[ai] = _mw_face_tn(
                        1, upd, s, b, live2, inv2, stack, holes,
                        f0, f1, f2, tc, rc, n_raises,
                        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                    )
                u = 0.0
                for i in range(4):
                    u += sig[i] * ua[i]
                _upd_legal(reg_tn[12, priv], sum_tn[12, priv], sig, ua, u, t, legal)
                return u
            ai = _sample(sig)
            if ai == 0:
                continue
            b = _mw_put(ai, pot, left)
            inv_w[s] += b
            return _mw_face_tn(
                1, upd, s, b, live_w, inv_w, stack, holes,
                f0, f1, f2, tc, rc, n_raises,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
        if _n_live(live_w) < 2:
            return _foldout_u(upd, live_w, inv_w)
        return _mw_go_rv(
                        upd, live_w, inv_w, stack, holes,
                        f0, f1, f2, tc, rc, n_raises,
                        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                    )

    @_nb.njit(cache=True)
    def _mw_go_tn(
        upd, live, inv, stack, holes,
        f0, f1, f2, tc, rc, n_raises,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        nlive = _n_live(live)
        if nlive < 2:
            return _foldout_u(upd, live, inv)
        if nlive == 2:
            return _mw_to_hu(
                upd, live, inv, stack, holes, f0, f1, f2, tc, rc, n_raises, 1,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
        chips = False
        for s in range(6):
            if live[s] == 1 and (stack - inv[s]) > 1e-9:
                chips = True
                break
        if not chips:
            return _mw_sd_u(upd, live, inv, holes, f0, f1, f2, tc, rc, st_is, st_hi)
        pot = 0.0
        for s in range(6):
            pot += inv[s]
        return _mw_round_tn(
            0, upd, live, inv, pot, stack, holes,
            f0, f1, f2, tc, rc, n_raises,
            reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
        )

    @_nb.njit(cache=True)
    def _mw_face_fl(
        k0, upd, bettor, bet, live, inv, stack, holes,
        f0, f1, f2, tc, rc, n_raises,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        live_w = np.empty(6, dtype=np.int8)
        inv_w = np.empty(6, dtype=np.float64)
        _cpy6i(live, live_w)
        _cpy6f(inv, inv_w)
        k = k0
        while k < 7:
            s = (bettor + k) % 6
            k += 1
            if live_w[s] == 0:
                continue
            left = stack - inv_w[s]
            if left <= 1e-9:
                continue
            to_put = bet
            if to_put > left:
                to_put = left
            legal = np.zeros(4, dtype=np.int8)
            legal[0] = 1
            legal[1] = 1
            priv = _mw_priv(0, holes[s, 0], holes[s, 1], f0, f1, f2, tc, rc, st_is, st_hi)
            sig = _rm_legal(reg_fl[13, priv], legal)
            if s == upd:
                ua = np.empty(4, dtype=np.float64)
                ua[0] = -inv_w[upd]
                ua[2] = 0.0
                ua[3] = 0.0
                live2 = np.empty(6, dtype=np.int8)
                inv2 = np.empty(6, dtype=np.float64)
                _cpy6i(live_w, live2)
                _cpy6f(inv_w, inv2)
                inv2[s] += to_put
                k2 = k
                folded_out = 0
                if _n_live(live2) < 2:
                    folded_out = 1
                    ua[1] = _foldout_u(upd, live2, inv2)
                while folded_out == 0 and k2 < 7:
                    s2 = (bettor + k2) % 6
                    k2 += 1
                    if live2[s2] == 0:
                        continue
                    left2 = stack - inv2[s2]
                    if left2 <= 1e-9:
                        continue
                    put2 = bet
                    if put2 > left2:
                        put2 = left2
                    legal2 = np.zeros(4, dtype=np.int8)
                    legal2[0] = 1
                    legal2[1] = 1
                    priv2 = _mw_priv(0, holes[s2, 0], holes[s2, 1], f0, f1, f2, tc, rc, st_is, st_hi)
                    sig2 = _rm_legal(reg_fl[13, priv2], legal2)
                    ai2 = _sample(sig2)
                    if ai2 == 0:
                        live2[s2] = 0
                        if _n_live(live2) < 2:
                            folded_out = 1
                            ua[1] = _foldout_u(upd, live2, inv2)
                    else:
                        inv2[s2] += put2
                if folded_out == 0:
                    if _n_live(live2) < 2:
                        ua[1] = _foldout_u(upd, live2, inv2)
                    else:
                        ua[1] = _mw_go_tn(
                        upd, live2, inv2, stack, holes,
                        f0, f1, f2, tc, rc, n_raises,
                        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                    )
                u = 0.0
                for i in range(4):
                    u += sig[i] * ua[i]
                _upd_legal(reg_fl[13, priv], sum_fl[13, priv], sig, ua, u, t, legal)
                return u
            ai = _sample(sig)
            if ai == 0:
                live_w[s] = 0
                if _n_live(live_w) < 2:
                    return _foldout_u(upd, live_w, inv_w)
            else:
                inv_w[s] += to_put
        if _n_live(live_w) < 2:
            return _foldout_u(upd, live_w, inv_w)
        return _mw_go_tn(
                        upd, live_w, inv_w, stack, holes,
                        f0, f1, f2, tc, rc, n_raises,
                        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                    )

    @_nb.njit(cache=True)
    def _mw_rest_fl(
        k0, upd, live, inv, pot, stack, holes,
        f0, f1, f2, tc, rc, n_raises,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        live_w = np.empty(6, dtype=np.int8)
        inv_w = np.empty(6, dtype=np.float64)
        _cpy6i(live, live_w)
        _cpy6f(inv, inv_w)
        k = k0
        while k < 6:
            s = (4 + k) % 6
            k += 1
            if live_w[s] == 0:
                continue
            left = stack - inv_w[s]
            if left <= 1e-9:
                continue
            legal = _mw_fi_legal(left, pot)
            priv = _mw_priv(0, holes[s, 0], holes[s, 1], f0, f1, f2, tc, rc, st_is, st_hi)
            sig = _rm_legal(reg_fl[12, priv], legal)
            ai = _sample(sig)
            if ai == 0:
                continue
            b = _mw_put(ai, pot, left)
            inv_w[s] += b
            return _mw_face_fl(
                1, upd, s, b, live_w, inv_w, stack, holes,
                f0, f1, f2, tc, rc, n_raises,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
        if _n_live(live_w) < 2:
            return _foldout_u(upd, live_w, inv_w)
        return _mw_go_tn(
                        upd, live_w, inv_w, stack, holes,
                        f0, f1, f2, tc, rc, n_raises,
                        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                    )

    @_nb.njit(cache=True)
    def _mw_round_fl(
        k0, upd, live, inv, pot, stack, holes,
        f0, f1, f2, tc, rc, n_raises,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        live_w = np.empty(6, dtype=np.int8)
        inv_w = np.empty(6, dtype=np.float64)
        _cpy6i(live, live_w)
        _cpy6f(inv, inv_w)
        k = k0
        while k < 6:
            s = (4 + k) % 6
            k += 1
            if live_w[s] == 0:
                continue
            left = stack - inv_w[s]
            if left <= 1e-9:
                continue
            legal = _mw_fi_legal(left, pot)
            priv = _mw_priv(0, holes[s, 0], holes[s, 1], f0, f1, f2, tc, rc, st_is, st_hi)
            sig = _rm_legal(reg_fl[12, priv], legal)
            if s == upd:
                ua = np.empty(4, dtype=np.float64)
                live2 = np.empty(6, dtype=np.int8)
                inv2 = np.empty(6, dtype=np.float64)
                _cpy6i(live_w, live2)
                _cpy6f(inv_w, inv2)
                ua[0] = _mw_rest_fl(
                    k, upd, live2, inv2, pot, stack, holes,
                    f0, f1, f2, tc, rc, n_raises,
                    reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                )
                for ai in range(1, 4):
                    if legal[ai] == 0:
                        ua[ai] = 0.0
                        continue
                    _cpy6i(live_w, live2)
                    _cpy6f(inv_w, inv2)
                    b = _mw_put(ai, pot, left)
                    inv2[s] += b
                    ua[ai] = _mw_face_fl(
                        1, upd, s, b, live2, inv2, stack, holes,
                        f0, f1, f2, tc, rc, n_raises,
                        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                    )
                u = 0.0
                for i in range(4):
                    u += sig[i] * ua[i]
                _upd_legal(reg_fl[12, priv], sum_fl[12, priv], sig, ua, u, t, legal)
                return u
            ai = _sample(sig)
            if ai == 0:
                continue
            b = _mw_put(ai, pot, left)
            inv_w[s] += b
            return _mw_face_fl(
                1, upd, s, b, live_w, inv_w, stack, holes,
                f0, f1, f2, tc, rc, n_raises,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
        if _n_live(live_w) < 2:
            return _foldout_u(upd, live_w, inv_w)
        return _mw_go_tn(
                        upd, live_w, inv_w, stack, holes,
                        f0, f1, f2, tc, rc, n_raises,
                        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                    )

    @_nb.njit(cache=True)
    def _mw_play(
        upd, live, inv, stack, holes, f0, f1, f2, tc, rc, n_raises,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        pot = 0.0
        for s in range(6):
            pot += inv[s]
        return _mw_round_fl(
            0, upd, live, inv, pot, stack, holes,
            f0, f1, f2, tc, rc, n_raises,
            reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
        )

    @_nb.njit(cache=True)
    def post_from_preflop(
        upd, live, inv, n_raises, raiser, stack, holes, f0, f1, f2, tc, rc,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
        cv_num, cv_den, t, st_is, st_hi,
    ):
        nlive = _n_live(live)
        if nlive < 2:
            return _foldout_u(upd, live, inv)
        if live[upd] == 0:
            return -inv[upd]
        if nlive > 2:
            chips = False
            for s in range(6):
                if live[s] == 1 and (stack - inv[s]) > 1e-9:
                    chips = True
                    break
            if not chips:
                return _mw_sd_u(upd, live, inv, holes, f0, f1, f2, tc, rc, st_is, st_hi)
            return _mw_play(
                upd, live, inv, stack, holes, f0, f1, f2, tc, rc, n_raises,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
        opp = -1
        best_inv = -1.0
        for s in range(6):
            if s != upd and live[s] == 1 and inv[s] > best_inv:
                best_inv = inv[s]
                opp = s
        if opp < 0:
            return _foldout_u(upd, live, inv)
        pot = 0.0
        for s in range(6):
            pot += inv[s]
        ia = inv[upd]
        ib = inv[opp]
        sa = stack - ia
        sb = stack - ib
        if sa < 0.0:
            sa = 0.0
        if sb < 0.0:
            sb = 0.0
        pot_type = 0 if n_raises <= 1 else (1 if n_raises == 2 else 2)
        oop_seat = -1
        for k in range(1, 7):
            s = (3 + k) % 6
            if s == upd or s == opp:
                oop_seat = s
                break
        oop = 0 if oop_seat == upd else 1
        oop_agg = 1 if raiser == oop_seat else 0
        u_upd = walk_post(
            0, 0, 1, oop, oop, 1, 0.0,
            pot, ia, ib, sa, sb, pot_type,
            holes[upd, 0], holes[upd, 1], holes[opp, 0], holes[opp, 1],
            f0, f1, f2, tc, rc,
            reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
            t, st_is, st_hi, oop_agg,
        )
        if pot > 1e-9:
            ha = bkt_flop(holes[upd, 0], holes[upd, 1], f0, f1, f2, st_is, st_hi)
            hb = bkt_flop(holes[opp, 0], holes[opp, 1], f0, f1, f2, st_is, st_hi)
            pt = pot_type
            if pt < 0:
                pt = 0
            if pt > 2:
                pt = 2
            cv_num[ha, hb, pt] += u_upd / pot
            cv_den[ha, hb, pt] += 1.0
        return u_upd

    @_nb.njit(cache=True)
    def iterate_n(
        reg_pf, sum_pf, reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
        cv_fl_num, cv_fl_den, hid_tab, st_is, st_hi, stack, n_iters, t0, seed,
    ):
        np.random.seed(seed)
        deck = np.empty(52, dtype=np.int32)
        holes = np.empty((6, 2), dtype=np.int32)
        live = np.empty(6, dtype=np.int8)
        inv = np.empty(6, dtype=np.float64)
        live2 = np.empty(6, dtype=np.int8)
        inv2 = np.empty(6, dtype=np.float64)
        ua = np.empty(4, dtype=np.float64)
        for it in range(n_iters):
            t = t0 + it + 1.0
            for i in range(52):
                deck[i] = i
            for i in range(51, 0, -1):
                j = np.random.randint(0, i + 1)
                tmp = deck[i]
                deck[i] = deck[j]
                deck[j] = tmp
            for s in range(6):
                holes[s, 0] = deck[2 * s]
                holes[s, 1] = deck[2 * s + 1]
            f0, f1, f2 = deck[12], deck[13], deck[14]
            tc, rc = deck[15], deck[16]
            upd = int(np.random.randint(0, 6))
            for s in range(6):
                live[s] = 1
                inv[s] = 0.0
            inv[4] = 0.5
            inv[5] = 1.0
            current = 1.0
            n_raises = 0
            bb_opt = 0
            raiser = -1
            s0 = 0
            steps = 0
            while steps < 24:
                steps += 1
                if _n_live(live) < 2:
                    break
                found = _next_pf_actor(s0, live, inv, current, n_raises, stack, bb_opt)
                if found < 0:
                    break
                s = found
                to_call = current - inv[s]
                if to_call < 0.0:
                    to_call = 0.0
                legal = _pf_legal(s, n_raises, to_call, stack, current, raiser)
                spot = _pf_spot(s, n_raises, current, stack, raiser)
                h = hid_tab[holes[s, 0], holes[s, 1]]
                sig = _pf_sig(reg_pf[spot, h], legal, s, n_raises)
                if s == upd:
                    for ai in range(4):
                        if legal[ai] == 0:
                            ua[ai] = 0.0
                            continue
                        for k in range(6):
                            live2[k] = live[k]
                            inv2[k] = inv[k]
                        cur = current
                        nr = n_raises
                        bb2 = bb_opt
                        raiser2 = raiser
                        _c, cur, nr, folded = apply_pf_action(s, ai, nr, cur, stack, inv2)
                        if ai == 2 or ai == 3:
                            raiser2 = s
                        if s == 5:
                            bb2 = 1
                        if folded == 1:
                            live2[s] = 0
                        if _n_live(live2) < 2:
                            ua[ai] = _foldout_u(upd, live2, inv2)
                        else:
                            ua[ai] = finish_preflop(
                                (s + 1) % 6, upd, live2, inv2, cur, nr, stack, bb2, raiser2,
                                holes, hid_tab, reg_pf, sum_pf, f0, f1, f2, tc, rc,
                                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
                                cv_fl_num, cv_fl_den, t, st_is, st_hi,
                            )
                    u = 0.0
                    for i in range(4):
                        u += sig[i] * ua[i]
                    _upd_legal(reg_pf[spot, h], sum_pf[spot, h], sig, ua, u, t, legal)
                    ai = _sample_legal(sig, legal)
                else:
                    ai = _sample(sig)
                _c2, current, n_raises, folded = apply_pf_action(s, ai, n_raises, current, stack, inv)
                if ai == 2 or ai == 3:
                    raiser = s
                if s == 5:
                    bb_opt = 1
                if folded == 1:
                    live[s] = 0
                s0 = (s + 1) % 6
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
                for i in range(reg_tn.shape[0]):
                    for j in range(reg_tn.shape[1]):
                        for k in range(reg_tn.shape[2]):
                            reg_tn[i, j, k] *= a
                for i in range(reg_rv.shape[0]):
                    for j in range(reg_rv.shape[1]):
                        for k in range(reg_rv.shape[2]):
                            reg_rv[i, j, k] *= a
        return t0 + n_iters

    @_nb.njit(cache=True)
    def iterate_hu_n(
        reg_pf, sum_pf, reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
        cv_fl_num, cv_fl_den, hid_tab, st_is, st_hi, stack, n_iters, t0, seed,
    ):
        """HU 100bb: BTN=0, BB=1. Same postflop tree as 6-max v3."""
        np.random.seed(seed)
        deck = np.empty(52, dtype=np.int32)
        ua = np.empty(4, dtype=np.float64)
        legal0 = np.zeros(4, dtype=np.int8)
        legal0[0] = 1
        legal0[1] = 0
        legal0[2] = 1
        legal0[3] = 1
        for it in range(n_iters):
            t = t0 + it + 1.0
            for i in range(52):
                deck[i] = i
            for i in range(51, 0, -1):
                j = np.random.randint(0, i + 1)
                tmp = deck[i]
                deck[i] = deck[j]
                deck[j] = tmp
            h0a, h1a = deck[0], deck[1]
            h0b, h1b = deck[2], deck[3]
            f0, f1, f2 = deck[4], deck[5], deck[6]
            tc, rc = deck[7], deck[8]
            upd = int(np.random.randint(0, 2))
            hid0 = hid_tab[h0a, h1a]
            sig0 = _rm_legal(reg_pf[0, hid0], legal0)
            if upd == 0:
                for ai in range(4):
                    if legal0[ai] == 0:
                        ua[ai] = 0.0
                        continue
                    ua[ai] = _hu_after_btn(
                        ai, upd, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
                        hid_tab, reg_pf, sum_pf, reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
                        cv_fl_num, cv_fl_den, t, st_is, st_hi,
                    )
                u = 0.0
                for i in range(4):
                    u += sig0[i] * ua[i]
                _upd_legal(reg_pf[0, hid0], sum_pf[0, hid0], sig0, ua, u, t, legal0)
            else:
                ai0 = _sample(sig0)
                _ = _hu_after_btn(
                    ai0, upd, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
                    hid_tab, reg_pf, sum_pf, reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
                    cv_fl_num, cv_fl_den, t, st_is, st_hi,
                )
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
                for i in range(reg_tn.shape[0]):
                    for j in range(reg_tn.shape[1]):
                        for k in range(reg_tn.shape[2]):
                            reg_tn[i, j, k] *= a
                for i in range(reg_rv.shape[0]):
                    for j in range(reg_rv.shape[1]):
                        for k in range(reg_rv.shape[2]):
                            reg_rv[i, j, k] *= a
        return t0 + n_iters

    @_nb.njit(cache=True)
    def _hu_flop(
        ia, ib, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
        expand_who, pot_type, cv_num, cv_den,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        pot = ia + ib
        sa = stack - ia
        sb = stack - ib
        if sa < 0.0:
            sa = 0.0
        if sb < 0.0:
            sb = 0.0
        pt = pot_type
        if pt < 0:
            pt = 0
        if pt > 2:
            pt = 2
        u = play_hu_streets(
            expand_who, 1, pot, ia, ib, sa, sb, pt,
            h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc,
            reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
            t, st_is, st_hi, 0, 1, 0, 1,
        )
        if pot > 1e-9:
            ha = bkt_flop(h0a, h1a, f0, f1, f2, st_is, st_hi)
            hb = bkt_flop(h0b, h1b, f0, f1, f2, st_is, st_hi)
            cv_num[ha, hb, pt] += u / pot
            cv_den[ha, hb, pt] += 1.0
        return u

    @_nb.njit(cache=True)
    def _hu_bb_vs_shove(
        ia, ib, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
        expand_who, pot_type, hid_tab, reg_pf, sum_pf,
        cv_num, cv_den, upd,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        legal = np.zeros(4, dtype=np.int8)
        legal[0] = 1
        legal[1] = 1
        facing_jam = ia >= stack - 1e-9
        if not facing_jam:
            legal[3] = 1
        hidb = hid_tab[h0b, h1b]
        node = 6 if facing_jam else 3
        sig = _rm_legal(reg_pf[node, hidb], legal)
        to = ia if ia < stack else stack
        dead = ib if ib < stack else stack
        if upd == 1:
            ua = np.empty(4, dtype=np.float64)
            ua[0] = -dead
            ua[1] = 0.0
            ua[2] = 0.0
            ua[3] = 0.0
            if legal[1] != 0:
                ua[1] = -_hu_flop(
                    to, to, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
                    expand_who, pot_type, cv_num, cv_den,
                    reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                )
            if legal[3] != 0:
                ua[3] = -_hu_flop(
                    stack, stack, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
                    expand_who, pot_type, cv_num, cv_den,
                    reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                )
            u = 0.0
            for i in range(4):
                u += sig[i] * ua[i]
            _upd_legal(reg_pf[node, hidb], sum_pf[node, hidb], sig, ua, u, t, legal)
        ai = _sample_trav(sig, legal, 1 if upd == 1 else 0)
        if ai == 0:
            return dead
        put = stack if ai == 3 else to
        return _hu_flop(
            put, put, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
            expand_who, pot_type, cv_num, cv_den,
            reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
        )

    @_nb.njit(cache=True)
    def _hu_btn_vs_3bet(
        ia, ib, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
        hid_tab, reg_pf, sum_pf, cv_num, cv_den, upd,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        legal = np.zeros(4, dtype=np.int8)
        legal[0] = 1
        legal[1] = 1
        facing_shove = ib >= stack - 1e-9
        if not facing_shove:
            legal[2] = 1
            legal[3] = 1
        hid0 = hid_tab[h0a, h1a]
        node = 7 if facing_shove else 2
        sig = _rm_legal(reg_pf[node, hid0], legal)
        expand = upd
        four = 35.0 if 35.0 < stack else stack
        if upd == 0:
            ua = np.empty(4, dtype=np.float64)
            ua[0] = -ia
            ua[1] = _hu_flop(
                ib, ib, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
                expand, 1, cv_num, cv_den,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
            ua[2] = 0.0
            ua[3] = 0.0
            if legal[2] != 0:
                ua[2] = _hu_bb_vs_shove(
                    four, ib, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
                    expand, 2, hid_tab, reg_pf, sum_pf, cv_num, cv_den, upd,
                    reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                )
            if legal[3] != 0:
                ua[3] = _hu_bb_vs_shove(
                    stack, ib, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
                    expand, 2, hid_tab, reg_pf, sum_pf, cv_num, cv_den, upd,
                    reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                )
            u = 0.0
            for i in range(4):
                u += sig[i] * ua[i]
            _upd_legal(reg_pf[node, hid0], sum_pf[node, hid0], sig, ua, u, t, legal)
        ai = _sample_trav(sig, legal, 1 if upd == 0 else 0)
        if ai == 0:
            return -ia
        if ai == 1:
            return _hu_flop(
                ib, ib, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
                expand, 1, cv_num, cv_den,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
        put = stack if ai == 3 else four
        return _hu_bb_vs_shove(
            put, ib, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
            expand, 2, hid_tab, reg_pf, sum_pf, cv_num, cv_den, upd,
            reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
        )

    @_nb.njit(cache=True)
    def _hu_btn_vs_iso(
        ib, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
        hid_tab, reg_pf, sum_pf, cv_num, cv_den, upd,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        legal = np.zeros(4, dtype=np.int8)
        legal[0] = 1
        legal[1] = 1
        legal[3] = 1
        hid0 = hid_tab[h0a, h1a]
        sig = _rm_legal(reg_pf[5, hid0], legal)
        expand = upd
        if upd == 0:
            ua = np.empty(4, dtype=np.float64)
            ua[0] = -1.0
            ua[1] = _hu_flop(
                ib, ib, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
                expand, 0, cv_num, cv_den,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
            ua[2] = ua[1]
            ua[3] = _hu_bb_vs_shove(
                stack, ib, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
                expand, 2, hid_tab, reg_pf, sum_pf, cv_num, cv_den, upd,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
            u = 0.0
            for i in range(4):
                u += sig[i] * ua[i]
            _upd_legal(reg_pf[5, hid0], sum_pf[5, hid0], sig, ua, u, t, legal)
        ai = _sample_trav(sig, legal, 1 if upd == 0 else 0)
        if ai == 0:
            return -1.0
        if ai == 3:
            return _hu_bb_vs_shove(
                stack, ib, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
                expand, 2, hid_tab, reg_pf, sum_pf, cv_num, cv_den, upd,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
        return _hu_flop(
            ib, ib, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
            expand, 0, cv_num, cv_den,
            reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
        )

    @_nb.njit(cache=True)
    def _hu_bb_limp(
        ai, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
        hid_tab, reg_pf, sum_pf, cv_num, cv_den, upd,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        if ai == 0:
            return _hu_flop(
                1.0, 1.0, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
                upd, 0, cv_num, cv_den,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
        ib = 4.0 if ai == 1 else stack
        if ib > stack:
            ib = stack
        return _hu_btn_vs_iso(
            ib, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
            hid_tab, reg_pf, sum_pf, cv_num, cv_den, upd,
            reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
        )

    @_nb.njit(cache=True)
    def _hu_bb_vs_open(
        ai, ia, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
        hid_tab, reg_pf, sum_pf, cv_num, cv_den, upd,
        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
    ):
        if ai == 0:
            return 1.0
        if ai == 1:
            return _hu_flop(
                ia, ia, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
                upd, 0, cv_num, cv_den,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
        ib = stack if ai == 3 else 11.0
        if ib > stack:
            ib = stack
        return _hu_btn_vs_3bet(
            ia, ib, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
            hid_tab, reg_pf, sum_pf, cv_num, cv_den, upd,
            reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
        )

    @_nb.njit(cache=True)
    def _hu_after_btn(
        ai0, upd, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
        hid_tab, reg_pf, sum_pf, reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv,
        cv_num, cv_den, t, st_is, st_hi,
    ):
        if ai0 == 0:
            return 0.5 if upd == 1 else -0.5
        if ai0 == 3:
            u_btn = _hu_bb_vs_shove(
                stack, 1.0, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
                upd, 2, hid_tab, reg_pf, sum_pf, cv_num, cv_den, upd,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
            return u_btn if upd == 0 else -u_btn
        if ai0 == 1:
            hidb = hid_tab[h0b, h1b]
            legal = np.zeros(4, dtype=np.int8)
            legal[0] = 1
            legal[1] = 1
            legal[2] = 1
            legal[3] = 1
            sig = _rm_legal(reg_pf[4, hidb], legal)
            if upd == 1:
                ua = np.empty(4, dtype=np.float64)
                for ai in range(4):
                    if legal[ai] == 0:
                        ua[ai] = 0.0
                        continue
                    ua[ai] = -_hu_bb_limp(
                        ai, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
                        hid_tab, reg_pf, sum_pf, cv_num, cv_den, upd,
                        reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
                    )
                u = 0.0
                for i in range(4):
                    u += sig[i] * ua[i]
                _upd_legal(reg_pf[4, hidb], sum_pf[4, hidb], sig, ua, u, t, legal)
            ai = _sample_trav(sig, legal, 1 if upd == 1 else 0)
            u_btn = _hu_bb_limp(
                ai, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
                hid_tab, reg_pf, sum_pf, cv_num, cv_den, upd,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
            return u_btn if upd == 0 else -u_btn
        ia = 2.5
        hidb = hid_tab[h0b, h1b]
        legal = np.zeros(4, dtype=np.int8)
        legal[0] = 1
        legal[1] = 1
        legal[2] = 1
        # No 100bb jam over a 2.5bb open — that polarized BB into jam-or-fold
        # and BTN into 72o-steal / KQo-fold.
        legal[3] = 0
        sig = _rm_legal(reg_pf[1, hidb], legal)
        ua_btn = np.empty(4, dtype=np.float64)
        ua_bb = np.empty(4, dtype=np.float64)
        for ai in range(4):
            if legal[ai] == 0:
                ua_btn[ai] = 0.0
                ua_bb[ai] = 0.0
                continue
            ua_btn[ai] = _hu_bb_vs_open(
                ai, ia, h0a, h1a, h0b, h1b, f0, f1, f2, tc, rc, stack,
                hid_tab, reg_pf, sum_pf, cv_num, cv_den, upd,
                reg_fl, sum_fl, reg_tn, sum_tn, reg_rv, sum_rv, t, st_is, st_hi,
            )
            ua_bb[ai] = -ua_btn[ai]
        u_bb = 0.0
        for i in range(4):
            u_bb += sig[i] * ua_bb[i]
        _upd_legal(reg_pf[1, hidb], sum_pf[1, hidb], sig, ua_bb, u_bb, t, legal)
        sig2 = _rm_legal(reg_pf[1, hidb], legal)
        u_btn = 0.0
        for i in range(4):
            u_btn += sig2[i] * ua_btn[i]
        return u_btn if upd == 0 else -u_btn

    COMPILED = True
except Exception:  # pragma: no cover
    COMPILED = False
    iterate_n = None
    iterate_hu_n = None
    walk_post = None


def hid_table() -> np.ndarray:
    return _HID


def straight_tables():
    return _ST_IS, _ST_HI


def warmup() -> bool:
    if not COMPILED or iterate_n is None:
        return False
    st_is, st_hi = straight_tables()
    iterate_n(
        np.zeros((N_PF, N_HANDS, PF_ACTS), dtype=np.float64),
        np.zeros((N_PF, N_HANDS, PF_ACTS), dtype=np.float64),
        np.zeros((N_FL_SPOT, N_FLOP, FL_ACTS), dtype=np.float64),
        np.zeros((N_FL_SPOT, N_FLOP, FL_ACTS), dtype=np.float64),
        np.zeros((N_TN_SPOT, N_TURN, TN_ACTS), dtype=np.float64),
        np.zeros((N_TN_SPOT, N_TURN, TN_ACTS), dtype=np.float64),
        np.zeros((N_RV_SPOT, N_RIVER, RV_ACTS), dtype=np.float64),
        np.zeros((N_RV_SPOT, N_RIVER, RV_ACTS), dtype=np.float64),
        np.zeros((N_FLOP, N_FLOP, 3), dtype=np.float64),
        np.zeros((N_FLOP, N_FLOP, 3), dtype=np.float64),
        hid_table(),
        st_is,
        st_hi,
        100.0,
        2,
        0.0,
        1,
    )
    if iterate_hu_n is not None:
        iterate_hu_n(
            np.zeros((HU_PF_NODES, N_HANDS, PF_ACTS), dtype=np.float64),
            np.zeros((HU_PF_NODES, N_HANDS, PF_ACTS), dtype=np.float64),
            np.zeros((N_FL_SPOT, N_FLOP, FL_ACTS), dtype=np.float64),
            np.zeros((N_FL_SPOT, N_FLOP, FL_ACTS), dtype=np.float64),
            np.zeros((N_TN_SPOT, N_TURN, TN_ACTS), dtype=np.float64),
            np.zeros((N_TN_SPOT, N_TURN, TN_ACTS), dtype=np.float64),
            np.zeros((N_RV_SPOT, N_RIVER, RV_ACTS), dtype=np.float64),
            np.zeros((N_RV_SPOT, N_RIVER, RV_ACTS), dtype=np.float64),
            np.zeros((N_FLOP, N_FLOP, 3), dtype=np.float64),
            np.zeros((N_FLOP, N_FLOP, 3), dtype=np.float64),
            hid_table(),
            st_is,
            st_hi,
            100.0,
            1,
            0.0,
            2,
        )
    return True
