"""6-max ~100bb blueprint with a real abstraction (not 8 coarse spots).

Infoset: (spot, card_bucket). Spots encode position × raise-level × pot-type.
Flop nodes use the 50-bucket card abstraction. Linear MCCFR via fast_loop.

Table policy stays combo trees until this mix is actually trained; the
checkpoint is beliefs + search leaves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from cfr.fast_loop import linear_update, regret_match
from cfr.mccfr import flop_continuation
from strategy.card_abs import flop_bucket, hand_id, load_clusters

DATA = Path(__file__).resolve().parents[1] / "Final_Bot" / "data"
PATH = DATA / "blueprint_6max_v2_100.npz"

N_HANDS = 169
N_FLOP = 50
# Preflop: 6 RFI + 6 vs_open + 6 vs_3bet = 18
N_PF = 18
PF_ACTS = 4  # fold, call, raise, jam
# Flop: 3 pot types × 4 nodes
N_FL_SPOT = 12
FL_ACTS = 4  # check/bet33/bet75/jam or fold/call/raise/jam


class SixMaxV2Trainer:
    """Position-aware 6-max 100bb linear MCCFR."""

    def __init__(self, stack: float = 100.0):
        load_clusters()
        self.stack = float(stack)
        self.reg_pf = np.zeros((N_PF, N_HANDS, PF_ACTS), dtype=np.float64)
        self.sum_pf = np.zeros((N_PF, N_HANDS, PF_ACTS), dtype=np.float64)
        self.reg_fl = np.zeros((N_FL_SPOT, N_FLOP, FL_ACTS), dtype=np.float64)
        self.sum_fl = np.zeros((N_FL_SPOT, N_FLOP, FL_ACTS), dtype=np.float64)
        self.cv_num = np.zeros((N_FLOP, N_FLOP, 3), dtype=np.float64)
        self.cv_den = np.zeros((N_FLOP, N_FLOP, 3), dtype=np.float64)
        self.iters = 0

    def _pf_spot(self, seat: int, n_raises: int) -> int:
        seat = int(max(0, min(5, seat)))
        if n_raises <= 0:
            return seat
        if n_raises == 1:
            return 6 + seat
        return 12 + seat

    def _fl_spot(self, pot_type: int, node: int) -> int:
        return int(max(0, min(2, pot_type))) * 4 + int(max(0, min(3, node)))

    def iterate_batch(self, n: int, rng: np.random.Generator) -> None:
        """n MCCFR hands. Uses the compiled kernel when available."""
        n = int(n)
        if n <= 0:
            return
        try:
            from cfr.kernel_6max import COMPILED, flop_edges, hid_table, iterate_n, straight_tables
        except Exception:
            COMPILED = False
            iterate_n = None
        if COMPILED and iterate_n is not None:
            st_is, st_hi = straight_tables()
            seed = int(rng.integers(1, 2**31 - 1))
            self.reg_pf = np.ascontiguousarray(self.reg_pf)
            self.sum_pf = np.ascontiguousarray(self.sum_pf)
            self.reg_fl = np.ascontiguousarray(self.reg_fl)
            self.sum_fl = np.ascontiguousarray(self.sum_fl)
            self.cv_num = np.ascontiguousarray(self.cv_num)
            self.cv_den = np.ascontiguousarray(self.cv_den)
            iterate_n(
                self.reg_pf,
                self.sum_pf,
                self.reg_fl,
                self.sum_fl,
                self.cv_num,
                self.cv_den,
                hid_table(),
                st_is,
                st_hi,
                flop_edges(),
                float(self.stack),
                n,
                float(self.iters),
                seed,
            )
            self.iters += n
            return
        for _ in range(n):
            self._iterate_python(rng)

    def iterate(self, rng: np.random.Generator) -> None:
        self.iterate_batch(1, rng)

    def _iterate_python(self, rng: np.random.Generator) -> None:
        cards = rng.choice(52, size=17, replace=False)
        holes = [(int(cards[2 * i]), int(cards[2 * i + 1])) for i in range(6)]
        flop = [int(x) for x in cards[12:15]]
        runout = [int(cards[15]), int(cards[16])]
        self.iters += 1
        t = float(self.iters)
        for upd in range(6):
            self._walk(holes, flop, runout, upd, t, rng)
        if self.iters % 400 == 0:
            a = t ** 1.5 / (t ** 1.5 + 1.0)
            self.reg_pf *= a
            self.reg_fl *= a

    def _walk(self, holes, flop, runout, upd, t, rng) -> None:
        live = [True] * 6
        inv = [0.0] * 6
        inv[4], inv[5] = 0.5, 1.0
        current = 1.0
        n_raises = 0
        opener = None
        three = None

        def hid(s):
            return hand_id(*holes[s])

        for s in (0, 1, 2, 3, 4, 5):
            if not live[s]:
                continue
            if n_raises == 0 and inv[s] >= current - 1e-9 and s == 5:
                continue
            spot = self._pf_spot(s, n_raises)
            h = hid(s)
            sig = regret_match(self.reg_pf[spot, h])
            # Chance utilities: fold / call / raise / jam vs a mean remaining equity.
            others = [j for j in range(6) if j != s and live[j]]
            eq = 0.55 if not others else 0.50
            pot = sum(inv) + 1.0
            u_a = np.zeros(4, dtype=np.float64)
            u_a[0] = -inv[s]
            extra = min(self.stack - current, pot * 0.70)
            u_a[1] = eq * (pot + max(0.0, current - inv[s]) + extra) - current - 0.5 * extra
            raise_to = min(self.stack, 2.5 if n_raises == 0 else current * 2.7)
            u_a[2] = eq * (pot + raise_to) - raise_to
            u_a[3] = eq * (2.0 * self.stack) - self.stack
            u = float(sig.dot(u_a))
            if s == upd:
                linear_update(self.reg_pf[spot, h], self.sum_pf[spot, h], sig, u_a, u, t, 1.0)
            # Sample action
            p = sig / max(float(sig.sum()), 1e-12)
            ai = int(np.searchsorted(np.cumsum(p), rng.random()))
            ai = min(3, max(0, ai))
            if ai == 0:
                live[s] = False
                continue
            if ai == 3:
                inv[s] = self.stack
                current = self.stack
                n_raises += 1
                if opener is None:
                    opener = s
                elif three is None:
                    three = s
                continue
            if ai == 2:
                inv[s] = raise_to
                current = raise_to
                n_raises += 1
                if opener is None:
                    opener = s
                elif three is None:
                    three = s
                continue
            inv[s] = current

        live_s = [i for i in range(6) if live[i]]
        if len(live_s) < 2:
            return
        pot_type = 0 if n_raises <= 1 else (1 if n_raises == 2 else 2)
        # Flop leaf among first two live: update flop infosets + CV.
        a, b = live_s[0], live_s[1]
        for node, player in ((0, b), (1, a), (2, a), (3, b)):
            hole = holes[player]
            priv = int(min(N_FLOP - 1, max(0, flop_bucket(hole, flop))))
            sp = self._fl_spot(pot_type, node)
            sig = regret_match(self.reg_fl[sp, priv])
            pot = float(sum(inv))
            stk = self.stack - inv[player]
            u_leaf = flop_continuation(
                holes[a], holes[b], flop, pot, inv[a], self.stack - inv[a], runout
            )
            if player != a:
                u_leaf = -u_leaf
            u_a = np.zeros(4, dtype=np.float64)
            u_a[0] = 0.0 if node < 2 else -inv[player]
            extra = min(stk, pot * 0.55)
            u_a[1] = u_leaf + 0.15 * extra
            u_a[2] = u_leaf + 0.35 * extra
            u_a[3] = u_leaf + 0.45 * min(stk, pot)
            u = float(sig.dot(u_a))
            if player == upd:
                linear_update(self.reg_fl[sp, priv], self.sum_fl[sp, priv], sig, u_a, u, t, 1.0)
        ha = min(N_FLOP - 1, flop_bucket(holes[a], flop))
        hb = min(N_FLOP - 1, flop_bucket(holes[b], flop))
        pot = float(sum(inv))
        u = flop_continuation(
            holes[a], holes[b], flop, pot, inv[a], self.stack - inv[a], runout
        )
        pt = int(max(0, min(2, pot_type)))
        self.cv_num[ha, hb, pt] += u / max(pot, 1.0)
        self.cv_den[ha, hb, pt] += 1.0

    def save(self, path: Path | None = None) -> None:
        path = Path(path or PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        cv = np.where(self.cv_den > 0, self.cv_num / np.maximum(self.cv_den, 1e-12), 0.0)
        # Also write the coarse 8-spot view so old loaders still work.
        coarse = np.zeros((8, N_HANDS, 4), dtype=np.float64)
        for seat in range(6):
            coarse[min(4, seat)] += self.sum_pf[seat]
            coarse[5 if seat == 5 else 6] += self.sum_pf[6 + seat]
            coarse[7] += self.sum_pf[12 + seat]
        np.savez(
            tmp,
            ssum=coarse,
            sum_pf=self.sum_pf,
            sum_fl=self.sum_fl,
            reg_pf=self.reg_pf,
            reg_fl=self.reg_fl,
            cv=cv,
            cv_num=self.cv_num,
            cv_den=self.cv_den,
            iters=np.array(self.iters),
            stack=np.array(self.stack),
            kind=np.array("6max_v2"),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "SixMaxV2Trainer":
        z = np.load(path)
        t = cls(stack=float(z["stack"]))
        t.iters = int(z["iters"])
        if "sum_pf" in z.files:
            t.sum_pf = np.array(z["sum_pf"], dtype=np.float64)
        elif "ssum" in z.files:
            s = np.array(z["ssum"], dtype=np.float64)
            for i in range(min(8, s.shape[0])):
                # scatter coarse back onto v2 seats (best-effort)
                t.sum_pf[min(17, i)] = s[i]
        if "sum_fl" in z.files:
            t.sum_fl = np.array(z["sum_fl"], dtype=np.float64)
        if "reg_pf" in z.files:
            t.reg_pf = np.array(z["reg_pf"], dtype=np.float64)
        if "reg_fl" in z.files:
            t.reg_fl = np.array(z["reg_fl"], dtype=np.float64)
        if "cv_num" in z.files and "cv_den" in z.files:
            t.cv_num = np.array(z["cv_num"], dtype=np.float64)
            t.cv_den = np.array(z["cv_den"], dtype=np.float64)
        elif "cv" in z.files:
            t.cv_num = np.array(z["cv"], dtype=np.float64)
            t.cv_den = np.ones_like(t.cv_num)
        return t
