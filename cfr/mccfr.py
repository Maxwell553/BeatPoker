"""Linear MCCFR with a real flop (limp is not instant showdown).

Utilities returned by the tree walk are always player-0 (BTN / first seat) chip
EV. Player 1 regrets are updated with the negation (zero-sum).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from poker.evaluator import evaluate
from strategy.card_abs import flop_bucket, hand_id, load_clusters

DATA = Path(__file__).resolve().parents[1] / "Final_Bot" / "data"

# Preflop nodes
# 0 BTN first: fold, limp, raise, jam
# 1 BB vs raise: fold, call, three, jam
# 2 BTN vs 3bet: fold, call, jam
# 3 BB vs jam: fold, call
# 4 BB vs limp: check, iso, jam
# 5 BTN vs iso: fold, call, jam
PF_ACTS = {
    0: ["fold", "limp", "raise", "jam"],
    1: ["fold", "call", "three", "jam"],
    2: ["fold", "call", "jam"],
    3: ["fold", "call"],
    4: ["check", "iso", "jam"],
    5: ["fold", "call", "jam"],
}
PF_PLAYER = {0: 0, 1: 1, 2: 0, 3: 1, 4: 1, 5: 0}

# Flop: 0 OOP first, 1 IP after check, 2 IP facing OOP bet, 3 OOP facing IP bet
FL_ACTS = {
    0: ["check", "bet33", "bet75", "jam"],
    1: ["check", "bet33", "bet75", "jam"],
    2: ["fold", "call", "jam"],
    3: ["fold", "call", "jam"],
}
FL_PLAYER = {0: 1, 1: 0, 2: 0, 3: 1}  # 0=BTN, 1=BB

N_HANDS = 169
N_FLOP_B = 50


def _rm(regrets: np.ndarray) -> np.ndarray:
    pos = np.maximum(regrets, 0.0)
    s = pos.sum()
    if s <= 1e-12:
        return np.ones_like(regrets) / max(len(regrets), 1)
    return pos / s


def _sd(h0, h1, board) -> float:
    a = evaluate([h0[0], h0[1]] + list(board))
    b = evaluate([h1[0], h1[1]] + list(board))
    if a > b:
        return 1.0
    if a < b:
        return 0.0
    return 0.5


def flop_continuation(h0, h1, flop, pot: float, inv0: float, stk: float, runout) -> float:
    """Street-complete leaf: run out remaining cards, then a slice of leftover stacks.

    This is *not* instant showdown on the current pot (the old limp bug).
    """
    board = list(flop) + list(runout)
    eq = _sd(h0, h1, board)
    extra = min(stk, pot * 0.70)
    return eq * (pot + extra) - inv0 - 0.5 * extra


class HUTrainer:
    """Heads-up 100bb linear MCCFR. Checkpoints average strategy + continuation table."""

    def __init__(self, stack: float = 100.0):
        load_clusters()
        self.stack = float(stack)
        self.open_to = 2.5
        self.three_to = 9.0
        self.iso_to = 4.0
        self.reg_pf = {n: np.zeros((N_HANDS, len(PF_ACTS[n]))) for n in PF_ACTS}
        self.sum_pf = {n: np.zeros((N_HANDS, len(PF_ACTS[n]))) for n in PF_ACTS}
        self.reg_fl = {n: np.zeros((N_FLOP_B, len(FL_ACTS[n]))) for n in FL_ACTS}
        self.sum_fl = {n: np.zeros((N_FLOP_B, len(FL_ACTS[n]))) for n in FL_ACTS}
        self.cv_num = np.zeros((N_FLOP_B, N_FLOP_B, 3), dtype=np.float64)
        self.cv_den = np.zeros((N_FLOP_B, N_FLOP_B, 3), dtype=np.float64)
        self.iters = 0

    def avg_pf(self) -> Dict[int, np.ndarray]:
        out = {}
        for n, s in self.sum_pf.items():
            tot = s.sum(axis=1, keepdims=True)
            k = s.shape[1]
            out[n] = np.where(tot > 0, s / tot, 1.0 / k)
        return out

    def avg_fl(self) -> Dict[int, np.ndarray]:
        out = {}
        for n, s in self.sum_fl.items():
            tot = s.sum(axis=1, keepdims=True)
            k = s.shape[1]
            out[n] = np.where(tot > 0, s / tot, 1.0 / k)
        return out

    def cv_table(self) -> np.ndarray:
        return np.where(self.cv_den > 0, self.cv_num / np.maximum(self.cv_den, 1e-12), 0.0)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        kw = {
            "iters": np.array(self.iters),
            "stack": np.array(self.stack),
            "cv": self.cv_table(),
            "kind": np.array("hu_flop"),
        }
        for n in PF_ACTS:
            kw[f"pf{n}"] = self.sum_pf[n]
        for n in FL_ACTS:
            kw[f"fl{n}"] = self.sum_fl[n]
        np.savez(tmp, **kw)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "HUTrainer":
        z = np.load(path)
        t = cls(stack=float(z["stack"]))
        t.iters = int(z["iters"])
        for n in PF_ACTS:
            key = f"pf{n}"
            if key in z.files:
                t.sum_pf[n] = np.array(z[key], dtype=np.float64)
        for n in FL_ACTS:
            key = f"fl{n}"
            if key in z.files:
                t.sum_fl[n] = np.array(z[key], dtype=np.float64)
        if "cv" in z.files:
            t.cv_num = np.array(z["cv"], dtype=np.float64)
            t.cv_den = np.ones_like(t.cv_num)
        return t

    def iterate(self, rng: np.random.Generator) -> None:
        cards = rng.choice(52, size=9, replace=False)
        h0 = (int(cards[0]), int(cards[1]))
        h1 = (int(cards[2]), int(cards[3]))
        board5 = [int(x) for x in cards[4:9]]
        self.iters += 1
        t = float(self.iters)
        self._cfr_pf(h0, h1, 0, 0, 0.5, 1.0, 0.5, board5, t)
        self._cfr_pf(h0, h1, 0, 1, 0.5, 1.0, 0.5, board5, t)
        if self.iters % 400 == 0:
            a = t ** 1.5 / (t ** 1.5 + 1.0)
            for n in self.reg_pf:
                self.reg_pf[n] *= a
            for n in self.reg_fl:
                self.reg_fl[n] *= a

    def _update(self, player: int, upd: int, regrets: np.ndarray, u_a: np.ndarray, u: float, ssum: np.ndarray, sig: np.ndarray, t: float) -> None:
        if player != upd:
            return
        ssum += t * sig
        delta = u_a - u
        if player == 1:
            delta = -delta
        regrets += delta

    def _cfr_pf(self, h0, h1, node, upd, inv0, inv1, to_call, board5, t) -> float:
        player = PF_PLAYER[node]
        hid = hand_id(*(h0 if player == 0 else h1))
        acts = PF_ACTS[node]
        sig = _rm(self.reg_pf[node][hid])
        u_a = np.zeros(len(acts))
        for i, name in enumerate(acts):
            u_a[i] = self._pf_child(name, h0, h1, node, upd, inv0, inv1, to_call, board5, t)
        u = float(sig.dot(u_a))
        self._update(player, upd, self.reg_pf[node][hid], u_a, u, self.sum_pf[node][hid], sig, t)
        return u

    def _pf_child(self, name, h0, h1, node, upd, inv0, inv1, to_call, board5, t) -> float:
        stack = self.stack
        flop = board5[:3]
        runout = board5[3:5]
        if name == "fold":
            player = PF_PLAYER[node]
            return -inv0 if player == 0 else (inv0 + inv1) - inv0
        if name == "limp":
            return self._cfr_pf(h0, h1, 4, upd, 1.0, 1.0, 0.0, board5, t)
        if name == "raise" and node == 0:
            return self._cfr_pf(h0, h1, 1, upd, self.open_to, 1.0, self.open_to - 1.0, board5, t)
        if name == "three":
            return self._cfr_pf(h0, h1, 2, upd, self.open_to, self.three_to, self.three_to - self.open_to, board5, t)
        if name == "iso":
            return self._cfr_pf(h0, h1, 5, upd, 1.0, self.iso_to, self.iso_to - 1.0, board5, t)
        if name == "jam":
            if node == 0:
                return self._cfr_pf(h0, h1, 3, upd, stack, 1.0, stack - 1.0, board5, t)
            if node == 1:
                return self._cfr_pf(h0, h1, 2, upd, self.open_to, stack, stack - self.open_to, board5, t)
            if node == 4:
                return self._cfr_pf(h0, h1, 5, upd, 1.0, stack, stack - 1.0, board5, t)
            if node == 2:
                return self._cfr_pf(h0, h1, 3, upd, stack, inv1, stack - inv1, board5, t)
            if node == 5:
                return self._cfr_pf(h0, h1, 3, upd, stack, inv1, stack - inv1, board5, t)
            eq = _sd(h0, h1, board5)
            return eq * (inv0 + inv1 + to_call) - (inv0 if PF_PLAYER[node] == 0 else 0) - (
                0 if PF_PLAYER[node] == 0 else 0
            )
        if name == "call":
            if node == 1:
                return self._flop(h0, h1, flop, runout, upd, 2 * self.open_to, self.open_to, stack - self.open_to, 0, t)
            if node == 2:
                return self._flop(h0, h1, flop, runout, upd, 2 * self.three_to, self.three_to, stack - self.three_to, 2, t)
            if node == 3:
                eq = _sd(h0, h1, board5)
                pot = inv0 + inv1 + to_call
                return eq * pot - (inv0 + (to_call if PF_PLAYER[node] == 0 else 0.0))
            if node == 5:
                return self._flop(h0, h1, flop, runout, upd, 2 * self.iso_to, self.iso_to, stack - self.iso_to, 1, t)
        if name == "check" and node == 4:
            return self._flop(h0, h1, flop, runout, upd, 2.0, 1.0, stack - 1.0, 1, t)
        eq = _sd(h0, h1, board5)
        return eq * (inv0 + inv1) - inv0

    def _flop(self, h0, h1, flop, runout, upd, pot, inv0, stk, pot_type, t) -> float:
        return self._cfr_fl(h0, h1, flop, runout, 0, upd, pot, inv0, stk, 0.0, pot_type, t)

    def _cfr_fl(self, h0, h1, flop, runout, node, upd, pot, inv0, stk, to_call, pot_type, t) -> float:
        player = FL_PLAYER[node]
        hole = h0 if player == 0 else h1
        priv = int(min(N_FLOP_B - 1, max(0, flop_bucket(hole, flop))))
        acts = FL_ACTS[node]
        sig = _rm(self.reg_fl[node][priv])
        u_a = np.zeros(len(acts))
        for i, name in enumerate(acts):
            u_a[i] = self._fl_child(
                name, h0, h1, flop, runout, node, upd, pot, inv0, stk, to_call, pot_type, t, player
            )
        u = float(sig.dot(u_a))
        self._update(player, upd, self.reg_fl[node][priv], u_a, u, self.sum_fl[node][priv], sig, t)
        return u

    def _fl_child(self, name, h0, h1, flop, runout, node, upd, pot, inv0, stk, to_call, pot_type, t, player) -> float:
        if name == "fold":
            return -inv0 if player == 0 else pot - inv0
        if name == "check":
            if node == 0:
                return self._cfr_fl(h0, h1, flop, runout, 1, upd, pot, inv0, stk, 0.0, pot_type, t)
            return self._leaf(h0, h1, flop, runout, pot, inv0, stk, pot_type)
        if name == "call":
            npot = pot + to_call
            ninv = inv0 + (to_call if player == 0 else 0.0)
            nstk = max(0.0, stk - to_call)
            if nstk <= 0.5:
                eq = _sd(h0, h1, list(flop) + list(runout))
                return eq * npot - ninv
            return self._leaf(h0, h1, flop, runout, npot, ninv, nstk, pot_type)
        frac = {"bet33": 0.33, "bet75": 0.75, "jam": 99.0}.get(name, 0.75)
        put = stk if name == "jam" else min(stk, max(1.0, frac * pot))
        if put >= stk * 0.92:
            put = stk
        npot = pot + put
        ninv = inv0 + (put if player == 0 else 0.0)
        nstk = max(0.0, stk - put)
        if node in (2, 3):
            eq = _sd(h0, h1, list(flop) + list(runout))
            return eq * npot - ninv
        facing = 2 if player == 1 else 3
        return self._cfr_fl(h0, h1, flop, runout, facing, upd, npot, ninv, nstk, put, pot_type, t)

    def _leaf(self, h0, h1, flop, runout, pot, inv0, stk, pot_type) -> float:
        u = flop_continuation(h0, h1, flop, pot, inv0, stk, runout)
        hb = min(N_FLOP_B - 1, flop_bucket(h0, flop))
        vb = min(N_FLOP_B - 1, flop_bucket(h1, flop))
        pt = int(max(0, min(2, pot_type)))
        self.cv_num[hb, vb, pt] += u / max(pot, 1.0)
        self.cv_den[hb, vb, pt] += 1.0
        return u


class SixMaxTrainer:
    """Coarse 6-max 100bb preflop MCCFR; flop leaf is equity × continuation, not limp-showdown."""

    SPOTS = 8
    ACTS = ["fold", "call", "raise", "jam"]

    def __init__(self, stack: float = 100.0):
        self.stack = float(stack)
        self.reg = np.zeros((self.SPOTS, N_HANDS, 4))
        self.ssum = np.zeros((self.SPOTS, N_HANDS, 4))
        self.iters = 0
        self._eq = None

    def _equity(self) -> np.ndarray:
        if self._eq is None:
            from strategy.preflop_cfr import _equity_table

            self._eq = _equity_table(n_boards=4, seed=2)
        return self._eq

    def avg(self) -> np.ndarray:
        tot = self.ssum.sum(axis=2, keepdims=True)
        return np.where(tot > 0, self.ssum / tot, 0.25)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            ssum=self.ssum,
            iters=np.array(self.iters),
            stack=np.array(self.stack),
            kind=np.array("6max"),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "SixMaxTrainer":
        z = np.load(path)
        t = cls(stack=float(z["stack"]))
        t.ssum = np.array(z["ssum"], dtype=np.float64)
        t.iters = int(z["iters"])
        return t

    def _spot(self, seat: int, n_raises: int) -> int:
        if n_raises == 0:
            return min(4, seat)
        if n_raises == 1:
            return 5 if seat == 5 else 6
        return 7

    def iterate(self, rng: np.random.Generator) -> None:
        cards = rng.choice(52, size=12, replace=False)
        holes = [(int(cards[2 * i]), int(cards[2 * i + 1])) for i in range(6)]
        self.iters += 1
        t = float(self.iters)
        eq_tab = self._equity()
        for upd in (0, 1):
            self._deal_tree(holes, upd, t, eq_tab)
        if self.iters % 400 == 0:
            self.reg *= t ** 1.5 / (t ** 1.5 + 1.0)

    def _deal_tree(self, holes, upd, t, eq_tab) -> float:
        """External-sampled 6-max: walk UTG→BB, then vs-open, then vs-3bet."""
        live = [True] * 6
        inv = [0.0] * 6
        inv[4], inv[5] = 0.5, 1.0
        current = 1.0
        n_raises = 0
        opener = None

        def hid(s):
            return hand_id(*holes[s])

        def eq_vs_live(s):
            others = [j for j in range(6) if j != s and live[j]]
            if not others:
                return 1.0
            return float(np.mean([eq_tab[hid(s), hid(j)] for j in others]))

        def apply_action(s, name):
            nonlocal current, n_raises, opener
            if name == "fold":
                live[s] = False
                return
            if name == "jam":
                inv[s] = self.stack
                current = self.stack
                n_raises += 1
                if opener is None:
                    opener = s
                return
            if name == "raise":
                target = 2.5 if n_raises == 0 else min(self.stack, current * 2.6 + 1.0)
                inv[s] = min(self.stack, max(current + 1.0, target))
                current = inv[s]
                n_raises += 1
                if opener is None:
                    opener = s
                return
            inv[s] = current

        order = [0, 1, 2, 3, 4, 5]
        u0 = 0.0
        for s in order:
            if not live[s]:
                continue
            if n_raises == 0 and inv[s] >= current - 1e-9 and s == 5:
                continue
            spot = self._spot(s, n_raises)
            h = hid(s)
            sig = _rm(self.reg[spot, h])
            eq = eq_vs_live(s)
            pot = sum(inv) + 1.0
            u_a = np.zeros(4)
            u_a[0] = -inv[s]
            call_to = current
            extra = min(self.stack - call_to, pot * 0.70)
            u_a[1] = eq * (pot + (call_to - inv[s]) + extra) - call_to - 0.5 * extra
            raise_to = min(self.stack, 2.5 if n_raises == 0 else current * 2.6)
            u_a[2] = eq * (pot + raise_to) - raise_to
            u_a[3] = eq * (2.0 * self.stack) - self.stack
            # Convert to seat-0-centric only for reporting; each player updates own u_a
            player_parity = s % 2
            if player_parity == upd:
                self.ssum[spot, h] += t * sig
                u = float(sig.dot(u_a))
                self.reg[spot, h] += u_a - u
            # chance-sample the action for the walk
            p = sig / max(sig.sum(), 1e-12)
            ai = int(np.argmax(p) if upd != player_parity else np.searchsorted(np.cumsum(p), np.random.random()))
            # Use a deterministic mix walk so both traversals see a closed game:
            ai = int(np.argmax(p)) if np.max(p) > 0.55 else int(np.argmax(p))
            apply_action(s, self.ACTS[ai])
            if sum(live) <= 1:
                winner = next(i for i in range(6) if live[i])
                return float(sum(inv) - inv[winner]) if winner == 0 else -inv[0]
        # flop continuation among remaining
        live_s = [i for i in range(6) if live[i]]
        if not live_s:
            return -inv[0]
        if 0 not in live_s:
            return -inv[0]
        eq = eq_vs_live(0)
        pot = sum(inv)
        extra = min(self.stack - inv[0], pot * 0.55)
        return eq * (pot + extra) - inv[0] - 0.5 * extra
