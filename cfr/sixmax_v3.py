"""6-max + HU v3 trainers: all-street linear MCCFR blueprints.

Checkpoints:
  Final_Bot/data/blueprint_6max_v3_100.npz
  Final_Bot/data/blueprint_hu_v3_100.npz

Tree v29: v28 plus facing a bet can raise/jam in 6-max HU collapse
so flopped sets stack. First-in jam still banned when SPR > 2.4.
v30: first-in EV mixes fold/call/raise. Vs a raise, THIS hand's
fold/call mix is used (AA calls, 72o folds) so PFR still cbets
value and BB cannot print by flatting 72o into a checking range.
v31: 100bb BTN cannot open-jam; multiway first-in jam only at SPR<=2.4.
HU tree 21: BTN cannot limp; SRP OOP checks to IP; unvisited IP cbets;
BTN raise CFV expands BB vs-open; facing a cbet defaults to MDF;
BB vs a 2.5bb open cannot jam.

Table policy uses these only when eval/gate_v3.py has allowed the chart.
Search leaves read the flop CV table under the same gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from strategy.card_abs import N_FLOP_V3, N_RIVER_V3, N_TURN_V3, load_clusters

DATA = Path(__file__).resolve().parents[1] / "Final_Bot" / "data"
SIX_V3_PATH = DATA / "blueprint_6max_v3_100.npz"
HU_V3_PATH = DATA / "blueprint_hu_v3_100.npz"
TREE_VERSION = 31
HU_TREE_VERSION = 21

N_HANDS = 169
N_PF = 150
PF_ACTS = 4
N_FL_SPOT = 14
N_TN_SPOT = 14
N_RV_SPOT = 14
HU_PF = 8


def _zeros():
    return (
        np.zeros((N_PF, N_HANDS, PF_ACTS), dtype=np.float64),
        np.zeros((N_PF, N_HANDS, PF_ACTS), dtype=np.float64),
        np.zeros((N_FL_SPOT, N_FLOP_V3, 4), dtype=np.float64),
        np.zeros((N_FL_SPOT, N_FLOP_V3, 4), dtype=np.float64),
        np.zeros((N_TN_SPOT, N_TURN_V3, 4), dtype=np.float64),
        np.zeros((N_TN_SPOT, N_TURN_V3, 4), dtype=np.float64),
        np.zeros((N_RV_SPOT, N_RIVER_V3, 4), dtype=np.float64),
        np.zeros((N_RV_SPOT, N_RIVER_V3, 4), dtype=np.float64),
        np.zeros((N_FLOP_V3, N_FLOP_V3, 3), dtype=np.float64),
        np.zeros((N_FLOP_V3, N_FLOP_V3, 3), dtype=np.float64),
    )


class SixMaxV3Trainer:
    def __init__(self, stack: float = 100.0):
        load_clusters()
        self.stack = float(stack)
        (
            self.reg_pf, self.sum_pf, self.reg_fl, self.sum_fl,
            self.reg_tn, self.sum_tn, self.reg_rv, self.sum_rv,
            self.cv_num, self.cv_den,
        ) = _zeros()
        self.iters = 0

    def iterate_batch(self, n: int, rng: np.random.Generator) -> None:
        n = int(n)
        if n <= 0:
            return
        try:
            from cfr.kernel_v3 import COMPILED, hid_table, iterate_n, straight_tables
        except Exception:
            COMPILED = False
            iterate_n = None
        if not COMPILED or iterate_n is None:
            self.iters += n
            return
        st_is, st_hi = straight_tables()
        seed = int(rng.integers(1, 2**31 - 1))
        self.reg_pf = np.ascontiguousarray(self.reg_pf)
        self.sum_pf = np.ascontiguousarray(self.sum_pf)
        self.reg_fl = np.ascontiguousarray(self.reg_fl)
        self.sum_fl = np.ascontiguousarray(self.sum_fl)
        self.reg_tn = np.ascontiguousarray(self.reg_tn)
        self.sum_tn = np.ascontiguousarray(self.sum_tn)
        self.reg_rv = np.ascontiguousarray(self.reg_rv)
        self.sum_rv = np.ascontiguousarray(self.sum_rv)
        self.cv_num = np.ascontiguousarray(self.cv_num)
        self.cv_den = np.ascontiguousarray(self.cv_den)
        iterate_n(
            self.reg_pf, self.sum_pf, self.reg_fl, self.sum_fl,
            self.reg_tn, self.sum_tn, self.reg_rv, self.sum_rv,
            self.cv_num, self.cv_den, hid_table(), st_is, st_hi,
            float(self.stack), n, float(self.iters), seed,
        )
        self.iters += n

    def iterate(self, rng: np.random.Generator) -> None:
        self.iterate_batch(1, rng)

    def save(self, path: Path | None = None) -> None:
        path = Path(path or SIX_V3_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        cv = np.where(self.cv_den > 0, self.cv_num / np.maximum(self.cv_den, 1e-12), 0.0)
        np.savez(
            tmp,
            sum_pf=self.sum_pf,
            sum_fl=self.sum_fl,
            sum_tn=self.sum_tn,
            sum_rv=self.sum_rv,
            reg_pf=self.reg_pf,
            reg_fl=self.reg_fl,
            reg_tn=self.reg_tn,
            reg_rv=self.reg_rv,
            cv=cv,
            cv_num=self.cv_num,
            cv_den=self.cv_den,
            iters=np.array(self.iters),
            stack=np.array(self.stack),
            kind=np.array("6max_v3"),
            tree=np.array(TREE_VERSION),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "SixMaxV3Trainer":
        z = np.load(path)
        tree = int(z["tree"]) if "tree" in z.files else 0
        if tree < TREE_VERSION:
            return cls()
        t = cls(stack=float(z["stack"]) if "stack" in z.files else 100.0)
        t.iters = int(z["iters"]) if "iters" in z.files else 0
        for name in (
            "sum_pf", "sum_fl", "sum_tn", "sum_rv",
            "reg_pf", "reg_fl", "reg_tn", "reg_rv",
            "cv_num", "cv_den",
        ):
            if name in z.files and getattr(t, name).shape == np.array(z[name]).shape:
                setattr(t, name, np.array(z[name], dtype=np.float64))
        if "cv_num" not in z.files and "cv" in z.files:
            cv = np.array(z["cv"], dtype=np.float64)
            if cv.shape == t.cv_num.shape:
                t.cv_num = cv
                t.cv_den = np.ones_like(cv)
        return t


class HUV3Trainer:
    def __init__(self, stack: float = 100.0):
        load_clusters()
        self.stack = float(stack)
        self.reg_pf = np.zeros((HU_PF, N_HANDS, PF_ACTS), dtype=np.float64)
        self.sum_pf = np.zeros((HU_PF, N_HANDS, PF_ACTS), dtype=np.float64)
        self.reg_fl = np.zeros((N_FL_SPOT, N_FLOP_V3, 4), dtype=np.float64)
        self.sum_fl = np.zeros((N_FL_SPOT, N_FLOP_V3, 4), dtype=np.float64)
        self.reg_tn = np.zeros((N_TN_SPOT, N_TURN_V3, 4), dtype=np.float64)
        self.sum_tn = np.zeros((N_TN_SPOT, N_TURN_V3, 4), dtype=np.float64)
        self.reg_rv = np.zeros((N_RV_SPOT, N_RIVER_V3, 4), dtype=np.float64)
        self.sum_rv = np.zeros((N_RV_SPOT, N_RIVER_V3, 4), dtype=np.float64)
        self.cv_num = np.zeros((N_FLOP_V3, N_FLOP_V3, 3), dtype=np.float64)
        self.cv_den = np.zeros((N_FLOP_V3, N_FLOP_V3, 3), dtype=np.float64)
        self.iters = 0

    def iterate_batch(self, n: int, rng: np.random.Generator) -> None:
        n = int(n)
        if n <= 0:
            return
        try:
            from cfr.kernel_v3 import COMPILED, hid_table, iterate_hu_n, straight_tables
        except Exception:
            COMPILED = False
            iterate_hu_n = None
        if not COMPILED or iterate_hu_n is None:
            self.iters += n
            return
        st_is, st_hi = straight_tables()
        seed = int(rng.integers(1, 2**31 - 1))
        for name in (
            "reg_pf", "sum_pf", "reg_fl", "sum_fl",
            "reg_tn", "sum_tn", "reg_rv", "sum_rv", "cv_num", "cv_den",
        ):
            setattr(self, name, np.ascontiguousarray(getattr(self, name)))
        iterate_hu_n(
            self.reg_pf, self.sum_pf, self.reg_fl, self.sum_fl,
            self.reg_tn, self.sum_tn, self.reg_rv, self.sum_rv,
            self.cv_num, self.cv_den, hid_table(), st_is, st_hi,
            float(self.stack), n, float(self.iters), seed,
        )
        self.iters += n

    def iterate(self, rng: np.random.Generator) -> None:
        self.iterate_batch(1, rng)

    def save(self, path: Path | None = None) -> None:
        path = Path(path or HU_V3_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        cv = np.where(self.cv_den > 0, self.cv_num / np.maximum(self.cv_den, 1e-12), 0.0)
        pack = {
            "sum_pf": self.sum_pf,
            "sum_fl": self.sum_fl,
            "sum_tn": self.sum_tn,
            "sum_rv": self.sum_rv,
            "reg_pf": self.reg_pf,
            "reg_fl": self.reg_fl,
            "reg_tn": self.reg_tn,
            "reg_rv": self.reg_rv,
            "cv": cv,
            "cv_num": self.cv_num,
            "cv_den": self.cv_den,
            "iters": np.array(self.iters),
            "stack": np.array(self.stack),
            "kind": np.array("hu_v3"),
            "tree": np.array(HU_TREE_VERSION),
        }
        # HU policy loader also reads pf0..pf5
        for n in range(HU_PF):
            pack[f"pf{n}"] = self.sum_pf[n]
        np.savez(tmp, **pack)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "HUV3Trainer":
        z = np.load(path)
        tree = int(z["tree"]) if "tree" in z.files else 0
        if tree < HU_TREE_VERSION:
            return cls()
        t = cls(stack=float(z["stack"]) if "stack" in z.files else 100.0)
        t.iters = int(z["iters"]) if "iters" in z.files else 0
        for name in (
            "sum_pf", "sum_fl", "sum_tn", "sum_rv",
            "reg_pf", "reg_fl", "reg_tn", "reg_rv",
            "cv_num", "cv_den",
        ):
            if name not in z.files:
                continue
            src = np.array(z[name], dtype=np.float64)
            dst = getattr(t, name)
            if dst.shape == src.shape:
                setattr(t, name, src)
            elif (
                len(dst.shape) == len(src.shape)
                and dst.shape[0] >= src.shape[0]
                and dst.shape[1:] == src.shape[1:]
            ):
                padded = np.zeros_like(dst)
                padded[: src.shape[0]] = src
                setattr(t, name, padded)
        if "sum_pf" not in z.files and "pf0" in z.files:
            for n in range(HU_PF):
                key = f"pf{n}"
                if key in z.files:
                    t.sum_pf[n] = np.array(z[key], dtype=np.float64)
        return t
