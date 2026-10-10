"""Heads-up preflop MCCFR blueprints (50bb and 100bb).

Trains a 169-hand strategy on a small NLH betting tree. Do NOT use as
table policy until limp is a real street (not instant showdown). See
`POLICY_ENABLED`. Combo trees in preflop_trees.py are the production policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from poker.game import Action, ActionType, PlayerView
from strategy.hands import HAND_INDEX, hole_to_169
from strategy.stacks import stack_bucket

DATA_DIR = Path(__file__).resolve().parents[1] / "Final_Bot" / "data"

# Limp utility is still instant showdown. Keep False until a flop continuation
# is trained. Production HU policy is the combo trees.
POLICY_ENABLED = False

N_NODES = 4
N_HANDS = 169
ACTS = {
    0: ["fold", "limp", "raise", "jam"],
    1: ["fold", "call", "three", "jam"],
    2: ["fold", "call", "jam"],
    3: ["fold", "call"],
}


def _data_path(stack_bb: float) -> Path:
    b = stack_bucket(stack_bb)
    tag = "50" if b == "50" else "100"
    return DATA_DIR / f"hu_preflop_{tag}.npz"


def _hand_id(c1: int, c2: int) -> int:
    return HAND_INDEX[hole_to_169(c1, c2)]


def _rm(regrets: np.ndarray) -> np.ndarray:
    pos = np.maximum(regrets, 0.0)
    s = pos.sum()
    if s <= 1e-12:
        return np.ones_like(regrets) / max(len(regrets), 1)
    return pos / s


def _equity_table(n_boards: int = 6, seed: int = 1) -> np.ndarray:
    """169×169 showdown equity via representative combos + sampled boards."""
    from poker.evaluator import evaluate
    from strategy.hands import HANDS_169, combos_of_169

    rng = np.random.default_rng(seed)
    eq = np.full((N_HANDS, N_HANDS), 0.5, dtype=np.float64)
    reps = []
    for code in HANDS_169:
        reps.append(combos_of_169(code)[0])
    for i in range(N_HANDS):
        for j in range(i, N_HANDS):
            h0, h1 = reps[i], reps[j]
            if len(set(h0) | set(h1)) < 4:
                # blocked: pick another combo for j
                found = False
                for alt in combos_of_169(HANDS_169[j]):
                    if len(set(h0) | set(alt)) == 4:
                        h1 = alt
                        found = True
                        break
                if not found:
                    continue
            dead = set(h0) | set(h1)
            rem = [c for c in range(52) if c not in dead]
            w = t = 0
            for _ in range(n_boards):
                pick = rng.choice(len(rem), size=5, replace=False)
                board = [rem[int(x)] for x in pick]
                a = evaluate([h0[0], h0[1]] + board)
                b = evaluate([h1[0], h1[1]] + board)
                if a > b:
                    w += 1
                elif a == b:
                    t += 1
            e = (w + 0.5 * t) / n_boards
            eq[i, j] = e
            eq[j, i] = 1.0 - e
    return eq


def train(iterations: int = 25_000, seed: int = 0, stack_bb: float = 100.0) -> Path:
    rng = np.random.default_rng(seed)
    regrets = {n: np.zeros((N_HANDS, len(ACTS[n]))) for n in range(N_NODES)}
    ssum = {n: np.zeros((N_HANDS, len(ACTS[n]))) for n in range(N_NODES)}
    EQ = _equity_table()
    stack = float(50.0 if stack_bucket(stack_bb) == "50" else 100.0)
    open_to = 2.0 if stack <= 55 else 2.5
    three_to = 7.5 if stack <= 55 else 9.0

    def deal():
        cards = rng.choice(52, size=4, replace=False)
        return (int(cards[0]), int(cards[1])), (int(cards[2]), int(cards[3]))

    def showdown_ev(h0, h1) -> float:
        return float(EQ[_hand_id(*h0), _hand_id(*h1)])

    def cfr(h0, h1, node: int, upd: int, invested0: float, invested1: float, to_call: float) -> float:
        hid = _hand_id(*h0) if node in (0, 2) else _hand_id(*h1)
        player = 0 if node in (0, 2) else 1
        acts = ACTS[node]
        sig = _rm(regrets[node][hid])
        if player == 0 and upd == 0:
            ssum[node][hid] += sig
        if player == 1 and upd == 1:
            ssum[node][hid] += sig

        def util_of(a_idx: int) -> float:
            name = acts[a_idx]
            if name == "fold":
                return -invested0 if player == 0 else invested1
            if name == "limp":
                # Not instant showdown: BB checks ~65% (IP realization on 2bb)
                # or iso-raises ~35%. Raise keeps the fold-equity this limp lacks.
                eq = showdown_ev(h0, h1)
                check_ev = (eq - 0.5) * 2.2
                iso_ev = (eq * 8.0 - 4.0) if eq >= 0.48 else -1.0
                return 0.65 * check_ev + 0.35 * iso_ev
            if name == "raise" and node == 0:
                return cfr(h0, h1, 1, upd, open_to, 1.0, open_to - 1.0)
            if name == "jam":
                if node == 0:
                    return cfr(h0, h1, 3, upd, stack, 1.0, stack - 1.0)
                if node == 1:
                    return cfr(h0, h1, 2, upd, open_to, stack, stack - open_to)
                eq = showdown_ev(h0, h1)
                return eq * (2.0 * stack) - stack
            if name == "call":
                eq = showdown_ev(h0, h1)
                if node == 1:
                    return eq * (2 * open_to) - open_to
                if node == 2:
                    return eq * (2 * three_to) - three_to
                return eq * (invested0 + invested1 + to_call) - (invested0 + to_call)
            if name == "three":
                return cfr(h0, h1, 2, upd, open_to, three_to, three_to - open_to)
            eq = showdown_ev(h0, h1)
            return eq * (invested0 + invested1) - invested0

        u_a = np.array([util_of(i) for i in range(len(acts))])
        u = float(sig.dot(u_a))
        if player == 0 and upd == 0:
            regrets[node][hid] += u_a - u
        if player == 1 and upd == 1:
            regrets[node][hid] += (-u_a) - (-u)
        return u

    for t in range(1, iterations + 1):
        h0, h1 = deal()
        cfr(h0, h1, 0, 0, 0.5, 1.0, 0.5)
        cfr(h0, h1, 0, 1, 0.5, 1.0, 0.5)
        if t % 400 == 0:
            a = t ** 1.5 / (t ** 1.5 + 1)
            for n in regrets:
                regrets[n] *= a

    dest = _data_path(stack)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dest, n0=ssum[0], n1=ssum[1], n2=ssum[2], n3=ssum[3], stack=np.array(stack))
    return dest


_cache: dict = {}


def load(stack_bb: float = 100.0):
    key = "50" if stack_bucket(stack_bb) == "50" else "100"
    if key in _cache:
        return _cache[key] or None
    path = _data_path(stack_bb)
    if not path.exists():
        _cache[key] = False
        return None
    z = np.load(path)

    def avg(arr):
        s = arr.sum(axis=1, keepdims=True)
        n = arr.shape[1]
        return np.where(s > 0, arr / s, 1.0 / n)

    _cache[key] = {0: avg(z["n0"]), 1: avg(z["n1"]), 2: avg(z["n2"]), 3: avg(z["n3"])}
    return _cache[key]


def blueprint_action(view: PlayerView, rng, stack_bb: Optional[float] = None) -> Optional[Action]:
    """HU-only. Returns None if disabled, missing, or the spot is not in the tree."""
    if not POLICY_ENABLED:
        return None
    if view.num_players != 2:
        return None
    sb = stack_bb if stack_bb is not None else view.effective_stack / max(view.bb, 1)
    tab = load(sb)
    if not tab:
        return None
    hid = _hand_id(*view.hole)
    n_raises = sum(
        1
        for _, a in view.action_history
        if a.type in (ActionType.RAISE, ActionType.BET)
        or (a.type == ActionType.ALL_IN and a.amount > view.bb)
    )
    pos = view.position
    if n_raises == 0 and pos in ("BTN", "SB") and (view.to_call == 0 or view.current_bet <= view.bb):
        node, names = 0, ACTS[0]
    elif n_raises == 1 and pos == "BB":
        node, names = 1, ACTS[1]
    elif n_raises == 2 and pos in ("BTN", "SB"):
        node, names = 2, ACTS[2]
    elif n_raises >= 3:
        node, names = 3, ACTS[3]
    else:
        return None
    sig = tab[node][hid]
    sig = np.maximum(sig[: len(names)], 0)
    if sig.sum() <= 0:
        return None
    sig = sig / sig.sum()
    name = names[int(rng.choices(range(len(names)), weights=list(sig), k=1)[0])]
    stack = view.stacks[view.hero]
    to_call = view.to_call
    bb = view.bb
    b = stack_bucket(sb)
    if name == "fold":
        return Action(ActionType.CHECK) if to_call <= 0 else Action(ActionType.FOLD)
    if name in ("limp", "call"):
        if to_call <= 0:
            return Action(ActionType.CHECK)
        return Action(ActionType.CALL, min(to_call, stack))
    if name == "jam":
        return Action(ActionType.ALL_IN, stack)
    if name == "raise":
        size = 2.0 if b == "50" else 2.5
        target = int(size * bb) - view.bets[view.hero]
        target = max(target, min(view.bb, stack))
        if target >= stack:
            return Action(ActionType.ALL_IN, stack)
        return Action(ActionType.BET if to_call <= 0 else ActionType.RAISE, target)
    if name == "three":
        size = 7.5 if b == "50" else 9.0
        target = int(size * bb) - view.bets[view.hero]
        if target >= stack:
            return Action(ActionType.ALL_IN, stack)
        return Action(ActionType.RAISE, max(target, to_call + view.bb))
    return None


if __name__ == "__main__":
    for st in (50.0, 100.0):
        print(f"Training HU preflop MCCFR at {st:g}bb...")
        p = train(20_000, stack_bb=st, seed=int(st))
        print("Wrote", p)
