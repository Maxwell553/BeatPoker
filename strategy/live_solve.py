"""Real-time nested public-tree CFR for the current street.

Laptop-scale Pluribus inner loop:
  * per-combo infosets for both players (no range-wide scalarized villain)
  * linear MCCFR averaging
  * mixed (not argmax) policy
  * depth-limited leaves: continuation value when streets remain, exact SD on river
  * two raises, then terminal / continuation
  * reverse-translated abstract sizes

Flop/turn build a pairwise equity matrix over sampled runouts. River is exact.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from poker.evaluator import evaluate
from poker.game import Action, ActionType, PlayerView, Street
from strategy.abstraction import (
    abstract_bet_puts,
    abstract_facing_puts,
    continuation_extra,
)
from strategy.cfr_core import payoff_matrix_fast
from strategy.nodelock import NodeLock, apply_to_strategy
from strategy.rake import RakeConfig
from strategy.strength import board_combo_score

Combo = Tuple[int, int]


def _val(combo: Combo, board: Sequence[int]) -> int:
    return evaluate([combo[0], combo[1]] + list(board))


def _blocks(a: Combo, b: Combo) -> bool:
    return a[0] == b[0] or a[0] == b[1] or a[1] == b[0] or a[1] == b[1]


def subsample(
    combos: Sequence[Combo],
    weights: Sequence[float],
    k: int,
    keep: Combo | None,
    board: Sequence[int],
) -> Tuple[List[Combo], np.ndarray, int]:
    """Keep `keep` plus a *board-strength*-stratified sample of the range."""
    live = [(combos[i], float(weights[i])) for i in range(len(combos)) if weights[i] > 1e-12]
    if keep is not None:
        ks = (min(keep[0], keep[1]), max(keep[0], keep[1]))
        if not any(c == ks or set(c) == set(ks) for c, _ in live):
            live.append((ks, 1.0))
    if len(live) <= k:
        cs = [c for c, _ in live]
        w = np.array([wt for _, wt in live], dtype=np.float64)
        idx = 0
        if keep is not None:
            ks = set(keep)
            for i, c in enumerate(cs):
                if set(c) == ks:
                    idx = i
                    break
        return cs, w, idx

    scored = [(board_combo_score(c, board) if len(board) >= 3 else 0.0, c, wt) for c, wt in live]
    scored.sort()
    keep_set = set(keep) if keep is not None else set()
    chosen: List[Tuple[Combo, float]] = []
    step = max(1, len(scored) // k)
    used = set()
    for i in range(0, len(scored), step):
        if len(chosen) >= k:
            break
        _, c, wt = scored[i]
        chosen.append((c, wt))
        used.add(c)
    if keep is not None:
        ks = (min(keep[0], keep[1]), max(keep[0], keep[1]))
        if ks not in used and tuple(sorted(keep)) not in used:
            chosen[0] = (ks, 1.0)
    while len(chosen) < k:
        for _, c, wt in reversed(scored):
            if c not in {x[0] for x in chosen}:
                chosen.append((c, wt))
                break
        else:
            break
    cs = [c for c, _ in chosen[:k]]
    w = np.array([wt for _, wt in chosen[:k]], dtype=np.float64)
    idx = 0
    for i, c in enumerate(cs):
        if keep is not None and set(c) == keep_set:
            idx = i
            w[i] = max(w[i], 1.0)
            break
    return cs, w, idx


def payoff_matrix(
    hero: Sequence[Combo],
    vill: Sequence[Combo],
    board: Sequence[int],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """P[i,j] = hero's expected pot *share* in [0,1]. Blocked pairs are NaN."""
    return payoff_matrix_fast(hero, vill, board, n_samples, rng)


def _regret_match(r: np.ndarray) -> np.ndarray:
    r = np.nan_to_num(np.asarray(r, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
    pos = np.maximum(r, 0.0)
    s = pos.sum(axis=-1, keepdims=True)
    n = r.shape[-1]
    out = np.full_like(pos, 1.0 / max(n, 1))
    np.divide(pos, s, out=out, where=s > 1e-12)
    return np.nan_to_num(out, nan=1.0 / max(n, 1), posinf=0.0, neginf=0.0)


def _as1d(x, n: int) -> np.ndarray:
    """Force a 1-D EV vector. Python sum() of mixed (n,) / (n,1) raises shape-mismatch."""
    a = np.nan_to_num(np.asarray(x, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if a.size == n:
        return a
    if a.size == 1:
        return np.full(n, float(a[0]))
    out = np.zeros(n, dtype=np.float64)
    m = min(a.size, n)
    out[:m] = a[:m]
    return out


def _prob_reach(x, n: int) -> np.ndarray:
    """Finite probability vector. Zero/inf reach used to explode extra @ r."""
    a = np.maximum(_as1d(x, n), 0.0)
    s = float(a.sum())
    if not np.isfinite(s) or s <= 1e-15:
        return np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    return a / s


def _blocked_shares(
    P: np.ndarray,
    r0n: np.ndarray,
    r1n: np.ndarray,
    extra_mat: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pot share and extra vs opponent reach with blocking combos removed.

    Filling blocked pairs with 0.5 let Ax/Kx keep phantom equity vs the
    same-suit cards that cannot exist, which made calling look too good.
    """
    H, V = P.shape
    live = np.isfinite(P)
    Psafe = np.clip(np.nan_to_num(P, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    w1 = r1n[None, :] * live
    s1 = w1.sum(axis=1, keepdims=True)
    w1 = np.divide(w1, s1, out=np.zeros_like(w1), where=s1 > 1e-15)
    empty1 = (s1.reshape(-1) <= 1e-15)
    if np.any(empty1):
        # Fully blocked vs this sample: no opponent, no phantom extra.
        w1[empty1] = 0.0
    hero_share = np.clip(np.sum(Psafe * w1, axis=1), 0.0, 1.0)

    w0 = r0n[:, None] * live
    s0 = w0.sum(axis=0, keepdims=True)
    w0 = np.divide(w0, s0, out=np.zeros_like(w0), where=s0 > 1e-15)
    empty0 = (s0.reshape(-1) <= 1e-15)
    if np.any(empty0):
        w0[:, empty0] = 0.0
    vill_share = np.clip(np.sum((1.0 - Psafe) * w0, axis=0), 0.0, 1.0)

    extra_h = np.zeros(H, dtype=np.float64)
    extra_v = np.zeros(V, dtype=np.float64)
    if extra_mat is not None:
        em = np.nan_to_num(np.asarray(extra_mat, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if em.shape == (H, V):
            extra_h = np.sum(em * w1, axis=1)
            extra_v = np.sum(em * w0, axis=0)
    return hero_share, vill_share, extra_h, extra_v


def _mix_columns(stacked: np.ndarray, masses) -> np.ndarray:
    """Reach-weighted mix of EV columns. Inf/NaN mass must not leak into BR."""
    m = np.nan_to_num(np.asarray(masses, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    m = np.maximum(m, 0.0)
    tot = float(m.sum())
    if stacked.ndim != 2 or stacked.shape[1] == 0:
        return np.zeros(stacked.shape[0] if stacked.ndim else 0, dtype=np.float64)
    if tot <= 1e-15 or not np.isfinite(tot):
        return np.nan_to_num(stacked[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    w = m / tot
    ev = stacked @ w
    return np.nan_to_num(ev, nan=0.0, posinf=0.0, neginf=0.0)


def _sizes(
    pot: float,
    stack: float,
    to_call: float,
    spr: float,
    library_raise: bool = False,
    ip: bool = True,
) -> List[Tuple[str, float]]:
    if to_call > 0:
        return abstract_facing_puts(pot, stack, to_call, spr, library_raise=library_raise)
    return abstract_bet_puts(pot, stack, spr, library_first=library_raise, ip=ip)


class _Node:
    __slots__ = (
        "kind",
        "player",
        "acts",
        "put",
        "inv0",
        "inv1",
        "pot",
        "folder",
        "stk0",
        "stk1",
        "children",
        "hid",
    )

    def __init__(self, kind: str):
        self.kind = kind  # 'dec' | 'fold' | 'sd'
        self.player = 0
        self.acts: List[str] = []
        self.put: List[float] = []
        self.inv0 = 0.0
        self.inv1 = 0.0
        self.pot = 0.0
        self.folder = -1
        self.stk0 = 0.0
        self.stk1 = 0.0
        self.children: List[int] = []
        self.hid = 0


def build_tree(
    pot: float,
    cap: float,
    to_call: float,
    hero_oop: bool,
    facing: bool,
    raises_left: int = 2,
    lock_spr: Optional[float] = None,
    library_raise: bool = False,
    stacks: Optional[Tuple[float, float]] = None,
) -> Tuple[List[_Node], int]:
    """Hero is always player 0. `raises_left` is how many remaining raises in the street.

    `stacks` is (hero, villain) remaining. Default clones `cap` for both.
    An all-in villain must be 0 — cloning cap priced a leftover donk after
    hero checked into a jammed side pot.
    """
    nodes: List[_Node] = []
    hid_ctr = [0]

    def add() -> int:
        nodes.append(_Node("dec"))
        hid_ctr[0] += 1
        nodes[-1].hid = hid_ctr[0]
        return len(nodes) - 1

    def term_fold(pot_now: float, inv0: float, inv1: float, folder: int, stk0: float, stk1: float) -> int:
        n = _Node("fold")
        n.pot, n.inv0, n.inv1, n.folder = pot_now, inv0, inv1, folder
        n.stk0, n.stk1 = stk0, stk1
        nodes.append(n)
        return len(nodes) - 1

    def term_sd(pot_now: float, inv0: float, inv1: float, stk0: float, stk1: float) -> int:
        n = _Node("sd")
        n.pot, n.inv0, n.inv1 = pot_now, inv0, inv1
        n.stk0, n.stk1 = stk0, stk1
        nodes.append(n)
        return len(nodes) - 1

    def rec(
        player: int,
        pot_now: float,
        inv: Tuple[float, float],
        stacks: Tuple[float, float],
        to_c: float,
        raises_left_n: int,
    ) -> int:
        nid = add()
        node = nodes[nid]
        node.player = player
        node.pot = pot_now
        node.stk0, node.stk1 = stacks[0], stacks[1]
        stk = stacks[player]
        spr = min(stacks[0], stacks[1]) / max(pot_now, 1.0)
        # Opponent already all-in: betting is uncalled and leftover cannot donk.
        # Cloning cap for both priced a phantom barrel into a jammed side pot.
        if to_c <= 0.5 and min(stacks[0], stacks[1]) <= 0.5:
            node.acts.append("check")
            node.put.append(0.0)
            node.children.append(term_sd(pot_now, inv[0], inv[1], stacks[0], stacks[1]))
            return nid
        # Table effective_stack includes all-in 0. LibraryGTO jams; acting-cap
        # SPR is deep and used to drop the all-in node so jam mass snapped to 125%.
        if lock_spr is not None and float(lock_spr) < spr:
            spr = float(lock_spr)
        raw = _sizes(
            pot_now,
            stk,
            to_c,
            spr,
            library_raise=library_raise,
            ip=(not hero_oop) if player == 0 else True,
        )
        for name, put in raw:
            if to_c > 0 and name not in ("fold", "call") and raises_left_n <= 0:
                continue
            node.acts.append(name)
            node.put.append(put)
            if name == "fold":
                node.children.append(
                    term_fold(pot_now, inv[0], inv[1], player, stacks[0], stacks[1])
                )
                continue
            pay = put
            ninv = [inv[0], inv[1]]
            ninv[player] += pay
            npot = pot_now + pay
            nstk = [stacks[0], stacks[1]]
            nstk[player] -= pay
            if name == "allin":
                nstk[player] = 0.0
            if name == "check" and to_c <= 0:
                if player == 0 and hero_oop:
                    node.children.append(
                        rec(1, npot, (ninv[0], ninv[1]), (nstk[0], nstk[1]), 0.0, raises_left_n)
                    )
                else:
                    # IP check closes: the OOP player already checked (or is
                    # all-in). Giving them a second first-in priced a phantom donk.
                    node.children.append(term_sd(npot, ninv[0], ninv[1], nstk[0], nstk[1]))
                continue
            if name == "call" or (to_c > 0 and pay <= to_c + 1e-6):
                node.children.append(term_sd(npot, ninv[0], ninv[1], nstk[0], nstk[1]))
                continue
            # Opponent already put `to_c` (in the pot). Seed that so a call
            # vs this raise is put - to_c, not a second full put.
            if to_c > 0:
                ninv[1 - player] = max(float(ninv[1 - player]), float(to_c))
            new_to_call = ninv[player] - ninv[1 - player]
            next_raises = raises_left_n - 1 if to_c > 0 else raises_left_n
            if new_to_call <= 0.5 or nstk[1 - player] <= 0.5:
                node.children.append(term_sd(npot, ninv[0], ninv[1], nstk[0], nstk[1]))
            else:
                node.children.append(
                    rec(
                        1 - player,
                        npot,
                        (ninv[0], ninv[1]),
                        (nstk[0], nstk[1]),
                        new_to_call,
                        next_raises,
                    )
                )
        if not node.acts:
            return term_sd(pot_now, inv[0], inv[1], stacks[0], stacks[1])
        return nid

    if stacks is None:
        hstk = vstk = float(cap)
    else:
        hstk, vstk = (max(0.0, float(x)) for x in stacks)
    root_stk = (hstk, vstk)
    if to_call > 0:
        root = rec(0, pot, (0.0, 0.0), root_stk, to_call, max(1, raises_left))
    else:
        root = rec(0, pot, (0.0, 0.0), root_stk, 0.0, max(1, raises_left))
    _ = facing
    return nodes, root


def _villain_library_strats(
    nodes: Sequence[_Node],
    vill_combos: Sequence[Combo],
    board: Sequence[int],
    pot_type: int,
    streets_left: int,
    hero_oop: bool,
    spr: float = 13.0,
    perc: Optional[Sequence[float]] = None,
) -> Dict[int, np.ndarray]:
    """Fix villain at the distilled library mix so hero best-responds to a library player."""
    try:
        from strategy.library import mix_for_combo, mix_on_acts
    except Exception:
        return {}
    street = {2: 0, 1: 1, 0: 2}.get(int(streets_left), 2)
    if perc is not None and len(perc) == len(vill_combos):
        perc_l = [float(x) for x in perc]
    else:
        sc = [board_combo_score(c, board) for c in vill_combos] if len(board) >= 3 else [0.5] * len(vill_combos)
        order = sorted(range(len(vill_combos)), key=lambda i: sc[i])
        n = max(len(vill_combos) - 1, 1)
        perc_l = [0.5] * len(vill_combos)
        for rank, i in enumerate(order):
            perc_l[i] = rank / n
    out: Dict[int, np.ndarray] = {}
    ip = hero_oop  # villain IP iff hero is OOP
    # LibraryGTO uses view.effective_stack, which includes all-in 0, so
    # spr≈0 is the jam gate. Tree cap is acting stacks (side-pot barrels);
    # do not lock a deep mix while they jam.
    spr_v = float(spr) if spr > 0.05 else 0.01
    table_jammed = spr_v <= 0.05
    for nid, node in enumerate(nodes):
        if node.kind != "dec" or node.player != 1:
            continue
        facing = "fold" in node.acts
        frac = 0.75 if facing else 0.0
        if facing and node.put:
            tc = 0.0
            for name, put in zip(node.acts, node.put):
                if name == "call":
                    tc = float(put)
                    break
            frac = tc / max(float(node.pot) - tc, 1.0) if tc > 0 else 0.75
        cap_n = min(float(node.stk0), float(node.stk1))
        if table_jammed or cap_n <= 0.05:
            node_spr = 0.01
        else:
            node_spr = cap_n / max(float(node.pot), 1.0)
            if node_spr < 0.05:
                node_spr = 0.01
        sig = np.zeros((len(vill_combos), len(node.acts)), dtype=np.float64)
        for i, combo in enumerate(vill_combos):
            mix = mix_for_combo(combo, board, perc_l[i], facing, frac, street, pot_type, ip, node_spr)
            mapped = mix_on_acts(mix, node.acts, facing)
            if mapped.shape[0] == sig.shape[1]:
                sig[i] = mapped
            else:
                sig[i] = 1.0 / max(sig.shape[1], 1)
        out[nid] = sig
    return out


def solve(
    P: np.ndarray,
    hw: np.ndarray,
    vw: np.ndarray,
    pot: float,
    cap: float,
    to_call: float,
    hero_oop: bool,
    iterations: int,
    streets_left: int = 0,
    raises_left: int = 2,
    hero_combos: Optional[Sequence[Combo]] = None,
    vill_combos: Optional[Sequence[Combo]] = None,
    board: Optional[Sequence[int]] = None,
    pot_type: int = 0,
    node_lock: Optional[NodeLock] = None,
    rake_percent: float = 0.0,
    rake_cap: float = 0.0,
    villain_library: bool = False,
    sixmax: bool = False,
    vill_perc: Optional[Sequence[float]] = None,
    lock_spr: Optional[float] = None,
    library_raise: bool = False,
    stacks: Optional[Tuple[float, float]] = None,
) -> Tuple[List[_Node], int, np.ndarray]:
    """Return (nodes, root, strategy[H, n_root_actions]).

    Per-combo regrets for both players. Linear MCCFR averaging. Depth-limited
    leaves when `streets_left > 0` use blueprint continuation values.
    """
    H = P.shape[0]
    V = P.shape[1]
    hw = np.nan_to_num(np.maximum(_as1d(hw, H), 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    vw = np.nan_to_num(np.maximum(_as1d(vw, V), 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    if float(hw.sum()) <= 1e-15:
        hw = np.ones(H, dtype=np.float64)
    if float(vw.sum()) <= 1e-15:
        vw = np.ones(V, dtype=np.float64)
    nodes, root = build_tree(
        pot, cap, to_call, hero_oop, to_call > 0, raises_left=raises_left,
        lock_spr=lock_spr, library_raise=library_raise, stacks=stacks,
    )
    dec = [i for i, n in enumerate(nodes) if n.kind == "dec"]
    regrets: Dict[int, np.ndarray] = {}
    strat_sum: Dict[int, np.ndarray] = {}
    for i in dec:
        n_a = len(nodes[i].acts)
        if nodes[i].player == 0:
            regrets[i] = np.zeros((H, n_a))
            strat_sum[i] = np.zeros((H, n_a))
        else:
            regrets[i] = np.zeros((V, n_a))
            strat_sum[i] = np.zeros((V, n_a))

    vill_lib: Dict[int, np.ndarray] = {}
    if villain_library and vill_combos is not None and board is not None and len(board) >= 3:
        spr = float(lock_spr) if lock_spr is not None else (float(cap) / max(float(pot), 1.0))
        vill_lib = _villain_library_strats(
            nodes, vill_combos, board, pot_type, streets_left, hero_oop, spr,
            perc=vill_perc,
        )
    cv_hv = None
    cv_is_chips = False
    if streets_left > 0 and hero_combos is not None and vill_combos is not None and board is not None:
        try:
            from strategy.library import continuation_matrix as lib_cv

            stk0, stk1 = (cap, cap) if stacks is None else (
                max(0.0, float(stacks[0])), max(0.0, float(stacks[1])),
            )
            mat = lib_cv(
                hero_combos, vill_combos, board, pot, stk0, stk1, pot_type, streets_left,
                sixmax=sixmax,
            )
            if mat is not None and float(np.sum(np.abs(mat))) > 1e-8:
                cv_hv = mat
                cv_is_chips = True
        except Exception:
            cv_hv = None
        if cv_hv is None:
            try:
                from strategy.blueprint import (
                    refresh_hu,
                    refresh_hu_v3,
                    refresh_six_v3,
                    v3_leaves_allowed,
                )
                from strategy.card_abs import street_bucket, street_bucket_v3

                pack = None
                if v3_leaves_allowed():
                    if sixmax:
                        pack = refresh_six_v3() or refresh_hu_v3()
                    else:
                        pack = refresh_hu_v3() or refresh_six_v3()
                if pack is None:
                    pack = refresh_hu()
                cv = pack.get("cv") if pack else None
                bkt_fn = street_bucket_v3 if cv is not None and cv.shape[0] >= 72 else street_bucket
                if cv is not None and float(np.sum(np.abs(cv))) > 1e-8:
                    hb = np.clip([bkt_fn(hc, board) for hc in hero_combos], 0, cv.shape[0] - 1)
                    vb = np.clip([bkt_fn(vc, board) for vc in vill_combos], 0, cv.shape[1] - 1)
                    pt = int(max(0, min(cv.shape[2] - 1, pot_type)))
                    cv_hv = cv[np.ix_(hb, vb)][:, :, pt]
                    cv_is_chips = float(np.mean(np.abs(cv_hv))) > 2.5
            except Exception:
                cv_hv = None

    def showdown(n: _Node, r0: np.ndarray, r1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r0n = _prob_reach(r0, H)
        r1n = _prob_reach(r1, V)

        def _net(gross: float) -> float:
            if rake_percent <= 1e-12:
                return gross
            return max(0.0, gross - min(rake_cap, gross * rake_percent))

        if cv_hv is not None:
            cap_s = min(float(n.stk0), float(n.stk1))
            if cap_s <= 0.5:
                extra_h = np.zeros(H, dtype=np.float64)
                extra_v = np.zeros(V, dtype=np.float64)
            else:
                scale = cap_s / max(float(cap), 1.0)
                # Library continuation_matrix is already chips. Remultiplying
                # by pot on small SPR pots (~few-chip extras) made calling
                # look like a 10x overstated future vs LibraryGTO barrels.
                if cv_is_chips:
                    extra_mat = cv_hv * scale
                else:
                    extra_mat = pot * cv_hv * max(streets_left, 1) * scale
                # Root-priced extras. After a cbet the leaf pot is larger;
                # keeping the 6bb future made calling cheap vs barrels.
                extra_mat = extra_mat * (float(n.pot) / max(float(pot), 1.0))
                extra_mat = np.clip(extra_mat, -cap_s, cap_s)
                extra_mat = np.nan_to_num(extra_mat, nan=0.0, posinf=cap_s, neginf=-cap_s)
            hero_share, vill_share, extra_h, extra_v = _blocked_shares(
                P, r0n, r1n, extra_mat if cap_s > 0.5 else None,
            )
            # Signed extra is additional EV, not extra chips both put in.
            # Clamping to ≥0 made dogs check-down (calling-station vs barrels).
            hero_u = hero_share * _net(n.pot) - n.inv0 + extra_h
            vill_u = vill_share * _net(n.pot) - n.inv1 + extra_v
            return _as1d(hero_u, H), _as1d(vill_u, V)
        # No combo CV: do not dump a geometric extra into the pot (that
        # made dogs check-down). Current-street showdown only.
        hero_share, vill_share, _eh, _ev = _blocked_shares(P, r0n, r1n, None)
        hero_u = hero_share * _net(n.pot) - n.inv0
        vill_u = vill_share * _net(n.pot) - n.inv1
        return _as1d(hero_u, H), _as1d(vill_u, V)

    def cfr(nid: int, r0: np.ndarray, r1: np.ndarray, upd: int, t: int) -> Tuple[np.ndarray, np.ndarray]:
        r0 = _as1d(r0, H)
        r1 = _as1d(r1, V)
        n = nodes[nid]
        if n.kind == "fold":
            if n.folder == 0:
                return np.full(H, -n.inv0), np.full(V, n.pot - n.inv1)
            return np.full(H, n.pot - n.inv0), np.full(V, -n.inv1)
        if n.kind == "sd":
            return showdown(n, r0, r1)
        n_a = len(n.acts)
        if n.player == 0:
            sig = _regret_match(regrets[nid])
            if upd == 0:
                strat_sum[nid] += t * r0[:, None] * sig
            au_h = np.zeros((n_a, H))
            au_v = np.zeros((n_a, V))
            for a in range(n_a):
                au_h[a], au_v[a] = cfr(n.children[a], r0 * sig[:, a], r1, upd, t)
            ev_h = np.einsum("ia,ai->i", sig, au_h)
            mass0 = float(r0.sum())
            if mass0 > 1e-15:
                w = np.array([float((r0 * sig[:, a]).sum()) / mass0 for a in range(n_a)])
                ev_v = w.dot(au_v)
            else:
                ev_v = au_v[0]
            if upd == 0:
                mass1 = float(r1.sum())
                regrets[nid] += mass1 * (au_h.T - ev_h[:, None])
            return ev_h, ev_v
        sig = _regret_match(regrets[nid])
        if n.player == 1 and nid in vill_lib:
            sig = vill_lib[nid]
        elif n.player == 1 and node_lock is not None:
            sig = apply_to_strategy(sig, n.acts, node_lock)
        if upd == 1:
            strat_sum[nid] += t * r1[:, None] * sig
        au_h = []
        au_v = []
        masses = np.zeros(n_a)
        for a in range(n_a):
            r1n = r1 * sig[:, a]
            masses[a] = r1n.sum()
            h, v = cfr(n.children[a], r0, r1n, upd, t)
            au_h.append(h)
            au_v.append(v)
        stacked_h = np.stack([_as1d(x, H) for x in au_h], axis=1)
        ev_h = _mix_columns(stacked_h, masses)
        stacked_v = np.stack([_as1d(x, V) for x in au_v], axis=1)
        ev_v = np.einsum("va,va->v", sig, stacked_v)
        if upd == 1 and nid not in vill_lib:
            m0 = float(r0.sum())
            regrets[nid] += m0 * (stacked_v - ev_v[:, None])
        return ev_h, ev_v

    r0 = np.nan_to_num(hw / max(float(hw.sum()), 1e-12), nan=0.0, posinf=0.0, neginf=0.0)
    r1 = np.nan_to_num(vw / max(float(vw.sum()), 1e-12), nan=0.0, posinf=0.0, neginf=0.0)
    # Locked library villains do not adapt. Live policy is _eval_h argmax;
    # MCCFR mix is discarded, so skip the walk (19601 speed, same BR).
    if not vill_lib:
        upd_players = (0, 1)
        for t in range(1, iterations + 1):
            for p in upd_players:
                cfr(root, r0, r1, p, t)
            if t % 5 == 0:
                a = t ** 1.5 / (t ** 1.5 + 1.0)
                for arr in regrets.values():
                    arr *= a

    if vill_lib:
        n_a = len(nodes[root].acts)
        last = {nid: _regret_match(regrets[nid]) for nid in regrets}

        def _eval_h(nid: int, rr0: np.ndarray, rr1: np.ndarray) -> np.ndarray:
            n = nodes[nid]
            if n.kind == "fold":
                if n.folder == 0:
                    return np.full(H, -n.inv0)
                return np.full(H, n.pot - n.inv0)
            if n.kind == "sd":
                hu, _vu = showdown(n, rr0, rr1)
                return hu
            na = len(n.acts)
            if n.player == 0:
                # Later hero nodes are a fresh live search (BR). Mixing the
                # last-iterate here undervalues check vs a bet-now that the
                # under-converged mix will overfold.
                parts = [
                    _as1d(_eval_h(n.children[a], rr0, rr1), H)
                    for a in range(na)
                ]
                stacked = np.nan_to_num(
                    np.stack(parts, axis=1), nan=-1e12, posinf=1e12, neginf=-1e12
                )
                return np.max(stacked, axis=1)
            sig = vill_lib[nid] if nid in vill_lib else last[nid]
            masses = np.zeros(na)
            parts = []
            for a in range(na):
                r1n = _as1d(rr1 * sig[:, a], V)
                masses[a] = r1n.sum()
                parts.append(_eval_h(n.children[a], rr0, r1n))
            stacked = np.stack([_as1d(p, H) for p in parts], axis=1)
            return _mix_columns(stacked, masses)

        evs = np.zeros((H, n_a))
        for a in range(n_a):
            evs[:, a] = _eval_h(nodes[root].children[a], r0, r1)
        evs = np.nan_to_num(evs, nan=-1e12, posinf=1e12, neginf=-1e12)
        strat = np.zeros_like(evs)
        strat[np.arange(H), np.argmax(evs, axis=1)] = 1.0
    else:
        ss = strat_sum[root]
        tot = ss.sum(axis=1, keepdims=True)
        n_a = ss.shape[1]
        strat = np.divide(ss, tot, out=np.full_like(ss, 1.0 / max(n_a, 1)), where=tot > 1e-12)
    # stash average strategies for exploitability
    solve.last_strat_sum = strat_sum  # type: ignore[attr-defined]
    solve.last_nodes = nodes  # type: ignore[attr-defined]
    return nodes, root, strat


def root_action_evs(
    nodes: Sequence[_Node],
    root: int,
    strat_sum: Dict[int, np.ndarray],
    P: np.ndarray,
    hw: np.ndarray,
    vw: np.ndarray,
    streets_left: int = 0,
) -> np.ndarray:
    """Hero EV of each root action vs villain's average strategy. Shape (H, n_a)."""
    H, V = P.shape
    hw = np.nan_to_num(np.maximum(_as1d(hw, H), 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    vw = np.nan_to_num(np.maximum(_as1d(vw, V), 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    if float(hw.sum()) <= 1e-15:
        hw = np.ones(H, dtype=np.float64)
    if float(vw.sum()) <= 1e-15:
        vw = np.ones(V, dtype=np.float64)
    r0 = _prob_reach(hw, H)
    r1 = _prob_reach(vw, V)

    def avg_sig(nid: int) -> np.ndarray:
        ss = strat_sum[nid]
        tot = ss.sum(axis=1, keepdims=True)
        n_a = ss.shape[1]
        return np.where(tot > 1e-12, ss / tot, 1.0 / n_a)

    def ev(nid: int, rr0: np.ndarray, rr1: np.ndarray) -> np.ndarray:
        n = nodes[nid]
        if n.kind == "fold":
            if n.folder == 0:
                return np.full(H, -n.inv0)
            return np.full(H, n.pot - n.inv0)
        if n.kind == "sd":
            extra = continuation_extra(n.pot, n.stk0, n.stk1, streets_left)
            extra_mat = np.full(P.shape, extra, dtype=np.float64)
            extra_mat = np.where(np.isfinite(P), extra_mat, np.nan)
            hero_share, _vs, extra_h, _ev = _blocked_shares(
                P, _prob_reach(rr0, H), _prob_reach(rr1, V), extra_mat,
            )
            return _as1d(hero_share * n.pot - n.inv0 + extra_h, H)
        n_a = len(n.acts)
        sig = avg_sig(nid)
        if n.player == 0:
            out = np.zeros((n_a, H))
            for a in range(n_a):
                out[a] = ev(n.children[a], rr0 * sig[:, a], rr1)
            return np.einsum("ia,ai->i", sig, out)
        masses = np.zeros(n_a)
        parts = []
        for a in range(n_a):
            r1n = rr1 * sig[:, a]
            masses[a] = r1n.sum()
            parts.append(ev(n.children[a], rr0, r1n))
        stacked = np.stack([_as1d(p, H) for p in parts], axis=1)
        return _mix_columns(stacked, masses)

    n = nodes[root]
    n_a = len(n.acts)
    out = np.zeros((H, n_a))
    sig_v_placeholder = None
    _ = sig_v_placeholder
    for a in range(n_a):
        out[:, a] = ev(n.children[a], r0, r1)
    return out


def best_response_gap(
    P: np.ndarray,
    hw: np.ndarray,
    vw: np.ndarray,
    pot: float,
    cap: float,
    to_call: float,
    hero_oop: bool,
    iterations: int,
    streets_left: int = 0,
    raises_left: int = 2,
) -> dict:
    """Nash-gap proxy in chips: BR_EV − mix_EV at the root vs villain average."""
    nodes, root, strat = solve(
        P, hw, vw, pot, cap, to_call, hero_oop, iterations,
        streets_left=streets_left, raises_left=raises_left,
    )
    strat_sum = getattr(solve, "last_strat_sum", None)
    if strat_sum is None:
        return {"spot": "facing" if to_call > 0 else "first_in", "gap": 0.0, "acts": nodes[root].acts}
    evs = root_action_evs(nodes, root, strat_sum, P, hw, vw, streets_left=streets_left)
    hw_n = hw / max(float(hw.sum()), 1e-12)
    br = float(np.dot(hw_n, evs.max(axis=1)))
    mix = float(np.dot(hw_n, (strat * evs).sum(axis=1)))
    gap = max(0.0, br - mix)
    return {
        "spot": "facing" if to_call > 0 else "first_in",
        "nash_ev": mix,
        "br_ev": br,
        "gap": gap,
        "gap_mbb": gap,  # caller converts with bb
        "acts": nodes[root].acts,
        "bet_freq": float(np.dot(hw_n, strat[:, [i for i, n in enumerate(nodes[root].acts) if n not in ("check", "fold")]].sum(axis=1)))
        if any(n not in ("check", "fold") for n in nodes[root].acts)
        else 0.0,
    }


def _acting_count(view: PlayerView) -> int:
    """Players who can still bet. All-in is in the pot but does not barrel."""
    n = int(view.num_players)
    all_in = getattr(view, "all_in", None)
    if all_in is None:
        all_in = [False] * n
    return sum(1 for i in range(n) if not view.folded[i] and not all_in[i])


def _acting_cap(view: PlayerView) -> float:
    """Min stack among players who can still bet.

    view.effective_stack includes all-in 0, which built a 1-chip HU tree
    in cloned leftover pots while the side pot was 100bb.
    """
    n = int(view.num_players)
    all_in = getattr(view, "all_in", None)
    if all_in is None:
        all_in = [False] * n
    caps = [
        float(view.stacks[i])
        for i in range(n)
        if not view.folded[i] and not all_in[i]
    ]
    if not caps:
        return float(max(view.effective_stack, 1))
    return float(max(min(caps), 1))


def _seat_oop(view: PlayerView, seat: int) -> bool:
    """True iff `seat` is first to act among live players (SB in 3-way)."""
    n = view.num_players
    for k in range(1, n + 1):
        i = (view.button + k) % n
        if not view.folded[i] and not view.all_in[i]:
            return i == seat
    return True


def _hero_oop(view: PlayerView) -> bool:
    return _seat_oop(view, view.hero)


def _to_engine(view: PlayerView, name: str, put: float) -> Action:
    stack = view.stacks[view.hero]
    to_call = view.to_call
    if name == "fold":
        return Action(ActionType.CHECK) if to_call <= 0 else Action(ActionType.FOLD)
    if name == "check":
        return Action(ActionType.CHECK) if to_call <= 0 else Action(ActionType.CALL, min(to_call, stack))
    if name == "call":
        if to_call <= 0:
            return Action(ActionType.CHECK)
        return Action(ActionType.CALL, min(to_call, stack))
    put_i = int(round(put))
    put_i = max(0, min(put_i, stack))
    if put_i >= stack:
        return Action(ActionType.ALL_IN, stack)
    if to_call <= 0:
        put_i = max(put_i, min(view.bb, stack))
        return Action(ActionType.BET, put_i) if put_i < stack else Action(ActionType.ALL_IN, stack)
    if put_i <= to_call:
        return Action(ActionType.CALL, min(to_call, stack))
    min_put = view.min_raise_to - view.bets[view.hero]
    if put_i < min_put:
        put_i = min_put if stack >= min_put else stack
    if put_i >= stack:
        return Action(ActionType.ALL_IN, stack)
    return Action(ActionType.RAISE, put_i)


_CACHE: dict = {}
_CACHE_MAX = 256


def _range_fp(combos, weights, k: int = 10):
    pairs = sorted(
        ((float(w), int(c[0]), int(c[1])) for c, w in zip(combos, weights) if w > 1e-12),
        reverse=True,
    )[:k]
    tot = round(sum(w for w, _, _ in pairs), 3)
    return (tot, tuple(pairs))


def _streets_left(street: Street) -> int:
    return {Street.FLOP: 2, Street.TURN: 1, Street.RIVER: 0}.get(street, 0)


def _sample_weights(probs, villain_library: bool) -> List[float]:
    """Vs a fixed library, play the best action (small mix hedge)."""
    p = np.maximum(np.asarray(probs, dtype=np.float64).reshape(-1), 0.0)
    if p.size == 0:
        return [1.0]
    s = float(p.sum())
    if s <= 1e-15:
        return [1.0] * int(p.size)
    p = p / s
    if villain_library:
        br = np.zeros_like(p)
        br[int(np.argmax(p))] = 1.0
        return [float(x) for x in br]
    return [float(x) for x in p]


def live_policy(
    view: PlayerView,
    hero_combos: Sequence[Combo],
    hero_w: Sequence[float],
    vill_combos: Sequence[Combo],
    vill_w: Sequence[float],
    rng,
    iterations: int = 40,
    max_combos: int = 40,
    runouts: int = 6,
    node_lock: Optional[NodeLock] = None,
    rake: Optional[RakeConfig] = None,
    villain_library: Optional[bool] = None,
    raises_left: Optional[int] = None,
) -> Tuple[List[str], List[float], np.ndarray, Optional[int]]:
    """Return (act_names, puts, probs, nuts_override_index)."""
    hole = view.hole
    keep = (min(hole), max(hole))
    board = list(view.board)
    use_lib = True if villain_library is None else bool(villain_library)
    # Locked villains do not adapt to our range. Extra hero combos do not
    # change this hole's BR and dominate payoff cost vs LibraryGTO.
    hero_k = 1 if use_lib else max_combos
    hc, hw, hi = subsample(hero_combos, hero_w, hero_k, keep, board)
    vc, vw, _ = subsample(vill_combos, vill_w, max_combos, None, board)
    gen = np.random.default_rng(rng.randrange(1 << 30))
    samples = runouts if len(board) < 5 else 1
    P = payoff_matrix(hc, vc, board, n_samples=samples, rng=gen)
    pot = float(view.pot)
    cap = _acting_cap(view)
    hstk = float(view.stacks[view.hero])
    n_pl = int(view.num_players)
    all_in = getattr(view, "all_in", None)
    if all_in is None:
        all_in = [False] * n_pl
    acting_others = [
        float(view.stacks[i])
        for i in range(n_pl)
        if i != view.hero and not view.folded[i] and not all_in[i]
    ]
    # All-in villain remaining is 0. Cloning cap priced a leftover donk.
    vstk = min(acting_others) if acting_others else 0.0
    to_call = float(view.to_call)
    oop = _hero_oop(view)
    sl = _streets_left(view.street)
    from strategy.pot_type import classify as pot_kind

    pt = {"srp": 0, "3bet": 1, "4bet": 2}.get(pot_kind(view), 0)
    if view.pot / max(view.bb, 1) <= 3.5:
        pt = 0
    rpct = rake.percent if rake is not None and rake.active else 0.0
    rcap = (rake.cap_bb * view.bb) if rake is not None and rake.active else 0.0
    n_raises = int(raises_left) if raises_left is not None else (3 if pt >= 1 else 2)
    vill_perc = None
    if use_lib and len(board) >= 3:
        try:
            from strategy.library import combo_percentiles

            vill_perc = combo_percentiles(vc, board, vill_combos, vill_w)
        except Exception:
            vill_perc = None
    # True 3-way: undamped library extras. HU 20k (num_players==2) keeps the
    # HU MCCFR blend. 6-max leftover vs LibraryGTO is still lib_6max_100 —
    # blending HU blueprint there priced a different continuation than they play.
    nodes, root, strat = solve(
        P, hw, vw, pot, cap, to_call, oop, iterations=iterations,
        streets_left=sl, raises_left=n_raises,
        hero_combos=hc, vill_combos=vc, board=board, pot_type=pt,
        node_lock=node_lock, rake_percent=rpct, rake_cap=rcap,
        villain_library=bool(use_lib),
        sixmax=_acting_count(view) >= 3 or (
            bool(use_lib) and int(view.num_players) != 2
        ),
        vill_perc=vill_perc,
        lock_spr=float(view.effective_stack) / max(pot, 1.0),
        library_raise=bool(use_lib) and int(view.num_players) != 2,
        stacks=(hstk, vstk),
    )
    probs = np.maximum(strat[hi], 0.0)
    if probs.sum() <= 0:
        probs = np.ones_like(probs)
    probs = probs / probs.sum()
    acts = nodes[root].acts
    puts = nodes[root].put
    nuts_i = None
    hv = _val(keep, board) if len(board) >= 3 else 0
    vill_vals = [_val(c, board) for c in vc] if len(board) >= 5 else None
    if vill_vals and hv >= max(vill_vals) and len(board) == 5:
        best_i, best_put = 0, -1.0
        for i, (name, put) in enumerate(zip(acts, puts)):
            if name == "fold":
                continue
            if put >= best_put:
                best_put = put
                best_i = i
        nuts_i = best_i
    return acts, puts, probs, nuts_i


def cached_live_action(
    view: PlayerView,
    hero_combos: Sequence[Combo],
    hero_w: Sequence[float],
    vill_combos: Sequence[Combo],
    vill_w: Sequence[float],
    rng,
    iterations: int = 40,
    max_combos: int = 40,
    runouts: int = 6,
    node_lock: Optional[NodeLock] = None,
    rake: Optional[RakeConfig] = None,
    villain_library: Optional[bool] = None,
    raises_left: Optional[int] = None,
) -> Action:
    hole = (min(view.hole), max(view.hole))
    spr_b = round(_acting_cap(view) / max(view.pot, 1), 0)
    pot_b = round(view.pot / max(view.bb, 1), 0)
    tc_b = round(view.to_call / max(view.bb, 1), 1)
    lock_k = (0.0, 0.0, 0.0, 0.0) if node_lock is None else (
        node_lock.fold_bias, node_lock.call_bias, node_lock.bet_bias, node_lock.strength,
    )
    rake_k = (0.0, 0.0) if rake is None else (rake.percent, rake.cap_bb)
    from strategy.abstraction import face_node

    face_k = face_node(view.to_call, view.pot) if view.to_call > 0 else "first"
    from strategy.pot_type import classify as pot_kind

    key = (
        view.street.value,
        tuple(view.board),
        hole,
        spr_b,
        pot_b,
        tc_b,
        face_k,
        int(_hero_oop(view)),
        pot_kind(view),
        _range_fp(hero_combos, hero_w),
        _range_fp(vill_combos, vill_w),
        iterations,
        max_combos,
        runouts,
        lock_k,
        rake_k,
        True if villain_library is None else bool(villain_library),
        int(raises_left) if raises_left is not None else -1,
        int(view.players_in_hand),
        int(_acting_count(view)),
        int(view.num_players),
        round(float(view.effective_stack) / max(view.pot, 1), 2),
    )
    hit = _CACHE.get(key)
    if hit is None:
        pack = live_policy(
            view,
            hero_combos,
            hero_w,
            vill_combos,
            vill_w,
            rng,
            iterations=iterations,
            max_combos=max_combos,
            runouts=runouts,
            node_lock=node_lock,
            rake=rake,
            villain_library=villain_library,
            raises_left=raises_left,
        )
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = pack
        hit = pack
    acts, puts, probs, nuts_i = hit
    if nuts_i is not None:
        return _to_engine(view, acts[nuts_i], puts[nuts_i])
    use_lib = True if villain_library is None else bool(villain_library)
    weights = _sample_weights(probs, use_lib)
    choice = int(rng.choices(range(len(acts)), weights=weights, k=1)[0])
    return _to_engine(view, acts[choice], puts[choice])


def live_action(
    view: PlayerView,
    hero_combos: Sequence[Combo],
    hero_w: Sequence[float],
    vill_combos: Sequence[Combo],
    vill_w: Sequence[float],
    rng,
    iterations: int = 40,
    max_combos: int = 40,
    runouts: int = 6,
    node_lock: Optional[NodeLock] = None,
    rake: Optional[RakeConfig] = None,
    villain_library: Optional[bool] = None,
    raises_left: Optional[int] = None,
) -> Action:
    acts, puts, probs, nuts_i = live_policy(
        view, hero_combos, hero_w, vill_combos, vill_w, rng,
        iterations=iterations, max_combos=max_combos, runouts=runouts,
        node_lock=node_lock, rake=rake, villain_library=villain_library,
        raises_left=raises_left,
    )
    if nuts_i is not None:
        return _to_engine(view, acts[nuts_i], puts[nuts_i])
    use_lib = True if villain_library is None else bool(villain_library)
    weights = _sample_weights(probs, use_lib)
    choice = int(rng.choices(range(len(acts)), weights=weights, k=1)[0])
    return _to_engine(view, acts[choice], puts[choice])
