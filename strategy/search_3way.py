"""True 3-player nested public-tree CFR vs two live ranges.

Hero + primary + extra villain each have per-combo regrets. Showdown uses a
3-way pot-share tensor (not a 2-player reduction). The street tree is bounded: hero acts, then each remaining player
fold/call/raise/jam. A raise reopens to players who already acted, and
a call pays the increment (bet_level - already in), not a second full put.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from poker.game import Action, ActionType, PlayerView
from strategy.cfr_core import payoff_tensor3
from strategy.live_solve import (
    _acting_cap,
    _acting_count,
    _as1d,
    _blocked_shares,
    _hero_oop,
    _mix_columns,
    _prob_reach,
    _range_fp,
    _regret_match,
    _sample_weights,
    _streets_left,
    _to_engine,
    payoff_matrix,
    subsample,
)
from strategy.nodelock import NodeLock, apply_to_strategy
from strategy.rake import RakeConfig

Combo = Tuple[int, int]


def _three_way_shares(S0, S1, S2, r0, r1, r2):
    """Pot shares with blocked triples removed, not filled as 1/3."""
    live = np.isfinite(S0)
    F0 = np.clip(np.nan_to_num(S0, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    F1 = np.clip(np.nan_to_num(S1, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    F2 = np.clip(np.nan_to_num(S2, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    w0 = r1[None, :, None] * r2[None, None, :] * live
    z0 = w0.sum(axis=(1, 2), keepdims=True)
    w0 = np.divide(w0, z0, out=np.zeros_like(w0), where=z0 > 1e-15)
    u0 = np.sum(F0 * w0, axis=(1, 2))
    w1 = r0[:, None, None] * r2[None, None, :] * live
    z1 = w1.sum(axis=(0, 2), keepdims=True)
    w1 = np.divide(w1, z1, out=np.zeros_like(w1), where=z1 > 1e-15)
    u1 = np.sum(F1 * w1, axis=(0, 2))
    w2 = r0[:, None, None] * r1[None, :, None] * live
    z2 = w2.sum(axis=(0, 1), keepdims=True)
    w2 = np.divide(w2, z2, out=np.zeros_like(w2), where=z2 > 1e-15)
    u2 = np.sum(F2 * w2, axis=(0, 1))
    return (
        np.clip(u0, 0.0, 1.0),
        np.clip(u1, 0.0, 1.0),
        np.clip(u2, 0.0, 1.0),
    )


def _pair_scale_after_dead(S: np.ndarray, r_dead: np.ndarray, axis: int) -> np.ndarray:
    """Fraction of the other villain's reach that leaves this hero–villain pair live.

    HU extras ignored the third player, so Ahxx vs Khxx kept full future EV
    even when the extra villain's range was full of those aces/kings.
    """
    live = np.isfinite(S)
    n_dead = int(S.shape[axis])
    r = np.maximum(_as1d(r_dead, n_dead), 0.0)
    tot = float(r.sum())
    if tot <= 1e-15:
        if axis == 2:
            return np.zeros((S.shape[0], S.shape[1]), dtype=np.float64)
        return np.zeros((S.shape[0], S.shape[2]), dtype=np.float64)
    if axis == 2:
        mass = (live * r[None, None, :]).sum(axis=2)
    else:
        mass = (live * r[None, :, None]).sum(axis=1)
    return mass / tot


def _masked_pair_extra(
    S0: np.ndarray,
    extra_mat: Optional[np.ndarray],
    r0n: np.ndarray,
    r_vill: np.ndarray,
    r_other: np.ndarray,
    dead_axis: int,
    cap_s: float,
    orig_cap: float,
    node_pot: float = 0.0,
    orig_pot: float = 0.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """HU extra vs one villain, with blocked pairs removed from reach.

    A 0.5 fill treated AhKd vs Ahxx as a live coin-flip and diluted extra
    across combos that cannot exist. Collapse S0 so those pairs are NaN.

    `extra_mat` is priced on the root pot. After a cbet the leaf pot is
    larger; keeping root extras made calling look cheap vs LibraryGTO
    barrels. Scale by node/root pot, then remaining stacks, then clip.
    """
    if extra_mat is None or cap_s <= 0.5:
        return None, None
    m = np.nan_to_num(np.asarray(extra_mat, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    n_hero = int(S0.shape[0])
    n_vill = int(S0.shape[1] if dead_axis == 2 else S0.shape[2])
    if m.ndim == 1:
        m = m.reshape(n_hero, -1)
    if m.shape != (n_hero, n_vill):
        return None, None
    if orig_pot > 0.5:
        m = m * (max(float(node_pot), 0.0) / orig_pot)
    if orig_cap > 0.5:
        m = m * (cap_s / orig_cap)
    cap_s = max(float(cap_s), 1.0)
    m = np.clip(m, -cap_s, cap_s)
    m = np.nan_to_num(m, nan=0.0, posinf=cap_s, neginf=-cap_s)
    scale = _pair_scale_after_dead(S0, r_other, dead_axis)
    if scale.shape != m.shape:
        return None, None
    m = m * scale
    Ppair = _collapse_dead(S0, r_other, dead_axis)
    if Ppair.shape != m.shape:
        return None, None
    _hs, _vs, eh, ev = _blocked_shares(Ppair, r0n, r_vill, m)
    return _as1d(eh, n_hero), _as1d(ev, n_vill)


def _collapse_dead(S: np.ndarray, r_dead: np.ndarray, axis: int) -> np.ndarray:
    """2-D equity vs the live villain, folding player's cards as blockers."""
    live = np.isfinite(S)
    F = np.clip(np.nan_to_num(S, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if axis == 2:
        w = r_dead[None, None, :] * live
        z = w.sum(axis=2)
        num = (F * w).sum(axis=2)
    else:
        w = r_dead[None, :, None] * live
        z = w.sum(axis=1)
        num = (F * w).sum(axis=1)
    out = np.full(num.shape, np.nan, dtype=np.float64)
    np.divide(num, z, out=out, where=z > 1e-15)
    return out


class _N:
    __slots__ = ("kind", "player", "acts", "put", "inv", "pot", "stk", "folder", "live", "children")

    def __init__(self, kind: str):
        self.kind = kind
        self.player = 0
        self.acts: List[str] = []
        self.put: List[float] = []
        self.inv = (0.0, 0.0, 0.0)
        self.pot = 0.0
        self.stk = (0.0, 0.0, 0.0)
        self.folder = -1
        self.live = (1, 1, 1)
        self.children: List[int] = []


def _sizes_first(
    pot: float, stack: float, spr: float, ip: bool = True
) -> List[Tuple[str, float]]:
    """Bounded 3-way first-in. Deep SPR keeps 125% (library overbet).

    LibraryGTO IP jams below SPR 2.35 (check/jam). OOP only jams value
    below 2.2; air still donks. Jam-only OOP snapped those donks onto
    check (mix_on_acts) and search overfolded when they stabbed.
    """
    acts: List[Tuple[str, float]] = [("check", 0.0)]
    if stack <= 0.5:
        return acts
    if ip and spr < 2.35:
        acts.append(("allin", stack))
        return acts
    # Same library sizes as HU: 25/33/55/75/125. Missing 25% snapped LibraryGTO
    # cbets onto 33% and made 3-way search play a different game.
    fracs = (0.25, 0.33, 0.55, 0.75, 1.25)
    for f in fracs:
        put = min(stack, max(1.0, f * pot))
        if put < stack * 0.90:
            acts.append((f"bet{int(round(f * 100))}", put))
    # 3bet-turn SPR~2.3 OOP: keep covering jam plus donk sizes (FIRST_ACTS is 7).
    if stack > 0 and spr <= 2.4:
        if not any(n == "allin" for n, _ in acts):
            acts.append(("allin", stack))
    return acts[:7]


def _sizes_face(
    pot: float,
    stack: float,
    to_call: float,
    raises_left: int,
    lock_spr: Optional[float] = None,
) -> List[Tuple[str, float]]:
    """Root facing. LibraryGTO always raises 2.8 / 0.75 pot (except overbet jam).

    HU facing uses 3.4x vs a 0.25x — a different node than the mix we lock,
    so 3-way BR over-raised small cbets.
    """
    acts: List[Tuple[str, float]] = [("fold", 0.0), ("call", min(to_call, stack))]
    if raises_left <= 0 or stack <= to_call + 1.0:
        return acts[:5]
    spr = stack / max(pot, 1.0)
    if lock_spr is not None and float(lock_spr) < spr:
        spr = float(lock_spr)
    frac = to_call / max(pot - to_call, 1.0)
    # face_mix raises when spr > 2.4. Jam-only below 2.5 dropped that
    # node at SPR 2.41–2.49 (3bet turn) and BR'd vs a shove they raise.
    if spr <= 2.4 or stack <= 3.2 * to_call or frac > 1.05:
        acts.append(("allin", stack))
        return acts[:5]
    raise_put = min(stack, max(to_call * 2.8, to_call + 0.75 * pot))
    if raise_put >= stack * 0.85 or raise_put <= to_call + 1.0:
        acts.append(("allin", stack))
    else:
        acts.append(("raise", raise_put))
        if stack > raise_put + 1.0:
            acts.append(("allin", stack))
    return acts[:5]


def _cover_stack(
    stk: Sequence[float],
    live: Sequence[int],
    p: int,
    inv: Optional[Sequence[float]] = None,
) -> float:
    """Chips `p` can put in now vs someone still contesting.

    A 100bb jam into two 50bb stacks left 50 uncalled in the pot, so covering
    jams looked 0.5*uncalled worse and hero under-jammed 3-way.

    After the first villain calls that jam, their remaining is 0. Covering
    from remaining stacks alone then collapsed the next villain to 0 — they
    either folded or got a free showdown. Count chips already in this street.
    """
    already = float(inv[p]) if inv is not None and int(p) < len(inv) else 0.0
    others = []
    for q in range(len(stk)):
        if q == int(p) or (q < len(live) and not live[q]):
            continue
        chip = float(stk[q])
        if inv is not None and q < len(inv):
            chip += float(inv[q])
        others.append(chip)
    if not others:
        return max(0.0, float(stk[p]))
    return max(0.0, min(float(stk[p]), max(others) - already))


def _size_spr_live(
    cover: float,
    pot_n: float,
    stk: Sequence[float],
    live: Sequence[int],
    lock_spr: Optional[float],
) -> float:
    """Covering SPR among players still in. Root lock_spr includes shorts.

    After a 20bb extra folds, clipping SB's node to that leftover SPR made
    the tree jam-only while LibraryGTO's effective_stack is deep again.
    All-in 0 stays live, so side-pot jam-only still matches.
    """
    size_spr = float(cover) / max(float(pot_n), 1.0)
    if lock_spr is None:
        return size_spr
    live_min = None
    for i in range(len(stk)):
        if i < len(live) and live[i]:
            v = float(stk[i])
            live_min = v if live_min is None else min(live_min, v)
    if live_min is None:
        return min(size_spr, float(lock_spr))
    live_spr = float(live_min) / max(float(pot_n), 1.0)
    if live_spr <= float(lock_spr) + 1e-6 and float(lock_spr) < size_spr:
        return float(lock_spr)
    return size_spr


def _matched_pots(
    inv: Sequence[float],
    live: Sequence[int],
    pot: float,
) -> Tuple[Tuple[float, float, float], float, List[Tuple[Tuple[int, ...], float]]]:
    """Return uncalled chips, then main + side-pot layers.

    Jam 100 vs 50/100: if the 100-stack folds, 50 uncalled must come back
    (else losing the short call costs 100 instead of 50). If both call, the
    short stack cannot scoop the 100 side pot — that is HU between the two
    covering stacks. Equal stacks collapse to one 3-way layer of the full pot.
    """
    inv_l = [float(x) for x in inv[:3]]
    while len(inv_l) < 3:
        inv_l.append(0.0)
    live_idx = [i for i in range(3) if i < len(live) and live[i]]
    pot_n = float(pot)
    if len(live_idx) >= 2:
        im = max(live_idx, key=lambda i: inv_l[i])
        second = max(inv_l[j] for j in live_idx if j != im)
        uncalled = max(0.0, inv_l[im] - second)
        if uncalled > 1e-9:
            inv_l[im] -= uncalled
            pot_n = max(0.0, pot_n - uncalled)
    prior = max(0.0, pot_n - sum(inv_l))
    layers: List[Tuple[Tuple[int, ...], float]] = []
    if prior > 1e-9 and live_idx:
        layers.append((tuple(live_idx), prior))
    rem = list(inv_l)
    while True:
        pos = [i for i in live_idx if rem[i] > 1e-9]
        if len(pos) < 2:
            break
        m = min(rem[i] for i in pos)
        layers.append((tuple(pos), m * len(pos)))
        for i in pos:
            rem[i] -= m
    return (inv_l[0], inv_l[1], inv_l[2]), pot_n, layers


def _hero_ev_from_layers(
    layers: Sequence[Tuple[Tuple[int, ...], float]],
    inv0: float,
    sh0: np.ndarray,
    hu01: Optional[np.ndarray] = None,
    hu02: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Hero chip EV from main + side pots, minus matched inv."""
    u = np.zeros(sh0.shape, dtype=np.float64)
    for who, amt in layers:
        if float(amt) <= 1e-12 or 0 not in who:
            continue
        if len(who) >= 3:
            u = u + sh0 * float(amt)
        elif 1 in who:
            share = hu01 if hu01 is not None else sh0
            u = u + share * float(amt)
        else:
            share = hu02 if hu02 is not None else sh0
            u = u + share * float(amt)
    return u - float(inv0)


def _ghost_scoop_ev(share, pot: float, inv0: float) -> np.ndarray:
    """Raise-fold vs chip actors still shows down the main pot vs a leftover jam.

    Uncalled raise chips are not at risk. EV is share * (pot - inv), not pot - inv
    (that was 100% scoop) and not share * pot - inv (puts the raise at risk).
    """
    prior = max(0.0, float(pot) - float(inv0))
    return np.asarray(share, dtype=np.float64) * prior


def _ghost_hu_remaining_ev(
    share_3: np.ndarray,
    share_hu: np.ndarray,
    pot: float,
    inv: Sequence[float],
    live: Sequence[int],
) -> np.ndarray:
    """Main pot includes the leftover all-in; current street is HU vs the actor."""
    inv_m, pot_m, _layers = _matched_pots(inv, live, pot)
    prior = max(0.0, float(pot_m) - sum(float(x) for x in inv_m))
    street = max(0.0, float(pot_m) - prior)
    return share_3 * prior + share_hu * street - float(inv_m[0])


def _ghost_prior_adjust(share_3, share_ghost, prior_amt: float) -> np.ndarray:
    """Call-call still shows down the leftover jam, not a 3-way scoop of it.

    3-way share of the prior is P(hero best of the chip players). Times HU
    vs the ghost (independent) is P(hero best of 4) without a 4-D tensor.
    Nuts stay ~1; air stays 0. Street side pots are unchanged.
    """
    sh = np.asarray(share_3, dtype=np.float64)
    gs = np.asarray(share_ghost, dtype=np.float64)
    return sh * max(0.0, float(prior_amt)) * (gs - 1.0)


def _remain_after_put(stk_p: float, put: float, cover: float) -> float:
    """Chips still in front after a put.

    Covering caps how much goes in (jam 50 into 50s). Remaining after a
    10bb cbet is still 90, not cover-put=40 — that clipped extras and the
    next covering jam to 40 instead of 50.
    """
    if float(put) >= float(cover) - 1e-6:
        return 0.0
    return max(0.0, float(stk_p) - float(put))


def _extra_already_matched(view: PlayerView) -> bool:
    """True when nobody behind hero still owes the current bet.

    BB facing a c-bet after a caller: extra already matched. Hero calling
    closes the action; a raise is an increment for both remaining villains.
    SB facing a c-bet: BB still owes, so extra is not matched.
    """
    if float(view.to_call) <= 0:
        return False
    current = float(view.current_bet)
    n = view.num_players
    for k in range(1, n):
        i = (view.hero + k) % n
        if i == view.hero or view.folded[i] or view.all_in[i]:
            continue
        if float(view.bets[i]) + 1e-6 < current:
            return False
    return True


def build_3way(
    pot: float,
    cap: float,
    to_call: float,
    hero_oop: bool,
    raises_left: int = 1,
    stacks: Optional[Tuple[float, float, float]] = None,
    p2_matched: bool = False,
    after_hero: Optional[Sequence[int]] = None,
    lock_spr: Optional[float] = None,
    vill_ips: Optional[Tuple[bool, bool]] = None,
) -> Tuple[List[_N], int]:
    nodes: List[_N] = []
    rl = max(0, int(raises_left))
    if after_hero is None:
        after = [1, 2]
    else:
        after = [int(p) for p in after_hero if int(p) in (1, 2)]
    if stacks is None:
        hstk = v1stk = v2stk = float(cap)
    else:
        hstk, v1stk, v2stk = (max(0.0, float(x)) for x in stacks)
    hero_ip = not bool(hero_oop)

    def add(kind: str) -> int:
        nodes.append(_N(kind))
        return len(nodes) - 1

    def sd_i(pot_n, inv, stk, live) -> int:
        i = add("sd")
        n = nodes[i]
        n.pot, n.inv, n.stk, n.live = pot_n, inv, stk, live
        return i

    def fold_i(pot_n, inv, stk, folder, live) -> int:
        i = add("fold")
        n = nodes[i]
        n.pot, n.inv, n.stk, n.folder, n.live = pot_n, inv, stk, folder, live
        return i

    def villains_respond(pot_n, inv, stk, bet_level, live, order: List[int], raises_left_n: int) -> int:
        order = [p for p in order if live[p] and float(stk[p]) > 0.5]
        if not order:
            return sd_i(pot_n, inv, stk, tuple(live))
        p = order[0]
        rest = order[1:]
        already = float(inv[p])
        to_c = max(0.0, float(bet_level) - already)
        if to_c <= 0.5:
            # Already matched: first-in, not facing. Lock uses first_mix.
            return first_in_chain([p] + rest, pot_n, inv, stk, live, raises_left_n)
        i = add("dec")
        n = nodes[i]
        n.player = p
        n.live = tuple(live)
        n.pot = pot_n
        n.stk = tuple(stk)
        cover = _cover_stack(stk, live, p, inv)
        n.acts = ["fold", "call"]
        n.put = [0.0, min(to_c, cover)]
        # LibraryGTO's 2-way mix includes raise. Mapping that onto jam made
        # 3-way search BR to a shove that the opponent never makes.
        raise_put = min(cover, max(to_c * 2.8, to_c + 0.75 * pot_n))
        size_spr = _size_spr_live(cover, pot_n, stk, live, lock_spr)
        can_raise = (
            size_spr > 2.4
            and raises_left_n > 0
            and cover > to_c + 1.0
            and to_c < cover * 0.90
            and raise_put < cover * 0.90
            and raise_put > to_c + 1.0
        )
        if can_raise:
            n.acts.append("raise")
            n.put.append(raise_put)
        can_jam = raises_left_n > 0 and cover > to_c + 1.0 and to_c < cover * 0.90
        if can_jam:
            n.acts.append("allin")
            n.put.append(cover)
        live_f = list(live)
        live_f[p] = 0
        if sum(live_f) <= 1:
            n.children.append(fold_i(pot_n, inv, stk, p, tuple(live_f)))
        else:
            n.children.append(villains_respond(pot_n, inv, stk, bet_level, live_f, rest, raises_left_n))
        inv_c = list(inv)
        stk_c = list(stk)
        pay = min(to_c, cover)
        inv_c[p] += pay
        stk_c[p] = _remain_after_put(stk[p], pay, cover)
        n.children.append(
            villains_respond(pot_n + pay, tuple(inv_c), tuple(stk_c), bet_level, live, rest, raises_left_n)
        )
        reopen = [q for q in rest if live[q]]
        # Last aggressor (p1) faces a squeeze before the caller (hero).
        # range(3) put hero first, so a CO squeeze was priced as hero-to-act
        # before UTG folded or 4-bet.
        for q in (1, 0, 2):
            if live[q] and q != p and q not in reopen:
                reopen.append(q)
        if can_raise:
            inv_r = list(inv)
            stk_r = list(stk)
            inv_r[p] += raise_put
            stk_r[p] = _remain_after_put(stk[p], raise_put, cover)
            n.children.append(
                villains_respond(
                    pot_n + raise_put, tuple(inv_r), tuple(stk_r), already + raise_put,
                    live, reopen, max(0, raises_left_n - 1),
                )
            )
        if can_jam:
            inv_j = list(inv)
            stk_j = list(stk)
            jam_put = cover
            inv_j[p] += jam_put
            stk_j[p] = _remain_after_put(stk[p], jam_put, cover)
            n.children.append(
                villains_respond(
                    pot_n + jam_put, tuple(inv_j), tuple(stk_j), already + jam_put,
                    live, reopen, 0,
                )
            )
        return i

    def first_in_chain(players: List[int], pot_n, inv, stk, live, raises_left_n: int) -> int:
        players = [p for p in players if live[p] and float(stk[p]) > 0.5]
        if not players:
            return sd_i(pot_n, inv, stk, tuple(live))
        p = players[0]
        rest = players[1:]
        i = add("dec")
        n = nodes[i]
        n.player = p
        n.live = tuple(live)
        n.pot = pot_n
        n.stk = tuple(stk)
        stk_p = _cover_stack(stk, live, p, inv)
        size_spr = _size_spr_live(stk_p, pot_n, stk, live, lock_spr)
        # Lock uses seat IP (OOP leftover still donks). Tree ip=True was
        # jam-only below 2.35, so those donks snapped onto check.
        if p == 0:
            ip_p = hero_ip
        elif vill_ips is not None:
            ip_p = bool(vill_ips[p - 1])
        else:
            ip_p = True
        for name, put in _sizes_first(pot_n, stk_p, size_spr, ip=ip_p):
            n.acts.append(name)
            n.put.append(put)
            if name == "check":
                n.children.append(first_in_chain(rest, pot_n, inv, stk, live, raises_left_n))
                continue
            inv_b = list(inv)
            stk_b = list(stk)
            inv_b[p] += put
            stk_b[p] = _remain_after_put(stk[p], put, stk_p)
            # Flop order after the bettor, wrapping. rest-first happened to
            # match when leftover was "everyone after hero", but SB check →
            # BB donk must face BTN then SB, not hero-then-BTN.
            before = [q for q in (1, 2) if q not in after]
            full: List[int] = []
            for q in before + [0] + list(after):
                if q not in full:
                    full.append(q)
            if p in full:
                pi = full.index(p)
                face = [q for q in (full[pi + 1 :] + full[:pi]) if live[q]]
            else:
                face = [q for q in full if q != p and live[q]]
            n.children.append(
                villains_respond(
                    pot_n + put, tuple(inv_b), tuple(stk_b), put, live, face, raises_left_n,
                )
            )
        return i

    cap3 = (hstk, v1stk, v2stk)
    inv0 = (0.0, 0.0, 0.0)
    live = [1, 1, 1]
    cover0 = _cover_stack(cap3, live, 0, inv0)
    if cover0 <= 0.5:
        cover0 = hstk
    root = add("dec")
    nodes[root].player = 0
    nodes[root].live = (1, 1, 1)
    nodes[root].pot = pot
    nodes[root].stk = cap3

    if to_call > 0:
        for name, put in _sizes_face(pot, cover0, to_call, max(1, rl), lock_spr=lock_spr):
            nodes[root].acts.append(name)
            nodes[root].put.append(put)
            if name == "fold":
                nodes[root].children.append(fold_i(pot, inv0, cap3, 0, (0, 1, 1)))
                continue
            inv = (put, to_call, to_call if p2_matched else 0.0)
            remain = _remain_after_put(hstk, put, cover0)
            stk = (remain, v1stk, v2stk)
            face = to_call if (name == "call" or put <= to_call + 1e-6) else put
            # Player 1 is the last aggressor. A call leaves only the extra
            # villain (2) still to act — unless they already matched (BB after
            # a caller). A raise reopens vs both.
            if name == "call":
                order = [] if p2_matched else [2]
            else:
                # Extra still to act must face the 3bet before the opener.
                # [1, 2] put UTG first after an HJ 3bet, so CO's call/4bet
                # was priced as if the opener had already acted.
                order = [1, 2] if p2_matched else [2, 1]
            # Calling does not consume a raise. left=0 made p2 fold/call
            # only, so hero BR'd as if LibraryGTO never squeezes.
            # Aggressor already put `to_call` (in the pot). Seed that so a
            # squeeze-call is raise-cbet, not a second full put. Same seed
            # on hero's raise. If extra already called, they pay the increment.
            nodes[root].children.append(
                villains_respond(pot + put, inv, stk, face, live, order, rl)
            )
        return nodes, root

    spr0 = cover0 / max(pot, 1.0) if lock_spr is None else min(cover0 / max(pot, 1.0), float(lock_spr))
    for name, put in _sizes_first(pot, cover0, spr0, ip=hero_ip):
        nodes[root].acts.append(name)
        nodes[root].put.append(put)
        if name == "check":
            # Remaining first-in only. Players before hero already checked
            # and must not get a second donk (BTN check used to price SB donk).
            nodes[root].children.append(first_in_chain(after, pot, inv0, cap3, live, rl))
        else:
            inv = (put, 0.0, 0.0)
            left = _remain_after_put(hstk, put, cover0)
            stk = (left, v1stk, v2stk)
            face = list(after)
            for q in (1, 2, 0):
                if q != 0 and q not in face:
                    face.append(q)
            nodes[root].children.append(
                villains_respond(pot + put, inv, stk, put, live, face, rl)
            )
    return nodes, root


def _lock_3way_library(
    nodes: Sequence[_N],
    c1: Sequence[Combo],
    c2: Sequence[Combo],
    board: Sequence[int],
    pot_type: int,
    streets_left: int,
    hero_oop: bool,
    spr: float,
    to_call: float = 0.0,
    vill_ips: Optional[Tuple[bool, bool]] = None,
    perc1: Optional[Sequence[float]] = None,
    perc2: Optional[Sequence[float]] = None,
) -> Dict[int, np.ndarray]:
    """Fix villains at the library mix so hero best-responds in 3-way pots."""
    try:
        from strategy.library import mix_for_combo, mix_on_acts
        from strategy.strength import board_combo_score
    except Exception:
        return {}
    street = {2: 0, 1: 1, 0: 2}.get(int(streets_left), 2)
    combos = {1: list(c1), 2: list(c2)}
    percs: Dict[int, List[float]] = {}
    given = {1: perc1, 2: perc2}
    for p, cs in combos.items():
        if not cs:
            percs[p] = []
            continue
        gp = given[p]
        if gp is not None and len(gp) == len(cs):
            percs[p] = [float(x) for x in gp]
            continue
        sc = [board_combo_score(c, board) for c in cs] if len(board) >= 3 else [0.5] * len(cs)
        order = sorted(range(len(cs)), key=lambda i: sc[i])
        n = max(len(cs) - 1, 1)
        perc = [0.5] * len(cs)
        for rank, i in enumerate(order):
            perc[i] = rank / n
        percs[p] = perc
    spr_v = float(spr) if spr > 0.05 else 0.01
    table_jammed = spr_v <= 0.05
    out: Dict[int, np.ndarray] = {}
    for nid, node in enumerate(nodes):
        if node.kind != "dec" or node.player not in (1, 2):
            continue
        cs = combos[node.player]
        if not cs:
            continue
        # LibraryGTO's mix uses ip = not first-to-act (binary). Lock to that,
        # not a 3-way "true IP". BB with SB live plays the IP table.
        # face_mix currently ignores ip; first_mix uses it for donk vs cbet.
        if vill_ips is not None:
            vill_ip = bool(vill_ips[0] if node.player == 1 else vill_ips[1])
        elif float(to_call) > 0:
            vill_ip = node.player == 1
        else:
            vill_ip = bool(hero_oop) if node.player == 2 else False
        facing = "fold" in node.acts
        frac = 0.75 if facing else 0.0
        if facing and node.put:
            tc = 0.0
            for name, put in zip(node.acts, node.put):
                if name == "call":
                    tc = float(put)
                    break
            frac = tc / max(float(node.pot) - tc, 1.0) if tc > 0 else 0.75
        # Folded extras are not in view.effective_stack. min(stk) kept a
        # 20bb leftover after BTN folded a BB cbet, so SB's lock was jam
        # while they play deep (raise 2.8). All-in 0 stays live and jams.
        live_stk = [
            float(node.stk[i])
            for i in range(len(node.stk))
            if i < len(node.live) and node.live[i]
        ]
        cap_n = min(live_stk) if live_stk else (min(node.stk) if node.stk else 0.0)
        # All-in 0 is LibraryGTO spr≈0 (jam), not acting-cap 100bb.
        # Table effective_stack includes that 0 even when tree stacks are deep.
        if table_jammed or cap_n <= 0.05:
            node_spr = 0.01
        else:
            node_spr = cap_n / max(float(node.pot), 1.0)
            if node_spr < 0.05:
                node_spr = 0.01
        sig = np.zeros((len(cs), len(node.acts)), dtype=np.float64)
        for i, combo in enumerate(cs):
            mix = mix_for_combo(
                combo, board, percs[node.player][i], facing, frac, street, pot_type, vill_ip, node_spr,
            )
            mapped = mix_on_acts(mix, node.acts, facing)
            sig[i] = mapped if mapped.shape[0] == sig.shape[1] else (1.0 / max(sig.shape[1], 1))
        out[nid] = sig
    return out


def solve_3way(
    S0: np.ndarray,
    S1: np.ndarray,
    S2: np.ndarray,
    hw: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    pot: float,
    cap: float,
    to_call: float,
    hero_oop: bool,
    iterations: int,
    rake_percent: float = 0.0,
    rake_cap: float = 0.0,
    lock: Optional[NodeLock] = None,
    streets_left: int = 0,
    extra_mat: Optional[np.ndarray] = None,
    extra_mat2: Optional[np.ndarray] = None,
    villain_library: bool = False,
    c1: Optional[Sequence[Combo]] = None,
    c2: Optional[Sequence[Combo]] = None,
    board: Optional[Sequence[int]] = None,
    pot_type: int = 0,
    stacks: Optional[Tuple[float, float, float]] = None,
    vill_ips: Optional[Tuple[bool, bool]] = None,
    p2_matched: bool = False,
    after_hero: Optional[Sequence[int]] = None,
    perc1: Optional[Sequence[float]] = None,
    perc2: Optional[Sequence[float]] = None,
    lock_spr: Optional[float] = None,
    ghost_P: Optional[np.ndarray] = None,
    ghost_w: Optional[np.ndarray] = None,
    ghost_G01: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ghost_G02: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[List[_N], int, np.ndarray]:
    H, A, B = S0.shape
    hw = np.nan_to_num(np.maximum(_as1d(hw, H), 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    w1 = np.nan_to_num(np.maximum(_as1d(w1, A), 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    w2 = np.nan_to_num(np.maximum(_as1d(w2, B), 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    if float(hw.sum()) <= 1e-15:
        hw = np.ones(H, dtype=np.float64)
    if float(w1.sum()) <= 1e-15:
        w1 = np.ones(A, dtype=np.float64)
    if float(w2.sum()) <= 1e-15:
        w2 = np.ones(B, dtype=np.float64)
    if stacks is None:
        orig_pair = {1: float(cap), 2: float(cap)}
    else:
        hs, s1, s2 = (max(0.0, float(x)) for x in stacks)
        orig_pair = {1: min(hs, s1), 2: min(hs, s2)}
    rl = 3 if int(pot_type) >= 1 else 2
    nodes, root = build_3way(
        pot, cap, to_call, hero_oop, raises_left=rl, stacks=stacks, p2_matched=p2_matched,
        after_hero=after_hero, lock_spr=lock_spr, vill_ips=vill_ips,
    )
    sizes = {0: H, 1: A, 2: B}
    regrets: Dict[int, np.ndarray] = {}
    strat_sum: Dict[int, np.ndarray] = {}
    for i, n in enumerate(nodes):
        if n.kind != "dec":
            continue
        na = len(n.acts)
        regrets[i] = np.zeros((sizes[n.player], na))
        strat_sum[i] = np.zeros((sizes[n.player], na))

    vill_lib: Dict[int, np.ndarray] = {}
    if villain_library and board is not None and len(board) >= 3 and c1 is not None and c2 is not None:
        spr = float(lock_spr) if lock_spr is not None else (float(cap) / max(float(pot), 1.0))
        vill_lib = _lock_3way_library(
            nodes, c1, c2, board, pot_type, streets_left, hero_oop, spr,
            to_call=to_call, vill_ips=vill_ips, perc1=perc1, perc2=perc2,
        )

    ghost_rg = None
    if ghost_P is not None and ghost_w is not None and int(getattr(ghost_P, "ndim", 0)) == 2:
        ghost_rg = _prob_reach(ghost_w, int(ghost_P.shape[1]))

    def net_pot(p: float) -> float:
        if rake_percent <= 1e-12:
            return p
        return max(0.0, p - min(rake_cap, p * rake_percent))

    def showdown(n: _N, r0, r1, r2):
        live = n.live
        r0n = _prob_reach(r0, H)
        r1n = _prob_reach(r1, A)
        r2n = _prob_reach(r2, B)

        def _leaf_extra(mat, r_vill, dead_axis: int, r_other: np.ndarray, vill_i: int):
            if streets_left <= 0 or mat is None or not n.stk:
                return None, None
            # HU clips extra to remaining stacks. Clipping to pot overstated
            # futures after a big bet (calling-station vs LibraryGTO barrels)
            # and kept extra on all-in leaves.
            # Min of all three zeroed HU futures vs a deep villain when the
            # extra was all-in (side pot still barrels). Use the pair.
            cap_s = min(float(n.stk[0]), float(n.stk[vill_i]))
            return _masked_pair_extra(
                S0, mat, r0n, r_vill, r_other, dead_axis, cap_s, orig_pair[vill_i],
                node_pot=float(n.pot), orig_pot=float(pot),
            )

        eh1, ev1 = _leaf_extra(extra_mat, r1n, 2, r2n, 1)
        eh2, ev2 = _leaf_extra(extra_mat2, r2n, 1, r1n, 2)
        # Signed extra is EV, not chips dumped into a shared pot. Inflating
        # the pot with a (possibly negative) extra made 3-way dogs check-down.
        inv_m, pot_m, layers = _matched_pots(n.inv, live, n.pot)
        pot_n = net_pot(pot_m)
        if live[0] and live[1] and live[2]:
            sh0, sh1, sh2 = _three_way_shares(S0, S1, S2, r0n, r1n, r2n)
            if pot_m > 1e-9 and abs(pot_n - pot_m) > 1e-9:
                scale = pot_n / pot_m
                layers = [(w, a * scale) for w, a in layers]
            hu01 = hu02 = None
            if any(len(w) == 2 and 0 in w and 1 in w for w, _ in layers):
                P01 = _collapse_dead(S0, r2n, 2)
                hu01, _, _, _ = _blocked_shares(P01, r0n, r1n, None)
            if any(len(w) == 2 and 0 in w and 2 in w for w, _ in layers):
                P02 = _collapse_dead(S0, r1n, 1)
                hu02, _, _, _ = _blocked_shares(P02, r0n, r2n, None)
            u0 = _hero_ev_from_layers(layers, inv_m[0], sh0, hu01, hu02)
            if ghost_rg is not None and ghost_P is not None:
                hs_g, _, _, _ = _blocked_shares(ghost_P, r0n, ghost_rg, None)
                prior_amt = max(0.0, float(pot_m) - sum(float(x) for x in inv_m))
                u0 = u0 + _ghost_prior_adjust(sh0, hs_g, prior_amt)
            u1 = sh1 * pot_n - inv_m[1]
            u2 = sh2 * pot_n - inv_m[2]
            # Two HU extras at the 3-way pot are still HU-priced (win ~1/2 of
            # futures). Three-way share is ~1/3, so scale by 2/3. Averaging
            # without that scale made 3-way dogs call down vs LibraryGTO.
            if eh1 is not None and eh2 is not None:
                # Weight by remaining stacks: a short leftover villain is
                # not half the future vs a deep one (3bet pots).
                s1 = max(float(n.stk[1]), 0.0)
                s2 = max(float(n.stk[2]), 0.0)
                tot_s = s1 + s2
                w1 = (s1 / tot_s) if tot_s > 0.5 else 0.5
                w2 = 1.0 - w1
                u0 = u0 + (2.0 / 3.0) * (w1 * eh1 + w2 * eh2)
            elif eh1 is not None:
                u0 = u0 + eh1
            elif eh2 is not None:
                u0 = u0 + eh2
            if ev1 is not None:
                u1 = u1 + ((2.0 / 3.0) * 0.5 * ev1 if eh2 is not None else ev1)
            if ev2 is not None:
                u2 = u2 + ((2.0 / 3.0) * 0.5 * ev2 if eh1 is not None else ev2)
            return _as1d(u0, H), _as1d(u1, A), _as1d(u2, B)
        if live[0] and live[1] and not live[2]:
            P01 = _collapse_dead(S0, r2n, 2)
            hero_share, vill_share, _eh, _ev = _blocked_shares(P01, r0n, r1n, None)
            u0 = hero_share * pot_n - inv_m[0]
            u1 = vill_share * pot_n - inv_m[1]
            if ghost_rg is not None and ghost_G01 is not None:
                sh0, _, _ = _three_way_shares(
                    ghost_G01[0], ghost_G01[1], ghost_G01[2], r0n, r1n, ghost_rg,
                )
                u0 = _ghost_hu_remaining_ev(sh0, hero_share, n.pot, n.inv, live)
            if eh1 is not None:
                u0 = u0 + eh1
            if ev1 is not None:
                u1 = u1 + ev1
            return _as1d(u0, H), _as1d(u1, A), np.full(B, -n.inv[2])
        if live[0] and live[2] and not live[1]:
            P02 = _collapse_dead(S0, r1n, 1)
            hero_share, vill_share, _eh, _ev = _blocked_shares(P02, r0n, r2n, None)
            u0 = hero_share * pot_n - inv_m[0]
            u2 = vill_share * pot_n - inv_m[2]
            if ghost_rg is not None and ghost_G02 is not None:
                sh0, _, _ = _three_way_shares(
                    ghost_G02[0], ghost_G02[1], ghost_G02[2], r0n, r2n, ghost_rg,
                )
                u0 = _ghost_hu_remaining_ev(sh0, hero_share, n.pot, n.inv, live)
            if eh2 is not None:
                u0 = u0 + eh2
            if ev2 is not None:
                u2 = u2 + ev2
            return _as1d(u0, H), np.full(A, -n.inv[1]), _as1d(u2, B)
        if live[1] and live[2] and not live[0]:
            return np.full(H, -n.inv[0]), np.full(A, n.pot * 0.5 - n.inv[1]), np.full(B, n.pot * 0.5 - n.inv[2])
        if live[0]:
            if ghost_rg is not None and ghost_P is not None:
                hs_g, _, _, _ = _blocked_shares(ghost_P, r0n, ghost_rg, None)
                u0 = _ghost_scoop_ev(hs_g, n.pot, n.inv[0])
                return _as1d(u0, H), np.full(A, -n.inv[1]), np.full(B, -n.inv[2])
            return np.full(H, pot_n - n.inv[0]), np.full(A, -n.inv[1]), np.full(B, -n.inv[2])
        if live[1]:
            return np.full(H, -n.inv[0]), np.full(A, pot_n - n.inv[1]), np.full(B, -n.inv[2])
        return np.full(H, -n.inv[0]), np.full(A, -n.inv[1]), np.full(B, pot_n - n.inv[2])

    def cfr(nid: int, r0, r1, r2, upd: int, t: int):
        r0 = _as1d(r0, H)
        r1 = _as1d(r1, A)
        r2 = _as1d(r2, B)
        n = nodes[nid]
        if n.kind in ("fold", "sd"):
            return showdown(n, r0, r1, r2)
        sig = _regret_match(regrets[nid])
        if n.player != 0 and nid in vill_lib:
            sig = vill_lib[nid]
        elif n.player != 0 and lock is not None:
            sig = apply_to_strategy(sig, n.acts, lock)
        reaches = (r0, r1, r2)
        rp = reaches[n.player]
        if upd == n.player:
            strat_sum[nid] += t * rp[:, None] * sig
        n_a = len(n.acts)
        ns = (H, A, B)
        au = [
            cfr(
                n.children[a],
                *((r0 * sig[:, a], r1, r2) if n.player == 0 else (r0, r1 * sig[:, a], r2) if n.player == 1 else (r0, r1, r2 * sig[:, a])),
                upd,
                t,
            )
            for a in range(n_a)
        ]
        ev = [np.zeros(ns[p], dtype=np.float64) for p in range(3)]
        for p in range(3):
            stacked = np.stack([_as1d(au[a][p], ns[p]) for a in range(n_a)], axis=1)
            if p == n.player:
                ev[p] = np.einsum("ca,ca->c", sig, stacked)
                if upd == n.player and nid not in vill_lib:
                    mass_opp = 1.0
                    for q in range(3):
                        if q != p:
                            mass_opp *= float(reaches[q].sum())
                    regrets[nid] += mass_opp * (stacked - ev[p][:, None])
            else:
                masses = np.array([float((rp * sig[:, a]).sum()) for a in range(n_a)])
                ev[p] = _mix_columns(stacked, masses)
        return ev[0], ev[1], ev[2]

    r0 = np.nan_to_num(hw / max(float(hw.sum()), 1e-12), nan=0.0, posinf=0.0, neginf=0.0)
    r1 = np.nan_to_num(w1 / max(float(w1.sum()), 1e-12), nan=0.0, posinf=0.0, neginf=0.0)
    r2 = np.nan_to_num(w2 / max(float(w2.sum()), 1e-12), nan=0.0, posinf=0.0, neginf=0.0)
    # Locked villains: live BR is _eval_h. MCCFR mix is discarded.
    if not vill_lib:
        upd_players = (0, 1, 2)
        for t in range(1, iterations + 1):
            for p in upd_players:
                cfr(root, r0, r1, r2, p, t)
            if t % 5 == 0:
                a = t ** 1.5 / (t ** 1.5 + 1.0)
                for arr in regrets.values():
                    arr *= a

    if vill_lib:
        n_a = len(nodes[root].acts) or 1
        if nodes[root].player != 0:
            return nodes, root, np.full((H, n_a), 1.0 / n_a)
        last = {nid: _regret_match(regrets[nid]) for nid in regrets}

        def _eval_h(nid: int, rr0: np.ndarray, rr1: np.ndarray, rr2: np.ndarray) -> np.ndarray:
            n = nodes[nid]
            if n.kind in ("fold", "sd"):
                u0, _u1, _u2 = showdown(n, rr0, rr1, rr2)
                return _as1d(u0, H)
            na = len(n.acts)
            if n.player == 0:
                parts = [
                    _as1d(_eval_h(n.children[a], rr0, rr1, rr2), H)
                    for a in range(na)
                ]
                stacked = np.nan_to_num(
                    np.stack(parts, axis=1), nan=-1e12, posinf=1e12, neginf=-1e12
                )
                return np.max(stacked, axis=1)
            sig = vill_lib[nid] if nid in vill_lib else last[nid]
            masses = np.zeros(na, dtype=np.float64)
            parts = []
            for a in range(na):
                if n.player == 1:
                    r1n = _as1d(rr1 * sig[:, a], A)
                    masses[a] = r1n.sum()
                    parts.append(_eval_h(n.children[a], rr0, r1n, rr2))
                else:
                    r2n = _as1d(rr2 * sig[:, a], B)
                    masses[a] = r2n.sum()
                    parts.append(_eval_h(n.children[a], rr0, rr1, r2n))
            stacked = np.stack([_as1d(p, H) for p in parts], axis=1)
            return _mix_columns(stacked, masses)

        evs = np.zeros((H, n_a), dtype=np.float64)
        for a in range(n_a):
            evs[:, a] = _eval_h(nodes[root].children[a], r0, r1, r2)
        evs = np.nan_to_num(evs, nan=-1e12, posinf=1e12, neginf=-1e12)
        strat = np.zeros_like(evs)
        strat[np.arange(H), np.argmax(evs, axis=1)] = 1.0
        return nodes, root, strat

    ss = strat_sum.get(root)
    if ss is None or nodes[root].player != 0:
        n_a = len(nodes[root].acts) or 1
        return nodes, root, np.full((H, n_a), 1.0 / n_a)
    tot = ss.sum(axis=1, keepdims=True)
    n_a = ss.shape[1]
    strat = np.where(tot > 1e-12, ss / tot, 1.0 / n_a)
    return nodes, root, strat


_CACHE3: dict = {}
_CACHE3_MAX = 256


def live_action_3way(
    view: PlayerView,
    hero_combos: Sequence[Combo],
    hero_w: Sequence[float],
    v1_combos: Sequence[Combo],
    v1_w: Sequence[float],
    v2_combos: Sequence[Combo],
    v2_w: Sequence[float],
    rng,
    iterations: int = 64,
    max_combos: int = 40,
    runouts: int = 4,
    lock: Optional[NodeLock] = None,
    rake: Optional[RakeConfig] = None,
    villain_library: bool = False,
    v1_stack: Optional[float] = None,
    v2_stack: Optional[float] = None,
    v1_ip: Optional[bool] = None,
    v2_ip: Optional[bool] = None,
    v1_seat: Optional[int] = None,
    v2_seat: Optional[int] = None,
    ghost_combos: Optional[Sequence[Combo]] = None,
    ghost_w: Optional[Sequence[float]] = None,
) -> Action:
    hole = (min(view.hole), max(view.hole))
    spr_b = round(_acting_cap(view) / max(view.pot, 1), 0)
    pot_b = round(view.pot / max(view.bb, 1), 0)
    tc_b = round(view.to_call / max(view.bb, 1), 1)
    rake_k = (0.0, 0.0) if rake is None else (rake.percent, rake.cap_bb)
    lock_k = (0.0, 0.0, 0.0, 0.0) if lock is None else (
        lock.fold_bias, lock.call_bias, lock.bet_bias, lock.strength,
    )
    from strategy.pot_type import classify as pot_kind

    p2_m = _extra_already_matched(view)
    after = None
    if v1_seat is not None and v2_seat is not None and float(view.to_call) <= 0:
        from strategy.mw_reduce import first_in_after_hero

        after = first_in_after_hero(view, int(v1_seat), int(v2_seat))
    key = (
        "3way",
        view.street.value,
        tuple(view.board),
        hole,
        spr_b,
        pot_b,
        tc_b,
        int(_hero_oop(view)),
        pot_kind(view),
        _range_fp(hero_combos, hero_w),
        _range_fp(v1_combos, v1_w),
        _range_fp(v2_combos, v2_w),
        iterations,
        max_combos,
        runouts,
        rake_k,
        lock_k,
        bool(villain_library),
        round(float(v1_stack or 0.0) / max(view.bb, 1), 0),
        round(float(v2_stack or 0.0) / max(view.bb, 1), 0),
        int(v1_ip) if v1_ip is not None else -1,
        int(v2_ip) if v2_ip is not None else -1,
        int(p2_m),
        int(view.players_in_hand),
        int(_acting_count(view)),
        tuple(after) if after is not None else (1, 2),
        round(float(view.effective_stack) / max(view.pot, 1), 2),
        _range_fp(ghost_combos, ghost_w) if ghost_combos is not None and ghost_w is not None else (),
    )
    hit = _CACHE3.get(key)
    if hit is None:
        hit = _solve_3way_policy(
            view, hero_combos, hero_w, v1_combos, v1_w, v2_combos, v2_w, rng,
            iterations, max_combos, runouts, lock, rake, villain_library,
            v1_stack, v2_stack, v1_ip, v2_ip, p2_m, after,
            ghost_combos, ghost_w,
        )
        if len(_CACHE3) >= _CACHE3_MAX:
            _CACHE3.pop(next(iter(_CACHE3)))
        _CACHE3[key] = hit
    acts, puts, probs = hit
    if not acts:
        if view.to_call > 0:
            return Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero]))
        return Action(ActionType.CHECK)
    weights = _sample_weights(probs, bool(villain_library))
    choice = int(rng.choices(range(len(acts)), weights=weights, k=1)[0])
    return _to_engine(view, acts[choice], puts[choice])


def _solve_3way_policy(
    view: PlayerView,
    hero_combos: Sequence[Combo],
    hero_w: Sequence[float],
    v1_combos: Sequence[Combo],
    v1_w: Sequence[float],
    v2_combos: Sequence[Combo],
    v2_w: Sequence[float],
    rng,
    iterations: int,
    max_combos: int,
    runouts: int,
    lock: Optional[NodeLock],
    rake: Optional[RakeConfig],
    villain_library: bool,
    v1_stack: Optional[float] = None,
    v2_stack: Optional[float] = None,
    v1_ip: Optional[bool] = None,
    v2_ip: Optional[bool] = None,
    p2_matched: bool = False,
    after_hero: Optional[Sequence[int]] = None,
    ghost_combos: Optional[Sequence[Combo]] = None,
    ghost_w: Optional[Sequence[float]] = None,
) -> Tuple[List[str], List[float], np.ndarray]:
    hole = view.hole
    keep = (min(hole), max(hole))
    board = list(view.board)
    hero_k = 1 if villain_library else max_combos
    hc, hw, hi = subsample(hero_combos, hero_w, hero_k, keep, board)
    c1, w1, _ = subsample(v1_combos, v1_w, max_combos, None, board)
    c2, w2, _ = subsample(v2_combos, v2_w, max_combos, None, board)
    gen = np.random.default_rng(rng.randrange(1 << 30))
    samples = runouts if len(board) < 5 else 1
    S0, S1, S2 = payoff_tensor3(hc, c1, c2, board, samples, gen)
    pot = float(view.pot)
    cap = _acting_cap(view)
    hstk = float(max(view.stacks[view.hero], 1))
    # 0 is a real all-in stack. `> 0 else cap` cloned them as 100bb in the
    # side pot after a shove.
    s1 = float(v1_stack) if v1_stack is not None else cap
    s2 = float(v2_stack) if v2_stack is not None else cap
    stacks = (hstk, s1, s2)
    to_call = float(view.to_call)
    rpct = rake.percent if rake is not None and rake.active else 0.0
    rcap = (rake.cap_bb * view.bb) if rake is not None and rake.active else 0.0
    sl = _streets_left(view.street)
    pt = 0
    extra_mat = None
    extra_mat2 = None
    try:
        from strategy.pot_type import classify as pot_kind

        pt = {"srp": 0, "3bet": 1, "4bet": 2}.get(pot_kind(view), 0)
    except Exception:
        pt = 0
    if sl > 0:
        try:
            from strategy.library import continuation_matrix

            extra_mat = continuation_matrix(hc, c1, board, pot, hstk, s1, pt, sl, sixmax=True)
            extra_mat2 = continuation_matrix(hc, c2, board, pot, hstk, s2, pt, sl, sixmax=True)
            # 4-way+ pots reduce to 3 players. HU extras on the full pot
            # price winning ~1/2 of futures; n-way share is ~1/n. Scale
            # by 3/n here; the 2/3 three-live factor then yields 1/n vs 1/2.
            # Count actors, not all-in: an all-in does not barrel, so a
            # 4-way with one shove is still 3-way futures (not 3/4).
            n_live = max(_acting_count(view), 3)
            if n_live > 3:
                scale_n = 3.0 / float(n_live)
                if extra_mat is not None:
                    extra_mat = extra_mat * scale_n
                if extra_mat2 is not None:
                    extra_mat2 = extra_mat2 * scale_n
        except Exception:
            extra_mat = None
            extra_mat2 = None
    perc1 = perc2 = None
    if villain_library and len(board) >= 3:
        try:
            from strategy.library import combo_percentiles

            perc1 = combo_percentiles(c1, board, v1_combos, v1_w)
            perc2 = combo_percentiles(c2, board, v2_combos, v2_w)
        except Exception:
            perc1 = perc2 = None
    ghost_P = ghost_Gw = ghost_G01 = ghost_G02 = None
    if ghost_combos is not None and ghost_w is not None and len(board) >= 3:
        try:
            gk = min(64, max(8, int(max_combos) // 4))
            cg, gw, _ = subsample(ghost_combos, ghost_w, gk, None, board)
            if cg:
                ghost_P = payoff_matrix(hc, cg, board, n_samples=samples, rng=gen)
                ghost_Gw = np.asarray(gw, dtype=np.float64)
                ghost_G01 = payoff_tensor3(hc, c1, cg, board, samples, gen)
                ghost_G02 = payoff_tensor3(hc, c2, cg, board, samples, gen)
        except Exception:
            ghost_P = ghost_Gw = ghost_G01 = ghost_G02 = None
    nodes, root, strat = solve_3way(
        S0, S1, S2, hw, w1, w2, pot, cap, to_call, _hero_oop(view),
        iterations=iterations, rake_percent=rpct, rake_cap=rcap, lock=lock,
        streets_left=sl, extra_mat=extra_mat, extra_mat2=extra_mat2, villain_library=villain_library,
        c1=c1, c2=c2, board=board, pot_type=pt, stacks=stacks,
        vill_ips=(bool(v1_ip), bool(v2_ip)) if v1_ip is not None and v2_ip is not None else None,
        p2_matched=p2_matched,
        after_hero=after_hero,
        perc1=perc1,
        perc2=perc2,
        lock_spr=float(view.effective_stack) / max(pot, 1.0),
        ghost_P=ghost_P,
        ghost_w=ghost_Gw,
        ghost_G01=ghost_G01,
        ghost_G02=ghost_G02,
    )
    n = nodes[root]
    if n.kind != "dec" or n.player != 0:
        return [], [], np.zeros(1)
    probs = np.maximum(strat[hi], 0.0)
    if probs.sum() <= 0:
        probs = np.ones_like(probs)
    probs = probs / probs.sum()
    return list(n.acts), list(n.put), probs
