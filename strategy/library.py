"""6-max 100bb postflop libraries: policy, beliefs, and search leaves.

Commercial trees (Pio / GTOWizard dumps) overlay the distilled floor when
present. Distilled mixes are combo-level — a keep-set is not a strategy.
Facing 0.25x vs 0.75x vs overbet changes the mix, not only the call amount.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from poker.equity import N_COMBOS, combo_from_index, combo_index
from poker.evaluator import category, evaluate
from poker.game import Action, ActionType, PlayerView, Street
from strategy.card_abs import street_bucket
from strategy.ranges import Range
from strategy.stacks import stack_bucket
from strategy.strength import board_combo_score
from strategy.texture import analyze

DATA = Path(__file__).resolve().parents[1] / "Final_Bot" / "data"
LIB_PATH = DATA / "library_6max_100.npz"
DUMP_DIR = DATA / "solver_dumps"

FIRST_ACTS: Tuple[str, ...] = ("check", "bet25", "bet33", "bet55", "bet75", "bet125", "jam")
FACE_ACTS: Tuple[str, ...] = ("fold", "call", "raise", "jam")
FACE_FRACS: Tuple[float, ...] = (0.25, 0.33, 0.50, 0.75, 1.00, 1.25, 2.00)
SIZE_FRACS: Tuple[float, ...] = (0.25, 0.33, 0.55, 0.75, 1.25)

N_STACK, N_POT, N_STREET, N_TEX, N_IP = 3, 3, 3, 8, 2
N_BKT = 50
N_FIRST, N_FACE = len(FIRST_ACTS), len(FACE_ACTS)
N_FACE_SZ = 6  # 25/33/50/75/100/125+ ; first-in is separate

_PACK: Optional[dict] = None
_DUMPS: Dict[tuple, np.ndarray] = {}


def facing_frac(to_call: float, pot: float) -> float:
    pot_before = max(pot - to_call, 1.0)
    return to_call / pot_before


def nearest_face_frac(frac: float) -> float:
    f = max(0.0, float(frac))
    return min(FACE_FRACS, key=lambda a: abs(a - f))


def face_size_index(frac: float) -> int:
    """0=25, 1=33, 2=50, 3=75, 4=100, 5=125+ — distinct strategy nodes."""
    f = nearest_face_frac(frac)
    return {0.25: 0, 0.33: 1, 0.50: 2, 0.75: 3, 1.00: 4}.get(f, 5)


def tex_id(board: Sequence[int]) -> int:
    if len(board) < 3:
        return 0
    t = analyze(board)
    if t.monotone:
        return 5
    if t.paired and t.wetness >= 0.45:
        return 7
    if t.paired:
        return 6
    if t.wetness >= 0.55 and t.two_tone:
        return 4
    if t.wetness >= 0.50:
        return 3
    if t.two_tone:
        return 2
    if t.highness < 0.42:
        return 1
    return 0


def stack_index(stack_bb: float) -> int:
    b = stack_bucket(stack_bb)
    return {"50": 0, "100": 1, "200": 2, "deep": 2}[b]


def pot_index(kind: str) -> int:
    return {"srp": 0, "3bet": 1, "4bet": 2}.get(kind, 0)


def street_index(street: Street) -> int:
    return {Street.FLOP: 0, Street.TURN: 1, Street.RIVER: 2}.get(street, 0)


def _norm(w: np.ndarray) -> np.ndarray:
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 1e-15:
        w[:] = 1.0 / len(w)
        return w
    w /= s
    return w


def _draw_flags(combo: Tuple[int, int], board: Sequence[int]) -> Tuple[bool, bool]:
    from poker.cards import rank_of, suit_of

    hs = (suit_of(combo[0]), suit_of(combo[1]))
    bs = [suit_of(c) for c in board]
    fd = False
    for s in range(4):
        if hs.count(s) >= 1 and hs.count(s) + bs.count(s) == 4:
            fd = True
            break
    ranks = sorted(set(rank_of(c) for c in list(combo) + list(board)))
    if 12 in ranks:
        ranks = sorted(set(ranks) | {-1})
    oesd = False
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            if (j - i + 1) == 4 and (ranks[j] - ranks[i]) == 3:
                oesd = True
    return fd, oesd


def _hero_oop(view: PlayerView) -> bool:
    n = view.num_players
    for k in range(1, n + 1):
        i = (view.button + k) % n
        if not view.folded[i] and not view.all_in[i]:
            return i == view.hero
    return True


def _seat_oop(view: PlayerView, seat: int) -> bool:
    n = view.num_players
    for k in range(1, n + 1):
        i = (view.button + k) % n
        if not view.folded[i] and not view.all_in[i]:
            return i == seat
    return True


def first_mix(
    perc: float,
    cat: int,
    fd: bool,
    oesd: bool,
    tex: int,
    street: int,
    pot: int,
    ip: bool,
    spr: float,
) -> np.ndarray:
    """Distilled 6-max solver first-in mix. Size nodes are distinct strategies."""
    w = np.zeros(N_FIRST, dtype=np.float64)
    wet = tex in (3, 4, 5, 7)
    dry = not wet
    river = street == 2
    flop = street == 0
    strong = perc > 0.82 or cat >= 4
    value = perc > 0.68 or cat >= 2
    mid = 0.35 < perc <= 0.68
    draw = (fd or oesd) and not river
    air = perc < 0.20 and cat == 0 and not draw

    if not ip:
        donk = 0.05 + (0.08 if pot >= 1 else 0.0) + (0.06 if wet and strong else 0.0)
        if river:
            donk = 0.32 if strong else (0.12 if air else 0.06)
        if spr < 2.2 and (value or strong):
            w[6] = 0.70
            w[0] = 0.30
            return _norm(w)
        w[0] = 1.0 - donk
        if strong:
            w[4] = donk * (0.55 if dry else 0.70)
            w[2] = donk * 0.25
            w[5] = donk * 0.20 if perc > 0.92 else 0.0
        elif draw:
            w[2] = donk * 0.70
            w[1] = donk * 0.30
        else:
            w[2] = donk
        return _norm(w)

    if spr < 2.35:
        jam = 0.0
        if strong or value:
            jam = 0.82
        elif draw:
            jam = 0.48
        elif air and perc < 0.10:
            jam = 0.28
        w[6] = jam
        w[0] = 1.0 - jam
        return _norm(w)

    if river:
        if strong:
            w[0] = 0.06
            w[4] = 0.42
            w[5] = 0.40 if perc > 0.90 else 0.18
            w[3] = 0.12
            return _norm(w)
        if value:
            w[0] = 0.28
            w[2] = 0.40
            w[3] = 0.22
            w[4] = 0.10
            return _norm(w)
        if air:
            bluff = 0.30 if dry else 0.14
            w[0] = 1.0 - bluff
            w[4] = bluff * 0.55
            w[5] = bluff * 0.30
            w[2] = bluff * 0.15
            return _norm(w)
        w[0] = 0.82
        w[2] = 0.18
        return _norm(w)

    cbet = 0.80 if (dry and flop) else 0.60
    if wet:
        cbet = 0.54 if flop else 0.40
    if pot >= 1:
        cbet = min(0.86, cbet + 0.08)
    if pot >= 2:
        cbet = min(0.90, cbet + 0.12)
    if street == 1:
        cbet *= 0.82

    if strong:
        w[0] = 0.08 if dry else 0.16
        w[2] = 0.38 if dry else 0.18
        w[4] = 0.36 if dry else 0.48
        w[5] = 0.10 if perc > 0.92 else 0.04
        w[3] = 1.0 - w[0] - w[2] - w[4] - w[5]
        return _norm(w)
    if value or (mid and dry and flop):
        if wet and mid:
            w[0] = 0.52
            w[2] = 0.28
            w[3] = 0.20
            return _norm(w)
        w[0] = max(0.08, 1.0 - cbet)
        w[1] = cbet * 0.18
        w[2] = cbet * 0.62
        w[3] = cbet * 0.20
        return _norm(w)
    if draw:
        w[0] = 0.28
        w[2] = 0.48
        w[1] = 0.12
        w[4] = 0.12
        return _norm(w)
    if air:
        bluff = cbet * (0.50 if dry else 0.22)
        w[0] = 1.0 - bluff
        w[1] = bluff * 0.22
        w[2] = bluff * 0.48
        w[4] = bluff * 0.30
        return _norm(w)
    w[0] = 1.0 - cbet * 0.35
    w[2] = cbet * 0.35
    return _norm(w)


def face_mix(
    perc: float,
    cat: int,
    fd: bool,
    oesd: bool,
    tex: int,
    street: int,
    pot: int,
    ip: bool,
    spr: float,
    face_frac: float,
) -> np.ndarray:
    """Facing mix. 0.25x continues middle pairs; 1.25x folds them. That is the point."""
    w = np.zeros(N_FACE, dtype=np.float64)
    river = street == 2
    draw = (fd or oesd) and not river
    strong = perc > 0.85 or cat >= 4
    value = perc > 0.70 or cat >= 2
    pair = cat >= 1
    air = perc < 0.18 and cat == 0 and not draw
    mdf = 1.0 / (1.0 + max(face_frac, 0.05))

    if face_frac <= 0.30:
        fold_cut, rv, ra = 0.10, 0.24, 0.10
    elif face_frac <= 0.42:
        fold_cut, rv, ra = 0.20, 0.14, 0.07
    elif face_frac <= 0.65:
        fold_cut, rv, ra = 0.36, 0.10, 0.04
    elif face_frac <= 1.05:
        fold_cut, rv, ra = 0.46, 0.08, 0.03
    else:
        fold_cut, rv, ra = 0.58, 0.04, 0.02

    if pot >= 1:
        fold_cut -= 0.07
    if pot >= 2:
        fold_cut -= 0.10
    if spr < 2.5:
        fold_cut -= 0.10
    if pot >= 1 and pair and perc > 0.48 and spr < 5.5:
        fold_cut = min(fold_cut, 0.14)
    fold_cut = max(0.04, fold_cut)

    continue_ok = perc >= fold_cut or draw or strong
    if pair and face_frac <= 0.40:
        continue_ok = True
    if pair and perc > 0.50 and face_frac <= 1.05:
        continue_ok = True
    if pair and face_frac > 1.05 and perc < 0.72 and cat < 4 and not strong:
        continue_ok = False
    if river and not strong and perc < (1.0 - mdf - 0.05):
        if not (pair and face_frac <= 0.40):
            continue_ok = perc >= (1.0 - mdf)

    if not continue_ok:
        w[0] = 0.94
        w[1] = 0.06
        return _norm(w)

    if spr < 2.2 and (value or draw or strong):
        w[3] = 0.72 if (strong or value) else 0.38
        w[1] = 1.0 - w[3]
        return _norm(w)

    want_raise = False
    if strong:
        want_raise = True
    elif value and face_frac <= 0.55 and perc > 0.88:
        want_raise = True
    elif air and face_frac <= 0.80 and perc < 0.10:
        want_raise = True
    elif draw and face_frac <= 0.50:
        want_raise = perc < 0.32

    if want_raise and spr > 2.4:
        rf = rv if (strong or value) else ra
        if strong:
            rf = min(0.88, rf + 0.50)
        if face_frac > 1.05:
            w[3] = rf
            w[1] = 1.0 - rf
            return _norm(w)
        w[2] = rf * (0.75 if spr > 6 else 0.55)
        w[3] = rf - w[2]
        w[1] = max(0.05, 1.0 - w[2] - w[3] - 0.02)
        w[0] = 0.02
        return _norm(w)

    w[1] = 0.90
    w[0] = 0.04
    w[2] = 0.06 if face_frac <= 0.55 else 0.02
    return _norm(w)


def combo_features(combo: Tuple[int, int], board: Sequence[int], perc: float) -> Tuple[int, bool, bool]:
    cat = 0
    if len(board) >= 3:
        cat = category(evaluate([combo[0], combo[1]] + list(board)))
    fd = oesd = False
    if len(board) < 5:
        fd, oesd = _draw_flags(combo, board)
    return cat, fd, oesd


def mix_for_combo(
    combo: Tuple[int, int],
    board: Sequence[int],
    perc: float,
    facing: bool,
    face_frac: float,
    street: int,
    pot: int,
    ip: bool,
    spr: float,
) -> np.ndarray:
    cat, fd, oesd = combo_features(combo, board, perc)
    tex = tex_id(board)
    dump = _dump_mix(combo, board, facing, face_frac, street, pot, ip)
    if dump is not None:
        return dump
    if facing:
        return face_mix(perc, cat, fd, oesd, tex, street, pot, ip, spr, face_frac)
    return first_mix(perc, cat, fd, oesd, tex, street, pot, ip, spr)


def mix_on_acts(mix: np.ndarray, acts: Sequence[str], facing: bool) -> np.ndarray:
    """Project a library mix onto a search-tree action list."""
    w = np.zeros(len(acts), dtype=np.float64)
    if not acts:
        return w
    if facing:
        has_raise = any(str(a).startswith("raise") for a in acts)
        for i, name in enumerate(acts):
            n = str(name)
            if n == "fold":
                w[i] = float(mix[0]) if len(mix) > 0 else 0.0
            elif n == "call":
                w[i] = float(mix[1]) if len(mix) > 1 else 0.0
            elif n.startswith("raise"):
                w[i] = float(mix[2]) if len(mix) > 2 else 0.0
            elif n in ("allin", "jam"):
                jam = float(mix[3]) if len(mix) > 3 else 0.0
                w[i] = jam + (0.0 if has_raise else 0.35 * (float(mix[2]) if len(mix) > 2 else 0.0))
        return _norm(w)
    lib_fracs = (0.0, 0.25, 0.33, 0.55, 0.75, 1.25, 9.0)
    act_fracs: List[float] = []
    for name in acts:
        n = str(name)
        if n == "check":
            act_fracs.append(0.0)
        elif n in ("allin", "jam"):
            act_fracs.append(9.0)
        elif n.startswith("bet"):
            digits = "".join(ch for ch in n if ch.isdigit())
            act_fracs.append(int(digits or "75") / 100.0)
        else:
            act_fracs.append(0.75)
    for li, lf in enumerate(lib_fracs):
        if li >= len(mix):
            break
        j = min(range(len(acts)), key=lambda i: abs(act_fracs[i] - lf))
        w[j] += float(mix[li])
    return _norm(w)


def p_observed_first(mix: np.ndarray, size_frac: float) -> float:
    if size_frac <= 0.06:
        return float(mix[0])
    if size_frac > 2.4:
        return float(mix[6] + 0.25 * mix[5])
    targets = ((0.25, 1), (0.33, 2), (0.55, 3), (0.75, 4), (1.25, 5))
    acc = tw = 0.0
    for f, i in targets:
        k = math.exp(-((size_frac - f) / 0.16) ** 2)
        acc += k * float(mix[i])
        tw += k
    jam = float(mix[6]) * (0.35 if size_frac > 1.4 else 0.05)
    return float(acc / max(tw, 1e-9) + jam)


def p_observed_face(mix: np.ndarray, act: str) -> float:
    name = {"fold": 0, "call": 1, "raise": 2, "jam": 3}.get(act, 1)
    return float(mix[name])


def percentiles(rg: Range, board: Sequence[int]) -> Tuple[np.ndarray, List[int]]:
    live = [i for i in range(N_COMBOS) if rg.w[i] > 1e-12]
    if not live:
        return np.zeros(N_COMBOS), []
    scored = [(board_combo_score(combo_from_index(i), board), i) for i in live]
    scored.sort()
    perc = np.zeros(N_COMBOS, dtype=np.float64)
    n = max(len(scored) - 1, 1)
    for rank, (_, i) in enumerate(scored):
        perc[i] = rank / n
    return perc, live


def combo_percentiles(
    combos: Sequence[Tuple[int, int]],
    board: Sequence[int],
    full_combos: Optional[Sequence[Tuple[int, int]]] = None,
    full_w: Optional[Sequence[float]] = None,
) -> List[float]:
    """Percentile of each combo in the full range — LibraryGTO's mix_for_combo input.

    Ranking only the search subsample made a 40th-percentile hand look like
    the nuts (or air) and BR'd to the wrong mix.
    """
    src_c = list(full_combos) if full_combos is not None else list(combos)
    if full_w is not None:
        src_w = [float(x) for x in full_w]
    else:
        src_w = [1.0] * len(src_c)
    rg = Range()
    for c, w in zip(src_c, src_w):
        if float(w) <= 1e-12:
            continue
        a, b = (int(c[0]), int(c[1])) if c[0] < c[1] else (int(c[1]), int(c[0]))
        rg.w[combo_index(a, b)] = float(w)
    if rg.total() <= 0:
        return [0.5] * len(combos)
    arr, _ = percentiles(rg, board)
    out: List[float] = []
    for c in combos:
        a, b = (int(c[0]), int(c[1])) if c[0] < c[1] else (int(c[1]), int(c[0]))
        out.append(float(arr[combo_index(a, b)]))
    return out


def apply_action(
    rg: Range,
    board: Sequence[int],
    act_type: ActionType,
    size_frac: float,
    street: int,
    pot: int,
    ip: bool,
    spr: float,
    observed: str = "",
) -> Range:
    """Bayes update: reach *= P(observed action | combo) from the library mix.

    `size_frac` on a raise is the bet they faced, not the raise put. Pass
    observed="jam" for a facing all-in — vs an overbet, face_mix only jams
    (mix[2]=0), so treating that shove as a raise inverted the range.
    """
    if rg.total() <= 0:
        return rg
    perc, live = percentiles(rg, board)
    out = Range()
    face_name = observed if observed in ("raise", "jam") else (
        "jam" if size_frac > 2.4 else "raise"
    )
    for i in live:
        combo = combo_from_index(i)
        if act_type == ActionType.CHECK:
            mix = mix_for_combo(combo, board, float(perc[i]), False, 0.0, street, pot, ip, spr)
            p = max(p_observed_first(mix, 0.0), 1e-6)
        elif act_type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            # Betting/raising as the aggressor (first-in or raise). Raise uses face mix.
            if act_type == ActionType.RAISE:
                mix = mix_for_combo(combo, board, float(perc[i]), True, size_frac, street, pot, ip, spr)
                p = max(p_observed_face(mix, face_name), 1e-6)
            else:
                mix = mix_for_combo(combo, board, float(perc[i]), False, 0.0, street, pot, ip, spr)
                p = max(p_observed_first(mix, size_frac), 1e-6)
        elif act_type == ActionType.CALL:
            mix = mix_for_combo(combo, board, float(perc[i]), True, size_frac, street, pot, ip, spr)
            p = max(p_observed_face(mix, "call"), 1e-6)
        elif act_type == ActionType.FOLD:
            mix = mix_for_combo(combo, board, float(perc[i]), True, size_frac, street, pot, ip, spr)
            p = max(p_observed_face(mix, "fold"), 1e-6)
        else:
            p = 1.0
        out.w[i] = rg.w[i] * p
    if out.total() <= 0:
        return rg
    return out


def replay_postflop(
    view: PlayerView,
    ranges: List[Range],
    dead: Sequence[int],
) -> Optional[List[Range]]:
    """Walk each public postflop action as library reach. None if no board yet."""
    if len(view.board) < 3:
        return None
    from strategy.pot_type import preflop_kind
    from strategy.range_model import iter_history

    out = [r.copy() for r in ranges]
    board_full = list(view.board)
    pot_est = float(view.bb + getattr(view, "sb", view.bb // 2))
    kind = preflop_kind(view)
    pidx = pot_index(kind)
    n_pl = len(view.stacks)
    left = [float(view.stacks[i] + view.committed[i]) for i in range(n_pl)]
    # Blinds are in committed but not in action_history. Without subtracting
    # them, BB leftover SPR is 1bb high and mix_for_combo's spr<2.5 jam gate
    # desyncs from library_policy (which uses view.effective_stack).
    sb_amt = float(getattr(view, "sb", view.bb // 2))
    bb_amt = float(view.bb)
    if n_pl <= 2:
        sb_seat = int(view.button) % n_pl
        bb_seat = (int(view.button) + 1) % n_pl
    else:
        sb_seat = (int(view.button) + 1) % n_pl
        bb_seat = (int(view.button) + 2) % n_pl
    if 0 <= sb_seat < n_pl:
        left[sb_seat] = max(0.0, left[sb_seat] - sb_amt)
    if 0 <= bb_seat < n_pl:
        left[bb_seat] = max(0.0, left[bb_seat] - bb_amt)
    # First-to-act at each history node, not view.folded (which already has
    # later folds). BB's cbet with SB live is IP; after SB folds it is not.
    can_act = [out[i].total() > 0 for i in range(n_pl)]
    street_put = [0.0] * n_pl
    bet_level = 0.0
    last_st = Street.PREFLOP

    def _freeze_broke() -> None:
        for i in range(n_pl):
            if left[i] <= 0.5:
                can_act[i] = False

    def _ip_at(seat: int) -> bool:
        for k in range(1, n_pl + 1):
            i = (view.button + k) % n_pl
            if can_act[i] and left[i] > 0.5:
                return i != seat
        return False

    _freeze_broke()
    for seat, act, st in iter_history(view):
        if st == Street.PREFLOP:
            if act.type in (ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                amt = float(max(act.amount, 0))
                pot_est += amt
                if 0 <= seat < n_pl:
                    left[seat] = max(0.0, left[seat] - amt)
                    if act.type == ActionType.ALL_IN or left[seat] <= 0.5:
                        can_act[seat] = False
            last_st = st
            continue
        if st != last_st:
            street_put = [0.0] * n_pl
            bet_level = 0.0
            last_st = st
            _freeze_broke()
        n_board = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}.get(st, 3)
        board = board_full[: min(n_board, len(board_full))]
        if len(board) < 3:
            continue
        sidx = street_index(st)
        if act.type == ActionType.FOLD:
            out[seat] = Range()
            if 0 <= seat < n_pl:
                can_act[seat] = False
            continue
        if out[seat].total() <= 0:
            continue
        ip = _ip_at(seat)
        cap = left[seat] if 0 <= seat < n_pl else float(view.effective_stack)
        for j in range(n_pl):
            if j != seat and out[j].total() > 0 and left[j] < cap:
                cap = left[j]
        spr = cap / max(pot_est, 1.0)
        already = street_put[seat] if 0 <= seat < n_pl else 0.0
        to_c = max(0.0, bet_level - already)
        if act.type == ActionType.CHECK:
            if to_c <= 0.5:
                out[seat] = apply_action(out[seat], board, ActionType.CHECK, 0.0, sidx, pidx, ip, spr)
        elif act.type == ActionType.CALL:
            frac = to_c / max(pot_est - to_c, 1.0) if to_c > 0 else 0.33
            out[seat] = apply_action(out[seat], board, ActionType.CALL, frac, sidx, pidx, ip, spr)
            amt = float(act.amount)
            pot_est += amt
            if 0 <= seat < n_pl:
                street_put[seat] = already + amt
                left[seat] = max(0.0, left[seat] - amt)
                if left[seat] <= 0.5:
                    can_act[seat] = False
        elif act.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            amt = float(act.amount)
            frac = amt / max(pot_est, 1.0)
            facing_now = to_c > 0.5
            if act.type == ActionType.RAISE or (act.type == ActionType.ALL_IN and facing_now):
                face = to_c / max(pot_est - to_c, 1.0) if to_c > 0 else frac
                # size_frac is the bet they faced. A shove is observed="jam";
                # vs overbet face_mix has mix[2]=0, so a raise observation
                # kept air and dropped the nuts.
                obs = "jam" if act.type == ActionType.ALL_IN or frac > 2.4 else "raise"
                out[seat] = apply_action(
                    out[seat], board, ActionType.RAISE, face, sidx, pidx, ip, spr, observed=obs,
                )
            else:
                # First-in jam at SPR 2.2–2.4 is stack/pot ≈ 2.3, which
                # p_observed_first treats as a 125% overbet (jam only if
                # size_frac > 2.4). 3bet-turn IP is jam-or-check; that
                # observation kept overbet air and dropped the jam range.
                if act.type == ActionType.ALL_IN:
                    frac = 9.0
                out[seat] = apply_action(out[seat], board, ActionType.BET, frac, sidx, pidx, ip, spr)
            pot_est += amt
            if 0 <= seat < n_pl:
                street_put[seat] = already + amt
                if street_put[seat] > bet_level + 0.5:
                    bet_level = street_put[seat]
                left[seat] = max(0.0, left[seat] - amt)
                if act.type == ActionType.ALL_IN or left[seat] <= 0.5:
                    can_act[seat] = False
        out[seat] = out[seat].mask_dead(dead)
    return out


def continuation_extra(
    hero: Tuple[int, int],
    vill: Tuple[int, int],
    board: Sequence[int],
    pot: float,
    cap: float,
    streets_left: int,
    pot_type: int = 0,
    sixmax: bool = False,
) -> float:
    """Library leaf: extra chips from remaining streets, not sampled showdown."""
    if streets_left <= 0 or cap <= 0.5:
        return 0.0
    hb = street_bucket(hero, board) / max(N_BKT - 1, 1)
    vb = street_bucket(vill, board) / max(N_BKT - 1, 1)
    vol = 0.28 + 0.72 * (0.5 * (hb + vb))
    realize = 0.52 + 0.14 * min(streets_left, 2)
    if pot_type >= 1:
        realize += 0.12
    tex = tex_id(board)
    if tex in (3, 4, 5, 7):
        vol += 0.08
    extra = (hb - vb) * min(cap, pot * 0.48 * streets_left * vol * realize)
    hi = min(N_BKT - 1, street_bucket(hero, board))
    vi = min(N_BKT - 1, street_bucket(vill, board))
    pack = _tables()
    if pack is not None:
        cv = pack["cv"]
        if cv.ndim == 5:
            si = stack_index(cap / max(pot / 13.0, 1.0) if pot > 0 else 100.0)
            pi = min(2, max(0, pot_type))
            table_v = float(cv[si, pi, tex, hi, vi])
            extra = min(cap, max(-cap, pot * table_v * streets_left))
    if not sixmax:
        try:
            from strategy.blueprint import continuation_cv

            bp = continuation_cv(hi, vi, pot_type, sixmax=False)
            if bp is not None:
                v = float(bp)
                mag = abs(v)
                if mag <= 2.5:
                    bp_extra = min(cap, max(-cap, pot * v * streets_left))
                else:
                    bp_extra = min(cap, max(-cap, v))
                extra = 0.55 * extra + 0.45 * bp_extra
        except Exception:
            pass
    return float(extra)


def continuation_matrix(
    hero_holes: Sequence[Tuple[int, int]],
    vill_holes: Sequence[Tuple[int, int]],
    board: Sequence[int],
    pot: float,
    stk0: float,
    stk1: float,
    pot_type: int,
    streets_left: int,
    sixmax: bool = False,
) -> Optional[np.ndarray]:
    if streets_left <= 0:
        return None
    cap = min(stk0, stk1)
    H, V = len(hero_holes), len(vill_holes)
    out = np.zeros((H, V), dtype=np.float64)
    for i, h in enumerate(hero_holes):
        for j, v in enumerate(vill_holes):
            if h[0] == v[0] or h[0] == v[1] or h[1] == v[0] or h[1] == v[1]:
                continue
            out[i, j] = continuation_extra(
                h, v, board, pot, cap, streets_left, pot_type, sixmax=sixmax
            )
    return out


def first_puts(pot: float, stack: float, mix: np.ndarray) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for name, p in zip(FIRST_ACTS, mix):
        if name == "check":
            out.append((name, 0.0))
            continue
        if name == "jam":
            out.append((name, stack))
            continue
        frac = {"bet25": 0.25, "bet33": 0.33, "bet55": 0.55, "bet75": 0.75, "bet125": 1.25}[name]
        put = min(stack, max(1.0, frac * pot))
        if put >= stack * 0.90:
            continue
        out.append((name, put))
        _ = p
    return out


def _to_engine(view: PlayerView, name: str, put: float) -> Action:
    stack = view.stacks[view.hero]
    to_call = view.to_call
    if name == "fold":
        return Action(ActionType.CHECK) if to_call <= 0 else Action(ActionType.FOLD)
    if name in ("check",):
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


def library_policy(view: PlayerView, hero_range: Range, rng) -> Optional[Action]:
    """Sample the distilled/imported mix for hero's combo. None if preflop."""
    if len(view.board) < 3:
        return None
    from strategy.pot_type import classify

    hole = (min(view.hole), max(view.hole))
    board = list(view.board)
    perc_arr, live = percentiles(hero_range if hero_range.total() > 0 else Range.from_hole(*hole), board)
    perc = float(perc_arr[combo_index(*hole)]) if combo_index(*hole) < N_COMBOS else 0.5
    if live and combo_index(*hole) not in live:
        perc = 0.5
    ip = not _hero_oop(view)
    kind = classify(view)
    pidx = pot_index(kind)
    sidx = street_index(view.street)
    spr = view.effective_stack / max(view.pot, 1)
    facing = view.to_call > 0
    frac = facing_frac(view.to_call, view.pot) if facing else 0.0
    mix = mix_for_combo(hole, board, perc, facing, frac, sidx, pidx, ip, spr)
    stack = float(view.stacks[view.hero])
    pot = float(view.pot)
    if facing:
        names = list(FACE_ACTS)
        puts = [0.0, min(float(view.to_call), stack)]
        raise_put = min(stack, max(view.to_call * 2.8, view.to_call + 0.75 * pot))
        if frac > 1.05:
            puts.extend([stack, stack])
        else:
            puts.extend([raise_put, stack])
    else:
        names = list(FIRST_ACTS)
        puts = [p for _, p in first_puts(pot, stack, mix)]
        if len(puts) < len(names):
            puts = [0.0]
            for n in names[1:-1]:
                f = {"bet25": 0.25, "bet33": 0.33, "bet55": 0.55, "bet75": 0.75, "bet125": 1.25}[n]
                puts.append(min(stack, max(1.0, f * pot)))
            puts.append(stack)
    p = np.maximum(mix[: len(names)], 0.0)
    if p.sum() <= 0:
        p = np.ones_like(p)
    p = p / p.sum()
    choice = int(rng.choices(range(len(names)), weights=[float(x) for x in p], k=1)[0])
    return _to_engine(view, names[choice], puts[choice] if choice < len(puts) else 0.0)


def _tables() -> Optional[dict]:
    global _PACK
    if _PACK is not None:
        return _PACK
    if not LIB_PATH.exists():
        build_tables(LIB_PATH)
    if not LIB_PATH.exists():
        return None
    z = np.load(LIB_PATH)
    _PACK = {k: z[k] for k in z.files}
    _load_dumps()
    return _PACK


def build_tables(path: Path | None = None) -> Path:
    """Materialize bucket tables from the distilled mix for search CV lookup."""
    path = path or LIB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)
    cv = np.zeros((N_STACK, N_POT, N_TEX, N_BKT, N_BKT), dtype=np.float64)
    mix_f = np.zeros((N_STACK, N_POT, N_STREET, N_TEX, N_IP, N_BKT, N_FIRST), dtype=np.float64)
    mix_c = np.zeros((N_STACK, N_POT, N_STREET, N_TEX, N_FACE_SZ, N_BKT, N_FACE), dtype=np.float64)
    for si in range(N_STACK):
        spr = (2.2, 13.0, 26.0)[si]
        for pi in range(N_POT):
            if pi == 1:
                spr_p = spr * 0.35
            elif pi == 2:
                spr_p = spr * 0.18
            else:
                spr_p = spr
            for tex in range(N_TEX):
                for hb in range(N_BKT):
                    perc = hb / max(N_BKT - 1, 1)
                    cat = 4 if perc > 0.88 else (2 if perc > 0.7 else (1 if perc > 0.45 else 0))
                    fd = perc < 0.4 and hb % 2 == 1
                    oesd = perc < 0.35 and hb % 3 == 0
                    for st in range(N_STREET):
                        for ip in (0, 1):
                            mix_f[si, pi, st, tex, ip, hb] = first_mix(
                                perc, cat, fd, oesd, tex, st, pi, bool(ip), spr_p
                            )
                        for fs, frac in enumerate((0.25, 0.33, 0.50, 0.75, 1.00, 1.25)):
                            mix_c[si, pi, st, tex, fs, hb] = face_mix(
                                perc, cat, fd, oesd, tex, st, pi, True, spr_p, frac
                            )
                    for vb in range(N_BKT):
                        hs = hb / 49.0
                        vs = vb / 49.0
                        cv[si, pi, tex, hb, vb] = (hs - vs) * (0.55 + 0.2 * pi)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        cv=cv,
        mix_first=mix_f,
        mix_face=mix_c,
        kind=np.array("lib_6max_100"),
        seed=np.array(int(rng.integers(1, 10))),
    )
    tmp.replace(path)
    global _PACK
    _PACK = None
    return path


def _dump_key(board: Sequence[int], facing: bool, face_frac: float, street: int, pot: int, ip: bool) -> tuple:
    b = tuple(int(x) for x in board[:5])
    fs = face_size_index(face_frac) if facing else -1
    return (b, int(facing), fs, street, pot, int(ip))


def _dump_mix(
    combo: Tuple[int, int],
    board: Sequence[int],
    facing: bool,
    face_frac: float,
    street: int,
    pot: int,
    ip: bool,
) -> Optional[np.ndarray]:
    if not _DUMPS:
        return None
    key = _dump_key(board, facing, face_frac, street, pot, ip)
    tab = _DUMPS.get(key)
    if tab is None:
        return None
    i = combo_index(*combo)
    if i < 0 or i >= tab.shape[0]:
        return None
    row = tab[i]
    if float(row.sum()) <= 1e-12:
        return None
    return _norm(np.array(row, dtype=np.float64))


def _load_dumps() -> None:
    """Import Pio/GTOWizard JSON dumps from Final_Bot/data/solver_dumps/."""
    global _DUMPS
    if _DUMPS or not DUMP_DIR.exists():
        return
    for p in DUMP_DIR.glob("*.json"):
        try:
            import_pio_json(p)
        except Exception:
            continue


def import_pio_json(path: Path) -> None:
    """Merge a solver dump.

    {
      "street": "flop",
      "board": "Ks7h2c",
      "pot_type": "srp",
      "ip": true,
      "facing_frac": 0.0,
      "strategy": {"AhAd": {"check": 0.1, "bet33": 0.9}, ...}
    }
    """
    from poker.cards import parse_cards
    from strategy.hands import combos_of_169

    data = json.loads(Path(path).read_text())
    board = parse_cards(data["board"]) if isinstance(data["board"], str) else list(data["board"])
    street = {"flop": 0, "turn": 1, "river": 2}.get(str(data.get("street", "flop")).lower(), 0)
    pot = pot_index(str(data.get("pot_type", "srp")))
    ip = bool(data.get("ip", True))
    facing_f = float(data.get("facing_frac", 0.0))
    facing = facing_f > 0.02
    names = list(FACE_ACTS if facing else FIRST_ACTS)
    tab = np.zeros((N_COMBOS, len(names)), dtype=np.float64)
    strat = data.get("strategy") or data.get("strategy_169") or {}
    for hand, freqs in strat.items():
        vec = np.zeros(len(names), dtype=np.float64)
        for k, v in freqs.items():
            k2 = str(k).lower().replace(" ", "")
            for i, n in enumerate(names):
                if k2 == n or k2.startswith(n) or n.startswith(k2):
                    vec[i] += float(v)
        if vec.sum() <= 0:
            continue
        vec = _norm(vec)
        if len(hand) <= 4 and not hand[:2].isdigit():
            try:
                for c1, c2 in combos_of_169(hand):
                    tab[combo_index(c1, c2)] = vec
            except Exception:
                cards = parse_cards(hand)
                tab[combo_index(cards[0], cards[1])] = vec
        else:
            cards = parse_cards(hand)
            tab[combo_index(cards[0], cards[1])] = vec
    _DUMPS[_dump_key(board, facing, facing_f, street, pot, ip)] = tab


def loaded() -> bool:
    return _tables() is not None


def refresh() -> None:
    global _PACK, _DUMPS
    _PACK = None
    _DUMPS = {}
    _tables()
