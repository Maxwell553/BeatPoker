"""Load MCCFR blueprint checkpoints and replay them as beliefs / preflop mix.

Missing or off-tree spots fall back to combo trees in range_model / preflop_trees.
Never loads the old 4-node limp-showdown `hu_preflop_*.npz` files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from poker.equity import combo_index
from poker.game import Action, ActionType, PlayerView
from strategy.card_abs import flop_bucket, hand_id, street_bucket
from strategy.hands import HANDS_169, combos_of_169, hole_to_169
from strategy.ranges import Range
from strategy.stacks import stack_bucket

DATA = Path(__file__).resolve().parents[1] / "Final_Bot" / "data"
HU_PATH = DATA / "blueprint_hu_100.npz"
SIX_PATH = DATA / "blueprint_6max_100.npz"
SIX_V2_PATH = DATA / "blueprint_6max_v2_100.npz"
SIX_V3_PATH = DATA / "blueprint_6max_v3_100.npz"
HU_V3_PATH = DATA / "blueprint_hu_v3_100.npz"
SIX_V3_BEST_PATH = DATA / "blueprint_6max_v3_best.npz"
HU_V3_BEST_PATH = DATA / "blueprint_hu_v3_best.npz"
V3_GATE_PATH = DATA / "v3_gate.json"
V3_TREE_VERSION = 30
V3_HU_TREE_VERSION = 21

# Blend v3 into table policy only after the eval+gate file says the chart is sane.
V3_POLICY_MIN = 25_000_000
V3_POLICY_FULL = 100_000_000

PF_ACTS = {
    0: ["fold", "limp", "raise", "jam"],
    1: ["fold", "call", "three", "jam"],
    2: ["fold", "call", "raise", "jam"],
    3: ["fold", "call", "raise", "jam"],
    4: ["check", "iso", "jam"],
    5: ["fold", "call", "jam"],
    6: ["fold", "call", "raise", "jam"],
    7: ["fold", "call", "raise", "jam"],
}

_HU = None
_SIX = None
_V2 = None
_V3 = None
_HU_V3 = None
_HU_MTIME = None
_SIX_MTIME = None
_V2_MTIME = None
_V3_MTIME = None
_HU_V3_MTIME = None
_GATE = None
_GATE_MTIME = None


def _mtime(path: Path):
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def hu_loaded() -> bool:
    refresh_hu()
    return _HU is not None and int(_HU.get("iters", 0)) > 0


def six_loaded() -> bool:
    refresh_six()
    return _SIX is not None and int(_SIX.get("iters", 0)) > 0


def refresh_hu(path: Path | None = None) -> Optional[dict]:
    global _HU, _HU_MTIME
    p = path or HU_PATH
    mt = _mtime(p)
    if mt is None:
        _HU = None
        _HU_MTIME = None
        return None
    if _HU is not None and mt == _HU_MTIME:
        return _HU
    z = np.load(p)
    kind = str(z["kind"]) if "kind" in z.files else ""
    if kind and "hu" not in kind:
        return _HU
    pack = {"iters": int(z["iters"]), "stack": float(z["stack"]), "cv": np.array(z["cv"]) if "cv" in z.files else None}
    for n in PF_ACTS:
        key = f"pf{n}"
        if key in z.files:
            s = np.array(z[key], dtype=np.float64)
            tot = s.sum(axis=1, keepdims=True)
            k = s.shape[1]
            pack[n] = np.where(tot > 0, s / tot, 1.0 / k)
    _HU = pack
    _HU_MTIME = mt
    return _HU


def refresh_six(path: Path | None = None) -> Optional[dict]:
    global _SIX, _SIX_MTIME
    p = path or SIX_PATH
    mt = _mtime(p)
    if mt is None:
        _SIX = None
        _SIX_MTIME = None
        return None
    if _SIX is not None and mt == _SIX_MTIME:
        return _SIX
    z = np.load(p)
    s = np.array(z["ssum"], dtype=np.float64)
    tot = s.sum(axis=2, keepdims=True)
    avg = np.divide(s, tot, out=np.full_like(s, 0.25), where=tot > 0)
    _SIX = {
        "iters": int(z["iters"]),
        "stack": float(z["stack"]),
        "avg": avg,
    }
    _SIX_MTIME = mt
    return _SIX


def refresh_six_v2(path: Path | None = None) -> Optional[dict]:
    global _V2, _V2_MTIME
    p = path or SIX_V2_PATH
    mt = _mtime(p)
    if mt is None:
        _V2 = None
        _V2_MTIME = None
        return None
    if _V2 is not None and mt == _V2_MTIME:
        return _V2
    z = np.load(p)
    pack = {
        "iters": int(z["iters"]) if "iters" in z.files else 0,
        "cv": np.array(z["cv"]) if "cv" in z.files else None,
        "sum_fl": np.array(z["sum_fl"]) if "sum_fl" in z.files else None,
        "sum_pf": np.array(z["sum_pf"]) if "sum_pf" in z.files else None,
    }
    _V2 = pack
    _V2_MTIME = mt
    return _V2


def hu_mix(node: int, hid: int) -> Optional[np.ndarray]:
    pack = refresh_hu()
    if pack is None or node not in pack:
        return None
    row = pack[node]
    if hid < 0 or hid >= row.shape[0]:
        return None
    return row[hid]


def six_mix(spot: int, hid: int) -> Optional[np.ndarray]:
    pack = refresh_six()
    if pack is None:
        return None
    avg = pack["avg"]
    if spot < 0 or spot >= avg.shape[0] or hid < 0 or hid >= avg.shape[1]:
        return None
    return avg[spot, hid]


def continuation_cv(
    hero_bkt: int, vill_bkt: int, pot_type: int = 0, *, sixmax: bool = False
) -> Optional[float]:
    """Blueprint EV/pot for a street-complete flop leaf, or None if missing.

    HU v3 has more iters than the frozen 6-max snapshot, so a max-iters
    pick blended HU-priced futures into 6-max extras (overcall vs LibraryGTO).
    `sixmax=True` prefers the 6-max pack.
    """
    v3 = refresh_six_v3() if v3_leaves_allowed() else None
    hu3 = refresh_hu_v3() if v3_leaves_allowed() else None

    def _from_pack(pack) -> Optional[Tuple[float, int]]:
        if pack is None or pack.get("cv") is None:
            return None
        cv = pack["cv"]
        if cv.size <= 0 or float(np.sum(np.abs(cv))) < 1e-8:
            return None
        n = int(pack.get("iters") or 0)
        hb = int(max(0, min(cv.shape[0] - 1, hero_bkt)))
        vb = int(max(0, min(cv.shape[1] - 1, vill_bkt)))
        pt = int(max(0, min(cv.shape[2] - 1, pot_type)))
        return float(cv[hb, vb, pt]), n

    if sixmax:
        for pack in (v3, hu3):
            got = _from_pack(pack)
            if got is not None and got[1] >= 10_000:
                return got[0]
    else:
        best = None
        best_n = -1
        for pack in (v3, hu3):
            got = _from_pack(pack)
            if got is None:
                continue
            val, n = got
            if n >= best_n:
                best, best_n = val, n
        if best is not None and best_n >= 10_000:
            return best
    pack = refresh_hu()
    hu_v = None
    if pack is not None and pack.get("cv") is not None:
        cv = pack["cv"]
        if cv.size > 0:
            hb = int(max(0, min(cv.shape[0] - 1, hero_bkt)))
            vb = int(max(0, min(cv.shape[1] - 1, vill_bkt)))
            pt = int(max(0, min(cv.shape[2] - 1, pot_type)))
            hu_v = float(cv[hb, vb, pt])
            if abs(hu_v) < 1e-12 and float(np.sum(np.abs(cv))) < 1e-8:
                hu_v = None
    v2_v = None
    v2 = refresh_six_v2()
    if v2 is not None and v2.get("cv") is not None:
        cv2 = v2["cv"]
        if cv2.size > 0:
            hb = int(max(0, min(cv2.shape[0] - 1, hero_bkt)))
            vb = int(max(0, min(cv2.shape[1] - 1, vill_bkt)))
            pt = int(max(0, min(cv2.shape[2] - 1, pot_type)))
            v2_v = float(cv2[hb, vb, pt])
            if abs(v2_v) < 1e-12 and float(np.sum(np.abs(cv2))) < 1e-8:
                v2_v = None
    if hu_v is None:
        return v2_v
    if v2_v is None:
        return hu_v
    n = int(v2.get("iters") or 0)
    w2 = 0.30
    if n >= 5_000_000:
        w2 = 0.55
    elif n >= 1_000_000:
        w2 = 0.42
    return (1.0 - w2) * hu_v + w2 * v2_v


def _v3_policy_path(best: Path, live: Path) -> Path:
    """Prefer a frozen production snapshot so the trainer can keep writing live files."""
    return best if best.exists() else live


def refresh_six_v3(path: Path | None = None) -> Optional[dict]:
    global _V3, _V3_MTIME
    p = path or _v3_policy_path(SIX_V3_BEST_PATH, SIX_V3_PATH)
    mt = _mtime(p)
    if mt is None:
        _V3 = None
        _V3_MTIME = None
        return None
    if mt == _V3_MTIME:
        return _V3
    z = np.load(p)
    kind = str(z["kind"]) if "kind" in z.files else ""
    if kind and "v3" not in kind:
        return _V3
    tree = int(z["tree"]) if "tree" in z.files else 0
    if tree < V3_TREE_VERSION:
        _V3 = None
        _V3_MTIME = mt
        return None
    pack = {
        "tree": tree,
        "iters": int(z["iters"]) if "iters" in z.files else 0,
        "cv": np.array(z["cv"]) if "cv" in z.files else None,
        "sum_pf": np.array(z["sum_pf"]) if "sum_pf" in z.files else None,
        "sum_fl": np.array(z["sum_fl"]) if "sum_fl" in z.files else None,
        "sum_tn": np.array(z["sum_tn"]) if "sum_tn" in z.files else None,
        "sum_rv": np.array(z["sum_rv"]) if "sum_rv" in z.files else None,
    }
    _V3 = pack
    _V3_MTIME = mt
    return _V3


def refresh_hu_v3(path: Path | None = None) -> Optional[dict]:
    global _HU_V3, _HU_V3_MTIME
    p = path or _v3_policy_path(HU_V3_BEST_PATH, HU_V3_PATH)
    mt = _mtime(p)
    if mt is None:
        _HU_V3 = None
        _HU_V3_MTIME = None
        return None
    if mt == _HU_V3_MTIME:
        return _HU_V3
    z = np.load(p)
    kind = str(z["kind"]) if "kind" in z.files else ""
    if kind and "hu_v3" not in kind and kind != "hu_v3":
        return _HU_V3
    tree = int(z["tree"]) if "tree" in z.files else 0
    if tree < V3_HU_TREE_VERSION:
        _HU_V3 = None
        _HU_V3_MTIME = mt
        return None
    pack = {
        "tree": tree,
        "iters": int(z["iters"]) if "iters" in z.files else 0,
        "cv": np.array(z["cv"]) if "cv" in z.files else None,
        "sum_pf": np.array(z["sum_pf"]) if "sum_pf" in z.files else None,
        "sum_fl": np.array(z["sum_fl"]) if "sum_fl" in z.files else None,
        "sum_tn": np.array(z["sum_tn"]) if "sum_tn" in z.files else None,
        "sum_rv": np.array(z["sum_rv"]) if "sum_rv" in z.files else None,
    }
    if pack["sum_pf"] is None and "pf0" in z.files:
        rows = []
        for n in range(8):
            key = f"pf{n}"
            if key in z.files:
                rows.append(np.array(z[key], dtype=np.float64))
        if rows:
            pack["sum_pf"] = np.stack(rows, axis=0)
    _HU_V3 = pack
    _HU_V3_MTIME = mt
    return _HU_V3


def v3_hu_action_range(node: int, acts: Sequence[int]) -> Optional[Range]:
    """Reach-weighted HU range from the live v3 mix (what LibraryGTO actually opens)."""
    if not v3_preflop_allowed():
        return None
    pack = refresh_hu_v3()
    sp = pack.get("sum_pf") if pack else None
    if sp is None or node < 0 or node >= int(sp.shape[0]):
        return None
    want = [int(a) for a in acts]
    rg = Range()
    n_a = int(sp.shape[-1])
    for i, code in enumerate(HANDS_169):
        row = sp[node, i]
        tot = float(np.sum(row))
        if tot <= 1e-12:
            continue
        p = float(sum(float(row[a]) for a in want if 0 <= a < n_a)) / tot
        if p <= 1e-12:
            continue
        for c in combos_of_169(code):
            rg.w[combo_index(*c)] = p
    return rg if rg.total() > 0 else None


def v3_6max_action_range(
    spot: int,
    acts: Sequence[int],
    *,
    rfi: bool = False,
    vs_ep: bool = False,
    fill: Optional[Range] = None,
) -> Optional[Range]:
    """Live 6-max v3 range after the same limp/jam zeros as try_v3_preflop.

    Hands whose mix collapses take `fill`. When `fill` is set, beliefs are
    the combo tree LibraryGTO actually plays — not `max(v3, tree)`, which
    turned a 18% BB flat vs BTN into the live 68% calling-station range,
    and not leftover v3 RFI mass the tree never opens.
    """
    if fill is not None:
        return fill.copy() if fill.total() > 0 else None
    if not v3_preflop_allowed():
        return None
    pack = refresh_six_v3()
    sp = pack.get("sum_pf") if pack else None
    if sp is None or spot < 0 or spot >= int(sp.shape[0]):
        return None
    want = [int(a) for a in acts]
    rg = Range()
    n_a = int(sp.shape[-1])
    for i, code in enumerate(HANDS_169):
        row = np.array(sp[spot, i], dtype=np.float64)
        tot = float(np.sum(row))
        mix = row / tot if tot > 1e-12 else None
        if mix is not None:
            if rfi:
                if mix.shape[0] > 1:
                    mix[1] = 0.0
                if mix.shape[0] > 3:
                    mix[3] = 0.0
            if vs_ep and mix.shape[0] > 1:
                mix[1] = 0.0
            s = float(mix.sum())
            if s > 1e-12:
                mix = mix / s
                p = float(sum(float(mix[a]) for a in want if 0 <= a < mix.shape[0]))
            else:
                mix = None
                p = 0.0
        else:
            p = 0.0
        for c in combos_of_169(code):
            idx = combo_index(*c)
            if p > 1e-12:
                rg.w[idx] = p
    return rg if rg.total() > 0 else None


def refresh_v3_gate(path: Path | None = None) -> Optional[dict]:
    """Live allow/deny for v3 table policy. Missing file => deny."""
    global _GATE, _GATE_MTIME
    p = path or V3_GATE_PATH
    mt = _mtime(p)
    if mt is None:
        _GATE = None
        _GATE_MTIME = None
        return None
    if _GATE is not None and mt == _GATE_MTIME:
        return _GATE
    try:
        pack = json.loads(p.read_text())
    except Exception:
        _GATE = None
        _GATE_MTIME = mt
        return None
    _GATE = pack
    _GATE_MTIME = mt
    return _GATE


def v3_preflop_allowed() -> bool:
    gate = refresh_v3_gate()
    return bool(gate and gate.get("allow_preflop"))


def v3_leaves_allowed() -> bool:
    gate = refresh_v3_gate()
    if gate is None:
        return False
    if "allow_leaves" in gate:
        return bool(gate.get("allow_leaves"))
    return bool(gate.get("allow_preflop"))


def v3_search_scale() -> float:
    gate = refresh_v3_gate()
    if not gate:
        return 1.0
    try:
        return float(max(1.0, min(1.5, gate.get("search_scale") or 1.0)))
    except Exception:
        return 1.0


def v3_weight(iters: int) -> float:
    if not v3_preflop_allowed():
        return 0.0
    if iters < V3_POLICY_MIN:
        return 0.0
    return float(min(0.88, (iters - V3_POLICY_MIN) / max(V3_POLICY_FULL - V3_POLICY_MIN, 1)))


def _avg_row(sum_tab: np.ndarray, *idx) -> Optional[np.ndarray]:
    try:
        row = sum_tab[idx]
    except Exception:
        return None
    s = float(np.sum(row))
    if s <= 1e-12:
        return np.full(row.shape, 1.0 / max(row.size, 1))
    return row / s


def continuation_extra(
    pot: float,
    stack0: float,
    stack1: float,
    streets_left: int,
    hero_bkt: Optional[int] = None,
    vill_bkt: Optional[int] = None,
    pot_type: int = 0,
) -> Optional[float]:
    """Implied extra chips from the blueprint CV table. None → caller uses geometric."""
    if streets_left <= 0:
        return 0.0
    cap = min(stack0, stack1)
    if cap <= 0.5:
        return 0.0
    if hero_bkt is None or vill_bkt is None:
        return None
    cv = continuation_cv(hero_bkt, vill_bkt, pot_type)
    if cv is None:
        return None
    # CV is chip-EV / pot. Negative when behind — do not abs() that into
    # extra money; search would call too much vs a library that barrels.
    extra = min(cap, max(-cap, pot * cv * streets_left))
    return float(extra)


def continuation_matrix(
    hero_holes: Sequence[Tuple[int, int]],
    vill_holes: Sequence[Tuple[int, int]],
    board: Sequence[int],
    pot: float,
    inv0: float,
    stk0: float,
    stk1: float,
    pot_type: int,
    streets_left: int,
) -> Optional[np.ndarray]:
    """(H, V) hero chip EV from blueprint buckets. None if no checkpoint CV."""
    pack = None
    if v3_leaves_allowed():
        pack = refresh_six_v3() or refresh_hu_v3()
    if pack is None:
        pack = refresh_hu()
    if pack is None or pack.get("cv") is None or streets_left <= 0:
        return None
    cv = pack["cv"]
    if float(np.sum(np.abs(cv))) < 1e-8:
        return None
    from strategy.card_abs import street_bucket, street_bucket_v3

    bkt = street_bucket_v3 if cv.shape[0] >= 72 else street_bucket
    H = len(hero_holes)
    V = len(vill_holes)
    hb = [min(cv.shape[0] - 1, bkt(h, board)) for h in hero_holes]
    vb = [min(cv.shape[1] - 1, bkt(v, board)) for v in vill_holes]
    pt = int(max(0, min(cv.shape[2] - 1, pot_type)))
    spr = min(stk0, stk1) / max(pot, 1.0)
    scale = min(1.0, spr / 4.0) * (streets_left / 2.0)
    out = np.zeros((H, V), dtype=np.float64)
    for i in range(H):
        for j in range(V):
            # EV/pot → chips at this pot, SPR-scaled
            out[i, j] = float(cv[hb[i], vb[j], pt]) * pot * (0.55 + 0.45 * scale) - (inv0 - 0.5 * pot) * 0.15
    return out


def _sample(mix: np.ndarray, rng) -> int:
    p = np.maximum(mix, 0.0)
    s = float(p.sum())
    if s <= 1e-12:
        return 0
    p = p / s
    u = rng.random() if hasattr(rng, "random") else rng.random()
    acc = 0.0
    for i, pi in enumerate(p):
        acc += float(pi)
        if u <= acc:
            return i
    return len(p) - 1


def _is_hu_100(view: PlayerView) -> bool:
    if view.num_players != 2:
        return False
    bb = view.effective_stack / max(view.bb, 1)
    return 70.0 <= bb <= 140.0 or stack_bucket(bb) == "100"


def _is_6max_100(view: PlayerView) -> bool:
    if view.num_players != 6:
        return False
    bb = view.effective_stack / max(view.bb, 1)
    return 70.0 <= bb <= 140.0 or stack_bucket(bb) == "100"


def _n_raises(view: PlayerView) -> int:
    n = 0
    for _, act in view.action_history:
        if act.type in (ActionType.RAISE, ActionType.BET):
            n += 1
        elif act.type == ActionType.ALL_IN and act.amount > view.bb:
            n += 1
    return n


def _facing_limp_only(view: PlayerView) -> bool:
    if view.current_bet > view.bb:
        return False
    for _, act in view.action_history:
        if act.type in (ActionType.RAISE, ActionType.BET, ActionType.ALL_IN):
            return False
        if act.type == ActionType.CALL:
            return True
    return False


def _put_raise(view: PlayerView, raise_to_bb: float) -> Action:
    target = int(round(raise_to_bb * view.bb))
    target = max(target, view.min_raise_to)
    target = min(target, view.max_raise_to)
    put = max(0, target - view.bets[view.hero])
    put = min(put, view.stacks[view.hero])
    if put <= 0:
        if view.can_check:
            return Action(ActionType.CHECK)
        return Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero]))
    if put >= view.stacks[view.hero]:
        return Action(ActionType.ALL_IN, view.stacks[view.hero])
    if view.to_call <= 0:
        return Action(ActionType.BET, put)
    return Action(ActionType.RAISE, put)


def _all_in(view: PlayerView) -> Action:
    return Action(ActionType.ALL_IN, view.stacks[view.hero])


def _name_to_action(view: PlayerView, name: str, node: int) -> Action:
    if name == "fold":
        if view.can_check:
            return Action(ActionType.CHECK)
        return Action(ActionType.FOLD)
    if name in ("check",):
        return Action(ActionType.CHECK) if view.can_check else Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero]))
    if name in ("call", "limp"):
        if view.to_call <= 0:
            return Action(ActionType.CALL, view.bb) if view.current_bet <= view.bb else Action(ActionType.CHECK)
        return Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero]))
    if name == "jam":
        return _all_in(view)
    if name == "raise":
        size = 2.5
        # HU node 2 corresponds to BTN vs 3-bet, where "raise" is a 4-bet.
        if node == 2:
            size = 35.0
        return _put_raise(view, size)
    if name == "three":
        # HU node 1 corresponds to BB vs open, where "three" is a 3-bet.
        return _put_raise(view, 11.0)
    if name == "iso":
        return _put_raise(view, 4.0)
    return Action(ActionType.FOLD) if view.to_call > 0 else Action(ActionType.CHECK)


def hu_node_for_view(view: PlayerView) -> Optional[int]:
    if not _is_hu_100(view):
        return None
    nr = _n_raises(view)
    pos = view.position
    to_bb = view.to_call / max(view.bb, 1)
    jammed = to_bb >= 40.0 or view.to_call >= view.stacks[view.hero] - 1e-9
    if pos == "BTN":
        if nr == 0:
            return 0
        if _facing_limp_only(view) or (
            nr == 1 and any(a.type == ActionType.CALL for _, a in view.action_history)
        ):
            return 5
        if jammed:
            return 7
        return 2
    if pos == "BB":
        if nr == 0 and _facing_limp_only(view):
            return 4
        if nr == 0:
            return 4 if view.to_call <= 0 else None
        if jammed:
            return 6
        if nr == 1:
            return 1
        return 3
    return None


def try_hu_action(view: PlayerView, rng) -> Optional[Action]:
    """Sample HU 100bb blueprint mix. None → tree fallback."""
    if not hu_loaded():
        return None
    node = hu_node_for_view(view)
    if node is None:
        return None
    hid = hand_id(*view.hole)
    mix = hu_mix(node, hid)
    if mix is None:
        return None
    names = PF_ACTS[node]
    i = _sample(mix, rng)
    name = names[min(i, len(names) - 1)]
    if node == 0 and view.effective_stack / max(view.bb, 1) >= 70:
        # Old HU checkpoint over-jams/limps; v3 live path does not use this.
        if name == "jam":
            name = "raise"
        if name in ("fold", "limp"):
            from strategy.preflop_trees import rfi_range

            if rfi_range("BTN", 2, view.effective_stack / max(view.bb, 1)).contains_hole(*view.hole) >= 0.45:
                name = "raise"
    return _name_to_action(view, name, node)


def _six_spot(view: PlayerView) -> Optional[int]:
    if not _is_6max_100(view):
        return None
    nr = _n_raises(view)
    pos = view.position
    seat = {"UTG": 0, "HJ": 1, "CO": 2, "BTN": 3, "SB": 4, "BB": 5}.get(pos)
    if seat is None:
        return None
    if nr == 0:
        return min(4, seat)
    if nr == 1:
        return 5 if pos == "BB" else 6
    return 7


def try_6max_action(view: PlayerView, rng) -> Optional[Action]:
    if not six_loaded():
        return None
    spot = _six_spot(view)
    if spot is None:
        return None
    hid = hand_id(*view.hole)
    mix = six_mix(spot, hid)
    if mix is None:
        return None
    names = ["fold", "call", "raise", "jam"]
    i = _sample(mix, rng)
    name = names[i]
    if spot <= 4 and view.to_call <= view.bb:
        if name == "raise":
            return _put_raise(view, 2.5)
        if name == "jam":
            return _all_in(view)
        if name == "fold":
            return Action(ActionType.FOLD) if view.to_call > 0 else Action(ActionType.CHECK)
        if name == "call":
            if view.to_call <= 0:
                return Action(ActionType.CHECK)
            return Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero]))
    if name == "fold":
        return Action(ActionType.FOLD) if view.to_call > 0 else Action(ActionType.CHECK)
    if name == "call":
        if view.to_call <= 0:
            return Action(ActionType.CHECK)
        return Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero]))
    if name == "jam":
        return _all_in(view)
    size = 2.5 if _n_raises(view) == 0 else (9.0 if _n_raises(view) == 1 else 22.0)
    return _put_raise(view, size)


def range_from_169(weights: np.ndarray) -> Range:
    rg = Range()
    for i, code in enumerate(HANDS_169):
        w = float(weights[i]) if i < len(weights) else 0.0
        if w <= 0:
            continue
        for c1, c2 in combos_of_169(code):
            rg.w[combo_index(c1, c2)] = w
    return rg


def replay_hu_reach(view: PlayerView) -> Optional[List[Range]]:
    """Per-seat 1326 weights from replaying history through the HU average strategy."""
    if not hu_loaded() or view.num_players != 2:
        return None
    pack = refresh_hu()
    if pack is None:
        return None
    n = 2
    reach = [np.ones(169, dtype=np.float64) for _ in range(n)]
    nr = 0
    btn = 0 if view.positions[0] == "BTN" else (0 if view.button == 0 else 1)
    # seats: identify BTN/BB
    btn_seat = next(i for i in range(n) if view.positions[i] == "BTN")
    bb_seat = 1 - btn_seat
    node_at = 0
    for seat, act in view.action_history:
        if act.type == ActionType.FOLD:
            reach[seat][:] = 0.0
            continue
        is_btn = seat == btn_seat
        if nr == 0 and is_btn:
            node = 0
            names = PF_ACTS[0]
            mix = pack.get(0)
        elif nr == 0 and not is_btn and act.type == ActionType.CALL:
            node = 4
            names = PF_ACTS[4]
            mix = pack.get(4)
        elif nr == 1 and not is_btn:
            node = 1
            names = PF_ACTS[1]
            mix = pack.get(1)
        elif nr >= 2 and is_btn:
            node = 2
            names = PF_ACTS[2]
            mix = pack.get(2)
        elif nr >= 1 and is_btn and act.type in (ActionType.CALL, ActionType.FOLD, ActionType.ALL_IN):
            node = 5 if nr == 1 else 2
            names = PF_ACTS[node]
            mix = pack.get(node)
        else:
            node = 1
            names = PF_ACTS[1]
            mix = pack.get(1)
        if mix is None:
            return None
        if act.type in (ActionType.BET, ActionType.RAISE) or (
            act.type == ActionType.ALL_IN and act.amount > view.bb
        ):
            if nr == 0:
                want = ["raise", "jam", "iso", "three"]
            else:
                want = ["three", "jam", "raise", "iso"]
            nr += 1
        elif act.type == ActionType.CALL:
            want = ["call", "limp", "check"]
        elif act.type == ActionType.CHECK:
            want = ["check"]
        else:
            want = ["fold"]
        idx = [i for i, nm in enumerate(names) if nm in want]
        if not idx:
            idx = list(range(len(names)))
        fac = mix[:, idx].sum(axis=1) if mix.ndim == 2 else np.ones(169)
        reach[seat] *= np.maximum(fac, 1e-8)
        _ = (node_at, btn)
    dead = list(view.hole) + list(view.board)
    out = []
    for i in range(n):
        rg = range_from_169(reach[i])
        rg = rg.mask_dead(dead)
        out.append(rg)
    if out[view.hero].total() <= 0:
        return None
    return out


def replay_6max_reach(view: PlayerView) -> Optional[List[Range]]:
    if not six_loaded() or view.num_players != 6:
        return None
    pack = refresh_six()
    if pack is None:
        return None
    avg = pack["avg"]
    n = 6
    reach = [np.ones(169, dtype=np.float64) for _ in range(n)]
    pos_seat = {view.positions[i]: i for i in range(n)}
    nr = 0
    for seat, act in view.action_history:
        pos = view.positions[seat]
        st = {"UTG": 0, "HJ": 1, "CO": 2, "BTN": 3, "SB": 4, "BB": 5}.get(pos, 3)
        if nr == 0:
            spot = min(4, st)
        elif nr == 1:
            spot = 5 if pos == "BB" else 6
        else:
            spot = 7
        mix = avg[spot]
        names = ["fold", "call", "raise", "jam"]
        if act.type == ActionType.FOLD:
            reach[seat][:] = 0.0
            continue
        if act.type in (ActionType.BET, ActionType.RAISE) or (
            act.type == ActionType.ALL_IN and act.amount > view.bb
        ):
            want = [2, 3]
            nr += 1
        elif act.type in (ActionType.CALL, ActionType.CHECK):
            want = [1]
        else:
            want = [0]
        fac = mix[:, want].sum(axis=1)
        reach[seat] *= np.maximum(fac, 1e-8)
        _ = pos_seat
    dead = list(view.hole) + list(view.board)
    out = [range_from_169(reach[i]).mask_dead(dead) for i in range(n)]
    if all(r.total() <= 0 for r in out):
        return None
    return out


def replay_reach(view: PlayerView) -> Optional[List[Range]]:
    """Blueprint beliefs, or None if the spot is off-tree / missing checkpoint."""
    bb = view.effective_stack / max(view.bb, 1)
    if bb > 160 or view.num_players not in (2, 6):
        return None
    if view.num_players == 2:
        return replay_hu_reach(view)
    return replay_6max_reach(view)


_CFR_SEAT = {"UTG": 0, "HJ": 1, "CO": 2, "BTN": 3, "SB": 4, "BB": 5}


def _v3_facing_bin(view: PlayerView) -> int:
    nr = _n_raises(view)
    if nr <= 0:
        return 0
    to_bb = view.current_bet / max(view.bb, 1.0)
    stack_bb = view.effective_stack / max(view.bb, 1.0)
    if view.to_call >= view.stacks[view.hero] - 1e-9 or to_bb >= min(90.0, stack_bb - 0.5):
        return 4
    if to_bb >= 20.0:
        return 3
    if to_bb >= 6.0:
        return 2
    return 1


def _v3_last_raiser(view: PlayerView) -> int:
    raiser = 0
    for seat, act in view.action_history:
        if act.type not in (ActionType.RAISE, ActionType.BET, ActionType.ALL_IN):
            continue
        if act.type == ActionType.ALL_IN and act.amount <= view.bb:
            continue
        if seat < 0 or seat >= len(view.positions):
            continue
        r = _CFR_SEAT.get(view.positions[seat])
        if r is not None:
            raiser = r
    return raiser


def _v3_pf_spot_6max(view: PlayerView) -> Optional[int]:
    if not _is_6max_100(view):
        return None
    seat = _CFR_SEAT.get(view.position)
    if seat is None:
        return None
    fb = _v3_facing_bin(view)
    if fb <= 0:
        return seat
    r = _v3_last_raiser(view)
    sp = 6 + (fb - 1) * 36 + r * 6 + seat
    return int(min(149, max(6, sp)))


def try_v3_preflop(view: PlayerView, rng) -> Optional[Action]:
    """Sample v3 blueprint mix, blended with None (caller uses trees) by weight."""
    if view.street.name != "PREFLOP":
        return None
    if view.num_players == 6:
        pack = refresh_six_v3()
        if pack is None or pack.get("sum_pf") is None:
            return None
        w = v3_weight(int(pack.get("iters") or 0))
        if w <= 0.0:
            return None
        spot = _v3_pf_spot_6max(view)
        if spot is None or spot >= pack["sum_pf"].shape[0]:
            return None
        hid = hand_id(*view.hole)
        mix = _avg_row(pack["sum_pf"], spot, hid)
        if mix is None or mix.shape[0] < 4:
            return None
        mix = np.array(mix, dtype=np.float64)

        def _in_range_mix(
            t_j: float,
            t_3: float,
            t_c: float,
            jam_thresh: float = 0.45,
            take_thresh: float = 0.5,
        ) -> np.ndarray:
            """Match LibraryGTO _from_trees: independent sequential in_range.

            One-hotting 3bet when w>=0.5 dropped fractional jams (0.2 jam +
            0.6 3bet became 100% 3bet; they jam 20% then 3bet the rest).
            vs open: jam 0.45 / 3bet+call 0.5. vs 3bet: 4bet+call 0.45.
            vs 4bet: jam+call 0.40.
            """
            from strategy.preflop_trees import sequential_action_probs

            p_j, p_3, p_c, p_f = sequential_action_probs(
                t_j, t_3, t_c, jam_thresh=jam_thresh, take_thresh=take_thresh,
            )
            out = np.zeros(4, dtype=np.float64)
            out[3] = p_j
            out[2] = p_3
            out[1] = p_c
            out[0] = p_f
            return out

        seat = {"UTG": 0, "HJ": 1, "CO": 2, "BTN": 3, "SB": 4, "BB": 5}.get(view.position, 0)
        nr = _n_raises(view)
        if nr <= 0 and seat < 5:
            mix[1] = 0.0
        if nr <= 0:
            # Tree-30 BTN may jam AKo in the average mix. 100bb open-jam
            # is not a steal; renormalize onto raise/fold.
            mix[3] = 0.0
            # Live tree-30 BTN averages ~8% (nit) and UTG ~21% (wide vs
            # the 16% combo tree). Snap each combo to the tree so 6-max
            # preflop matches LibraryGTO; search is the edge.
            if seat < 5:
                try:
                    from strategy.preflop_trees import rfi_range

                    stack_bb = float(view.effective_stack) / max(float(view.bb), 1.0)
                    tw = float(rfi_range(view.position, 6, stack_bb).w[combo_index(*view.hole)])
                    mix[1] = 0.0
                    mix[3] = 0.0
                    # in_range thresh 0.5: tree weight >= 0.5 always opens.
                    if tw >= 0.5:
                        mix[2] = 1.0
                        mix[0] = 0.0
                    else:
                        mix[2] = tw
                        mix[0] = max(0.0, 1.0 - tw)
                    # Folded-to-SB: LibraryGTO steals first (in_range SB RFI),
                    # then completes the BTN chart. Always-complete when
                    # cw>0.25 zeroed mixed steals (tw=0.4, cw=0.8 → limp).
                    if (
                        seat == 4
                        and view.to_call > 0
                        and view.current_bet <= view.bb + 1e-9
                        and tw < 0.5
                    ):
                        cw = float(rfi_range("BTN", 6, stack_bb).w[combo_index(*view.hole)])
                        if cw > 0.25:
                            mix[3] = 0.0
                            mix[2] = tw
                            mix[1] = max(0.0, 1.0 - tw)
                            mix[0] = 0.0
                        elif cw > 0:
                            rem = max(0.0, 1.0 - tw)
                            mix[3] = 0.0
                            mix[2] = tw
                            mix[1] = rem * cw
                            mix[0] = rem * (1.0 - cw)
                except Exception:
                    pass
            elif _facing_limp_only(view):
                # BB vs a complete: LibraryGTO isos the CO chart at 4bb.
                # v3 BB-option mix checks AA and lets the complete see a flop.
                try:
                    from strategy.preflop_trees import rfi_range

                    stack_bb = float(view.effective_stack) / max(float(view.bb), 1.0)
                    tw = float(rfi_range("CO", 6, stack_bb).w[combo_index(*view.hole)])
                    mix[3] = 0.0
                    mix[1] = 0.0
                    if tw >= 0.5:
                        mix[2] = 1.0
                        mix[0] = 0.0
                    else:
                        mix[2] = tw
                        mix[0] = max(0.0, 1.0 - tw)
                except Exception:
                    pass
        if nr == 1 and _v3_last_raiser(view) <= 1:
            if view.to_call < view.stacks[view.hero] - 1e-9:
                mix[1] = 0.0
        if nr == 1:
            # Tree-30 zeros flats vs EP and folds 22 vs CO. Combo trees
            # set-mine. Lift this combo to the tree if v3 continues less.
            try:
                from strategy.preflop_trees import apply_blockers, vs_open_action_ranges

                opener = {0: "UTG", 1: "HJ", 2: "CO", 3: "BTN", 4: "SB", 5: "BB"}.get(
                    _v3_last_raiser(view)
                )
                if opener is not None:
                    stack_bb = float(view.effective_stack) / max(float(view.bb), 1.0)
                    jam, three, call = vs_open_action_ranges(
                        view.position, opener, 6, stack_bb
                    )
                    # LibraryGTO 3bets Axs more with the ace. Raw tree weights
                    # under-3bet those combos and leak vs the same charts.
                    three = apply_blockers(three, view.hole, "3bet_bluff")
                    jam = apply_blockers(jam, view.hole, "value_4bet")
                    idx = combo_index(*view.hole)
                    t_j = float(jam.w[idx])
                    t_3 = float(three.w[idx])
                    t_c = float(call.w[idx])
                    mix = _in_range_mix(t_j, t_3, t_c)
            except Exception:
                pass
        if nr == 2:
            # v3 may overfold or polarize vs 3bets. LibraryGTO uses combo
            # 4bet/call trees; 100bb jam over 11bb is illegal in this kernel.
            try:
                from strategy.preflop_trees import apply_blockers, vs_3bet_ranges

                threebettor = {0: "UTG", 1: "HJ", 2: "CO", 3: "BTN", 4: "SB", 5: "BB"}.get(
                    _v3_last_raiser(view)
                )
                if threebettor is not None:
                    stack_bb = float(view.effective_stack) / max(float(view.bb), 1.0)
                    four, call = vs_3bet_ranges(view.position, threebettor, stack_bb)
                    four = apply_blockers(four, view.hole, "value_4bet")
                    idx = combo_index(*view.hole)
                    t_4 = float(four.w[idx])
                    t_c = float(call.w[idx])
                    mix = _in_range_mix(0.0, t_4, t_c, take_thresh=0.45)
            except Exception:
                pass
        if nr >= 3:
            try:
                from strategy.preflop_trees import apply_blockers, vs_4bet_ranges

                stack_bb = float(view.effective_stack) / max(float(view.bb), 1.0)
                jam, call = vs_4bet_ranges(stack_bb)
                jam = apply_blockers(jam, view.hole, "value_4bet")
                idx = combo_index(*view.hole)
                t_j = float(jam.w[idx])
                t_c = float(call.w[idx])
                mix = _in_range_mix(t_j, 0.0, t_c, jam_thresh=0.4, take_thresh=0.4)
            except Exception:
                pass
        if view.to_call >= view.stacks[view.hero] - 1e-9:
            mix[2] = 0.0
            mix[3] = 0.0
        if mix.sum() <= 1e-15:
            return None
        names = ["fold", "call", "raise", "jam"]
        name = names[_sample(mix, rng)]
        if name == "fold":
            return Action(ActionType.FOLD) if view.to_call > 0 else Action(ActionType.CHECK)
        if name == "call":
            if view.to_call <= 0:
                return Action(ActionType.CHECK)
            return Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero]))
        if name == "jam":
            return _all_in(view)
        nr_now = _n_raises(view)
        if nr_now == 0:
            if _facing_limp_only(view):
                size = 4.0
            else:
                from strategy.stacks import open_size_bb

                stack_bb = float(view.effective_stack) / max(float(view.bb), 1.0)
                size = float(open_size_bb(view.position, view.num_players, stack_bb))
        elif nr_now == 1:
            pos = view.position
            size = 9.0 if pos in ("BTN", "CO", "BB") else 11.0
            opener = {0: "UTG", 1: "HJ", 2: "CO", 3: "BTN", 4: "SB", 5: "BB"}.get(
                _v3_last_raiser(view), "BTN"
            )
            if opener in ("UTG", "UTG1", "MP"):
                size += 1.0
            if pos == "SB":
                size = 12.0
        elif nr_now == 2:
            # Combo trees 4bet ~22bb (25 SB). Kernel 35bb is a different SPR.
            size = 22.0 if view.position != "SB" else 25.0
        else:
            size = 35.0
        return _put_raise(view, size)

    if view.num_players == 2:
        pack = refresh_hu_v3()
        if pack is None or pack.get("sum_pf") is None:
            return None
        w = v3_weight(int(pack.get("iters") or 0))
        if w <= 0.0 or rng.random() > w:
            return None
        node = hu_node_for_view(view)
        if node is None:
            return None
        hid = hand_id(*view.hole)
        mix = _avg_row(pack["sum_pf"], node, hid)
        if mix is None:
            return None
        names = PF_ACTS.get(node, ["fold", "call", "raise", "jam"])
        i = _sample(mix, rng)
        name = names[min(i, len(names) - 1)]
        return _name_to_action(view, name, node)
    return None


def _hero_oop(view: PlayerView) -> bool:
    n = view.num_players
    for k in range(1, n + 1):
        i = (view.button + k) % n
        if not view.folded[i] and not view.all_in[i]:
            return i == view.hero
    return True


def _last_bet_seat(view: PlayerView) -> Optional[int]:
    agg = None
    for seat, act in view.action_history:
        if act.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            agg = seat
    return agg


def _v3_post_node(facing: bool, oop: bool, pfr: bool, tree: int, hu: bool) -> int:
    """HU collapse spots: 0 OOP caller FI, 1 IP FI, 2 facing, 3 OOP PFR FI (tree 23+)."""
    if facing:
        return 2
    if oop and pfr and (not hu) and tree >= 23:
        return 3
    if oop:
        return 0
    return 1


def try_v3_postflop(view: PlayerView, rng) -> Optional[Action]:
    """v3 flop/turn/river mix when search did not run.

    2-way uses HU/6-max collapse spots 0–11. 3+ live 6-max uses MW
    first-in / face spots 12–13 (fold-or-call when facing).
    """
    if view.street.name == "PREFLOP":
        return None
    mw = view.players_in_hand > 2
    if mw and view.num_players != 6:
        return None
    pack = refresh_hu_v3() if view.num_players == 2 else refresh_six_v3()
    if pack is None:
        return None
    w = v3_weight(int(pack.get("iters") or 0))
    if w <= 0.0:
        return None
    if view.num_players != 6 and rng.random() > w:
        return None
    from strategy.card_abs import street_bucket_v3
    from strategy.pot_type import classify

    street = {"FLOP": "sum_fl", "TURN": "sum_tn", "RIVER": "sum_rv"}.get(view.street.name)
    if street is None or pack.get(street) is None:
        return None
    tab = pack[street]
    kind = classify(view)
    pt = {"srp": 0, "3bet": 1, "4bet": 2}.get(kind, 0)
    facing = view.to_call > 0
    if mw:
        sp = 13 if facing else 12
    else:
        oop = _hero_oop(view)
        pfr = _last_bet_seat(view) == view.hero
        node = _v3_post_node(facing, oop, pfr, int(pack.get("tree") or 0), view.num_players == 2)
        sp = min(11, pt * 4 + node)
    priv = street_bucket_v3(view.hole, view.board)
    if sp >= tab.shape[0] or priv >= tab.shape[1]:
        return None
    mix = _avg_row(tab, sp, priv)
    if mix is None:
        return None
    mix = np.array(mix, dtype=np.float64)
    if mw and facing:
        mix[2] = 0.0
        mix[3] = 0.0
        s = float(mix.sum())
        if s <= 1e-12:
            return None
        mix /= s
    elif mw and not facing:
        spr = view.effective_stack / max(float(view.pot), 1.0)
        if spr > 2.4:
            mix[3] = 0.0
            s = float(mix.sum())
            if s <= 1e-12:
                return None
            mix /= s
    i = _sample(mix, rng)
    if not facing:
        names = ["check", "bet33", "bet75", "jam"]
        name = names[min(i, 3)]
        if name == "check":
            return Action(ActionType.CHECK) if view.can_check else Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero]))
        if name == "jam":
            return _all_in(view)
        frac = 0.33 if name == "bet33" else 0.75
        put = int(max(view.bb, frac * view.pot))
        put = min(put, view.stacks[view.hero])
        if put <= 0:
            return Action(ActionType.CHECK)
        if view.to_call <= 0:
            return Action(ActionType.BET, put)
        return Action(ActionType.RAISE, put)
    names = ["fold", "call", "raise", "jam"]
    name = names[min(i, 3)]
    if name == "fold":
        return Action(ActionType.FOLD) if view.to_call > 0 else Action(ActionType.CHECK)
    if name == "call":
        return Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero]))
    if name == "jam":
        return _all_in(view)
    return _put_raise(view, max(2.5, view.to_call / max(view.bb, 1) * 2.7))
