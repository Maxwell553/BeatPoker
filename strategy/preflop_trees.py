"""Combo-level preflop trees (solver range-string mixes, not ranking percentiles).

50 / 100 / 200 / deep are separate policies. Optional JSON at
Final_Bot/data/preflop_trees.json can override any spot:

    {"stack": "50", "players": 6, "spot": "rfi", "position": "UTG", "open": "22+,ATs+,..."}
    {"stack": "100", "players": 6, "spot": "vs_open", "position": "BB", "opener": "BTN",
     "3bet": "...", "call": "...", "jam": "..."}
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from poker.cards import rank_of
from poker.equity import combo_index
from strategy.hands import hole_to_169, parse_hand169
from strategy.rake import RakeConfig
from strategy.ranges import Range, ranked_hands, top_percent_range
from strategy.stacks import nearest_table, stack_bucket

JSON_PATH = Path(__file__).resolve().parents[1] / "Final_Bot" / "data" / "preflop_trees.json"

_JSON: Optional[dict] = None


def load_tree_json(path: Optional[Path] = None) -> None:
    """Load optional Pio/GTOWizard-style dump. Missing file is a no-op."""
    global _JSON
    p = path or JSON_PATH
    if not p.exists():
        _JSON = {}
        return
    data = json.loads(p.read_text())
    index: dict = {}
    rows = data if isinstance(data, list) else data.get("spots", [])
    for row in rows:
        key = (
            str(row.get("stack", "100")),
            int(row.get("players", 6)),
            str(row.get("spot", "rfi")),
            str(row.get("position", "")),
            str(row.get("opener", "")),
        )
        index[key] = row
    _JSON = index


def _json_row(bucket: str, n: int, spot: str, pos: str, opener: str = "") -> Optional[dict]:
    global _JSON
    if _JSON is None:
        load_tree_json()
    if not _JSON:
        return None
    n2 = nearest_table(n)
    for key in (
        (bucket, n, spot, pos, opener),
        (bucket, n2, spot, pos, opener),
        (bucket, n2, spot, pos, ""),
    ):
        if key in _JSON:
            return _JSON[key]
    return None


@lru_cache(maxsize=1024)
def _parse(spec: str) -> Range:
    return Range.from_string(spec) if spec else Range()


def parse_range(spec: str) -> Range:
    return _parse(spec).copy()


def _keep_strong(rg: Range, frac: float, stack_bb: float) -> Range:
    if frac >= 0.99 or rg.total() <= 0:
        return rg
    ranking = ranked_hands(stack_bb)
    # Re-weight: keep strongest frac of current mass using 169 ranking.
    from strategy.hands import combos_of_169

    target = rg.total() * frac
    out = Range()
    acc = 0.0
    for code in ranking:
        for a, b in combos_of_169(code):
            i = combo_index(a, b)
            w = rg.w[i]
            if w <= 0:
                continue
            if acc >= target:
                return out
            out.w[i] = w
            acc += w
    return out


# --- RFI strings (combo mixes) ------------------------------------------------

def _rfi_spec(bucket: str, n: int, pos: str) -> str:
    n = nearest_table(n)
    if n == 2:
        return _rfi_hu(bucket, pos)
    if n == 9:
        return _rfi_9(bucket, pos)
    return _rfi_6(bucket, pos)


def _rfi_hu(bucket: str, pos: str) -> str:
    if pos == "BB":
        return ""
    # HU BTN/SB open: ~78–84%. 86s in; 72o out.
    if bucket == "50":
        return (
            "22+,A2s+,A7o+,K4s+,K9o+,Q6s+,QTo+,J7s+,JTo,T7s+,T9o,97s+,87s,86s,76s,75s,"
            "65s,64s,54s,53s,43s,A2o-A6o:0.45,K5o-K8o:0.4,Q8o-Q9o:0.45,J8o-J9o:0.4,"
            "T8o:0.35,98o:0.3,32s:0.35,42s:0.4,52s:0.4,62s:0.35,72s:0.25"
        )
    extra = ""
    if bucket in ("200", "deep"):
        extra = ",32s,42s,52s,62s,72s:0.5,J5s,T5s,96s,85s,74s,63s"
    return (
        "22+,A2s+,A2o+,K2s+,K8o+,Q5s+,Q9o+,J6s+,J8o+,T6s+,T8o+,96s+,97o+,86s+,87o:0.55,"
        "76s,75s,65s,64s,54s,53s,43s,K2o-K7o:0.45,Q6o-Q8o:0.4,J7o:0.45,T7o:0.4,98o:0.45"
        + extra
    )


def _rfi_6(bucket: str, pos: str) -> str:
    if pos == "BB":
        return ""
    if bucket == "50":
        return {
            "UTG": "22+,A9s+,A5s:0.45,A4s:0.25,AJo+,KJs+,KQo,QJs,JTs,T9s,ATo:0.55,KJo:0.35,QTs:0.55,98s:0.35",
            "UTG1": "22+,A9s+,A5s:0.5,AJo+,KTs+,KQo,QTs+,JTs,T9s,ATo:0.65,KJo:0.5,98s:0.5,87s:0.3",
            "MP": "22+,A8s+,A5s:0.55,ATo+,KTs+,KQo,QTs+,JTs,T9s,98s,KJo,A9o:0.4,87s:0.45",
            "HJ": "22+,A4s+,ATo+,KTs+,KJo+,QTs+,J9s+,T8s+,98s,87s,76s:0.45,A9o,KTo:0.45,QJo:0.5",
            "CO": "22+,A2s+,A9o+,K9s+,KTo+,Q9s+,QTo+,J9s+,JTo,T8s+,97s+,87s,76s,65s:0.5,A8o:0.4,K9o:0.45",
            "BTN": (
                "22+,A2s+,A8o+,K4s+,K9o+,Q8s+,QTo+,J8s+,JTo,T8s+,T9o,97s+,98o:0.45,"
                "87s,86s,76s,65s,54s:0.55,A5o-A7o:0.4,K8o:0.4,Q9o:0.5,J9o:0.45"
            ),
            "SB": "22+,A2s+,A9o+,K9s+,KJo+,Q9s+,QJo,J9s+,JTo,T8s+,97s+,87s,76s:0.5,A8o:0.35,KTo:0.5",
        }.get(pos, "22+,A9s+,AJo+,KJs+,KQo,QJs,JTs")
    if bucket in ("200", "deep"):
        deep = {
            "UTG": "22+,A2s+,ATo+,K9s+,KQo,QTs+,QJo,JTs,T9s,98s,87s,76s,65s:0.6,54s:0.4,A9o:0.4,KJo:0.7",
            "HJ": "22+,A2s+,A9o+,K9s+,KTo+,Q9s+,QJo,J9s+,JTo,T8s+,98s,87s,76s,65s,54s,43s:0.4,A8o:0.35",
            "CO": "22+,A2s+,A5o+,K6s+,KTo+,Q8s+,QTo+,J8s+,JTo,T7s+,97s+,86s+,76s,65s,54s,43s,32s:0.35,A4o:0.3",
            "BTN": (
                "22+,A2s+,A2o+,K2s+,K8o+,Q4s+,Q9o+,J6s+,J8o+,T6s+,T8o+,96s+,86s+,76s,75s,"
                "65s,64s,54s,53s,43s,32s,K5o-K7o:0.5,Q8o:0.55,J7o:0.4"
            ),
            "SB": "22+,A2s+,A8o+,K8s+,KTo+,Q9s+,QTo+,J8s+,JTo,T8s+,97s+,87s,76s,65s,54s,A5o-A7o:0.4",
        }
        if pos in deep:
            return deep[pos]
        if pos in ("UTG1", "MP"):
            return deep["HJ"]
    # 100bb 6-max
    return {
        "UTG": "22+,A9s+,A5s:0.85,A4s:0.7,A3s:0.5,A2s:0.4,ATo+,K9s+,KQo,QTs+,QJo,JTs,T9s,98s,87s,76s:0.45,A9o:0.25,KJo:0.55",
        "UTG1": "22+,A2s+,ATo+,K9s+,KTo+,QTs+,QJo,JTs,T9s,98s,87s,76s,65s:0.4,A9o:0.4,KJo:0.7",
        "MP": "22+,A2s+,ATo+,K9s+,KTo+,Q9s+,QJo,J9s+,JTo,T8s+,98s,87s,76s,65s,A9o,KJo",
        "HJ": "22+,A2s+,A9o+,K9s+,KTo+,Q9s+,QTo+,J9s+,JTo,T8s+,97s+,87s,76s,65s,54s:0.5,A8o:0.3,K9o:0.35",
        "CO": (
            "22+,A2s+,A8o+,K7s+,KTo+,Q8s+,QTo+,J8s+,JTo,T8s+,T9o:0.45,97s+,87s,86s,"
            "76s,75s,65s,54s,A5o-A7o:0.35,K9o,Q9o:0.5,J9o:0.4"
        ),
        "BTN": (
            "22+,A2s+,A4o+,K2s+,K9o+,Q6s+,QTo+,J7s+,J9o+,T7s+,T8o+,97s+,98o,86s+,87o:0.4,"
            "76s,75s,65s,64s,54s,53s,43s,A2o-A3o:0.35,K7o-K8o:0.45,Q9o,J8o:0.45"
        ),
        "SB": "22+,A2s+,A9o+,K9s+,KJo+,Q9s+,QJo,J9s+,JTo,T8s+,97s+,87s,76s,65s:0.55,54s:0.4,A8o:0.4,KTo,QTo:0.5",
    }.get(pos, "22+,A2s+,ATo+,KTs+,KQo,QTs+,JTs")


def _rfi_9(bucket: str, pos: str) -> str:
    if pos == "BB":
        return ""
    if bucket == "50":
        ep = "22+,ATs+,A5s:0.35,AJo+,KJs+,KQo,QJs,JTs,ATo:0.4,KJo:0.25,T9s:0.35"
        return {
            "UTG": ep,
            "UTG1": "22+,A9s+,A5s:0.4,AJo+,KTs+,KQo,QJs,JTs,T9s,ATo:0.5,KJo:0.4",
            "UTG2": "22+,A9s+,A5s:0.5,ATo+,KTs+,KQo,QTs+,JTs,T9s,KJo:0.55,98s:0.4",
            "MP": "22+,A8s+,A5s:0.55,ATo+,KTs+,KJo+,QTs+,JTs,T9s,98s,87s:0.4,A9o:0.35",
            "HJ": "22+,A4s+,ATo+,KTs+,KJo+,QTs+,J9s+,T8s+,98s,87s,76s:0.4,A9o,KTo:0.4",
            "CO": "22+,A2s+,A9o+,K9s+,KTo+,Q9s+,QTo+,J9s+,JTo,T8s+,97s+,87s,76s,A8o:0.35",
            "BTN": "22+,A2s+,A8o+,K5s+,K9o+,Q8s+,QTo+,J8s+,JTo,T8s+,97s+,87s,76s,65s,54s:0.45,A5o-A7o:0.35",
            "SB": "22+,A2s+,A9o+,K9s+,KJo+,Q9s+,QJo,J9s+,JTo,T8s+,97s+,87s,76s:0.45,KTo:0.5",
        }.get(pos, ep)
    # 100bb+ 9-max: UTG tighter than 6-max
    return {
        "UTG": "22+,ATs+,A5s:0.5,A4s:0.3,AJo+,KJs+,KQo,QJs,JTs,T9s,ATo:0.45,KJo:0.3,98s:0.35,87s:0.2",
        "UTG1": "22+,A9s+,A5s:0.55,ATo+,KTs+,KQo,QTs+,JTs,T9s,98s,A9o:0.25,KJo:0.5,87s:0.35",
        "UTG2": "22+,A8s+,A5s:0.6,ATo+,KTs+,KJo+,QTs+,J9s+,T9s,98s,87s,76s:0.3,A9o:0.4",
        "MP": "22+,A4s+,ATo+,K9s+,KTo+,QTs+,QJo,J9s+,T8s+,98s,87s,76s,A9o,KJo",
        "HJ": "22+,A2s+,A9o+,K9s+,KTo+,Q9s+,QJo,J9s+,JTo,T8s+,97s+,87s,76s,65s:0.5,A8o:0.3",
        "CO": "22+,A2s+,A8o+,K8s+,KTo+,Q8s+,QTo+,J8s+,JTo,T8s+,97s+,87s,76s,65s,54s:0.45,K9o:0.4",
        "BTN": (
            "22+,A2s+,A5o+,K4s+,K9o+,Q7s+,QTo+,J7s+,J9o+,T7s+,T8o+,97s+,86s+,76s,65s,"
            "54s,43s:0.4,A4o:0.3,K8o:0.4,Q9o:0.55"
        ),
        "SB": "22+,A2s+,A9o+,K9s+,KJo+,Q9s+,QJo,J9s+,JTo,T8s+,97s+,87s,76s,65s:0.5,A8o:0.35,KTo",
    }.get(pos, "22+,ATs+,AJo+,KJs+,KQo,QJs,JTs")


def rfi_range(position: str, n_players: int, stack_bb: float, rake: RakeConfig | None = None) -> Range:
    b = stack_bucket(stack_bb)
    n = nearest_table(n_players)
    row = _json_row(b, n_players, "rfi", position)
    spec = (row or {}).get("open") or _rfi_spec(b, n, position)
    rg = parse_range(spec)
    if rake is not None and rake.active:
        rg = _keep_strong(rg, rake.open_mult(position), stack_bb)
    return rg


def rfi_percent(position: str, n_players: int, stack_bb: float, rake: RakeConfig | None = None) -> float:
    return rfi_range(position, n_players, stack_bb, rake=rake).frac_of_full()


# --- vs open / 3bet / 4bet ----------------------------------------------------

def _vs_open_hu(bucket: str, our: str) -> Tuple[str, str, str]:
    """HU BB vs BTN: defend ~88–92%. BTN vs BB 3bet is handled in vs_3bet."""
    if our != "BB":
        # BTN facing a limp-iso or leftover; treat as value-heavy.
        jam = "QQ+,AKs,AKo:0.6" if bucket == "50" else ""
        three = "JJ+,AQs+,AKo,A5s:0.4,AQo:0.4"
        call = "22-TT,ATs+,KQs,AJo+"
        return jam, three, call
    if bucket == "50":
        jam = "QQ+,AKs,AKo:0.75,JJ:0.45,AQs:0.25"
        three = "JJ-99,AQs,AJs:0.55,A5s-A3s:0.5,KQs,AQo:0.45,TT:0.3,76s:0.2"
        call = (
            "22-TT,A2s-ATs,A2o-ATo,K2s+,K8o+,Q5s+,Q9o+,J7s+,J9o+,T7s+,T9o,"
            "97s+,87s,86s,76s,75s,65s,64s,54s,53s,43s,32s:0.45,"
            "K2o-K7o:0.4,Q6o-Q8o:0.45,J8o:0.4,T8o:0.35,98o:0.4"
        )
        return jam, three, call
    jam = ""
    three = (
        "QQ+,AKs,AKo:0.55,AQs,AJs:0.5,A5s-A2s:0.55,KQs,KJs:0.45,AQo:0.5,"
        "99-JJ:0.25,76s:0.22,54s:0.22,T9s:0.18,ATo:0.2,KQo:0.25,AJo:0.15"
    )
    call = (
        "22-JJ,A2s-ATs,A2o-ATo,K2s+,K7o+,Q4s+,Q9o+,J6s+,J8o+,T6s+,T8o+,"
        "96s+,97o,86s+,87o:0.55,76s,75s,65s,64s,54s,53s,43s,32s:0.5,"
        "K2o-K6o:0.45,Q6o-Q8o:0.5,J7o:0.45,T7o:0.4,98o:0.45,85s:0.4,74s:0.35"
    )
    return jam, three, call


def _vs_open_specs(bucket: str, our: str, opener: str, n: int) -> Tuple[str, str, str]:
    """Return (jam, threebet, call) range strings."""
    n = nearest_table(n)
    if n == 2:
        return _vs_open_hu(bucket, our)

    steal = opener in ("BTN", "CO", "SB")
    bb = our == "BB"
    ip = our in ("BTN", "CO", "BB") and opener not in ("BB",)

    if bucket == "50":
        jam = "QQ+,AKs,AKo:0.85,JJ:0.6,TT:0.35,AQs:0.35"
        if our in ("UTG", "UTG1", "UTG2", "MP") or opener in ("UTG", "UTG1", "UTG2"):
            jam = "QQ+,AKs,AKo,JJ:0.75,AQs:0.2"
        three = "JJ-99,AQs-AJs,AQo,A5s-A3s:0.55,KQs,KJs:0.5,ATo:0.2,KQo:0.35,AKo:0.2"
        if not bb and our != "SB":
            # 3bet-or-fold from most 50bb seats; tiny flats IP vs steal only.
            call = "TT-88:0.35,AQs:0.3,KQs:0.25" if ip and steal else ""
        elif our == "SB":
            call = "TT-77:0.3,AQs,KQs:0.4,JTs:0.25"
        else:
            # BB still flats a real range at 50bb
            call = (
                "22-TT,ATs-A2s,K9s+,Q9s+,J9s+,T8s+,97s+,87s,76s,65s,54s:0.55,"
                "AJo-ATo,KQo,KJo:0.45,QJo:0.35,JTo:0.3,A9o:0.25"
            )
            if opener in ("UTG", "UTG1", "UTG2", "MP"):
                call = "22-JJ,ATs+,A5s,KJs+,QJs,JTs,T9s,AJo,KQo,AQo:0.5"
        if bb and steal:
            three = "QQ+,AKs,AKo:0.5,AQs,AJs:0.6,A5s-A2s:0.5,KQs,KJs:0.45,AQo:0.45,99-JJ:0.3,76s:0.2,54s:0.2"
        return jam, three, call

    # 100bb+ flats exist IP / BB
    jam = ""
    if bucket == "100":
        three = {
            ("BB", "BTN"): "QQ+,AKs,AKo:0.65,AQs,AJs:0.55,A5s-A2s:0.5,KQs,KJs:0.4,AQo:0.5,99-JJ:0.25,76s:0.18,54s:0.18,ATo:0.15,KQo:0.25",
            ("BB", "CO"): "QQ+,AKs,AKo:0.7,AQs,AJs:0.5,A5s-A4s:0.4,KQs,AQo:0.45,TT+:0.2",
            ("BB", "HJ"): "QQ+,AKs,AKo,AQs:0.6,AJs:0.35,A5s:0.35,KQs:0.35,AQo:0.4",
            ("BB", "SB"): "QQ+,AJs+,AKo,AQo:0.6,A5s-A2s:0.55,KQs+,KJs:0.5,99+:0.3,76s:0.22,T9s:0.2",
            ("BTN", "CO"): "QQ+,AKs,AKo:0.7,AQs,AJs:0.5,A5s-A4s:0.45,KQs,AQo:0.4,JJ:0.3",
            ("BTN", "HJ"): "QQ+,AKs,AKo,AQs:0.55,A5s:0.4,KQs:0.4,AQo:0.4",
            ("BTN", "UTG"): "QQ+,AKs,AKo,AQs:0.45,AQo:0.3,JJ:0.25",
            ("CO", "HJ"): "QQ+,AKs,AKo,AQs:0.5,A5s:0.35,KQs:0.35,AQo:0.35",
            ("CO", "UTG"): "QQ+,AKs,AKo,AQs:0.4,JJ:0.2",
            ("SB", "BTN"): "QQ+,AJs+,AKo,AQo:0.55,A5s-A3s:0.5,KQs,KJs:0.45,TT+:0.25,76s:0.2,ATo:0.2",
            ("SB", "CO"): "QQ+,AKs,AKo,AQs,AJs:0.45,A5s:0.4,KQs:0.4,AQo:0.4",
            ("SB", "UTG"): "QQ+,AKs,AKo,AQs:0.4,AQo:0.3",
        }.get((our, opener), "QQ+,AKs,AKo:0.7,AQs:0.5,A5s:0.4,KQs:0.35,AQo:0.35")
        call = ""
        if bb:
            call = (
                "22-JJ,ATs-A2s,K7s+,Q8s+,J8s+,T8s+,97s+,87s,76s,65s,54s,43s:0.45,"
                "ATo+,KTo+,QTo+,JTo,T9o,98o:0.4,K9o:0.3,A9o:0.5,Q9o:0.25"
            )
            if opener in ("UTG", "UTG1", "UTG2"):
                call = "22-JJ,A9s-A2s,KTs+,QTs+,JTs,T9s,98s,87s:0.5,AJo,KQo,KJo:0.4"
        elif ip:
            call = "TT-22,AQs-ATs,KQs-KTs,QJs,JTs,T9s,98s,87s,76s:0.5,65s:0.4,AJo:0.45,KQo:0.5"
            if opener in ("UTG", "UTG1"):
                call = "TT-77,AQs,AJs,KQs,QJs,JTs,AQo:0.4"
        elif our == "SB":
            call = "TT-77,AQs-ATs,KQs,QJs,JTs,T9s,87s:0.4,AJo:0.3"
        return jam, three, call

    # 200 / deep: fewer 3bets, more flats, almost no jam
    jam = ""
    three = "QQ+,AKs,AKo:0.55,AQs:0.4,A5s:0.35,KQs:0.3,AQo:0.3,JJ:0.2"
    if opener in ("UTG", "UTG1", "UTG2"):
        three = "QQ+,AKs,AKo,AQs:0.35,AQo:0.2"
    call = ""
    if bb:
        call = (
            "22-JJ,A2s-ATs,K6s+,Q8s+,J8s+,T8s+,97s+,86s+,76s,65s,54s,43s,32s:0.4,"
            "ATo+,KTo+,QTo+,JTo,T9o,98o,A9o,K9o:0.45"
        )
    elif ip:
        call = "22-QQ:0.7,AQs-A8s,KQs-K9s,QJs-Q9s,JTs,T9s,98s,87s,76s,65s,54s,AJo+,KQo,KJo:0.5"
    elif our == "SB":
        call = "22-JJ,AQs-ATs,KQs,QJs,JTs,T9s,98s,87s,76s:0.5,AJo:0.4"
    if bucket == "deep":
        three = "KK+,AKs,AKo:0.4,AQs:0.25,A5s:0.2"
    return jam, three, call


def vs_open_action_ranges(
    our_pos: str, opener: str, n_players: int, stack_bb: float, rake: RakeConfig | None = None
) -> Tuple[Range, Range, Range]:
    b = stack_bucket(stack_bb)
    row = _json_row(b, n_players, "vs_open", our_pos, opener)
    if row:
        jam_s, tb_s, cl_s = row.get("jam", ""), row.get("3bet", "") or row.get("threebet", ""), row.get("call", "")
    else:
        jam_s, tb_s, cl_s = _vs_open_specs(b, our_pos, opener, n_players)
    jam, three, call = parse_range(jam_s), parse_range(tb_s), parse_range(cl_s)
    if rake is not None and rake.active:
        call = _keep_strong(call, rake.defend_mult(), stack_bb)
        three.w *= 0.95
    three.w = three.w * (jam.w < 0.5)
    call.w = call.w * (jam.w < 0.5) * (three.w < 0.5)
    return jam, three, call


def vs_open_ranges(
    our_pos: str, opener: str, n_players: int, stack_bb: float, rake: RakeConfig | None = None
) -> Tuple[Range, Range]:
    import numpy as np

    jam, three, call = vs_open_action_ranges(our_pos, opener, n_players, stack_bb, rake=rake)
    agg = jam.copy()
    agg.w = np.maximum(agg.w, three.w)
    return agg, call


def _vs_3bet_specs(bucket: str, our: str, threebettor: str) -> Tuple[str, str]:
    """Position-specific 6-max continue vs a 3bet: (4bet, call). MCCFR stays leaves."""
    ip = our in ("BTN", "CO", "HJ") and threebettor in ("BB", "SB")
    vs_ep = threebettor in ("UTG", "UTG1", "UTG2", "MP")
    vs_steal = threebettor in ("BTN", "CO", "SB")
    if bucket == "50":
        four = "QQ+,AKs,AKo,JJ:0.75,AQs:0.4,A5s:0.35"
        call = "JJ-TT:0.35,AQs:0.3,KQs:0.2"
        if vs_ep:
            four = "QQ+,AKs,AKo,JJ:0.55"
            call = "JJ:0.25"
        if ip and vs_steal:
            call = "TT-99:0.45,AQs,KQs:0.35,AJs:0.25"
        return four, call
    if bucket == "deep":
        four = "KK+,AKs:0.45,AKo:0.3"
        call = "QQ-JJ,AQs,AKo:0.65,KQs,AJs:0.45,TT:0.55,87s:0.3,76s:0.3"
        if vs_ep:
            four = "AA,KK:0.5,AKs:0.25"
            call = "QQ-JJ,AQs:0.4"
        return four, call
    if bucket == "200":
        four = "QQ+,AKs,AKo:0.55,AQs:0.22,A5s:0.22"
        call = "JJ-99,AQs,AJs,KQs,AQo:0.5,TT:0.7,87s:0.4,76s:0.35,65s:0.25"
        if our == "BTN" and threebettor == "BB":
            four = "QQ+,AKs,AKo:0.6,AQs:0.3,A5s:0.35,A4s:0.25"
            call = "JJ-88,AQs-ATs,KQs,QJs,JTs,T9s,98s,87s,76s,AQo:0.55"
        if vs_ep:
            four = "QQ+,AKs,AKo:0.4"
            call = "JJ-TT,AQs:0.4,KQs:0.25"
        return four, call
    # 100bb 6-max — the solver-grinder tree
    table = {
        ("BTN", "BB"): (
            "QQ+,AKs,AKo:0.72,AQs:0.45,A5s-A4s:0.45,KQs:0.22,JJ:0.4,AJs:0.15",
            "JJ-77,AQs-ATs,KQs-KJs,QJs,JTs,T9s,98s,87s,76s:0.4,65s:0.3,AQo:0.55,AJo:0.2",
        ),
        ("BTN", "SB"): (
            "QQ+,AKs,AKo:0.7,AQs:0.4,A5s:0.4,JJ:0.4",
            "JJ-88,AQs-ATs,KQs,QJs,JTs,87s:0.35,AQo:0.5",
        ),
        ("CO", "BB"): (
            "QQ+,AKs,AKo:0.7,AQs:0.35,A5s:0.35,JJ:0.35",
            "JJ-88,AQs-ATs,KQs,QJs,JTs,T9s,87s:0.3,AQo:0.45",
        ),
        ("CO", "BTN"): (
            "QQ+,AKs,AKo:0.65,AQs:0.3,JJ:0.3",
            "JJ-99,AQs,KQs,QJs,JTs,AQo:0.4",
        ),
        ("HJ", "BB"): (
            "QQ+,AKs,AKo:0.68,AQs:0.3,A5s:0.25,JJ:0.3",
            "JJ-99,AQs,KQs,QJs,JTs,AQo:0.4",
        ),
        ("HJ", "CO"): (
            "QQ+,AKs,AKo:0.6,AQs:0.25,JJ:0.25",
            "JJ-TT,AQs,KQs:0.35,AQo:0.3",
        ),
        ("UTG", "BB"): (
            "QQ+,AKs,AKo:0.55,AQs:0.2,JJ:0.25",
            "JJ-TT,AQs:0.35,KQs:0.2",
        ),
        ("UTG", "BTN"): (
            "QQ+,AKs,AKo:0.5,JJ:0.2",
            "JJ,AQs:0.25",
        ),
        ("BB", "BTN"): (
            "QQ+,AKs,AKo:0.6,AQs:0.25,A5s:0.3,JJ:0.3",
            "JJ-88,AQs-ATs,KQs,QJs,JTs,AQo:0.4",
        ),
        ("SB", "BTN"): (
            "QQ+,AKs,AKo:0.62,AQs:0.28,A5s:0.3,JJ:0.3",
            "JJ-88,AQs,KQs,QJs,AQo:0.35",
        ),
        ("SB", "CO"): (
            "QQ+,AKs,AKo:0.58,AQs:0.25,JJ:0.25",
            "JJ-99,AQs,KQs:0.3",
        ),
    }
    if (our, threebettor) in table:
        return table[(our, threebettor)]
    four = "QQ+,AKs,AKo:0.75,AQs:0.4,A5s-A4s:0.4,KQs:0.2,JJ:0.45"
    call = "JJ-88,AQs-ATs,KQs,QJs,JTs,AQo:0.55,AJo:0.25,76s:0.3,65s:0.25"
    if ip:
        call = "JJ-77,AQs-ATs,KQs,QJs,JTs,T9s,98s,87s,76s:0.35,AQo:0.5"
    if vs_ep:
        four = "QQ+,AKs,AKo:0.55,AQs:0.25,JJ:0.3"
        call = "JJ-TT,AQs,KQs:0.3,AQo:0.3"
    if vs_steal and our in ("BB", "SB"):
        four = "QQ+,AKs,AKo:0.65,AQs:0.4,A5s-A3s:0.4,KQs:0.25,JJ:0.35"
        call = "JJ-88,AQs-A9s,KQs-KTs,QJs,JTs,T9s,87s:0.35,AQo:0.45"
    return four, call


def vs_3bet_ranges(
    our_pos: str, threebettor: str, stack_bb: float, rake: RakeConfig | None = None
) -> Tuple[Range, Range]:
    b = stack_bucket(stack_bb)
    row = _json_row(b, 6, "vs_3bet", our_pos, threebettor)
    if row:
        four = parse_range(row.get("4bet", "") or row.get("jam", ""))
        call = parse_range(row.get("call", ""))
    else:
        four_s, call_s = _vs_3bet_specs(b, our_pos, threebettor)
        four, call = parse_range(four_s), parse_range(call_s)
    if rake is not None and rake.active:
        call = _keep_strong(call, rake.defend_mult(), stack_bb)
    call.w = call.w * (four.w < 0.5)
    return four, call


def vs_4bet_ranges(stack_bb: float) -> Tuple[Range, Range]:
    b = stack_bucket(stack_bb)
    row = _json_row(b, 6, "vs_4bet", "", "")
    if row:
        jam = parse_range(row.get("jam", ""))
        call = parse_range(row.get("call", ""))
    elif b == "50":
        jam = parse_range("QQ+,AKs,AKo")
        call = parse_range("JJ:0.25")
    elif b == "deep":
        jam = parse_range("AA,KK:0.4")
        call = parse_range("KK-QQ,AKs,AKo,JJ:0.6")
    elif b == "200":
        jam = parse_range("KK+,AKs:0.5")
        call = parse_range("QQ,AKs,AKo,JJ:0.5")
    else:
        jam = parse_range("KK+,AKs,AKo:0.6,QQ:0.45")
        call = parse_range("QQ,JJ:0.5,AKs:0.4,AKo:0.4")
    call.w = call.w * (jam.w < 0.5)
    return jam, call


def apply_blockers(rg: Range, hole: Tuple[int, int], kind: str) -> Range:
    """Bump this combo's mix: Ace/King blockers for 3bet-bluffs, pairs for value 4bets."""
    out = rg.copy()
    i = combo_index(*hole)
    ranks = (rank_of(hole[0]), rank_of(hole[1]))
    code = hole_to_169(*hole)
    hi, lo, suited, is_pair = parse_hand169(code)
    cur = float(out.w[i])
    if kind == "3bet_bluff":
        # Axs / Kxs / small SC: more 3bet when holding the Ace/King
        bluffy = (not is_pair) and ((hi == 12 and lo <= 8) or (hi == 11 and lo <= 9) or (suited and hi - lo <= 2 and hi <= 9))
        if bluffy:
            bump = 0.0
            if 12 in ranks:
                bump += 0.12
            if 11 in ranks:
                bump += 0.06
            out.w[i] = min(1.0, cur * (1.0 + bump * 2) + bump)
    elif kind == "value_4bet":
        if is_pair and hi >= 10:
            out.w[i] = min(1.0, max(cur, 0.85) * 1.1)
        if code in ("AKs", "AKo"):
            out.w[i] = min(1.0, max(cur, 0.7))
    return out


def apply_blockers_range(rg: Range, kind: str) -> Range:
    """Apply per-combo blocker bumps as if each combo were the hole.

    LibraryGTO 3bets Axs more than the raw tree. Reconstructing the raw
    tree made 3bet ranges too value-heavy and search overfolded.
    """
    from poker.equity import combo_from_index

    out = rg.copy()
    for i, w in enumerate(rg.w):
        if float(w) <= 1e-12:
            continue
        out.w[i] = apply_blockers(rg, combo_from_index(i), kind).w[i]
    return out


def sequential_action_probs(
    t_j: float,
    t_3: float,
    t_c: float,
    jam_thresh: float = 0.45,
    take_thresh: float = 0.5,
) -> Tuple[float, float, float, float]:
    """P(jam), P(3bet/4bet), P(call), P(fold) matching LibraryGTO in_range.

    in_range is sequential: jam at jam_thresh, then raise at take_thresh,
    then call at take_thresh. A combo that always 3bets (w>=0.5) never calls.
    """
    p_j = 1.0 if t_j >= jam_thresh else max(0.0, min(1.0, float(t_j)))
    rem = max(0.0, 1.0 - p_j)
    p_3 = rem if t_3 >= take_thresh else rem * max(0.0, min(1.0, float(t_3)))
    rem = max(0.0, 1.0 - p_j - p_3)
    p_c = rem if t_c >= take_thresh else rem * max(0.0, min(1.0, float(t_c)))
    p_f = max(0.0, 1.0 - p_j - p_3 - p_c)
    return p_j, p_3, p_c, p_f


def sequential_mix_weights(
    jam_w,
    three_w,
    call_w,
    jam_thresh: float = 0.45,
    take_thresh: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized sequential_action_probs over combo-weight arrays."""
    t_j = np.clip(np.asarray(jam_w, dtype=np.float64), 0.0, 1.0)
    t_3 = np.clip(np.asarray(three_w, dtype=np.float64), 0.0, 1.0)
    t_c = np.clip(np.asarray(call_w, dtype=np.float64), 0.0, 1.0)
    p_j = np.where(t_j >= jam_thresh, 1.0, t_j)
    rem = np.maximum(0.0, 1.0 - p_j)
    p_3 = np.where(t_3 >= take_thresh, rem, rem * t_3)
    rem = np.maximum(0.0, 1.0 - p_j - p_3)
    p_c = np.where(t_c >= take_thresh, rem, rem * t_c)
    p_f = np.maximum(0.0, 1.0 - p_j - p_3 - p_c)
    return p_j, p_3, p_c, p_f


def in_range(rg: Range, hole: Tuple[int, int], rng, thresh: float = 0.5) -> bool:
    w = rg.contains_hole(*hole)
    if w >= thresh:
        return True
    if w <= 0:
        return False
    return rng.random() < w


def export_tree_json(path: Optional[Path] = None) -> Path:
    """Write solver-style combo dumps so `preflop_trees.json` is the loaded source."""
    from poker.game import positions_for_table

    dest = path or JSON_PATH
    spots: list = []
    stacks = [("50", 50.0), ("100", 100.0), ("200", 200.0), ("deep", 500.0)]
    fourbet = {
        "50": ("QQ+,AKs,AKo", "JJ:0.25"),
        "100": ("KK+,AKs,AKo:0.6,QQ:0.45", "QQ,JJ:0.5,AKs:0.4,AKo:0.4"),
        "200": ("KK+,AKs:0.5", "QQ,AKs,AKo,JJ:0.5"),
        "deep": ("AA,KK:0.4", "KK-QQ,AKs,AKo,JJ:0.6"),
    }
    for bucket, _stack in stacks:
        for n in (2, 6, 9):
            names = positions_for_table(n)
            for pos in names:
                spec = _rfi_spec(bucket, n, pos)
                if spec:
                    spots.append({"stack": bucket, "players": n, "spot": "rfi", "position": pos, "open": spec})
            for our in names:
                for opener in names:
                    if our == opener:
                        continue
                    jam, three, call = _vs_open_specs(bucket, our, opener, n)
                    spots.append({
                        "stack": bucket, "players": n, "spot": "vs_open",
                        "position": our, "opener": opener, "jam": jam, "3bet": three, "call": call,
                    })
            for our in names:
                for threebettor in names:
                    if our == threebettor:
                        continue
                    four, call4 = _vs_3bet_specs(bucket, our, threebettor)
                    spots.append({
                        "stack": bucket, "players": n, "spot": "vs_3bet",
                        "position": our, "opener": threebettor, "4bet": four, "call": call4,
                    })
            jam_s, call_s = fourbet[bucket]
            spots.append({
                "stack": bucket, "players": n, "spot": "vs_4bet",
                "position": "", "opener": "", "jam": jam_s, "call": call_s,
            })
    payload = {
        "source": "solver-style combo mixes distilled for Final_Bot (not a licensed Pio/GTOWizard dump)",
        "spots": spots,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2))
    global _JSON
    _JSON = None
    load_tree_json(dest)
    return dest


if __name__ == "__main__":
    p = export_tree_json()
    print("Wrote", p)
