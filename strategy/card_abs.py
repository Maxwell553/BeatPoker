"""Card abstraction: EHS/potential buckets and infoset indexing.

Preflop is lossless (169). Flop/turn/river map (hand, board) into a small
integer bucket so MCCFR and live search share the same public/private index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from poker.evaluator import CAT_BASE, category, evaluate
from strategy.hands import HAND_INDEX, hole_to_169
from strategy.strength import board_combo_score

DATA = Path(__file__).resolve().parents[1] / "Final_Bot" / "data"
CLUSTER_PATH = DATA / "card_clusters.npz"

# v2 (legacy checkpoint / library 50-bin tables)
N_FLOP = 50
N_TURN = 40
N_RIVER = 20

# v3: category × in-class strength × draw. Fast, no MC, distinct flop/turn/river.
# 9 ranks × 4 strength × 3 draw = 108 flop; turn drops a draw nibble; river is made-hand only.
N_FLOP_V3 = 108
N_TURN_V3 = 72
N_RIVER_V3 = 36
_KICK_SPAN = 13 ** 5  # 371_293, packed-kicker range

_CLUSTERS = None


def hand_id(c1: int, c2: int) -> int:
    return HAND_INDEX[hole_to_169(c1, c2)]


def _draw_flags(hole: Tuple[int, int], board: Sequence[int]) -> Tuple[int, int]:
    from poker.cards import rank_of, suit_of

    hs = (suit_of(hole[0]), suit_of(hole[1]))
    bs = [suit_of(c) for c in board]
    fd = 0
    for s in range(4):
        total = hs.count(s) + bs.count(s)
        if hs.count(s) >= 1 and total == 4:
            fd = 1
            break
    ranks = sorted(set(rank_of(c) for c in list(hole) + list(board)))
    if 12 in ranks:
        ranks = sorted(set(ranks) | {-1})
    oesd = 0
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            if (j - i + 1) == 4 and (ranks[j] - ranks[i]) == 3:
                oesd = 1
    return fd, oesd


def _str4(value: int) -> int:
    cat = int(value) // int(CAT_BASE)
    kick = int(value) - cat * int(CAT_BASE)
    q = kick * 4 // _KICK_SPAN
    return int(min(3, max(0, q)))


def flop_bucket_v3(hole: Tuple[int, int], board: Sequence[int]) -> int:
    v = evaluate([hole[0], hole[1]] + list(board[:3]))
    cat = min(8, category(v))
    fd, oesd = _draw_flags(hole, board[:3])
    draw = 2 if (fd and oesd) else (1 if (fd or oesd) else 0)
    return int(min(N_FLOP_V3 - 1, cat * 12 + _str4(v) * 3 + draw))


def turn_bucket_v3(hole: Tuple[int, int], board: Sequence[int]) -> int:
    v = evaluate([hole[0], hole[1]] + list(board[:4]))
    cat = min(8, category(v))
    fd, oesd = _draw_flags(hole, board[:4])
    draw = 1 if (fd or oesd) else 0
    return int(min(N_TURN_V3 - 1, cat * 8 + _str4(v) * 2 + draw))


def river_bucket_v3(hole: Tuple[int, int], board: Sequence[int]) -> int:
    v = evaluate([hole[0], hole[1]] + list(board[:5]))
    cat = min(8, category(v))
    return int(min(N_RIVER_V3 - 1, cat * 4 + _str4(v)))


def street_bucket_v3(hole: Tuple[int, int], board: Sequence[int]) -> int:
    n = len(board)
    if n >= 5:
        return river_bucket_v3(hole, board)
    if n == 4:
        return turn_bucket_v3(hole, board)
    if n == 3:
        return flop_bucket_v3(hole, board)
    return hand_id(*hole)


def flop_features(hole: Tuple[int, int], board: Sequence[int]) -> np.ndarray:
    """Cheap EHS/potential proxy: made-hand rank + draws. No Monte Carlo."""
    made = float(evaluate([hole[0], hole[1]] + list(board))) / float(CAT_BASE * 9)
    fd, oesd = _draw_flags(hole, board)
    cat = category(evaluate([hole[0], hole[1]] + list(board))) / 8.0
    return np.array([made, cat, float(fd), float(oesd)], dtype=np.float64)


def _fit_clusters(n_samples: int = 4000, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    flop_x, turn_x, river_x = [], [], []
    for _ in range(n_samples):
        cards = rng.choice(52, size=9, replace=False)
        h = (int(cards[0]), int(cards[1]))
        flop = [int(x) for x in cards[2:5]]
        turn = flop + [int(cards[5])]
        river = turn + [int(cards[6])]
        flop_x.append(flop_features(h, flop)[0])
        turn_x.append(float(board_combo_score(h, turn)))
        river_x.append(float(evaluate([h[0], h[1]] + river)))
    def edges(arr, k):
        a = np.sort(np.asarray(arr, dtype=np.float64))
        qs = np.linspace(0.0, 1.0, k + 1)[1:-1]
        return np.quantile(a, qs)

    return {
        "flop_edges": edges(flop_x, N_FLOP),
        "turn_edges": edges(turn_x, N_TURN),
        "river_edges": edges(river_x, N_RIVER),
    }


def load_clusters(path: Path | None = None) -> dict:
    global _CLUSTERS
    if _CLUSTERS is not None:
        return _CLUSTERS
    p = path or CLUSTER_PATH
    if p.exists():
        z = np.load(p)
        _CLUSTERS = {k: z[k] for k in z.files}
        return _CLUSTERS
    DATA.mkdir(parents=True, exist_ok=True)
    cl = _fit_clusters()
    np.savez(p, **cl)
    _CLUSTERS = cl
    return cl


def save_clusters(path: Path | None = None) -> Path:
    p = path or CLUSTER_PATH
    cl = load_clusters(p if p.exists() else None)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **cl)
    return p


def _bin(value: float, edges: np.ndarray) -> int:
    return int(np.searchsorted(edges, value, side="right"))


def flop_bucket(hole: Tuple[int, int], board: Sequence[int]) -> int:
    cl = load_clusters()
    feat = flop_features(hole, board)
    b = _bin(feat[0], cl["flop_edges"])
    fd, oesd = int(feat[2]), int(feat[3])
    # 50 bins: 25 strength × draw nibble
    return int(min(N_FLOP - 1, (b % 25) * 2 + (1 if fd or oesd else 0)))


def turn_bucket(hole: Tuple[int, int], board: Sequence[int]) -> int:
    cl = load_clusters()
    return min(N_TURN - 1, _bin(float(board_combo_score(hole, board)), cl["turn_edges"]))


def river_bucket(hole: Tuple[int, int], board: Sequence[int]) -> int:
    cl = load_clusters()
    v = float(evaluate([hole[0], hole[1]] + list(board)))
    return min(N_RIVER - 1, _bin(v, cl["river_edges"]))


def street_bucket(hole: Tuple[int, int], board: Sequence[int]) -> int:
    n = len(board)
    if n >= 5:
        return river_bucket(hole, board)
    if n == 4:
        return turn_bucket(hole, board)
    if n == 3:
        return flop_bucket(hole, board)
    return hand_id(*hole)


def board_public_id(board: Sequence[int]) -> int:
    """Coarse public flop texture id (paired / monotone / wet)."""
    from strategy.texture import analyze

    if len(board) < 3:
        return 0
    tex = analyze(board)
    return (
        int(tex.paired)
        + 2 * int(tex.monotone)
        + 4 * int(round(tex.wetness * 3))
        + 16 * int(round(tex.highness * 3))
    )


def infoset_index(player: int, street: int, pub: int, history: int, priv: int) -> Tuple[int, ...]:
    return (int(player), int(street), int(pub), int(history), int(priv))
