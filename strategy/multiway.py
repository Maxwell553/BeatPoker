"""Position-and-count-specific multiway continue policy.

Not "cut bluffs × 0.55". Continue percentiles and c-bet frequencies are
tables of (n_live, IP/OOP, street, facing-bet vs first-in).
"""

from __future__ import annotations

from dataclasses import dataclass

from poker.game import Street


@dataclass(frozen=True)
class MultiwayPolicy:
    min_continue_perc: float
    extra_equity: float
    cbet_freq: float
    bluff_mult: float
    value_perc: float
    raise_perc: float
    air_bluff: bool


def policy(n_live: int, ip: bool, street: Street, facing: bool) -> MultiwayPolicy:
    n_live = max(2, int(n_live))
    river = street == Street.RIVER
    flop = street == Street.FLOP

    if n_live <= 2:
        if facing:
            perc = 0.42 if flop else (0.48 if not river else 0.50)
            return MultiwayPolicy(perc, 0.0, 0.55, 1.0, 0.70, 0.88, True)
        cbet = 0.62 if ip else 0.42
        if flop:
            cbet += 0.06
        if river:
            cbet = 0.32 if ip else 0.24
        return MultiwayPolicy(0.35, 0.0, cbet, 1.0, 0.70, 0.88, True)

    if n_live == 3:
        if facing:
            perc = 0.58 if flop else (0.64 if not river else 0.72)
            eq = 0.08 if flop else 0.10
            return MultiwayPolicy(perc, eq, 0.28, 0.18, 0.82, 0.92, False)
        if ip:
            cbet = 0.38 if flop else (0.28 if not river else 0.18)
            return MultiwayPolicy(0.50, 0.06, cbet, 0.22, 0.78, 0.90, False)
        cbet = 0.16 if flop else (0.12 if not river else 0.08)
        return MultiwayPolicy(0.55, 0.08, cbet, 0.08, 0.84, 0.93, False)

    # 4+
    if facing:
        perc = 0.68 if flop else (0.74 if not river else 0.82)
        return MultiwayPolicy(perc, 0.14, 0.16, 0.0, 0.88, 0.95, False)
    if ip:
        cbet = 0.28 if flop else (0.18 if not river else 0.10)
        return MultiwayPolicy(0.62, 0.10, cbet, 0.04, 0.86, 0.94, False)
    return MultiwayPolicy(0.70, 0.14, 0.10 if flop else 0.06, 0.0, 0.90, 0.96, False)
