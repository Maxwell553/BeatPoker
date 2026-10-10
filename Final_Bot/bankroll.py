"""Bankroll sizing from win rate and NLH variance.

Uses the standard diffusion risk-of-ruin formula:

    RoR ≈ exp(-2 μ B / σ²)

μ, σ per hand in big blinds; B is bankroll in big blinds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BankrollPlan:
    buyins_5pct_ror: int
    buyins_1pct_ror: int
    recommended: int
    hands_for_long_run: int
    assumed_bb100: float
    assumed_stdev_bb100: float
    note: str


def buyins_for_ror(
    winrate_bb100: float,
    stdev_bb100: float = 110.0,
    ror: float = 0.05,
    bi_bb: float = 100.0,
) -> int:
    if winrate_bb100 <= 0:
        return 10**9
    mu = winrate_bb100 / 100.0
    sigma = stdev_bb100 / 10.0  # stdev per 100 hands → per hand
    # B = -ln(ror) * σ² / (2μ)
    b_bb = -math.log(ror) * (sigma**2) / (2.0 * mu)
    return max(20, int(math.ceil(b_bb / bi_bb)))


def plan(
    winrate_bb100: float = 8.0,
    stdev_bb100: float = 110.0,
    bi_bb: float = 100.0,
) -> BankrollPlan:
    """Default: after-rake ~8 bb/100 in a soft casino, NLH stdev ~110 bb/100."""
    b5 = buyins_for_ror(winrate_bb100, stdev_bb100, 0.05, bi_bb)
    b1 = buyins_for_ror(winrate_bb100, stdev_bb100, 0.01, bi_bb)
    # Long run: SE of bb/100 = stdev / sqrt(N/100)  want SE < winrate/2
    # stdev / sqrt(N/100) < wr/2  → N/100 > (2 stdev / wr)^2
    if winrate_bb100 > 0:
        n100 = (2.0 * stdev_bb100 / winrate_bb100) ** 2
        hands = int(math.ceil(n100 * 100))
    else:
        hands = 10**9
    rec = max(b5, 30)
    return BankrollPlan(
        buyins_5pct_ror=b5,
        buyins_1pct_ror=b1,
        recommended=rec,
        hands_for_long_run=hands,
        assumed_bb100=winrate_bb100,
        assumed_stdev_bb100=stdev_bb100,
        note=(
            f"{rec} buy-ins of {bi_bb:.0f}bb (~{rec * bi_bb:.0f}bb) for ~5% risk of "
            f"ruin at {winrate_bb100:.0f} bb/100 after rake. "
            f"Need ~{hands:,} hands before the win rate is distinguishable from noise."
        ),
    )


def recommend(game: str = "soft_casino") -> BankrollPlan:
    """Bankroll for a 100bb cash game by pool toughness (after rake)."""
    presets = {
        "soft_casino": (8.0, 110.0, 30),
        "mixed": (4.0, 110.0, 50),
        "tough": (1.5, 120.0, 80),
        "solver": (0.5, 110.0, 100),
    }
    if game not in presets:
        raise ValueError(f"unknown game {game!r}; expected {sorted(presets)}")
    wr, sd, floor = presets[game]
    p = plan(wr, sd)
    rec = max(p.recommended, floor)
    return BankrollPlan(
        buyins_5pct_ror=p.buyins_5pct_ror,
        buyins_1pct_ror=p.buyins_1pct_ror,
        recommended=rec,
        hands_for_long_run=p.hands_for_long_run,
        assumed_bb100=p.assumed_bb100,
        assumed_stdev_bb100=p.assumed_stdev_bb100,
        note=(
            f"{rec} buy-ins of 100bb for a {game.replace('_', ' ')} game at "
            f"~{wr:g} bb/100 after rake (σ≈{sd:.0f} bb/100)."
        ),
    )
