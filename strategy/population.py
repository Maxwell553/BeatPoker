"""Population node-locks. Independent of rake.

Live 1-2 overfolds rivers and calls too wide; online mid-stakes does not.
Rake config is not a calling-station prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from strategy.rake import RakeConfig


@dataclass(frozen=True)
class Population:
    """Frequency priors used to rebuild villain ranges and to scale exploits."""

    name: str
    vpip: float
    pfr: float
    threebet: float
    cbet: float
    fold_to_cbet: float
    river_fold: float
    call_wide: float  # 0 Nash, 1 calling-station
    overfold_river: float  # 0 Nash, 1 live-rec overfold
    bluff_freq: float  # relative to GTO; <1 underbluffs

    @classmethod
    def unknown(cls) -> "Population":
        """Nash prior — do not invent a leak vs unknown regulars / solvers."""
        return cls(
            name="unknown",
            vpip=0.24,
            pfr=0.18,
            threebet=0.08,
            cbet=0.55,
            fold_to_cbet=0.42,
            river_fold=0.52,
            call_wide=0.0,
            overfold_river=0.0,
            bluff_freq=1.0,
        )

    @classmethod
    def live_low(cls) -> "Population":
        """Live 1-2 / 1-3: high VPIP, overfolds rivers, calls too wide flop/turn."""
        return cls(
            name="live_low",
            vpip=0.38,
            pfr=0.14,
            threebet=0.05,
            cbet=0.48,
            fold_to_cbet=0.55,
            river_fold=0.68,
            call_wide=0.55,
            overfold_river=0.70,
            bluff_freq=0.55,
        )

    @classmethod
    def online_mid(cls) -> "Population":
        """Online mid-stakes: closer to GTO, not a station prior."""
        return cls(
            name="online_mid",
            vpip=0.26,
            pfr=0.20,
            threebet=0.09,
            cbet=0.58,
            fold_to_cbet=0.40,
            river_fold=0.50,
            call_wide=0.10,
            overfold_river=0.08,
            bluff_freq=0.92,
        )

    @classmethod
    def from_rake(cls, rake: Optional[RakeConfig]) -> "Population":
        """Rake is not a calling-station prior. Default Nash vs unknown regs.

        Use live_low() only after HUD says rec, or when the caller opts in.
        """
        _ = rake
        return cls.unknown()

    @classmethod
    def from_hud(cls, hud, seat: int, n_seats: int, folded, fallback: Optional["Population"] = None) -> "Population":
        """live_low only after a seat looks like a rec; else Nash."""
        from strategy.hud import HUD

        _ = n_seats
        pop = fallback or cls.unknown()
        if hud is None:
            return pop
        st = hud.stats(seat) if isinstance(hud, HUD) else None
        if st is None or st.hands < 12:
            return pop
        if st.is_station() or (
            st.vpip() >= 0.34 and st.agg_ratio() <= 0.50 and st.river_fold() >= 0.58
        ):
            return cls.live_low()
        if st.is_solver_like():
            return cls.unknown()
        return pop
