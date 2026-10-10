"""Cash-game rake.

Live rooms typically take ~10% of the pot up to a cap (e.g. $5 at 1/2, $8–15
at 2/5). Most rooms are no-flop-no-drop: uncontested preflop pots are unraked.

Rake makes marginal preflop hands and thin flop peels negative. It does *not*
change that AA still raises.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RakeConfig:
    """`percent` is a fraction (0.10 = 10%). `cap_bb` is the cap in big blinds."""

    percent: float = 0.10
    cap_bb: float = 3.0
    no_flop_no_drop: bool = True

    @classmethod
    def none(cls) -> "RakeConfig":
        return cls(percent=0.0, cap_bb=0.0)

    @classmethod
    def casino(cls) -> "RakeConfig":
        """Typical live 1/2–2/5: 10% up to ~3 bb, no flop no drop."""
        return cls(percent=0.10, cap_bb=3.0, no_flop_no_drop=True)

    @classmethod
    def casino_heavy(cls) -> "RakeConfig":
        """Harsh live cap (10% to 2 bb) — tighter strategy."""
        return cls(percent=0.10, cap_bb=2.0, no_flop_no_drop=True)

    @classmethod
    def online(cls) -> "RakeConfig":
        """Typical mid-stakes online: ~5% cap 3 bb."""
        return cls(percent=0.05, cap_bb=3.0, no_flop_no_drop=True)

    @property
    def active(self) -> bool:
        return self.percent > 1e-9

    def amount(self, pot_chips: int, bb: int, saw_flop: bool) -> int:
        if not self.active or pot_chips <= 0:
            return 0
        if self.no_flop_no_drop and not saw_flop:
            return 0
        cap = int(round(self.cap_bb * bb))
        return min(cap, int(pot_chips * self.percent))

    def net_pot(self, pot_chips: int, bb: int, saw_flop: bool = True) -> int:
        return max(0, pot_chips - self.amount(pot_chips, bb, saw_flop))

    def equity_needed_to_call(self, to_call: int, pot: int, bb: int) -> float:
        """Break-even equity to call after rake is taken from the future pot."""
        if to_call <= 0:
            return 0.0
        future = pot + to_call
        net = self.net_pot(future, bb, saw_flop=True)
        if net <= 0:
            return 1.0
        return to_call / net

    def open_mult(self, position: str) -> float:
        """Scale RFI frequency. SB and EP pay the most; BTN is almost untouched."""
        if not self.active:
            return 1.0
        harsh = min(1.0, self.percent / 0.10) * min(1.2, 3.0 / max(self.cap_bb, 0.75))
        by_pos = {
            "UTG": 0.86,
            "UTG1": 0.88,
            "UTG2": 0.90,
            "MP": 0.91,
            "HJ": 0.93,
            "CO": 0.95,
            "BTN": 0.97,
            "SB": 0.80,
            "BB": 1.0,
        }
        base = by_pos.get(position, 0.92)
        return 1.0 - (1.0 - base) * harsh

    def defend_mult(self) -> float:
        """Scale calling vs opens (3bets stay; peeling junk to see a raked flop does not)."""
        if not self.active:
            return 1.0
        harsh = min(1.0, self.percent / 0.10)
        return 1.0 - 0.16 * harsh

    def bluff_mult(self) -> float:
        """Fold equity is worth slightly less when the pot is raked."""
        if not self.active:
            return 1.0
        return 1.0 - 0.08 * min(1.0, self.percent / 0.10)
