"""HUD node-locks: change the search tree, not the last action.

Villain strategy at facing-bet / first-in nodes is mixed toward observed
frequencies. Hero then best-responds. Strength is capped so a misread
station does not blow up vs a solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from poker.game import PlayerView
from strategy.hud import HUD, SeatStats
from strategy.population import Population


@dataclass(frozen=True)
class NodeLock:
    """Additive biases on villain action weights, mixed in at `strength`."""

    fold_bias: float = 0.0
    call_bias: float = 0.0
    bet_bias: float = 0.0
    strength: float = 0.0

    @property
    def active(self) -> bool:
        return self.strength > 1e-9

    @classmethod
    def nash(cls) -> "NodeLock":
        return cls()


def from_hud(
    hud: Optional[HUD],
    population: Optional[Population],
    view: PlayerView,
) -> NodeLock:
    """Mild lock from the best-sampled live opponent. Unknown / solver → Nash."""
    pop = population or Population.unknown()
    st: Optional[SeatStats] = None
    if hud is not None:
        for i in range(view.num_players):
            if i == view.hero or view.folded[i]:
                continue
            cand = hud.stats(i)
            if st is None or cand.hands > st.hands:
                st = cand
    if st is None or st.hands < 12:
        if pop.name == "unknown":
            return NodeLock.nash()
        # Explicit live_low prior (caller opted in) — still mild.
        return NodeLock(
            fold_bias=0.12 * pop.overfold_river - 0.10 * pop.call_wide,
            call_bias=0.12 * pop.call_wide - 0.08 * pop.overfold_river,
            bet_bias=-0.06 * pop.call_wide,
            strength=0.12,
        )
    extra_fold = 0.0
    extra_bet = 0.0
    if st.donk_d >= 8 and st.donk() >= 0.18:
        extra_bet += 0.04
    if st.probe_d >= 8 and st.probe() <= 0.10:
        extra_fold += 0.04
    if st.river_bet_ip_d >= 6 and st.river_bet_when_checked() <= 0.18:
        extra_fold += 0.05
    if st.bet_frac_n >= 8 and st.avg_bet_frac() >= 0.90:
        extra_fold += 0.03
    # Leaks before solver-like: HU GTO VPIP looks loose; do not Nash-lock a station.
    if st.is_station():
        return NodeLock(
            fold_bias=-0.18,
            call_bias=0.22,
            bet_bias=-0.06 + extra_bet,
            strength=min(0.16, 0.14 + 0.02),
        )
    # Solver-like before overfolder: 6 noisy river folds must not lock
    # LibraryGTO as a leak (same order as FinalBot._use_villain_library).
    if st.is_solver_like() or st._sixmax_solver_chart():
        return NodeLock.nash()
    if st.is_tight_reg():
        under = st.river_bet_ip_d >= 6 and st.river_bet_when_checked() <= 0.22
        if under:
            return NodeLock(
                fold_bias=min(0.18, 0.12 + extra_fold),
                call_bias=-0.08,
                bet_bias=0.02,
                strength=min(0.14, 0.11 + extra_fold),
            )
        return NodeLock(
            fold_bias=0.02,
            call_bias=0.06,
            bet_bias=0.14 + extra_bet,
            strength=0.12,
        )
    if st.is_nit() or st.is_overfolder():
        return NodeLock(
            fold_bias=min(0.22, 0.16 + extra_fold),
            call_bias=-0.10,
            bet_bias=0.03,
            strength=min(0.16, 0.12 + extra_fold),
        )
    if extra_fold > 0.04 or extra_bet > 0.03:
        return NodeLock(
            fold_bias=0.08 * extra_fold * 4,
            call_bias=-0.04,
            bet_bias=0.04 * extra_bet * 8,
            strength=0.10,
        )
    return NodeLock.nash()


def apply_to_strategy(sig: np.ndarray, acts: Sequence[str], lock: Optional[NodeLock]) -> np.ndarray:
    """Mix `sig` (C, n_a) toward the locked weights. Identity if lock is off."""
    if lock is None or not lock.active or sig.size == 0:
        return sig
    out = np.array(sig, dtype=np.float64, copy=True)
    scale = np.ones(out.shape[-1], dtype=np.float64)
    for a, name in enumerate(acts):
        if name == "fold":
            scale[a] = max(0.20, 1.0 + lock.fold_bias)
        elif name == "call":
            scale[a] = max(0.20, 1.0 + lock.call_bias)
        elif name not in ("check", "fold"):
            scale[a] = max(0.20, 1.0 + lock.bet_bias)
    locked = out * scale
    tot = locked.sum(axis=-1, keepdims=True)
    n = locked.shape[-1]
    np.divide(locked, tot, out=locked, where=tot > 1e-12)
    locked = np.where(tot > 1e-12, locked, 1.0 / max(n, 1))
    mixed = (1.0 - lock.strength) * out + lock.strength * locked
    s = mixed.sum(axis=-1, keepdims=True)
    np.divide(mixed, s, out=mixed, where=s > 1e-12)
    return mixed
