"""Session HUD: VPIP, PFR, 3bet, c-bet, fold-to-cbet, river fold.

Stats persist across hands on a FinalBot instance. Folded heroes still see
later-street events when the match runner broadcasts observe_action / end_hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from poker.game import Action, ActionType, Street


@dataclass
class SeatStats:
    hands: int = 0
    vpip_n: int = 0
    vpip_d: int = 0
    pfr_n: int = 0
    pfr_d: int = 0
    threebet_n: int = 0
    threebet_d: int = 0
    cbet_n: int = 0
    cbet_d: int = 0
    fold_cbet_n: int = 0
    fold_cbet_d: int = 0
    river_fold_n: int = 0
    river_fold_d: int = 0
    fold_3bet_n: int = 0
    fold_3bet_d: int = 0
    donk_n: int = 0
    donk_d: int = 0
    probe_n: int = 0
    probe_d: int = 0
    river_bet_ip_n: int = 0
    river_bet_ip_d: int = 0
    bet_frac_sum: float = 0.0
    bet_frac_n: int = 0

    def _rate(self, n: int, d: int, default: float) -> float:
        return n / d if d >= 8 else default

    def vpip(self, default: float = 0.24) -> float:
        return self._rate(self.vpip_n, self.vpip_d, default)

    def pfr(self, default: float = 0.18) -> float:
        return self._rate(self.pfr_n, self.pfr_d, default)

    def threebet(self, default: float = 0.08) -> float:
        return self._rate(self.threebet_n, self.threebet_d, default)

    def cbet(self, default: float = 0.55) -> float:
        return self._rate(self.cbet_n, self.cbet_d, default)

    def fold_to_cbet(self, default: float = 0.42) -> float:
        return self._rate(self.fold_cbet_n, self.fold_cbet_d, default)

    def river_fold(self, default: float = 0.52) -> float:
        return self._rate(self.river_fold_n, self.river_fold_d, default)

    def fold_to_3bet(self, default: float = 0.55) -> float:
        return self._rate(self.fold_3bet_n, self.fold_3bet_d, default)

    def donk(self, default: float = 0.08) -> float:
        return self._rate(self.donk_n, self.donk_d, default)

    def probe(self, default: float = 0.22) -> float:
        return self._rate(self.probe_n, self.probe_d, default)

    def river_bet_when_checked(self, default: float = 0.35) -> float:
        return self._rate(self.river_bet_ip_n, self.river_bet_ip_d, default)

    def avg_bet_frac(self, default: float = 0.55) -> float:
        if self.bet_frac_n < 6:
            return default
        return self.bet_frac_sum / self.bet_frac_n

    def samples(self) -> int:
        return self.hands

    def agg_ratio(self) -> float:
        """PFR / VPIP. Solvers sit ~0.7–0.9; stations limp-call (~0.3). Table-size invariant."""
        v = self.vpip()
        if v < 0.05:
            # 0/12 VPIP is no-limp (pfr==vpip), not a 0.75 limp ratio that
            # tags LibraryGTO as a nit before the 12-hand chart can fire.
            if self.vpip_n == self.pfr_n:
                return 1.0
            return 0.75
        return self.pfr() / v

    def is_station(self) -> bool:
        if self.hands < 10 or self.vpip_d < 10:
            return False
        v = self.vpip_n / self.vpip_d
        r = self.agg_ratio()
        rf = (self.river_fold_n / self.river_fold_d) if self.river_fold_d >= 8 else 1.0
        limp_call = v >= 0.38 and r <= 0.48
        calling_river = v >= 0.36 and rf <= 0.38 and self.river_fold_d >= 8
        return limp_call or calling_river

    def is_nit(self) -> bool:
        if self.hands < 10 or self.vpip_d < 10:
            return False
        # No-limp low VPIP in <24 hands is often 6-max LibraryGTO noise
        # (binomial 23% → 10%), not a nit. True nits limp (agg << 0.92).
        if self.agg_ratio() >= 0.92 and self.hands < 24:
            return False
        return (self.vpip_n / self.vpip_d) <= 0.16 and (self.pfr_n / max(self.pfr_d, 1)) <= 0.12

    def _sixmax_solver_chart(self) -> bool:
        """6-max combo trees: ~22–26% VPIP, almost no limp (PFR≈VPIP).

        12-hand VPIP noise often lands outside 17–32% (binomial se ~12pp).
        Keep the no-limp ratio as the signature; widen VPIP so LibraryGTO
        is not HUD-scaled into a different range.
        """
        if self.hands < 12 or self.vpip_d < 12:
            return False
        v = self.vpip()
        r = self.agg_ratio()
        # 12-hand binomial 23% → 0–1 hits (~8%) used to miss 0.12.
        # Cap below HU library ~45% no-limp so that chart is not 6-max.
        return 0.0 <= v <= 0.42 and 0.92 <= r <= 1.05

    def is_solver_like(self) -> bool:
        """Library / solver-trained: this tree's HU VPIP is ~31%, not 50%.

        Need a lot of flop/river samples before a frequency leak can veto Nash.
        Eight noisy river folds is how SolverPro got node-locked as an overfolder.
        6-max GTO folds more multiway — that is not a rec leak.
        """
        if self.hands < 16:
            return False
        v = self.vpip()
        r = self.agg_ratio()
        # HU library ~31% VPIP. 6-max combo trees ~22–26% with almost no limp
        # (PFR≈VPIP). A 0.25 floor tagged those seats as TAG and collapsed
        # HU-pot search to 48 iters vs LibraryGTO.
        hu_ok = v > 0.25 and 0.55 <= r <= 1.05
        six_ok = self._sixmax_solver_chart()
        if not (hu_ok or six_ok):
            return False
        if six_ok:
            return True
        if self.fold_cbet_d >= 16:
            fc = self.fold_cbet_n / self.fold_cbet_d
            if fc >= 0.55 or fc <= 0.22:
                return False
        if self.river_fold_d >= 16:
            rf = self.river_fold_n / self.river_fold_d
            if rf >= 0.64 or rf <= 0.34:
                return False
        return True

    def is_tight_reg(self) -> bool:
        """Winning TAG: very high PFR/VPIP. This tree's HU library sits ~0.73, not 0.90.

        A 0.24 VPIP cutoff missed TAG seeds at 25–27% VPIP and Nash-locked them.
        """
        if self.hands < 20 or self.vpip_d < 20:
            return False
        if self.is_station() or self.is_solver_like():
            return False
        return self.agg_ratio() >= 0.88 and self.vpip() <= 0.30

    def is_overfolder(self) -> bool:
        """River overfold / underbluff leak — typical winning low-stakes regular.

        Tight-regs overfold air but c-bet too often and call pairs — handled as
        `is_tight_reg()`, not this flag. Keep 6@75% for true nits; do not treat
        8 noisy 60% river folds as a solver-trained pro leak.
        """
        if self.hands < 8:
            return False
        if self._sixmax_solver_chart():
            return False
        # 6-max LibraryGTO folds more multiway. 6 river samples at 8 hands
        # must not unlock exploit 3bets before the 12-hand chart can fire.
        if self.hands < 24 and self.agg_ratio() >= 0.92:
            return False
        if self.river_fold_d >= 6 and (self.river_fold_n / self.river_fold_d) >= 0.75:
            return True
        if self.river_fold_d >= 16 and (self.river_fold_n / self.river_fold_d) >= 0.62:
            return True
        if self.fold_cbet_d >= 16 and (self.fold_cbet_n / self.fold_cbet_d) >= 0.52:
            return True
        if self.fold_3bet_d >= 12 and (self.fold_3bet_n / self.fold_3bet_d) >= 0.65:
            return True
        return False

    def is_underbluffer(self) -> bool:
        return self.is_overfolder()


@dataclass
class _HandAcc:
    n: int
    pf_raises: int = 0
    pf_aggressor: Optional[int] = None
    vpip: Optional[List[bool]] = None
    pfr: Optional[List[bool]] = None
    faced_open: Optional[List[bool]] = None
    threebet: Optional[List[bool]] = None
    cbet_opp: Optional[List[bool]] = None
    cbet: Optional[List[bool]] = None
    faced_cbet: Optional[List[bool]] = None
    folded_cbet: Optional[List[bool]] = None
    faced_river: Optional[List[bool]] = None
    folded_river: Optional[List[bool]] = None
    faced_3bet: Optional[List[bool]] = None
    folded_3bet: Optional[List[bool]] = None
    flop_cbet_done: bool = False
    flop_checked: bool = True
    flop_bettor: Optional[int] = None
    donk_opp: Optional[List[bool]] = None
    donk: Optional[List[bool]] = None
    probe_opp: Optional[List[bool]] = None
    probe: Optional[List[bool]] = None
    river_checked_to: Optional[List[bool]] = None
    river_bet: Optional[List[bool]] = None
    size_frac: Optional[List[float]] = None
    size_n: Optional[List[int]] = None

    def __post_init__(self) -> None:
        z = [False] * self.n
        self.vpip = list(z)
        self.pfr = list(z)
        self.faced_open = list(z)
        self.threebet = list(z)
        self.cbet_opp = list(z)
        self.cbet = list(z)
        self.faced_cbet = list(z)
        self.folded_cbet = list(z)
        self.faced_river = list(z)
        self.folded_river = list(z)
        self.faced_3bet = list(z)
        self.folded_3bet = list(z)
        self.donk_opp = list(z)
        self.donk = list(z)
        self.probe_opp = list(z)
        self.probe = list(z)
        self.river_checked_to = list(z)
        self.river_bet = list(z)
        self.size_frac = [0.0] * self.n
        self.size_n = [0] * self.n


class HUD:
    """Per-seat session stats. One HUD per FinalBot (same table seats)."""

    def __init__(self, n_seats: int = 9):
        self.n = n_seats
        self.seats: List[SeatStats] = [SeatStats() for _ in range(n_seats)]
        self._hand: Optional[_HandAcc] = None

    def _ensure(self, n: int) -> None:
        if n <= len(self.seats):
            return
        for _ in range(n - len(self.seats)):
            self.seats.append(SeatStats())
        self.n = n

    def stats(self, seat: int) -> SeatStats:
        self._ensure(seat + 1)
        return self.seats[seat]

    def begin_hand(self, n_seats: int) -> None:
        self._ensure(n_seats)
        self._hand = _HandAcc(n=n_seats)

    def observe_action(self, seat: int, action: Action, street: Street, n_seats: int, pot: Optional[int] = None) -> None:
        self._ensure(n_seats)
        if self._hand is None or self._hand.n != n_seats:
            self.begin_hand(n_seats)
        h = self._hand
        aggressive = action.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)
        voluntary = action.type in (
            ActionType.CALL,
            ActionType.BET,
            ActionType.RAISE,
            ActionType.ALL_IN,
        )
        if pot is not None and aggressive and pot > 0:
            h.size_frac[seat] += action.amount / max(float(pot), 1.0)
            h.size_n[seat] += 1
        if street == Street.PREFLOP:
            if voluntary:
                h.vpip[seat] = True
            if aggressive:
                h.pfr[seat] = True
                if h.pf_raises == 1:
                    h.threebet[seat] = True
                h.pf_raises += 1
                h.pf_aggressor = seat
            elif h.pf_raises == 1 and action.type in (ActionType.FOLD, ActionType.CALL):
                h.faced_open[seat] = True
            if h.pf_raises >= 2 and not aggressive:
                h.faced_3bet[seat] = True
                if action.type == ActionType.FOLD:
                    h.folded_3bet[seat] = True
        elif street == Street.FLOP:
            if h.pf_aggressor is not None and seat == h.pf_aggressor and not h.flop_cbet_done:
                h.cbet_opp[seat] = True
                if aggressive:
                    h.cbet[seat] = True
                h.flop_cbet_done = True
            agg = h.pf_aggressor
            if agg is not None and seat != agg and not h.flop_cbet_done:
                h.donk_opp[seat] = True
                if aggressive:
                    h.donk[seat] = True
            if aggressive:
                h.flop_checked = False
                h.flop_bettor = seat
            agg = h.pf_aggressor
            if agg is not None and h.cbet[agg] and seat != agg:
                if action.type == ActionType.FOLD:
                    h.faced_cbet[seat] = True
                    h.folded_cbet[seat] = True
                elif voluntary:
                    h.faced_cbet[seat] = True
            elif h.flop_bettor is not None and seat != h.flop_bettor:
                if action.type == ActionType.FOLD:
                    h.faced_cbet[seat] = True
                    h.folded_cbet[seat] = True
                elif voluntary:
                    h.faced_cbet[seat] = True
        elif street == Street.TURN:
            if h.flop_checked and aggressive:
                h.probe_opp[seat] = True
                h.probe[seat] = True
        elif street == Street.RIVER:
            if action.type == ActionType.FOLD:
                h.faced_river[seat] = True
                h.folded_river[seat] = True
            elif voluntary and not aggressive:
                h.faced_river[seat] = True
            if not aggressive and action.type == ActionType.CHECK:
                h.river_checked_to[seat] = True
            if aggressive:
                h.river_bet[seat] = True

    def end_hand(self, n_seats: Optional[int] = None) -> None:
        h = self._hand
        self._hand = None
        if h is None:
            return
        n = n_seats or h.n
        self._ensure(n)
        for i in range(h.n):
            st = self.seats[i]
            st.hands += 1
            st.vpip_d += 1
            if h.vpip[i]:
                st.vpip_n += 1
            st.pfr_d += 1
            if h.pfr[i]:
                st.pfr_n += 1
            if h.faced_open[i] or h.threebet[i]:
                st.threebet_d += 1
                if h.threebet[i]:
                    st.threebet_n += 1
            if h.cbet_opp[i]:
                st.cbet_d += 1
                if h.cbet[i]:
                    st.cbet_n += 1
            if h.faced_cbet[i]:
                st.fold_cbet_d += 1
                if h.folded_cbet[i]:
                    st.fold_cbet_n += 1
            if h.faced_river[i]:
                st.river_fold_d += 1
                if h.folded_river[i]:
                    st.river_fold_n += 1
            if h.faced_3bet[i]:
                st.fold_3bet_d += 1
                if h.folded_3bet[i]:
                    st.fold_3bet_n += 1
            if h.donk_opp[i]:
                st.donk_d += 1
                if h.donk[i]:
                    st.donk_n += 1
            if h.probe_opp[i]:
                st.probe_d += 1
                if h.probe[i]:
                    st.probe_n += 1
            if h.river_checked_to[i] or h.river_bet[i]:
                st.river_bet_ip_d += 1
                if h.river_bet[i]:
                    st.river_bet_ip_n += 1
            if h.size_n[i]:
                st.bet_frac_sum += h.size_frac[i]
                st.bet_frac_n += h.size_n[i]

    def villain_seats(self, hero: int, n: int, folded: Sequence[bool]) -> List[int]:
        return [i for i in range(n) if i != hero and not folded[i]]
