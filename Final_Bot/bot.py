"""Final_Bot — production NLH cash strategy.

Public API
----------
    from Final_Bot import FinalBot, RakeConfig, Population
    bot = FinalBot(seat=0)                          # casino rake, Nash pop
    action = bot.act(view)                          # PlayerView -> Action
    bot.observe_action(seat, action, street, n)
    bot.end_hand()
    bot.reset()                                     # between hands (HUD persists)

GTO floor (combo trees + nested HU search) plus a population/HUD
exploit layer. Default population is Nash (`unknown`); live_low only
after HUD says rec, or when the caller opts in.
"""

from __future__ import annotations

import random
import traceback
from pathlib import Path
from typing import Optional

from poker.game import Action, ActionType, PlayerView, Street
from strategy.exploit import apply as exploit_apply
from strategy.exploit import bluff_mult, river_defend_mult, value_threshold_adj
from strategy.hud import HUD
from strategy.live_solve import cached_live_action
from strategy.nodelock import from_hud as node_lock_from_hud
from strategy.population import Population
from strategy.postflop import decide_postflop
from strategy.pot_type import classify as pot_kind
from strategy.preflop import decide_preflop
from strategy.rake import RakeConfig
from strategy.range_model import extra_villain_ranges, primary_villain_range, reconstruct


def _log_search_fail(tag: str, e: BaseException) -> None:
    try:
        log = Path(__file__).resolve().parent / "data" / "search_fail.log"
        with log.open("a") as fh:
            fh.write(f"{tag} {type(e).__name__}: {e}\n{traceback.format_exc(limit=8)}\n")
    except Exception:
        pass


class FinalBot:
    """Production NLH agent for 2–9 players and 50–1000bb stacks."""

    name = "Final_Bot"

    def __init__(
        self,
        seat: int = 0,
        seed: Optional[int] = None,
        quality: str = "high",
        rake: Optional[RakeConfig] = None,
        population: Optional[Population] = None,
    ):
        self.seat = seat
        self.rng = random.Random(seed)
        self.quality = quality if quality in ("high", "fast") else "high"
        self.rake = rake if rake is not None else RakeConfig.casino()
        self.population = population if population is not None else Population.unknown()
        self.hud = HUD(n_seats=9)
        self._last_board_len = -1
        self._hand_open = False

    def reset(self) -> None:
        """Between hands: do not wipe HUD."""
        self._last_board_len = -1

    def observe_action(self, seat: int, action: Action, street: Street, n_seats: int, pot=None) -> None:
        if not self._hand_open:
            self.hud.begin_hand(n_seats)
            self._hand_open = True
        self.hud.observe_action(seat, action, street, n_seats, pot=pot)

    def end_hand(self, n_seats: Optional[int] = None) -> None:
        self.hud.end_hand(n_seats)
        self._hand_open = False

    def act(self, view: PlayerView) -> Action:
        try:
            action = self._act_inner(view)
        except Exception as e:
            _log_search_fail("act", e)
            action = self._safe_fallback(view)
        return self._sanitize(view, action)

    def _effective_population(self, view: PlayerView) -> Population:
        """Nash vs unknown regs. live_low only after HUD looks like a rec."""
        pop = self.population
        best = None
        for i in range(view.num_players):
            if i == view.hero or view.folded[i]:
                continue
            st = self.hud.stats(i)
            if best is None or st.hands > best.hands:
                best = st
        if best is None or best.hands < 12:
            return pop
        if best.is_station() or (
            best.vpip() >= 0.34 and best.agg_ratio() <= 0.50 and best.river_fold() >= 0.58
        ):
            return Population.live_low()
        if best.is_solver_like() or best._sixmax_solver_chart():
            return Population.unknown()
        return pop

    def _act_inner(self, view: PlayerView) -> Action:
        pop = self._effective_population(view)
        if view.street == Street.PREFLOP:
            # LibraryGTO's trees are unraked. Casino-rake _from_trees
            # overfolds vs it when the overlay misses.
            a = decide_preflop(view, self.rng, rake=self._search_rake(view))
            return exploit_apply(
                a, view, self.hud, pop, self.rng,
                vs_library=self._use_villain_library(view),
            )

        ranges = reconstruct(view, hud=self.hud, population=pop)
        v_combos, v_w = primary_villain_range(view, ranges)
        h_combos, h_w = ranges[view.hero].combo_weights()
        extra = max(0, view.players_in_hand - 2)
        extras = extra_villain_ranges(view, ranges, (v_combos, v_w))
        defend = river_defend_mult(self.hud, pop, view)
        bmult = bluff_mult(self.hud, pop, view)
        vadj = value_threshold_adj(self.hud, pop, view)

        heuristic = decide_postflop(
            view,
            v_combos,
            v_w,
            h_combos,
            h_w,
            self.rng,
            extra_villains=extra,
            rake=self.rake,
            extra_villain_ranges=extras,
            river_defend_mult=defend,
            bluff_scale=bmult,
            value_adj=vadj,
        )

        gto = heuristic
        searched = None
        lock = node_lock_from_hud(self.hud, pop, view)
        # Nested search: HU 2-player; 3+ way reduces to hero + two relevant ranges.
        extra = max(0, view.players_in_hand - 2)
        vs_lib = self._use_villain_library(view)
        if extra == 0:
            searched = self._maybe_search(view, h_combos, h_w, v_combos, v_w, lock)
        else:
            searched = self._maybe_search_3way(view, h_combos, h_w, v_combos, v_w, extras, lock, ranges)
            # HU collapse of a 3-way pot vs LibraryGTO cbets as if heads-up.
            if searched is None and extra <= 1 and not vs_lib:
                searched = self._maybe_search(view, h_combos, h_w, v_combos, v_w, lock)
        if searched is not None:
            gto = searched
        else:
            # v3 MW is fold/call + 33/75. LibraryGTO plays 25/55/raise.
            # Prefer that mix over a different game when search misses,
            # including HU leftover pots at a 6-max table.
            if vs_lib:
                try:
                    from strategy.library import library_policy

                    lib_a = library_policy(view, ranges[view.hero], self.rng)
                    if lib_a is not None:
                        gto = lib_a
                except Exception:
                    pass
            if gto is heuristic:
                try:
                    from strategy.blueprint import try_v3_postflop

                    bp = try_v3_postflop(view, self.rng)
                    if bp is not None:
                        gto = bp
                except Exception:
                    pass
            if gto is heuristic:
                try:
                    from strategy.library import library_policy

                    lib_a = library_policy(view, ranges[view.hero], self.rng)
                    if lib_a is not None:
                        gto = lib_a
                except Exception:
                    pass
        return exploit_apply(
            gto, view, self.hud, pop, self.rng,
            from_search=searched is not None, vs_library=vs_lib,
        )

    def _search_budget(self, view, extra: int):
        kind = pot_kind(view)
        if self.quality == "high":
            iters, combos, runouts = 360, 240, 20
            if view.street == Street.RIVER:
                runouts = 1
                combos = 280
                iters = max(iters, 400)
            if kind in ("3bet", "4bet"):
                iters, combos = max(iters, 400), max(combos, 240)
            if extra >= 1:
                # Hero-only CFR vs two locked library ranges. The 3-way tree
                # is ~1600 nodes; 320 iters under-converged raise EV vs LibraryGTO.
                iters, combos, runouts = 480, 192, 12
                if view.street == Street.RIVER:
                    runouts = 1
                    iters, combos = 320, 192
            elif extra == 0:
                # Hero-only BR vs locked library; 6 first-in sizes need more
                # iterations than the old 33/75/125 tree.
                iters = max(iters, 480)
                combos = max(combos, 240)
        else:
            iters, combos, runouts = 48, 56, 4
            if view.street == Street.RIVER:
                runouts = 1
                iters, combos = 72, 80
            if kind in ("3bet", "4bet"):
                iters, combos, runouts = 80, 96, 4
            if extra == 0 and view.to_call <= 0 and kind == "srp":
                iters, combos, runouts = 96, 112, 5
            if extra == 0 and view.to_call > 0:
                iters = max(iters, 72)
                combos = max(combos, 80)
            if extra >= 1:
                iters, combos, runouts = 36, 32, 2
                if kind in ("3bet", "4bet") or view.to_call > 0:
                    iters, combos, runouts = 48, 40, 3
        st = self._primary_hud(view)
        # Do not starve HU leftover vs LibraryGTO. A noisy TAG tag used to
        # collapse facing search to 48 iters while the villain stayed locked.
        if (
            extra == 0
            and st is not None
            and st.is_tight_reg()
            and not st.is_solver_like()
            and not st._sixmax_solver_chart()
        ):
            if view.to_call > 0:
                iters, combos, runouts = 48, 56, 3
                if view.street == Street.RIVER:
                    runouts = 1
                    iters, combos = 56, 64
        try:
            from strategy.blueprint import v3_search_scale

            scale = v3_search_scale()
            if scale > 1.0:
                iters = int(iters * scale)
                combos = int(combos * scale)
        except Exception:
            pass
        # H=1 vs a locked library; spend the combo budget on villains.
        if self.quality == "high" and extra >= 0 and self._use_villain_library(view):
            if extra >= 1:
                combos = max(combos, 400)
                if view.street != Street.RIVER:
                    runouts = max(runouts, 16)
            else:
                combos = max(combos, 320)
        return iters, combos, runouts

    def _use_villain_library(self, view) -> bool:
        """Best-respond to the library mix vs unknown/solver seats. Node-lock recs."""
        st = self._primary_hud(view)
        if st is None or st.hands < 12:
            return True
        # Solver-like first: 6 noisy river folds must not unlock Nash vs LibraryGTO.
        if st.is_solver_like():
            return True
        # 6-max no-limp ~23% VPIP folds more multiway. That chart is LibraryGTO,
        # not a TAG/overfolder — keep the library lock and full search budget.
        if st._sixmax_solver_chart():
            return True
        # 12–23 no-limp hands is LibraryGTO noise, not a leak unlock.
        if st.hands < 24 and st.agg_ratio() >= 0.92:
            return True
        if st.is_station() or st.is_overfolder() or st.is_nit():
            return False
        return True

    def _raises_left(self, view) -> int:
        kind = pot_kind(view)
        if kind in ("3bet", "4bet"):
            return 3
        return 2

    def _primary_hud(self, view):
        st = None
        for i in range(view.num_players):
            if i == view.hero or view.folded[i]:
                continue
            cand = self.hud.stats(i)
            if st is None or cand.hands > st.hands:
                st = cand
        return st

    def _search_rake(self, view):
        """LibraryGTO does not price rake. Searching raked overfolds rivers vs it."""
        if self._use_villain_library(view):
            from strategy.rake import RakeConfig

            return RakeConfig.none()
        return self.rake

    def _maybe_search(self, view, h_combos, h_w, v_combos, v_w, lock=None) -> Optional[Action]:
        if view.street not in (Street.FLOP, Street.TURN, Street.RIVER):
            return None
        kind = pot_kind(view)
        extra = max(0, view.players_in_hand - 2)
        # Fast multiway first-in SRP stays on the library. HU first-in searches
        # (that's the edge vs a 6-max distill).
        if self.quality == "fast" and view.street != Street.RIVER and kind == "srp" and view.to_call <= 0:
            if extra >= 1:
                return None
            st = self._primary_hud(view)
            if st is not None and st.is_tight_reg():
                return None
        iters, combos, runouts = self._search_budget(view, extra)
        try:
            return cached_live_action(
                view,
                h_combos,
                h_w,
                v_combos,
                v_w,
                self.rng,
                iterations=iters,
                max_combos=combos,
                runouts=runouts,
                node_lock=lock,
                rake=self._search_rake(view),
                villain_library=self._use_villain_library(view),
                raises_left=self._raises_left(view),
            )
        except Exception as e:
            self.search_errors = getattr(self, "search_errors", 0) + 1
            _log_search_fail("hu", e)
            return None

    def _maybe_search_3way(self, view, h_combos, h_w, v_combos, v_w, extras, lock=None, ranges=None) -> Optional[Action]:
        if view.street not in (Street.FLOP, Street.TURN, Street.RIVER):
            return None
        kind = pot_kind(view)
        extra_n = max(0, view.players_in_hand - 2)
        # Nested search on every postflop decision, including 6-max first-in SRP
        # (reduced to hero + two relevant ranges). Fast first-in uses a smaller budget.
        iters, combos, runouts = self._search_budget(view, extra=max(1, extra_n))
        if self.quality == "fast" and view.street != Street.RIVER and kind == "srp" and view.to_call <= 0:
            iters, combos, runouts = min(iters, 28), min(combos, 24), min(runouts, 2)
        v2_c, v2_w = None, None
        if ranges is not None:
            try:
                from strategy.mw_reduce import apply_blocker_mass, two_relevant_ranges

                primary, secondary, others = two_relevant_ranges(view, ranges)
                v_combos, v_w = primary
                v2_c, v2_w = secondary
                if others:
                    other_rgs = [ranges[i] for i in others]
                    v_combos, v_w = apply_blocker_mass(v_combos, v_w, other_rgs)
                    v2_c, v2_w = apply_blocker_mass(v2_c, v2_w, other_rgs)
            except Exception:
                v2_c = v2_w = None
        v1_stk = v2_stk = None
        v1_ip = v2_ip = None
        v1_seat = v2_seat = None
        if ranges is not None:
            try:
                from strategy.mw_reduce import two_relevant_seats
                from strategy.live_solve import _seat_oop

                a, b = two_relevant_seats(view, ranges)
                v1_stk = float(view.stacks[a])
                v2_stk = float(view.stacks[b])
                v1_ip = not _seat_oop(view, a)
                v2_ip = not _seat_oop(view, b)
                v1_seat, v2_seat = int(a), int(b)
            except Exception:
                v1_stk = v2_stk = None
                v1_ip = v2_ip = None
                v1_seat = v2_seat = None
        ghost_c = ghost_w = None
        if ranges is not None and v1_seat is not None and v2_seat is not None:
            try:
                from strategy.mw_reduce import ghost_allin_combos

                gw = ghost_allin_combos(view, ranges, v1_seat, v2_seat)
                if gw is not None:
                    ghost_c, ghost_w = gw
            except Exception:
                ghost_c = ghost_w = None
        if v1_seat is not None and v1_seat == v2_seat:
            # One acting opponent (all-in/dead are blockers). A 3-way tree
            # vs a cloned villain prices a phantom donk/raise.
            try:
                from strategy.live_solve import cached_live_action

                return cached_live_action(
                    view,
                    h_combos,
                    h_w,
                    v_combos,
                    v_w,
                    self.rng,
                    iterations=iters,
                    max_combos=combos,
                    runouts=runouts,
                    node_lock=lock,
                    rake=self._search_rake(view),
                    villain_library=self._use_villain_library(view),
                    raises_left=self._raises_left(view),
                )
            except Exception as e:
                self.search_errors = getattr(self, "search_errors", 0) + 1
                _log_search_fail("hu-allin", e)
                return None
        if v2_c is None:
            if not extras:
                return None
            v2_c, v2_w = extras[0]
        try:
            from strategy.search_3way import live_action_3way

            return live_action_3way(
                view, h_combos, h_w, v_combos, v_w, v2_c, v2_w, self.rng,
                iterations=iters, max_combos=combos, runouts=runouts,
                lock=lock, rake=self._search_rake(view),
                villain_library=self._use_villain_library(view),
                v1_stack=v1_stk, v2_stack=v2_stk,
                v1_ip=v1_ip, v2_ip=v2_ip,
                v1_seat=v1_seat, v2_seat=v2_seat,
                ghost_combos=ghost_c, ghost_w=ghost_w,
            )
        except Exception as e:
            self.search_errors = getattr(self, "search_errors", 0) + 1
            _log_search_fail("3way", e)
            return None

    def _safe_fallback(self, view: PlayerView) -> Action:
        if view.to_call > 0:
            if view.to_call <= max(view.bb, view.pot // 6):
                return Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero]))
            return Action(ActionType.FOLD)
        return Action(ActionType.CHECK)

    def _sanitize(self, view: PlayerView, action: Action) -> Action:
        stack = view.stacks[view.hero]
        to_call = view.to_call
        if action.type == ActionType.FOLD:
            if to_call <= 0:
                return Action(ActionType.CHECK)
            return action
        if action.type == ActionType.CHECK:
            if to_call > 0:
                return Action(ActionType.CALL, min(to_call, stack))
            return action
        if action.type == ActionType.CALL:
            if to_call <= 0:
                return Action(ActionType.CHECK)
            return Action(ActionType.CALL, min(to_call, stack))
        put = int(action.amount)
        put = max(0, min(put, stack))
        if put <= 0:
            return Action(ActionType.CHECK) if to_call <= 0 else Action(ActionType.FOLD)
        if put >= stack:
            return Action(ActionType.ALL_IN, stack)
        if to_call <= 0:
            put = max(put, min(view.bb, stack))
            if put >= stack:
                return Action(ActionType.ALL_IN, stack)
            return Action(ActionType.BET, put)
        if put < to_call:
            return Action(ActionType.CALL, min(to_call, stack))
        if put == to_call:
            return Action(ActionType.CALL, put)
        min_put = view.min_raise_to - view.bets[view.hero]
        if put < min_put:
            if stack >= min_put:
                put = min_put
            else:
                return Action(ActionType.ALL_IN, stack)
        if put >= stack:
            return Action(ActionType.ALL_IN, stack)
        return Action(ActionType.RAISE, put)
