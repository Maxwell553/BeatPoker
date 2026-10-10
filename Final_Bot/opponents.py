"""Opponent archetypes used to measure Final_Bot's win rate.

These are intentionally *not* GTO — they are the styles a strong bot must
crush, plus a tight-aggressive heuristic that approximates a competent human.
"""

from __future__ import annotations

import random
from typing import Optional

from poker.game import Action, ActionType, PlayerView
from strategy.hands import chen_score, hole_to_169
from strategy.preflop import _raise_to_put, rfi_percent
from strategy.ranges import Range, ranked_hands, top_percent_range


class _Base:
    name = "base"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def _sanitize(self, view: PlayerView, action: Action) -> Action:
        stack = view.stacks[view.hero]
        to_call = view.to_call
        if action.type == ActionType.FOLD and to_call <= 0:
            return Action(ActionType.CHECK)
        if action.type == ActionType.CHECK and to_call > 0:
            return Action(ActionType.CALL, min(to_call, stack))
        if action.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            put = min(action.amount, stack)
            if put <= 0:
                return Action(ActionType.CHECK) if to_call <= 0 else Action(ActionType.FOLD)
            if put >= stack:
                return Action(ActionType.ALL_IN, stack)
            if to_call <= 0:
                return Action(ActionType.BET, max(put, min(view.bb, stack)))
            if put < to_call:
                return Action(ActionType.CALL, min(to_call, stack))
            if put == to_call:
                return Action(ActionType.CALL, put)
            min_put = view.min_raise_to - view.bets[view.hero]
            if put < min_put:
                if stack > to_call:
                    return Action(ActionType.ALL_IN, stack) if stack < min_put else Action(ActionType.RAISE, min_put)
                return Action(ActionType.CALL, min(to_call, stack))
            return Action(ActionType.RAISE, put)
        if action.type == ActionType.CALL:
            if to_call <= 0:
                return Action(ActionType.CHECK)
            return Action(ActionType.CALL, min(to_call, stack))
        return action


class RandomBot(_Base):
    name = "Random"

    def act(self, view: PlayerView) -> Action:
        opts = []
        if view.to_call > 0:
            opts.append(Action(ActionType.FOLD))
            opts.append(Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero])))
            if view.stacks[view.hero] > view.to_call:
                opts.append(Action(ActionType.ALL_IN, view.stacks[view.hero]))
        else:
            opts.append(Action(ActionType.CHECK))
            if view.stacks[view.hero] > 0:
                opts.append(Action(ActionType.BET, min(view.pot, view.stacks[view.hero])))
                opts.append(Action(ActionType.ALL_IN, view.stacks[view.hero]))
        return self._sanitize(view, self.rng.choice(opts))


class CallingStation(_Base):
    name = "CallingStation"

    def act(self, view: PlayerView) -> Action:
        # Never folds; rarely raises; calls everything.
        if view.to_call > 0:
            return self._sanitize(view, Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero])))
        code = hole_to_169(*view.hole)
        if chen_score(code) >= 8 and self.rng.random() < 0.35:
            put = min(max(view.bb, view.pot // 2), view.stacks[view.hero])
            return self._sanitize(view, Action(ActionType.BET, put))
        return Action(ActionType.CHECK)


class Nit(_Base):
    name = "Nit"

    def act(self, view: PlayerView) -> Action:
        code = hole_to_169(*view.hole)
        premium = code in ("AA", "KK", "QQ", "AKs", "AKo")
        strong = premium or code in ("JJ", "TT", "AQs", "AQo")
        if view.to_call > 0:
            if premium:
                if view.stacks[view.hero] > view.to_call and self.rng.random() < 0.7:
                    return self._sanitize(view, Action(ActionType.RAISE, min(view.stacks[view.hero], view.to_call * 3)))
                return self._sanitize(view, Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero])))
            if strong and view.to_call <= view.pot:
                return self._sanitize(view, Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero])))
            return Action(ActionType.FOLD)
        if premium or (strong and view.street.name == "PREFLOP"):
            put = min(view.stacks[view.hero], max(view.bb * 3, view.pot // 2))
            return self._sanitize(view, Action(ActionType.BET, put))
        return Action(ActionType.CHECK) if view.to_call <= 0 else Action(ActionType.FOLD)


class TAG(_Base):
    """Tight-aggressive 'competent human' — opens a tighter-than-GTO range,
    c-bets too often, under-bluffs rivers, overfolds to big bets.
    Typical of a winning low/mid-stakes regular, not a solver specialist.
    """

    name = "TAG"

    def act(self, view: PlayerView) -> Action:
        code = hole_to_169(*view.hole)
        ch = chen_score(code)
        stack_bb = view.effective_stack / view.bb
        pos = view.position
        # Tighter than GTO opens
        if view.street.name == "PREFLOP" and (view.to_call == 0 or view.current_bet <= view.bb):
            pct = rfi_percent(pos, view.num_players, stack_bb) * 0.78
            rg = top_percent_range(ranked_hands(stack_bb), pct, stack_bb)
            if rg.contains_hole(*view.hole) > 0.5:
                target = 3.0 if pos == "SB" else 2.5
                put = int(target * view.bb) - view.bets[view.hero]
                return self._sanitize(view, Action(ActionType.RAISE if view.to_call > 0 else ActionType.BET, max(put, view.bb)))
            if view.to_call > 0:
                return Action(ActionType.FOLD)
            return Action(ActionType.CHECK)
        if view.street.name == "PREFLOP" and view.to_call > 0:
            if ch >= 10:  # JJ+ AK
                put = min(view.stacks[view.hero], view.to_call * 3)
                return self._sanitize(view, Action(ActionType.RAISE, put))
            if ch >= 7:
                return self._sanitize(view, Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero])))
            return Action(ActionType.FOLD)

        # Postflop
        if view.to_call > 0:
            # Overfold vs big bets
            frac = view.to_call / max(view.pot, 1)
            need = 7.5 if frac < 0.5 else 9.0
            if frac > 1.0:
                need = 11.0
            if ch >= need or (view.board and _board_pair(view)):
                return self._sanitize(view, Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero])))
            return Action(ActionType.FOLD)
        # C-bet too often with any piece / air IP
        if ch >= 6 or self.rng.random() < 0.55:
            put = min(view.stacks[view.hero], max(view.bb, int(0.66 * view.pot)))
            return self._sanitize(view, Action(ActionType.BET, put))
        return Action(ActionType.CHECK)


class LAG(_Base):
    name = "LAG"

    def act(self, view: PlayerView) -> Action:
        code = hole_to_169(*view.hole)
        ch = chen_score(code)
        if view.to_call > 0:
            if ch >= 5 or self.rng.random() < 0.35:
                if self.rng.random() < 0.22 and view.stacks[view.hero] > view.to_call:
                    return self._sanitize(
                        view, Action(ActionType.RAISE, min(view.stacks[view.hero], int(view.to_call * 2.8)))
                    )
                return self._sanitize(view, Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero])))
            return Action(ActionType.FOLD)
        if self.rng.random() < 0.55:
            put = min(view.stacks[view.hero], max(view.bb, int(0.75 * max(view.pot, view.bb))))
            return self._sanitize(view, Action(ActionType.BET, put))
        return Action(ActionType.CHECK) if view.to_call <= 0 else Action(ActionType.FOLD)


class Maniac(_Base):
    name = "Maniac"

    def act(self, view: PlayerView) -> Action:
        if self.rng.random() < 0.55 and view.stacks[view.hero] > 0:
            put = view.stacks[view.hero] if self.rng.random() < 0.25 else min(
                view.stacks[view.hero], max(view.bb, int(view.pot * 1.1) or view.bb * 3)
            )
            if view.to_call > 0:
                if put <= view.to_call:
                    return self._sanitize(view, Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero])))
                return self._sanitize(view, Action(ActionType.RAISE, put))
            return self._sanitize(view, Action(ActionType.BET, put))
        if view.to_call > 0:
            if self.rng.random() < 0.6:
                return self._sanitize(view, Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero])))
            return Action(ActionType.FOLD)
        return Action(ActionType.CHECK)


def _board_pair(view: PlayerView) -> bool:
    if not view.board:
        return False
    hr = {view.hole[0] % 13, view.hole[1] % 13}
    br = {c % 13 for c in view.board}
    return bool(hr & br)


class StrongReg(_Base):
    """Board-aware thinking regular: equity vs leftover deck (not vs a range),
    slightly tight preflop, under-bluffs rivers, overfolds to overbets.
    This is closer to a winning human than TAG, and still has classic leaks.
    """

    name = "StrongReg"

    def act(self, view: PlayerView) -> Action:
        from poker.equity import monte_carlo_equity, all_combos

        code = hole_to_169(*view.hole)
        ch = chen_score(code)
        stack_bb = view.effective_stack / view.bb
        pos = view.position
        if view.street.name == "PREFLOP":
            if view.to_call == 0 or view.current_bet <= view.bb:
                pct = rfi_percent(pos, view.num_players, stack_bb) * 0.90
                rg = top_percent_range(ranked_hands(stack_bb), pct, stack_bb)
                if rg.contains_hole(*view.hole) > 0.5:
                    target = 3.0 if pos == "SB" else 2.5
                    put = int(target * view.bb) - view.bets[view.hero]
                    return self._sanitize(
                        view,
                        Action(ActionType.RAISE if view.to_call > 0 else ActionType.BET, max(put, view.bb)),
                    )
                if view.can_check:
                    return Action(ActionType.CHECK)
                return Action(ActionType.FOLD)
            if ch >= 9.5:
                put = min(view.stacks[view.hero], max(view.to_call * 3, int(9 * view.bb)))
                return self._sanitize(view, Action(ActionType.RAISE, put))
            if ch >= 6.5:
                return self._sanitize(view, Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero])))
            if view.can_check:
                return Action(ActionType.CHECK)
            return Action(ActionType.FOLD)

        dead = list(view.hole) + list(view.board)
        pool = [c for c in all_combos() if c[0] not in dead and c[1] not in dead]
        eq = monte_carlo_equity(view.hole, pool, view.board, n_samples=80, rng=self.rng)
        if view.to_call > 0:
            odds = view.to_call / max(view.pot + view.to_call, 1)
            frac = view.to_call / max(view.pot, 1)
            # Overfold vs overbets — human leak
            if frac > 1.0 and eq < 0.42:
                return Action(ActionType.FOLD)
            if eq + 0.03 >= odds or eq > 0.38:
                if eq > 0.78 and view.stacks[view.hero] > view.to_call:
                    return self._sanitize(view, Action(ActionType.RAISE, min(view.stacks[view.hero], view.to_call * 3)))
                return self._sanitize(view, Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero])))
            return Action(ActionType.FOLD)
        # Under-bluff: bet for value, rarely as a bluff
        if eq > 0.62 or (eq > 0.55 and self.rng.random() < 0.6):
            put = min(view.stacks[view.hero], max(view.bb, int(0.66 * view.pot)))
            return self._sanitize(view, Action(ActionType.BET, put))
        if eq < 0.22 and self.rng.random() < 0.12:
            put = min(view.stacks[view.hero], max(view.bb, int(0.75 * view.pot)))
            return self._sanitize(view, Action(ActionType.BET, put))
        return Action(ActionType.CHECK)


class SolverPro(_Base):
    """Solver-trained professional proxy.

    Same preflop charts a studying pro would memorize, and the same
    GTO-structured flop/turn/river *heuristics* they apply from memory
    (range advantage, MDF, SPR). No live public-tree search. That gap —
    recalling a similar spot vs solving this exact one — is the computer's
    edge against a solver-trained human.
    """

    name = "SolverPro"

    def act(self, view: PlayerView) -> Action:
        from strategy.postflop import decide_postflop
        from strategy.preflop import decide_preflop
        from strategy.range_model import primary_villain_range, reconstruct

        if view.street.name == "PREFLOP":
            return self._sanitize(view, decide_preflop(view, self.rng))
        ranges = reconstruct(view)
        v_combos, v_w = primary_villain_range(view, ranges)
        h_combos, h_w = ranges[view.hero].combo_weights()
        extra = max(0, view.players_in_hand - 2)
        return self._sanitize(
            view,
            decide_postflop(view, v_combos, v_w, h_combos, h_w, self.rng, extra_villains=extra),
        )


class LibraryGTO(_Base):
    """Solver-trained proxy that is *not* this repo's heuristic SolverPro clone.

    Plays combo preflop trees plus the imported/distilled 6-max 100bb postflop
    library. No nested search, no HUD, no exploit layer. This is the floor a
    studying 200nl–2k grinder is aiming at.
    """

    name = "LibraryGTO"

    def act(self, view: PlayerView) -> Action:
        from strategy.library import library_policy
        from strategy.preflop import decide_preflop
        from strategy.range_model import reconstruct
        from strategy.ranges import Range

        if view.street.name == "PREFLOP":
            # 6-max: combo trees (solver charts). try_v3 is the live snapshot
            # — tree-30 BTN ~8% — not a 100bb pro open. HU v3 (tree 21) is.
            if view.num_players != 2:
                from strategy.preflop import _from_trees

                return self._sanitize(view, _from_trees(view, self.rng, None))
            return self._sanitize(view, decide_preflop(view, self.rng))
        ranges = reconstruct(view)
        hero_rg = ranges[view.hero] if ranges[view.hero].total() > 0 else Range.from_hole(*view.hole)
        a = library_policy(view, hero_rg, self.rng)
        if a is None:
            from strategy.postflop import decide_postflop
            from strategy.range_model import primary_villain_range

            v_c, v_w = primary_villain_range(view, ranges)
            h_c, h_w = hero_rg.combo_weights()
            a = decide_postflop(view, v_c, v_w, h_c, h_w, self.rng)
        return self._sanitize(view, a)


def make_opponent(name: str, seed: Optional[int] = None):
    name = name.lower()
    return {
        "random": RandomBot,
        "station": CallingStation,
        "callingstation": CallingStation,
        "nit": Nit,
        "tag": TAG,
        "lag": LAG,
        "maniac": Maniac,
        "reg": StrongReg,
        "strongreg": StrongReg,
        "solver": SolverPro,
        "solverpro": SolverPro,
        "pro": SolverPro,
        "library": LibraryGTO,
        "librarygto": LibraryGTO,
        "gtowizard": LibraryGTO,
    }[name](seed=seed)
