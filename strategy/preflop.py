"""Combo-level preflop policy for 2–9 players and 50–1000bb stacks.

The Nash floor is the combo trees. v3 MCCFR is table policy only when
Final_Bot/data/v3_gate.json allow_preflop is true (eval+gate loop).
"""

from __future__ import annotations

from typing import Optional

from poker.game import Action, ActionType, PlayerView
from strategy.hands import hole_to_169
from strategy.preflop_trees import (
    apply_blockers,
    in_range,
    rfi_percent,
    rfi_range,
    vs_3bet_ranges,
    vs_4bet_ranges,
    vs_open_action_ranges,
    vs_open_ranges,
)
from strategy.rake import RakeConfig
from strategy.stacks import open_size_bb, stack_bucket


def _n_raises(view: PlayerView) -> int:
    n = 0
    for _, act in view.action_history:
        if act.type in (ActionType.RAISE, ActionType.BET):
            n += 1
        elif act.type == ActionType.ALL_IN and act.amount > view.bb:
            n += 1
    return n


def _opener_pos(view: PlayerView) -> Optional[str]:
    for seat, act in view.action_history:
        if act.type in (ActionType.RAISE, ActionType.BET, ActionType.ALL_IN):
            return view.positions[seat]
    return None


def _last_aggressor_pos(view: PlayerView) -> Optional[str]:
    pos = None
    for seat, act in view.action_history:
        if act.type in (ActionType.RAISE, ActionType.BET, ActionType.ALL_IN):
            pos = view.positions[seat]
    return pos


def _facing_limp_only(view: PlayerView) -> bool:
    if view.current_bet > view.bb:
        return False
    for _, act in view.action_history:
        if act.type in (ActionType.RAISE, ActionType.BET, ActionType.ALL_IN):
            return False
        if act.type == ActionType.CALL:
            return True
    return False


def _raise_to_put(view: PlayerView, raise_to_bb: float) -> int:
    target = int(round(raise_to_bb * view.bb))
    target = max(target, view.min_raise_to)
    target = min(target, view.max_raise_to)
    put = target - view.bets[view.hero]
    return max(put, 0)


def _bet_put(view: PlayerView, put: int) -> Action:
    put = min(put, view.stacks[view.hero])
    if put <= 0:
        if view.can_check:
            return Action(ActionType.CHECK)
        return Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero]))
    if put >= view.stacks[view.hero]:
        return Action(ActionType.ALL_IN, view.stacks[view.hero])
    if view.to_call <= 0:
        return Action(ActionType.BET, put)
    return Action(ActionType.RAISE, put)


def _all_in(view: PlayerView) -> Action:
    return Action(ActionType.ALL_IN, view.stacks[view.hero])


def _from_trees(view: PlayerView, rng, rake: RakeConfig | None) -> Action:
    eff_bb = view.effective_stack / view.bb
    pos = view.position
    n = view.num_players
    hole = view.hole
    to_call = view.to_call
    can_check = view.can_check
    bucket = stack_bucket(eff_bb)
    n_raises = _n_raises(view)

    if to_call == 0 or (to_call > 0 and view.current_bet <= view.bb and n_raises == 0):
        if can_check and pos == "BB" and view.current_bet <= view.bb:
            if _facing_limp_only(view):
                iso = rfi_range("CO", n, eff_bb, rake=rake)
                if in_range(iso, hole, rng):
                    return _bet_put(view, _raise_to_put(view, 4.0 if n >= 3 else 3.5))
                return Action(ActionType.CHECK)
            return Action(ActionType.CHECK)

        rg = rfi_range(pos, n, eff_bb, rake=rake)
        if in_range(rg, hole, rng):
            size = open_size_bb(pos, n, eff_bb)
            return _bet_put(view, _raise_to_put(view, size))
        if to_call > 0 and view.current_bet <= view.bb:
            if pos == "SB":
                pct = 0.50
                if rake is not None and rake.active:
                    pct *= rake.defend_mult()
                complete = rfi_range("BTN", n, eff_bb, rake=rake)
                # Completing is wider than RFI; mix in a slice of the BTN range.
                if complete.contains_hole(*hole) > 0.25 or in_range(complete, hole, rng, thresh=0.35):
                    return Action(ActionType.CALL, to_call)
            return Action(ActionType.FOLD) if to_call > 0 else Action(ActionType.CHECK)
        if can_check:
            return Action(ActionType.CHECK)
        return Action(ActionType.FOLD)

    opener = _opener_pos(view) or "BTN"

    if n_raises == 1:
        jam, three, call = vs_open_action_ranges(pos, opener, n, eff_bb, rake=rake)
        three = apply_blockers(three, hole, "3bet_bluff")
        jam = apply_blockers(jam, hole, "value_4bet")
        if in_range(jam, hole, rng, thresh=0.45):
            return _all_in(view)
        if in_range(three, hole, rng):
            size_to = 9.0 if pos in ("BTN", "CO", "BB") else 11.0
            if opener in ("UTG", "UTG1", "MP"):
                size_to += 1.0
            if pos == "SB":
                size_to = 12.0
            if bucket == "50":
                size_to = 7.5 if pos in ("BTN", "BB") else 8.5
            return _bet_put(view, _raise_to_put(view, size_to))
        if in_range(call, hole, rng):
            if to_call >= view.stacks[view.hero]:
                return Action(ActionType.CALL, view.stacks[view.hero])
            return Action(ActionType.CALL, to_call)
        if can_check:
            return Action(ActionType.CHECK)
        return Action(ActionType.FOLD)

    if n_raises == 2:
        threebettor = _last_aggressor_pos(view) or opener
        four, call = vs_3bet_ranges(pos, threebettor, eff_bb, rake=rake)
        four = apply_blockers(four, hole, "value_4bet")
        if in_range(four, hole, rng, thresh=0.45):
            if bucket == "50" or eff_bb <= 70:
                return _all_in(view)
            return _bet_put(view, _raise_to_put(view, 22.0 if pos != "SB" else 25.0))
        if in_range(call, hole, rng, thresh=0.45):
            return Action(ActionType.CALL, min(to_call, view.stacks[view.hero]))
        return Action(ActionType.FOLD) if to_call > 0 else Action(ActionType.CHECK)

    jam, call = vs_4bet_ranges(eff_bb)
    jam = apply_blockers(jam, hole, "value_4bet")
    if in_range(jam, hole, rng, thresh=0.4):
        return _all_in(view)
    if in_range(call, hole, rng, thresh=0.4):
        return Action(ActionType.CALL, min(to_call, view.stacks[view.hero]))
    if can_check:
        return Action(ActionType.CHECK)
    return Action(ActionType.FOLD)


def decide_preflop(view: PlayerView, rng, rake: RakeConfig | None = None) -> Action:
    try:
        from strategy.blueprint import try_v3_preflop

        a = try_v3_preflop(view, rng)
        if a is not None:
            return a
    except Exception:
        pass
    return _from_trees(view, rng, rake)


# Re-exports used by range_model / opponents
__all__ = [
    "decide_preflop",
    "rfi_range",
    "rfi_percent",
    "vs_open_ranges",
    "vs_3bet_ranges",
    "vs_4bet_ranges",
    "_raise_to_put",
    "hole_to_169",
]
