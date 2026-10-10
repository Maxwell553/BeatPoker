"""4-way / 5-way reduction to a 3-player nested subgame.

Heuristic "cut air" is not a pro-table policy. Multiway pots with 4+ players
collapse to hero plus the two most relevant live ranges:

  First-in: next-to-act, then the other live opponent. A live all-in stays
  as p2 (0 stack) so folding the actors is showdown, not a scoop of the jam.
  Facing: last aggressor, then the extra. A call leaves only that extra to act.
  A leftover all-in that is not p1/p2 still contests the main pot (ghost
  showdown) so raise-fold does not scoop the jam.

Everyone else is dead-money / blocker mass (their cards are masked out of
the two ranges), not a third strategic player and not a c-bet frequency cut.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from poker.game import ActionType, PlayerView
from strategy.ranges import Range

ComboW = Tuple[List[Tuple[int, int]], List[float]]


def last_aggressor_seat(view: PlayerView) -> Optional[int]:
    hero = view.hero
    for seat, act in reversed(view.action_history):
        if seat == hero or view.folded[seat]:
            continue
        if act.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            return seat
    return None


def _invested(view: PlayerView, seat: int) -> float:
    if hasattr(view, "committed"):
        return float(view.committed[seat])
    return float(view.bets[seat])


def _next_to_act_after(view: PlayerView, after: int) -> List[int]:
    n = view.num_players
    order = []
    for k in range(1, n):
        i = (after + k) % n
        if not view.folded[i] and not view.all_in[i] and i != view.hero:
            order.append(i)
    return order


def _first_in_relevance(view: PlayerView, seat: int) -> int:
    """Later IP can squeeze; BB is a bigger defender than SB.

    4-way CO: rest is SB/BB → BB. 5-way HJ: rest is BTN/SB/BB → BTN
    (always-BB used to drop the squeeze seat).
    """
    pos = getattr(view, "positions", None)
    name = str(pos[seat]) if pos is not None else ""
    return {"BTN": 6, "CO": 5, "HJ": 4, "UTG": 3, "BB": 2, "SB": 1}.get(name, 0)


def two_relevant_seats(view: PlayerView, ranges: Sequence[Range]) -> Tuple[int, int]:
    """Seats of the two opponents who define the 3-player subgame."""
    hero = view.hero
    all_in = getattr(view, "all_in", [False] * view.num_players)
    acting = [
        i
        for i in range(view.num_players)
        if i != hero and not view.folded[i] and not all_in[i] and ranges[i].total() > 0
    ]
    present = [
        i
        for i in range(view.num_players)
        if i != hero and not view.folded[i] and ranges[i].total() > 0
    ]
    if len(acting) == 1:
        allin_others = [i for i in present if i != acting[0]]
        # Facing a shove: keep the all-in range plus the one player who can act.
        # First-in used to drop the all-in as blockers, then HU leftover
        # scooped the main pot when the live villain folded. Keep them as
        # p2 with 0 stack; first_in_chain / villains_respond skip stk<=0.5.
        if allin_others:
            if view.to_call > 0:
                agg = last_aggressor_seat(view)
                if agg in allin_others:
                    return agg, acting[0]
                return allin_others[0], acting[0]
            ai = max(allin_others, key=lambda s: (_invested(view, s), ranges[s].total()))
            return acting[0], ai
        return acting[0], acting[0]
    if not acting:
        others = present or [i for i in range(view.num_players) if i != hero]
        a = others[0] if others else 0
        return a, a

    upcoming = _next_to_act_after(view, view.hero)
    upcoming = [s for s in upcoming if s in acting]

    # First-in: tree acts p1 then p2 after hero. That is next-to-act, then
    # the other live opponent — not last-aggressor (a preflop raiser) first.
    # A live all-in cannot act, but dropping them made both actors folding
    # scoop the jammed main pot. Keep them as p2 (0 stack); other actors
    # are blocker mass. Same as the one-actor + all-in case above.
    if view.to_call <= 0 and upcoming:
        a = upcoming[0]
        allin_others = [i for i in present if all_in[i]]
        if allin_others:
            ai = max(allin_others, key=lambda s: (_invested(view, s), ranges[s].total()))
            return a, ai
        rest = [s for s in acting if s != a]
        if not rest:
            return a, a
        return a, max(rest, key=lambda s: _first_in_relevance(view, s))

    agg = last_aggressor_seat(view)
    if agg is None or agg not in present:
        agg = max(acting, key=lambda s: (_invested(view, s), ranges[s].total()))

    rest = [s for s in acting if s != agg]
    if not rest:
        return agg, agg

    upcoming = [s for s in upcoming if s in rest]
    if upcoming:
        # Tree on a call leaves p2 to act next. Default is the next seat,
        # not the widest remaining range (BTN vs CO cbet used to skip SB
        # and keep BB, so the squeeze was only blocker mass).
        # A call leaves only p2 to act. That must be the next seat, not
        # BTN-as-squeeze: HJ vs UTG, CO acts first. Jumping to BTN BR'd vs
        # a shove CO makes and a fold BTN never faces. BTN vs CO still
        # keeps SB (next_rel 1), not BB — that was the skip-SB leak.
        # First-in still uses BTN as the second extra (p1 then p2 after hero).
        b = upcoming[0]
    else:
        b = max(rest, key=lambda s: ranges[s].total())
    return agg, b


def first_in_after_hero(view: PlayerView, v1_seat: int, v2_seat: int) -> Tuple[int, ...]:
    """Tree players who still act if hero checks (first-in).

    Anyone before hero in flop order already checked; they do not get a
    second first-in donk. BTN after both blinds check: empty (check closes).
    """
    mapping = {int(view.hero): 0}
    mapping[int(v1_seat)] = 1
    if int(v2_seat) != int(v1_seat):
        mapping[int(v2_seat)] = 2
    n = view.num_players
    all_in = getattr(view, "all_in", [False] * n)
    order: List[int] = []
    for k in range(1, n + 1):
        i = (view.button + k) % n
        if i not in mapping or view.folded[i] or all_in[i]:
            continue
        p = mapping[i]
        if p not in order:
            order.append(p)
    if 0 not in order:
        return (1, 2)
    return tuple(order[order.index(0) + 1 :])


def apply_blocker_mass(
    combos: Sequence[Tuple[int, int]],
    weights: Sequence[float],
    other_ranges: Sequence[Range],
) -> ComboW:
    """Downweight combos that other live (non-searched) ranges hold.

    Those players are blockers / dead money, not a fourth strategic actor.
    """
    if not other_ranges or not combos:
        return list(combos), [float(w) for w in weights]
    mass = {}
    for rg in other_ranges:
        if rg is None or rg.total() <= 0:
            continue
        cs, ws = rg.combo_weights()
        tot = max(sum(ws), 1e-12)
        for c, w in zip(cs, ws):
            key = (min(c[0], c[1]), max(c[0], c[1]))
            mass[key] = mass.get(key, 0.0) + float(w) / tot
    out_c = []
    out_w = []
    for c, w in zip(combos, weights):
        key = (min(c[0], c[1]), max(c[0], c[1]))
        scale = max(0.0, 1.0 - mass.get(key, 0.0))
        nw = float(w) * scale
        if nw > 1e-15:
            out_c.append(key)
            out_w.append(nw)
    if not out_c:
        return list(combos), [float(w) for w in weights]
    return out_c, out_w


def ghost_allin_combos(
    view: PlayerView,
    ranges: Sequence[Range],
    v1_seat: int,
    v2_seat: int,
) -> Optional[ComboW]:
    """All-in not in the 3-way tree. They still contest the main pot.

    Facing a cbet with a leftover jam: p1/p2 stay the chip actors so a call
    does not close. Folding both of them used to scoop that jam. Return the
    biggest leftover all-in's range for ghost showdown.
    """
    all_in = getattr(view, "all_in", [False] * view.num_players)
    seats = [
        i
        for i in range(view.num_players)
        if i != view.hero
        and i != int(v1_seat)
        and i != int(v2_seat)
        and not view.folded[i]
        and all_in[i]
        and i < len(ranges)
        and ranges[i].total() > 0
    ]
    if not seats:
        return None
    seat = max(seats, key=lambda s: (_invested(view, s), ranges[s].total()))
    return ranges[seat].combo_weights()


def two_relevant_ranges(
    view: PlayerView,
    ranges: Sequence[Range],
) -> Tuple[ComboW, ComboW, List[int]]:
    """Return (primary, secondary, other_live_seats).

    `other_live_seats` are masked as blockers, not searched.
    """
    a, b = two_relevant_seats(view, ranges)
    others = [
        i
        for i in range(view.num_players)
        if i != view.hero and i != a and i != b and not view.folded[i]
    ]
    primary = ranges[a].combo_weights()
    secondary = ranges[b].combo_weights()
    return primary, secondary, others


def reduction_note(n_live: int) -> str:
    if n_live <= 3:
        return "full nested tree"
    return (
        f"{n_live}-way reduced to hero + next-to-act + most relevant remaining; "
        "other live ranges are blockers only"
    )
