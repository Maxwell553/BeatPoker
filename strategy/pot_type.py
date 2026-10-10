"""Single-raised vs 3bet/4bet pot classification and postflop libraries.

Solver grinders live in 3bet pots. SPR gates from a 100bb SRP are the wrong
tree: at SPR ~1.5–4, top pair and draws play for stacks.
"""

from __future__ import annotations

from typing import Optional

from poker.game import Action, ActionType, PlayerView, Street


def preflop_pot_bb(view: PlayerView) -> float:
    this_street = float(sum(view.bets))
    return max(view.pot - this_street, view.bb) / max(view.bb, 1)


def _preflop_raises(view: PlayerView) -> int:
    n = 0
    hist = view.action_history
    streets = getattr(view, "street_history", None) or []
    for i, item in enumerate(hist):
        act = item[1]
        if len(item) > 2:
            st = item[2]
        elif i < len(streets):
            st = streets[i]
        else:
            st = Street.PREFLOP
        if st != Street.PREFLOP:
            continue
        if act.type in (ActionType.RAISE, ActionType.BET):
            n += 1
        elif act.type == ActionType.ALL_IN and act.amount > view.bb:
            n += 1
    return n


def preflop_kind(view: PlayerView) -> str:
    """Pot type from preflop raises only.

    Replay of past streets must not use the current SPR gate: a river SPR
    of 1.5 turned an SRP flop cbet into a 4bet mix and made LibraryGTO's
    air disappear from the range.
    """
    if view.action_history:
        n_r = _preflop_raises(view)
        if n_r >= 3:
            return "4bet"
        if n_r >= 2:
            return "3bet"
        return "srp"
    p = preflop_pot_bb(view)
    if p >= 32:
        return "4bet"
    if p >= 11.0:
        return "3bet"
    return "srp"


def classify(view: PlayerView) -> str:
    """'srp' | '3bet' | '4bet' from preflop raise count, else pot/SPR."""
    spr = view.effective_stack / max(view.pot, 1)
    # Jammed pots play for stacks regardless of how they got here.
    if spr <= 1.55:
        return "4bet"
    if view.action_history:
        return preflop_kind(view)
    p = preflop_pot_bb(view)
    if p >= 32:
        return "4bet"
    if p >= 11.0 or (p >= 8.0 and spr <= 5.5):
        return "3bet"
    return "srp"


def decide_3bet_pot(
    view: PlayerView,
    eq: float,
    perc: float,
    cat: int,
    fd: bool,
    oesd: bool,
    ip: bool,
    rng,
    bluff_scale: float = 1.0,
    extra_villains: int = 0,
) -> Optional[Action]:
    """Heuristic 3bet/4bet flop/turn/river when search is not used.

    Returns None to fall through to the SRP policy.
    """
    kind = classify(view)
    if kind == "srp":
        return None
    pot = max(view.pot, 1)
    to_call = view.to_call
    stack = view.stacks[view.hero]
    spr = view.effective_stack / pot
    four = kind == "4bet"
    n_live = view.players_in_hand
    if extra_villains > 0:
        n_live = max(n_live, 2 + extra_villains)

    strong = cat >= 4 or perc > 0.88
    top_pairish = cat >= 1 and perc > 0.50
    draw = fd or oesd
    air = perc < 0.16 and cat == 0 and not draw

    if to_call > 0:
        frac = to_call / max(pot - to_call, 1)
        stack_off = False
        if four or spr < 2.2:
            stack_off = eq >= 0.38 or perc > 0.42 or top_pairish or draw or strong
        elif spr < 4.0:
            stack_off = eq >= 0.42 or perc > 0.52 or cat >= 2 or (top_pairish and perc > 0.58) or (draw and perc > 0.25)
        else:
            stack_off = perc > 0.70 or strong or (draw and spr < 6)
        if n_live >= 3:
            stack_off = stack_off and (perc > 0.62 or cat >= 2 or strong)
        if stack_off and stack > to_call and (four or spr < 2.8 or perc > 0.82 or strong):
            return Action(ActionType.ALL_IN, stack)
        if stack_off:
            amt = min(to_call, stack)
            if amt >= stack:
                return Action(ActionType.ALL_IN, stack)
            return Action(ActionType.CALL, amt)
        if eq + 0.03 >= to_call / (pot + to_call) and (top_pairish or draw or perc > 0.40):
            return Action(ActionType.CALL, min(to_call, stack))
        if frac > 1.1 and perc < 0.55 and cat < 2:
            return Action(ActionType.FOLD)
        return Action(ActionType.FOLD)

    # First-in: high-frequency small c-bet IP; more checks OOP; jam low SPR.
    cbet = 0.78 if ip else 0.52
    if view.street == Street.TURN:
        cbet = 0.62 if ip else 0.40
    if view.street == Street.RIVER:
        cbet = 0.55 if (strong or perc > 0.72) else (0.22 * bluff_scale if air else 0.08)
    if n_live >= 3:
        cbet *= 0.45
    cbet *= max(0.15, bluff_scale) if air else 1.0

    want = False
    if strong or top_pairish or (cat >= 2):
        want = True
    elif draw and view.street != Street.RIVER:
        want = rng.random() < min(0.85, cbet + 0.15)
    elif air:
        want = rng.random() < cbet * 0.55
    else:
        want = ip and view.street == Street.FLOP and rng.random() < cbet * 0.40

    if not want:
        return Action(ActionType.CHECK)

    if four or spr < 2.2:
        if eq >= 0.40 or strong or top_pairish or (air and spr < 1.4):
            return Action(ActionType.ALL_IN, stack)
    frac = 0.33 if not strong else (0.75 if spr < 4 else 0.55)
    if view.street == Street.RIVER and (strong or perc > 0.85):
        frac = 0.75 if spr > 2 else 1.0
    put = int(round(frac * pot))
    put = max(put, min(view.bb, stack))
    put = min(put, stack)
    if put >= stack:
        return Action(ActionType.ALL_IN, stack)
    return Action(ActionType.BET, put)
