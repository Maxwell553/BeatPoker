"""Flop/turn (and river fallback) policy.

GTO-structured heuristics used by modern solvers, evaluated with exact /
Monte Carlo equity against tracked ranges:

* Range advantage → high-frequency small bets (especially IP, dry boards).
* Nut advantage → larger, more polarized sizes.
* MDF when facing bets; continue with the top of our range plus draws.
* SPR gates commitment (don't stack off top pair at 1000bb).
* Multiway: cut bluffs, thicken value, demand more equity.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from poker.equity import combo_index, equity_vs_range, hand_strength_percentile, range_vs_range_equity
from poker.evaluator import category, evaluate
from poker.game import Action, ActionType, PlayerView, Street
from strategy.multiway import policy as mw_policy
from strategy.rake import RakeConfig
from strategy.stacks import stack_bucket
from strategy.texture import analyze


def _pot_odds(to_call: int, pot: int) -> float:
    if to_call <= 0:
        return 0.0
    return to_call / (pot + to_call)


def _mdf(to_call: int, pot: int) -> float:
    """Minimum defense frequency vs a bet of `to_call` into `pot` (pot already includes the bet)."""
    if to_call <= 0:
        return 1.0
    # bet size as fraction of pot-before-bet
    pot_before = pot - to_call
    if pot_before <= 0:
        pot_before = max(pot, 1)
        bet = to_call
    else:
        bet = to_call
    # If view.pot includes the opponent's bet, pot_before = pot - last_bet.
    return 1.0 - bet / (pot_before + bet) if (pot_before + bet) else 0.5


def geometric_fraction(spr: float, streets_left: int) -> float:
    """Bet fraction of pot that gets stacks in over remaining streets."""
    if streets_left <= 0:
        return min(1.5, max(0.33, spr))
    # Want (1+f)^streets * pot ≈ pot + stack = pot*(1+spr)
    # (1+f)^n ≈ 1+spr  => f ≈ (1+spr)^(1/n) - 1
    f = (1.0 + max(spr, 0.1)) ** (1.0 / streets_left) - 1.0
    return float(min(2.0, max(0.25, f)))


def _draw_outs(hole: Sequence[int], board: Sequence[int]) -> Tuple[int, bool, bool]:
    """Very cheap out counter: flush draw, OESD/gutshot. Returns (outs, flush_draw, oesd)."""
    from poker.cards import rank_of, suit_of

    suits = [suit_of(c) for c in list(hole) + list(board)]
    hole_suits = [suit_of(c) for c in hole]
    board_suits = [suit_of(c) for c in board]
    flush_draw = False
    outs = 0
    for s in range(4):
        hs = hole_suits.count(s)
        bs = board_suits.count(s)
        if hs >= 1 and hs + bs == 4 and len(board) < 5:
            flush_draw = True
            outs += 9
            break
    ranks = sorted(set(rank_of(c) for c in list(hole) + list(board)))
    bits = 0
    for r in ranks:
        bits |= 1 << r
    if 12 in ranks:
        bits |= 1 << 0  # don't break wheel check below
    oesd = False
    gut = False
    # 4-rank windows
    for start in range(0, 9):
        window = [(start + k) % 13 for k in range(4)]
        have = sum(1 for r in window if r in ranks)
        # This is crude; check 4 consecutive
    # consecutive unique ranks
    uniq = sorted(set(rank_of(c) for c in list(hole) + list(board)))
    # include wheel A
    if 12 in uniq:
        uniq = sorted(set(uniq) | { -1 })
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            span = uniq[j] - uniq[i]
            cnt = j - i + 1
            if cnt == 4 and span == 3:
                oesd = True
            if cnt == 4 and span == 4:
                gut = True
            if cnt == 3 and span == 4:
                gut = True
    if oesd:
        outs += 8
    elif gut:
        outs += 4
    return min(outs, 15), flush_draw, oesd


def _made_cat(hole: Sequence[int], board: Sequence[int]) -> int:
    if len(board) < 3:
        return 0
    return category(evaluate(list(hole) + list(board)))


def _commitment_threshold(spr: float, n_players: int) -> float:
    """Minimum equity to stack off (approx). Higher multiway and deeper."""
    base = 0.38 + 0.10 * min(spr, 16) / 16.0
    if spr > 8:
        base += 0.08
    if spr > 20:
        base += 0.12
    if spr > 50:
        base += 0.10
    if n_players >= 3:
        base += 0.08
    if n_players >= 4:
        base += 0.06
    return min(0.92, base)


def decide_postflop(
    view: PlayerView,
    vill_combos: Sequence[Tuple[int, int]],
    vill_weights: Sequence[float],
    hero_combos: Sequence[Tuple[int, int]],
    hero_weights: Sequence[float],
    rng,
    extra_villains: int = 0,
    rake: RakeConfig | None = None,
    extra_villain_ranges=None,
    river_defend_mult: float = 0.97,
    bluff_scale: float = 1.0,
    value_adj: float = 0.0,
) -> Action:
    hole = view.hole
    board = view.board
    pot = max(view.pot, 1)
    to_call = view.to_call
    stack = view.stacks[view.hero]
    spr = view.effective_stack / pot
    tex = analyze(board)
    n_live = view.players_in_hand
    streets_left = {Street.FLOP: 2, Street.TURN: 1, Street.RIVER: 0}.get(view.street, 1)
    bucket = stack_bucket(view.effective_stack / max(view.bb, 1))
    eq = equity_vs_range(
        hole,
        vill_combos,
        board,
        n_samples=120 if len(board) < 5 else 40,
        rng=rng,
        villain_weights=vill_weights,
        extra_villains=extra_villains,
        extra_villain_ranges=extra_villain_ranges,
    )
    perc = hand_strength_percentile(hole, hero_combos, board, hero_weights) if len(board) >= 3 else 0.5
    try:
        range_eq = range_vs_range_equity(
            hero_combos, vill_combos, board, n_samples=60, rng=rng, wa=hero_weights, wb=vill_weights
        )
    except Exception:
        range_eq = 0.5
    range_adv = range_eq - 0.5  # positive => we have range advantage
    outs, fd, oesd = _draw_outs(hole, board)
    draw_eq = outs / (2 * (15 if view.street == Street.FLOP else 8) + 1e-9) if view.street != Street.RIVER else 0.0
    # rough additional equity from draws already in MC; still useful as a flag
    cat = _made_cat(hole, board)
    strong_made = cat >= 6  # boat+
    very_strong = cat >= 4 or (cat >= 2 and perc > 0.85)  # two pair+ / strong
    top_pairish = cat >= 1 and perc > 0.55
    air = perc < 0.18 and cat == 0 and outs <= 4

    # Nut advantage proxy: our range's top 10% vs theirs — use perc of nuts-ish
    nut_adv = 0.0
    if perc > 0.92:
        nut_adv += 0.4
    if cat >= 7:
        nut_adv += 0.5
    if tex.flush_possible and cat >= 5:
        nut_adv += 0.2
    if tex.wetness > 0.5:
        nut_adv += 0.15 * max(0, perc - 0.7)

    ip = not _oop(view)
    from strategy.library import library_policy
    from strategy.pot_type import decide_3bet_pot
    from strategy.ranges import Range as _Range

    hero_rg = _Range.from_combos(hero_combos) if hero_combos else _Range.from_hole(*hole)
    if hero_weights and hero_combos:
        hero_rg = _Range()
        for c, w in zip(hero_combos, hero_weights):
            hero_rg.w[combo_index(*c)] = float(w)
    lib_a = library_policy(view, hero_rg, rng)
    if lib_a is not None and extra_villains == 0:
        return lib_a

    threebet_a = decide_3bet_pot(
        view, eq, perc, cat, fd, oesd, ip, rng,
        bluff_scale=bluff_scale, extra_villains=extra_villains,
    )
    if threebet_a is not None:
        return threebet_a
    mw = mw_policy(n_live, ip, view.street, facing=to_call > 0)
    # --- Facing a bet ---
    if to_call > 0:
        pot_odds = _pot_odds(to_call, pot)
        mdf = _mdf(to_call, pot)
        bet_frac = to_call / max(pot - to_call, 1)
        # Implied odds: remaining stack * chance to win a big pot with a draw
        implied = 0.0
        if view.street != Street.RIVER and outs >= 8:
            implied = min(0.12, 0.4 * view.effective_stack / (pot + to_call) * (outs / 20.0))
            if rake is not None and rake.active:
                implied *= 0.75  # rake eats implied-odds EV on later streets
            if bucket == "deep":
                implied *= 1.25
            if bucket == "50":
                implied *= 0.55
        if rake is not None and rake.active:
            needed = rake.equity_needed_to_call(to_call, pot, view.bb) - implied
        else:
            needed = pot_odds - implied
        needed += mw.extra_equity

        # Raise for value
        want_raise = False
        if perc > mw.raise_perc or strong_made:
            want_raise = True
        if cat >= 5 and perc > 0.75:  # flush+
            want_raise = True
        if view.street != Street.RIVER and oesd and fd and perc > 0.3 and n_live == 2 and spr < 6:
            want_raise = rng.random() < 0.35  # combo-draw raise
        # Don't raise thin when deep
        if (spr > 15 or bucket == "deep") and cat < 4 and perc < 0.93:
            want_raise = False
        if n_live >= 3 and perc < mw.raise_perc and cat < 6:
            want_raise = False
        if bucket == "50" and top_pairish and spr < 4:
            want_raise = want_raise or perc > 0.70

        if want_raise and stack > to_call:
            size = _raise_size(view, bet_frac, spr, tex, nut_adv)
            return size

        # Call / fold
        continue_ok = eq + 0.02 >= needed
        if perc < mw.min_continue_perc and not (fd or oesd) and not very_strong:
            continue_ok = False
        if view.street == Street.RIVER:
            defend = mdf * river_defend_mult
            if perc >= 1.0 - defend and eq >= needed - 0.04:
                continue_ok = True
            if perc < 1.0 - defend - 0.04 and not very_strong:
                continue_ok = False
            # Overbet: still defend the top of range; don't panic-fold second pair
            if bet_frac > 1.1 and perc < 0.62 and cat < 4:
                continue_ok = False
        else:
            # Flop/turn: continue with draws and made hands
            if fd or oesd:
                continue_ok = True
            if top_pairish:
                continue_ok = True
            if eq < needed - 0.12 and perc < 0.25 and outs < 6:
                continue_ok = False
            # Don't barrel-call off with weak pairs at huge SPR
            if spr > 20 and cat <= 1 and perc < 0.6 and outs < 8 and bet_frac > 0.6:
                continue_ok = False

        # Deep single-raised pots: one-pair does not play for 100bb+.
        # 50bb / low-SPR 3bet pots stack off top pair.
        committed_bb = (view.committed[view.hero] + to_call) / max(view.bb, 1)
        if bucket == "deep" or (bucket == "200" and spr >= 8.0):
            if committed_bb > 40 and cat <= 1 and perc < 0.90 and not strong_made:
                if not (fd or oesd) or committed_bb > 80:
                    continue_ok = False
            if committed_bb > 100 and cat < 4 and perc < 0.94:
                continue_ok = False
            if committed_bb > 200 and cat < 6 and perc < 0.97:
                continue_ok = False
        elif bucket == "50" or spr < 2.8:
            if eq >= 0.38 or perc > 0.45 or very_strong or fd or oesd or top_pairish:
                continue_ok = True
            if eq < 0.28 and perc < 0.30 and not (fd or oesd) and not very_strong:
                continue_ok = False
        elif spr >= 5.0:
            if committed_bb > 40 and cat <= 1 and perc < 0.90 and not strong_made:
                if not (fd or oesd) or committed_bb > 80:
                    continue_ok = False
            if committed_bb > 100 and cat < 4 and perc < 0.94:
                continue_ok = False

        if continue_ok:
            amt = min(to_call, stack)
            if amt >= stack:
                # stacking off: check commitment
                if eq >= _commitment_threshold(spr, n_live) or strong_made or perc > 0.9:
                    return Action(ActionType.ALL_IN, stack)
                if eq < _commitment_threshold(spr, n_live) - 0.08 and perc < 0.7:
                    return Action(ActionType.FOLD)
                return Action(ActionType.ALL_IN, stack)
            return Action(ActionType.CALL, amt)
        return Action(ActionType.FOLD)

    # --- We can check or bet ---
    mw = mw_policy(n_live, ip, view.street, facing=False)
    cbet_freq = mw.cbet_freq
    cbet_freq += 0.55 * max(-0.15, min(0.25, range_adv))
    cbet_freq -= 0.22 * tex.wetness
    if view.street == Street.RIVER:
        cbet_freq = 0.28 + 0.8 * max(0, perc - 0.55)
        if air:
            cbet_freq = (0.22 + 0.15 * (1.0 if tex.wetness < 0.35 else 0.0)) * mw.bluff_mult
        if rake is not None and rake.active:
            cbet_freq *= rake.bluff_mult()
    cbet_freq *= bluff_scale
    cbet_freq = max(0.04, min(0.92, cbet_freq))

    value_cut = mw.value_perc + value_adj
    value = perc > value_cut or cat >= 4 or (cat >= 2 and perc > 0.6)
    if (bucket == "deep" or spr > 25) and cat == 1:
        value = perc > 0.85
        thin_value = perc > 0.55
    else:
        thin_value = perc > max(0.50, value_cut - 0.14) and cat >= 1
    if bucket == "50" and top_pairish:
        value = True

    bluff = (air or (perc < 0.22 and (fd or oesd or view.street == Street.RIVER))) and mw.air_bluff
    from poker.cards import rank_of, suit_of

    ace_blocker = any(rank_of(c) == 12 for c in hole) and any(rank_of(c) == 12 for c in board)
    if ace_blocker and perc < 0.35 and mw.bluff_mult > 0.15:
        bluff = True
    flush_suit = _flush_suit(board)
    if flush_suit is not None:
        nf_blocker = any(rank_of(c) == 12 and suit_of(c) == flush_suit for c in hole)
        if nf_blocker and cat < 5 and perc < 0.55 and mw.bluff_mult > 0.15:
            bluff = True
            if view.street == Street.RIVER:
                cbet_freq = max(cbet_freq, 0.50 * mw.bluff_mult)

    want_bet = False
    if value:
        want_bet = True
    elif thin_value and view.street != Street.RIVER:
        want_bet = rng.random() < min(0.85, cbet_freq + 0.2)
    elif bluff:
        bf = cbet_freq * (0.55 if view.street == Street.RIVER else 0.7) * mw.bluff_mult * bluff_scale
        if rake is not None and rake.active:
            bf *= rake.bluff_mult()
        want_bet = rng.random() < bf
    else:
        if ip and view.street == Street.FLOP and n_live == 2 and rng.random() < cbet_freq * 0.35:
            want_bet = True

    if not want_bet:
        return Action(ActionType.CHECK)

    size_frac = _bet_frac(view, tex, range_adv, nut_adv, spr, value, bluff, streets_left)
    put = int(round(size_frac * pot))
    put = max(put, min(view.bb, stack))
    put = min(put, stack)
    if put >= stack:
        # Only jam if commitment threshold met or as a polar river shove
        if eq >= _commitment_threshold(spr, n_live) or strong_made or (bluff and spr < 1.2):
            return Action(ActionType.ALL_IN, stack)
        put = min(int(0.75 * pot), stack - 1)
        put = max(put, min(view.bb, stack))
        if put >= stack:
            return Action(ActionType.CHECK)
    return Action(ActionType.BET, put)


def _oop(view: PlayerView) -> bool:
    n = view.num_players
    for k in range(1, n + 1):
        i = (view.button + k) % n
        if not view.folded[i] and not view.all_in[i]:
            return i == view.hero
    return True


def _flush_suit(board: Sequence[int]) -> Optional[int]:
    if len(board) < 3:
        return None
    from poker.cards import suit_of

    counts = [0, 0, 0, 0]
    for c in board:
        counts[suit_of(c)] += 1
    s = max(range(4), key=lambda i: counts[i])
    return s if counts[s] >= 3 else None


def _bet_frac(view, tex, range_adv, nut_adv, spr, value, bluff, streets_left) -> float:
    if view.street == Street.RIVER:
        if value and nut_adv > 0.3:
            return 0.9 + 0.5 * min(1.0, nut_adv)
        if value:
            return 0.55 if tex.wetness < 0.4 else 0.75
        if bluff:
            return 0.75 if tex.wetness < 0.5 else 1.0
        return 0.33
    # Flop/turn
    geo = geometric_fraction(spr, max(1, streets_left))
    if tex.wetness < 0.25 and range_adv > 0.05:
        frac = 0.33
    elif nut_adv > 0.3:
        frac = 0.75
    elif tex.wetness > 0.55:
        # Wet without nut advantage is merged, not polar. Polarizing every
        # wet barrel is the main leak vs thinking regs.
        frac = 0.55
    else:
        frac = 0.55
    # Deep: smaller, pot-control
    if spr > 20:
        frac = min(frac, 0.45)
    if spr > 50:
        frac = min(frac, 0.33)
    # Geometric when we want to get money in (strong)
    if value and spr < 8:
        frac = max(frac, min(1.1, geo))
    return float(min(1.5, max(0.25, frac)))


def _raise_size(view: PlayerView, bet_frac: float, spr: float, tex, nut_adv: float) -> Action:
    stack = view.stacks[view.hero]
    to_call = view.to_call
    # Raise to ~3x the bet, or more if wet / nutty
    last_bet = to_call  # approximate
    mult = 2.8
    if tex.wetness > 0.5:
        mult = 3.2
    if nut_adv > 0.4:
        mult = 3.4
    target_put = to_call + int(mult * last_bet)
    # raise-to is hero_bet + put; engine wants chips put in
    put = target_put
    put = min(put, stack)
    if put <= to_call:
        return Action(ActionType.CALL, min(to_call, stack))
    if put >= stack or spr < 2.2:
        if spr < 3.5:
            return Action(ActionType.ALL_IN, stack)
    return Action(ActionType.RAISE, put)
