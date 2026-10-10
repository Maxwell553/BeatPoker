"""Reconstruct everyone's ranges from the public action history.

Villain ranges start from the combo-level trees, then scale to HUD frequencies
or the population prior. Not a padded copy of our own GTO range.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from poker.equity import combo_from_index, combo_index
from poker.game import ActionType, PlayerView, Street
from strategy.hud import HUD
from strategy.population import Population
from strategy.preflop import rfi_range, vs_3bet_ranges, vs_4bet_ranges
from strategy.preflop_trees import apply_blockers_range, sequential_mix_weights
from strategy.ranges import Range, ranked_hands, top_percent_range
from strategy.strength import board_combo_score


def table_vpip_prior(n: int) -> float:
    """Match this repo's combo trees, not generic online HU GTO (~50%)."""
    if n <= 2:
        return 0.32
    if n >= 7:
        return 0.18
    return 0.24


def starting_ranges(n: int) -> List[Range]:
    return [Range.uniform() for _ in range(n)]


def apply_dead(ranges: List[Range], dead: Sequence[int]) -> List[Range]:
    return [r.mask_dead(dead) for r in ranges]


def _scale_to_freq(rg: Range, target_frac: float, stack_bb: float, widen: bool) -> Range:
    """Stretch or squeeze a tree range toward a population/HUD frequency."""
    cur = rg.frac_of_full()
    if cur <= 1e-9:
        return rg
    if abs(target_frac - cur) < 0.02:
        return rg
    if target_frac > cur:
        extra = top_percent_range(ranked_hands(stack_bb), min(0.92, target_frac), stack_bb)
        out = rg.copy()
        mix = min(1.0, (target_frac - cur) / max(0.05, 1.0 - cur))
        out.w = np.maximum(out.w, extra.w * (0.35 + 0.65 * mix))
        return out
    if not widen:
        # Tighter than the tree: keep the strongest slice
        keep = max(0.35, target_frac / cur)
        idxs = np.argsort(-rg.w)
        tot = rg.w.sum()
        acc = 0.0
        out = Range()
        for i in idxs:
            if rg.w[i] <= 0:
                break
            out.w[i] = rg.w[i]
            acc += rg.w[i]
            if acc >= keep * tot:
                break
        return out
    return rg


def reconstruct(
    view: PlayerView,
    hud: Optional[HUD] = None,
    population: Optional[Population] = None,
) -> List[Range]:
    """Replay preflop actions to get current ranges."""
    n = view.num_players
    dead = list(view.hole) + list(view.board)
    # Tree replay is the Nash-floor range. Blueprint reach is fold/jam-collapsed
    # and made search think villains are polar vs every opponent type.
    ranges = starting_ranges(n)
    ranges = apply_dead(ranges, dead)

    stack_bb = view.effective_stack / max(view.bb, 1)
    start_bb = max(stack_bb, (view.stacks[view.hero] + view.committed[view.hero]) / view.bb)
    pop = population or Population.unknown()

    n_raises = 0
    opener_pos = None
    threebettor_pos = None

    def _hu_v3(node: int, acts) -> Optional[Range]:
        if n != 2:
            return None
        try:
            from strategy.blueprint import v3_hu_action_range

            rg = v3_hu_action_range(node, acts)
            return rg.mask_dead(dead) if rg is not None and rg.total() > 0 else None
        except Exception:
            return None

    def _six_v3(pos: str, acts, *, rfi: bool = False, vs_ep: bool = False, fill: Optional[Range] = None) -> Optional[Range]:
        if n != 6:
            return None
        try:
            from strategy.blueprint import _CFR_SEAT, v3_6max_action_range

            st = _CFR_SEAT.get(pos)
            if st is None:
                return None
            if six_nr <= 0:
                spot = int(st)
            else:
                fb = 1 if six_nr == 1 else (2 if six_nr == 2 else 3)
                if six_jammed:
                    fb = 4
                spot = int(min(149, max(6, 6 + (fb - 1) * 36 + six_raiser * 6 + st)))
            rg = v3_6max_action_range(spot, acts, rfi=rfi, vs_ep=vs_ep, fill=fill)
            return rg.mask_dead(dead) if rg is not None and rg.total() > 0 else None
        except Exception:
            return None

    def _path_and(prior: Range, fill: Range) -> Range:
        """Keep only combos that took the prior action, then this one.

        Replacing a BTN flat with opener-vs-3bet stuffed 87s/AJo into a
        squeeze-call. Uniform prior * fill is fill; a limp (~83%) must
        still intersect. Empty overlap keeps the prior, not a disjoint chart.
        """
        if fill is None or fill.total() <= 0:
            return prior if prior is not None and prior.total() > 0 else fill
        if prior is None or prior.total() <= 0:
            return fill
        out = prior.copy()
        out.w = prior.w * np.clip(np.asarray(fill.w, dtype=np.float64), 0.0, 1.0)
        return out if out.total() > 1e-12 else prior

    six_nr = 0
    six_raiser = 0
    six_jammed = False

    for seat, act, st in iter_history(view):
        if st != Street.PREFLOP:
            # Flop cbets used to walk this as another preflop raise, so a
            # BTN steal became a 3bet range and search overfolded the cbet.
            continue
        pos = view.positions[seat]
        if act.type == ActionType.FOLD:
            ranges[seat] = Range()
            continue
        if act.type == ActionType.CHECK:
            # BB vs a complete: LibraryGTO isos the CO chart, else checks.
            # Leaving BB uniform made every complete pot a 100% range.
            if n_raises == 0 and pos == "BB":
                iso = rfi_range("CO", n, start_bb)
                out = ranges[seat].copy() if ranges[seat].total() > 0 else Range.uniform()
                # Overlay isos at >=0.5 always, else with weight w. Check is the
                # complement — mixed CO combos must not stay at full prior.
                p_iso = np.where(iso.w >= 0.5, 1.0, np.clip(iso.w, 0.0, 1.0))
                out.w = out.w * np.maximum(0.0, 1.0 - p_iso)
                ranges[seat] = out.mask_dead(dead) if out.total() > 0 else ranges[seat].mask_dead(dead)
            continue
        if act.type in (ActionType.BET, ActionType.RAISE) or (
            act.type == ActionType.ALL_IN and n_raises >= 0 and act.amount > view.bb
        ):
            jammed = act.type == ActionType.ALL_IN and act.amount >= 40 * view.bb
            if n_raises == 0:
                # BB never RFIs. A raise here is an iso vs a complete; LibraryGTO
                # uses the CO chart, not the BB-option leftover default spec.
                fill = rfi_range("CO" if pos == "BB" else pos, n, start_bb).copy()
                fill.w = np.where(fill.w >= 0.5, 1.0, fill.w)
                rg = _six_v3(pos, (2,), rfi=True, fill=fill) or _hu_v3(0, (2, 3)) or fill
                ranges[seat] = _path_and(ranges[seat], rg).mask_dead(dead)
                opener_pos = pos
                n_raises = 1
            elif n_raises == 1:
                from strategy.preflop_trees import vs_open_action_ranges

                jam, three, call = vs_open_action_ranges(pos, opener_pos or "BTN", n, start_bb)
                vs_ep = six_raiser <= 1
                jam = apply_blockers_range(jam, "value_4bet")
                three = apply_blockers_range(three, "3bet_bluff")
                # Sequential in_range: always-jammers have P(3bet)=0, not
                # leftover three.w. Search used to treat those as 3bet value.
                p_j, p_3, _p_c, _ = sequential_mix_weights(jam.w, three.w, call.w)
                fill3 = jam.copy() if jammed else three.copy()
                fill3.w = p_j if jammed else p_3
                rg = _six_v3(pos, (2, 3), vs_ep=vs_ep, fill=fill3) or _hu_v3(1, (2, 3)) or fill3
                ranges[seat] = _path_and(ranges[seat], rg).mask_dead(dead)
                threebettor_pos = pos
                n_raises = 2
            elif n_raises == 2:
                # Continue vs the 3bettor, not vs the original opener label.
                four, call4 = vs_3bet_ranges(pos, threebettor_pos or opener_pos or "BB", start_bb)
                four = apply_blockers_range(four, "value_4bet")
                _pj, p_4, _pc, _ = sequential_mix_weights(
                    np.zeros_like(four.w), four.w, call4.w, take_thresh=0.45,
                )
                fill4 = four.copy()
                fill4.w = p_4
                rg = _six_v3(pos, (2, 3), fill=fill4) or _hu_v3(2, (2, 3)) or fill4
                ranges[seat] = _path_and(ranges[seat], rg).mask_dead(dead)
                n_raises = 3
            else:
                jam, call_j = vs_4bet_ranges(start_bb)
                jam = apply_blockers_range(jam, "value_4bet")
                p_j, _p3, _pc, _ = sequential_mix_weights(
                    jam.w, np.zeros_like(jam.w), call_j.w, jam_thresh=0.4, take_thresh=0.4,
                )
                fill_j = jam.copy()
                fill_j.w = p_j
                ranges[seat] = _path_and(ranges[seat], fill_j).mask_dead(dead)
                n_raises += 1
            try:
                from strategy.blueprint import _CFR_SEAT

                six_raiser = int(_CFR_SEAT.get(pos, six_raiser))
            except Exception:
                pass
            six_nr += 1
            six_jammed = bool(jammed)
        elif act.type == ActionType.CALL:
            if n_raises == 0:
                # LibraryGTO SB completes the BTN chart, not a 42% ranking
                # slice (that stuffed AA into a complete and search overbluffed).
                if pos == "SB":
                    complete = rfi_range("BTN", n, start_bb)
                    steal = rfi_range("SB", n, start_bb)
                    p_steal = np.where(steal.w >= 0.5, 1.0, np.clip(steal.w, 0.0, 1.0))
                    rem = np.maximum(0.0, 1.0 - p_steal)
                    cw = np.clip(complete.w, 0.0, 1.0)
                    out = complete.copy()
                    out.w = np.where(cw > 0.25, rem, rem * cw)
                    ranges[seat] = (out if out.total() > 0 else complete).mask_dead(dead)
                elif pos == "BB":
                    iso = rfi_range("CO", n, start_bb)
                    out = Range.uniform()
                    p_iso = np.where(iso.w >= 0.5, 1.0, np.clip(iso.w, 0.0, 1.0))
                    out.w = out.w * np.maximum(0.0, 1.0 - p_iso)
                    ranges[seat] = out.mask_dead(dead)
                else:
                    # LibraryGTO never limps EP. A limp is the complement of
                    # that seat's RFI, not a top-22% value slice (that put AA
                    # in a limp and search overfolded).
                    rfi = rfi_range(pos, n, start_bb)
                    out = Range.uniform()
                    p_open = np.where(rfi.w >= 0.5, 1.0, np.clip(rfi.w, 0.0, 1.0))
                    out.w = out.w * np.maximum(0.0, 1.0 - p_open)
                    ranges[seat] = (out if out.total() > 0 else rfi).mask_dead(dead)
            elif n_raises == 1:
                from strategy.preflop_trees import vs_open_action_ranges

                jam, three, call = vs_open_action_ranges(pos, opener_pos or "BTN", n, start_bb)
                vs_ep = six_raiser <= 1
                jam = apply_blockers_range(jam, "value_4bet")
                three = apply_blockers_range(three, "3bet_bluff")
                # Always-3bet combos (w>=0.5) never call. Raw call.w left
                # AA/KK in flats and search overfolded vs LibraryGTO.
                _pj, _p3, p_c, _ = sequential_mix_weights(jam.w, three.w, call.w)
                fill_c = call.copy()
                fill_c.w = p_c
                fill_c = (
                    _six_v3(pos, (1,), vs_ep=vs_ep, fill=fill_c) or _hu_v3(1, (1,)) or fill_c
                )
                ranges[seat] = _path_and(ranges[seat], fill_c).mask_dead(dead)
            elif n_raises == 2:
                _four, call = vs_3bet_ranges(pos, threebettor_pos or opener_pos or "BB", start_bb)
                _four = apply_blockers_range(_four, "value_4bet")
                _pj, _p4, p_c, _ = sequential_mix_weights(
                    np.zeros_like(_four.w), _four.w, call.w, take_thresh=0.45,
                )
                fill_c = call.copy()
                fill_c.w = p_c
                fill_c = _six_v3(pos, (1,), fill=fill_c) or _hu_v3(2, (1,)) or fill_c
                ranges[seat] = _path_and(ranges[seat], fill_c).mask_dead(dead)
            else:
                _jam, call = vs_4bet_ranges(start_bb)
                _jam = apply_blockers_range(_jam, "value_4bet")
                _pj, _p3, p_c, _ = sequential_mix_weights(
                    _jam.w, np.zeros_like(_jam.w), call.w, jam_thresh=0.4, take_thresh=0.4,
                )
                fill_c = call.copy()
                fill_c.w = p_c
                ranges[seat] = _path_and(ranges[seat], fill_c).mask_dead(dead)

    hero = view.hero
    hero_r = Range.from_hole(*view.hole)
    constructed = ranges[hero]
    if constructed.total() > 0:
        mix = constructed.normalize().scale(0.3)
        mix.w[combo_index(*view.hole)] += 0.7
        ranges[hero] = mix.mask_dead(view.board)
    else:
        ranges[hero] = hero_r.mask_dead(view.board)

    if view.street != Street.PREFLOP:
        ranges = _apply_postflop_actions(view, ranges, dead, hud=hud, population=pop)

    if view.street == Street.PREFLOP:
        for i in range(n):
            if i == hero or ranges[i].total() <= 0:
                continue
            gto = table_vpip_prior(n)
            if pop.name == "unknown":
                target_vpip = gto
            else:
                target_vpip = pop.vpip
            st = hud.stats(i) if hud is not None else None
            if st is not None and st.hands >= 12:
                if st.is_solver_like() or st._sixmax_solver_chart():
                    continue
                # Noisy 12-hand VPIP vs unknown/LibraryGTO must not squeeze
                # combo-tree beliefs. Scale only vs a classified leak.
                if not (st.is_station() or st.is_nit() or st.is_overfolder()):
                    continue
                target_vpip = st.vpip(target_vpip)
            factor = target_vpip / max(gto, 0.05)
            if abs(factor - 1.0) < 0.12:
                continue
            target_frac = min(0.85, ranges[i].frac_of_full() * factor)
            ranges[i] = _scale_to_freq(ranges[i], target_frac, start_bb, widen=factor > 1.0).mask_dead(dead)

    return ranges


def iter_history(view: PlayerView):
    """Yield (seat, action, street) from engine history or a parallel street list."""
    hist = view.action_history
    streets = getattr(view, "street_history", None) or []
    for i, item in enumerate(hist):
        seat, act = item[0], item[1]
        if len(item) > 2:
            st = item[2]
        elif i < len(streets):
            st = streets[i]
        else:
            st = Street.PREFLOP
        yield seat, act, st


def _board_at(street: Street, board: Sequence[int]) -> List[int]:
    n = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}.get(street, len(board))
    return list(board[: min(n, len(board))])


def _check_keep(rg: Range, board: Sequence[int]) -> Range:
    """Checking range: middle of the board-strength order plus a few traps."""
    order = _live_board_order(rg, board)
    if not order:
        return rg
    n = len(order)
    out = Range()
    # Drop the bottom ~18% (air that would c-bet) and most of the top 22% (value bets).
    lo = int(0.18 * n)
    hi = int(0.78 * n)
    for i in order[lo:hi]:
        out.w[i] = rg.w[i]
    trap = order[int(0.92 * n) :]
    for i in trap:
        out.w[i] = rg.w[i] * 0.35
    if out.total() <= 0:
        return rg
    return out


def _bet_keep(rg: Range, board: Sequence[int], size_frac: float) -> Range:
    """Betting range by size and texture.

    Small bets and wet boards are merged (thinking-reg leak was polarizing
    every barrel). Large bets on dry boards stay polar.
    """
    wet = 0.0
    if len(board) >= 3:
        from strategy.texture import analyze

        wet = float(analyze(board).wetness)
    if size_frac <= 0.42 or wet >= 0.55:
        keep = 0.74 if size_frac <= 0.42 else 0.68
        return _keep_top_by_board(rg, board, keep)
    if size_frac <= 0.80:
        return _polarize_by_board(rg, board, top=0.34, bot=0.14)
    return _polarize_by_board(rg, board, top=0.24, bot=0.11)


def _call_keep(rg: Range, board: Sequence[int], continue_frac: float) -> Range:
    return _keep_top_by_board(rg, board, max(0.28, min(0.92, continue_frac)))


def _raise_keep(rg: Range, board: Sequence[int]) -> Range:
    return _polarize_by_board(rg, board, top=0.22, bot=0.10)


def _apply_postflop_actions(
    view: PlayerView,
    ranges: List[Range],
    dead: Sequence[int],
    hud: Optional[HUD] = None,
    population: Optional[Population] = None,
) -> List[Range]:
    """Replay each postflop action as library reach (combo mix), not a keep-set."""
    pop = population or Population.unknown()
    try:
        from strategy.library import replay_postflop

        lib = replay_postflop(view, ranges, dead)
        if lib is not None and any(r.total() > 0 for r in lib):
            return lib
    except Exception:
        pass
    board_full = list(view.board)
    pot_est = float(view.bb + getattr(view, "sb", view.bb // 2))
    for seat, act, st in iter_history(view):
        if st == Street.PREFLOP:
            if act.type in (ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
                pot_est += float(max(act.amount, 0))
            continue
        if view.folded[seat] or ranges[seat].total() <= 0:
            if act.type == ActionType.FOLD:
                ranges[seat] = Range()
            continue
        board = _board_at(st, board_full)
        if len(board) < 3:
            continue
        st_stats = hud.stats(seat) if hud is not None else None
        fold_c = st_stats.fold_to_cbet(pop.fold_to_cbet) if st_stats is not None else pop.fold_to_cbet
        river_f = st_stats.river_fold(pop.river_fold) if st_stats is not None else pop.river_fold
        if act.type == ActionType.FOLD:
            ranges[seat] = Range()
            continue
        if act.type == ActionType.CHECK:
            ranges[seat] = _check_keep(ranges[seat], board)
        elif act.type == ActionType.CALL:
            cont = 1.0 - (river_f if st == Street.RIVER else fold_c)
            if pop.call_wide > 0.4:
                cont = min(0.92, cont + 0.12 * pop.call_wide)
            ranges[seat] = _call_keep(ranges[seat], board, cont)
        elif act.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            frac = act.amount / max(pot_est, 1.0)
            if act.type == ActionType.RAISE:
                ranges[seat] = _raise_keep(ranges[seat], board)
            else:
                ranges[seat] = _bet_keep(ranges[seat], board, frac)
            pot_est += act.amount
        ranges[seat] = ranges[seat].mask_dead(dead)
    return ranges


def _postflop_compress(
    view: PlayerView,
    ranges: List[Range],
    dead: Sequence[int],
    hud: Optional[HUD] = None,
    population: Optional[Population] = None,
    board: Optional[Sequence[int]] = None,
) -> List[Range]:
    """Back-compat wrapper. Production path is _apply_postflop_actions."""
    return _apply_postflop_actions(view, ranges, dead, hud=hud, population=population)


def _live_board_order(rg: Range, board: Sequence[int]) -> List[int]:
    """Combo indices sorted weak → strong vs this board."""
    live = [i for i in range(len(rg.w)) if rg.w[i] > 1e-12]
    if not live:
        return []
    scored = [(board_combo_score(combo_from_index(i), board), i) for i in live]
    scored.sort()
    return [i for _, i in scored]


def _keep_top_by_board(rg: Range, board: Sequence[int], frac: float) -> Range:
    if frac >= 0.99 or rg.total() <= 0:
        return rg
    order = _live_board_order(rg, board)
    if not order:
        return rg
    tot = rg.w.sum()
    acc = 0.0
    out = Range()
    for i in reversed(order):
        out.w[i] = rg.w[i]
        acc += rg.w[i]
        if acc >= frac * tot:
            break
    return out


def _polarize_by_board(rg: Range, board: Sequence[int], top: float, bot: float) -> Range:
    order = _live_board_order(rg, board)
    if not order:
        return rg
    n = len(order)
    n_top = max(1, int(top * n))
    n_bot = max(1, int(bot * n))
    out = Range()
    for i in order[-n_top:]:
        out.w[i] = rg.w[i]
    for i in order[:n_bot]:
        out.w[i] = rg.w[i]
    return out


def _keep_top(rg: Range, frac: float) -> Range:
    """Weight-rank keep (preflop squeeze). Prefer _keep_top_by_board postflop."""
    idxs = np.argsort(-rg.w)
    tot = rg.w.sum()
    if tot <= 0:
        return rg
    acc = 0.0
    out = Range()
    for i in idxs:
        if rg.w[i] <= 0:
            break
        out.w[i] = rg.w[i]
        acc += rg.w[i]
        if acc >= frac * tot:
            break
    return out


def _polarize(rg: Range, top: float, bot: float) -> Range:
    idxs = np.argsort(-rg.w)
    live = [i for i in idxs if rg.w[i] > 0]
    if not live:
        return rg
    n = len(live)
    n_top = max(1, int(top * n))
    n_bot = max(1, int(bot * n))
    out = Range()
    for i in live[:n_top]:
        out.w[i] = rg.w[i]
    for i in live[-n_bot:]:
        out.w[i] = rg.w[i]
    return out


def primary_villain_range(view: PlayerView, ranges: List[Range]) -> Tuple[List[Tuple[int, int]], List[float]]:
    """The most relevant opponent: last aggressor, else biggest live range."""
    hero = view.hero
    last_agg = None
    for seat, act in reversed(view.action_history):
        if seat != hero and not view.folded[seat] and act.type in (
            ActionType.BET,
            ActionType.RAISE,
            ActionType.ALL_IN,
        ):
            last_agg = seat
            break
    if last_agg is not None and ranges[last_agg].total() > 0:
        return ranges[last_agg].combo_weights()
    merged = Range()
    for i, r in enumerate(ranges):
        if i == hero or view.folded[i]:
            continue
        merged.w = np.maximum(merged.w, r.w)
    if merged.total() <= 0:
        return Range.uniform().mask_dead(list(view.hole) + list(view.board)).combo_weights()
    return merged.combo_weights()


def extra_villain_ranges(
    view: PlayerView, ranges: List[Range], primary: Tuple[List, List]
) -> List[Tuple[List[Tuple[int, int]], List[float]]]:
    """Other live opponents' combo ranges (not the primary villain)."""
    hero = view.hero
    prim_set = set(primary[0]) if primary[0] else set()
    out = []
    for i, r in enumerate(ranges):
        if i == hero or view.folded[i] or r.total() <= 0:
            continue
        cs, ws = r.combo_weights()
        if not cs:
            continue
        if set(cs) == prim_set and len(out) == 0:
            continue
        out.append((cs, ws))
    extra_n = max(0, view.players_in_hand - 2)
    return out[:extra_n]
