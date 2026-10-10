"""Action abstraction, reverse translation, and infoset keys.

Live search solves an abstracted betting tree. Off-tree sizes are mapped onto
distinct strategy nodes (0.25 / 0.33 / 0.75 / overbet) — not only snapping the
call amount onto 33/55/75/jam while leaving the mix unchanged.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

# Geometric / solver-standard fractions, plus overbet and jam.
ABSTRACT_FRACS: Tuple[float, ...] = (0.25, 0.33, 0.50, 0.75, 1.00, 1.25, 2.00)


def nearest_frac(frac: float) -> float:
    f = max(0.0, float(frac))
    return min(ABSTRACT_FRACS, key=lambda a: abs(a - f))


def facing_frac(to_call: float, pot: float) -> float:
    """Bet as a fraction of the pot *before* this bet (pot includes the bet)."""
    pot_before = max(pot - to_call, 1.0)
    return to_call / pot_before


def translate_facing(to_call: float, pot: float) -> float:
    """Map an off-tree bet onto a distinct facing-size node."""
    return nearest_frac(facing_frac(to_call, pot))


def face_node(to_call: float, pot: float) -> str:
    """Strategy-node label. 20% and 120% must not share a tree."""
    f = translate_facing(to_call, pot)
    return f"face{int(round(f * 100))}"


def abstract_bet_puts(
    pot: float,
    stack: float,
    spr: float,
    library_first: bool = False,
    ip: bool = True,
) -> List[Tuple[str, float]]:
    """First-in abstract bets. 25% / 75% / overbet are different mixes.

    HU 20k (library_first=False) keeps the old SPR<2.4 → 75%/jam bucket.
    6-max leftover vs LibraryGTO: IP jams below 2.35. OOP keeps donk
    sizes below 2.2 (air still stabs; only value jams).
    """
    acts: List[Tuple[str, float]] = [("check", 0.0)]
    if stack <= 0.5:
        return acts
    if library_first:
        if ip and spr < 2.35:
            acts.append(("allin", stack))
            return acts
        fracs = [0.25, 0.33, 0.55, 0.75, 1.25]
        for f in fracs:
            put = min(stack, max(1.0, f * pot))
            if put < stack * 0.90 and put >= 0.18 * pot:
                acts.append((f"bet{int(round(f * 100))}", put))
        if spr <= 2.4 and not any(n == "allin" for n, _ in acts):
            acts.append(("allin", stack))
        return _unique_puts(acts, 7)
    if spr < 2.4:
        fracs = [0.75]
    else:
        # Library first-in is always 25/33/55/75/125. SPR~4 is a 3bet-pot
        # flop; the old SPR<4.5 bucket (33/75/125) dropped 25/55 — LibraryGTO's
        # merged cbet — and search snapped that mix onto 33 or 75.
        # Deep SRP still needs 125% (OOP folds middle pair to 1.25x).
        fracs = [0.25, 0.33, 0.55, 0.75, 1.25]
    for f in fracs:
        put = min(stack, max(1.0, f * pot))
        if put < stack * 0.90 and put >= 0.18 * pot:
            acts.append((f"bet{int(round(f * 100))}", put))
    if stack > 0 and (spr <= 2.2 or stack <= 1.35 * pot or len(acts) == 1):
        acts.append(("allin", stack))
    elif stack > 0 and acts[-1][1] < stack * 0.92:
        # Library first-in jams only below SPR~2.35. Do not replace 25/55
        # with a jam at 3bet-pot SPR~4.
        if spr < 2.4:
            acts.append(("allin", stack))
        elif len(acts) < 6:
            put = min(stack, 2.00 * pot)
            if put < stack * 0.9 and not any(n == "bet200" for n, _ in acts):
                acts.append(("bet200", put))
            else:
                acts.append(("allin", stack))
    return _unique_puts(acts)


def abstract_facing_puts(
    pot: float,
    stack: float,
    to_call: float,
    spr: float,
    library_raise: bool = False,
) -> List[Tuple[str, float]]:
    """Fold / call (exact) / size-dependent raise set.

    vs 0.25x: raise is a real node (punish small). vs overbet: call or jam,
    no min-raise that would play the 75% tree.
    HU 20k uses 3.4x vs small. LibraryGTO always raises 2.8 / 0.75 pot.
    """
    acts: List[Tuple[str, float]] = [("fold", 0.0), ("call", min(to_call, stack))]
    frac = facing_frac(to_call, pot)
    node = face_node(to_call, pot)
    if stack <= to_call + 1:
        return _unique_puts(acts)[:5]
    # HU 20k keeps jam-only below 2.5. LibraryGTO face_mix raises when
    # spr > 2.4 — leftover 6-max must keep that raise node.
    jam_only = spr <= 2.4 if library_raise else spr < 2.5
    if jam_only or stack <= 3.2 * to_call:
        acts.append(("allin", stack))
        return _unique_puts(acts)[:5]
    if frac > 1.05:
        acts.append(("allin", stack))
        return _unique_puts(acts)[:5]
    if library_raise:
        # Match LibraryGTO / 3-way. Do not use on 2-player HU (20k bar).
        raise_put = min(stack, max(to_call * 2.8, to_call + 0.75 * pot))
    elif frac <= 0.40:
        raise_put = min(stack, max(to_call * 3.4, to_call + 1.10 * pot))
    elif frac <= 0.70:
        raise_put = min(stack, max(to_call * 2.8, to_call + 0.75 * pot))
    else:
        raise_put = min(stack, max(to_call * 2.4, to_call + 0.90 * pot))
    if raise_put >= stack * 0.85:
        acts.append(("allin", stack))
    else:
        acts.append((f"raise_{node}", raise_put))
        if stack > raise_put + 1:
            acts.append(("allin", stack))
    return _unique_puts(acts)[:5]


def _unique_puts(acts: Sequence[Tuple[str, float]], limit: int = 6) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    seen = set()
    for name, put in acts:
        key = round(put, 1)
        if key in seen and name not in ("fold", "check"):
            continue
        seen.add(key)
        out.append((name, put))
    return out[: int(limit)]


def continuation_extra(
    pot: float,
    stack0: float,
    stack1: float,
    streets_left: int,
    hero_bkt: int | None = None,
    vill_bkt: int | None = None,
    pot_type: int = 0,
) -> float:
    """Depth-limited leaf: library CV, else blueprint, else geometric."""
    if streets_left <= 0:
        return 0.0
    cap = min(stack0, stack1)
    if cap <= 0.5:
        return 0.0
    geo = min(cap, pot * 0.55 * streets_left)
    try:
        from strategy.library import continuation_extra as lib_extra
        from strategy.card_abs import street_bucket

        # Bucket-only path when we don't have the actual combos (scalar leaf).
        if hero_bkt is not None and vill_bkt is not None:
            hs = hero_bkt / 49.0
            vs = vill_bkt / 49.0
            vol = 0.28 + 0.72 * (0.5 * (hs + vs))
            realize = 0.52 + 0.14 * min(streets_left, 2) + (0.12 if pot_type >= 1 else 0.0)
            lib = min(cap, pot * 0.48 * streets_left * vol * realize)
            extra = (hs - vs) * lib
            return 0.85 * float(extra) + 0.15 * geo * (hs - vs)
        _ = (lib_extra, street_bucket)
    except Exception:
        pass
    try:
        from strategy.blueprint import continuation_extra as bp_extra

        got = bp_extra(pot, stack0, stack1, streets_left, hero_bkt, vill_bkt, pot_type)
        if got is not None:
            return 0.5 * float(got) + 0.5 * geo
    except Exception:
        pass
    return geo


def board_bucket(board: Sequence[int]) -> Tuple[int, ...]:
    """Coarse public-state bucket for infoset indexing / cache grouping."""
    from strategy.texture import analyze

    if len(board) < 3:
        return (0, 0, 0, 0)
    tex = analyze(board)
    return (
        len(board),
        int(tex.paired),
        int(tex.monotone),
        int(round(tex.wetness * 4)),
        int(round(tex.highness * 4)),
    )


def infoset_key(player: int, street: int, bucket: Tuple[int, ...], history: int, combo: int):
    """(player, street, board-bucket, action-history, combo) — Pluribus-style index."""
    return (player, street, bucket, history, combo)
