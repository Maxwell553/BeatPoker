"""Board-relative combo strength.

Postflop range compression and search subsampling must rank hands *versus
this board* (made hand + draw), not by preflop combo weight.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from poker.cards import rank_of, suit_of
from poker.evaluator import CAT_BASE, evaluate

Combo = Tuple[int, int]


def board_combo_score(combo: Combo, board: Sequence[int]) -> float:
    """Higher = stronger on this board. Draw bonuses sit below a pair (CAT 1)."""
    if len(board) < 3:
        return 0.0
    made = float(evaluate([combo[0], combo[1]] + list(board)))
    if len(board) >= 5:
        return made
    return made + _draw_bonus(combo, board)


def _draw_bonus(combo: Combo, board: Sequence[int]) -> float:
    """Flush-draw / straight-draw addend. Always < one pair category."""
    bonus = 0.0
    hole_s = (suit_of(combo[0]), suit_of(combo[1]))
    board_s = [suit_of(c) for c in board]
    for s in range(4):
        hs = hole_s.count(s)
        bs = board_s.count(s)
        if hs >= 1 and hs + bs == 4:
            bonus += 0.12 * CAT_BASE  # nut-ish FD still < pair
        elif hs == 2 and bs == 1:
            bonus += 0.04 * CAT_BASE
    ranks = sorted(set(rank_of(c) for c in list(combo) + list(board)))
    if 12 in ranks:
        ranks = sorted(set(ranks) | {-1})
    oesd = gut = False
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            span = ranks[j] - ranks[i]
            cnt = j - i + 1
            if cnt == 4 and span == 3:
                oesd = True
            elif cnt == 4 and span == 4:
                gut = True
            elif cnt == 3 and span == 4:
                gut = True
    if oesd:
        bonus += 0.10 * CAT_BASE
    elif gut:
        bonus += 0.04 * CAT_BASE
    return bonus
