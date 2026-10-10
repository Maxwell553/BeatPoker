"""Stack-depth buckets. 50bb is its own game, not a 100bb chart with a multiplier."""

from __future__ import annotations

from typing import Literal

StackBucket = Literal["50", "100", "200", "deep"]


def stack_bucket(stack_bb: float) -> StackBucket:
    if stack_bb <= 65:
        return "50"
    if stack_bb <= 140:
        return "100"
    if stack_bb <= 280:
        return "200"
    return "deep"


def nearest_table(n_players: int) -> int:
    """Map 2–9 to the trees we actually encode (2 / 6 / 9)."""
    if n_players <= 2:
        return 2
    if n_players <= 5:
        return 6
    if n_players <= 7:
        return 6
    return 9


def open_size_bb(position: str, n_players: int, stack_bb: float) -> float:
    b = stack_bucket(stack_bb)
    if n_players == 2:
        return 2.0 if b == "50" else 2.5
    if position == "SB":
        return 2.5 if b == "50" else 3.0
    if b == "50":
        return 2.2
    return 2.5
