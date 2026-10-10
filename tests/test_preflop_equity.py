from hu_bot.equity import preflop_equity
from poker.cards import parse_cards
from strategy.ranges import Range


def test_aces_vs_kings_equity_and_reproducibility():
    hole=parse_cards("AcAd")
    weights=Range.from_string("KK").w
    a=preflop_equity(hole,weights,samples=10000,seed=42)
    b=preflop_equity(hole,weights,samples=10000,seed=42)
    assert a==b and .80<a<.84


def test_symmetry_with_identical_pairs():
    a=preflop_equity(parse_cards("AcAd"),Range.from_hole(*parse_cards("AhAs")).w,samples=10000)
    assert .49<a<.51
