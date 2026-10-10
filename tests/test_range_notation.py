from poker.cards import parse_cards
from strategy.ranges import Range, expand_token
from strategy.preflop_trees import rfi_range


def test_kicker_plus_includes_broadways():
    assert expand_token("K8o+")==["K8o","K9o","KTo","KJo","KQo"]
    assert expand_token("Q9o+")==["Q9o","QTo","QJo"]
    assert expand_token("J7s+")==["J7s","J8s","J9s","JTs"]


def test_connector_and_pair_plus():
    assert expand_token("T9s+")==["T9s","JTs","QJs","KQs","AKs"]
    assert expand_token("QQ+")==["QQ","KK","AA"]


def test_hu_button_opens_offsuit_broadways():
    for depth in [50,100,200,500,1000]:
        r=rfi_range("BTN",2,depth)
        for hand in ["KcQd","QcJh","AcKs"]:
            assert r.contains_hole(*parse_cards(hand)) == 1.0
