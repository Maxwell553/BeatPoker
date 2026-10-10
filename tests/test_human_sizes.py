import pytest
from play import parse_human
from poker.game import GameConfig, NLHGame, ActionType


def view(stack=1000):
    game=NLHGame(GameConfig(num_players=2,stack_bb=stack))
    game.start_hand()
    return game.view(game.to_act)


def test_deep_stack_amounts_are_unambiguously_big_blinds():
    state=view()
    assert parse_human("r500",state).amount==49950
    assert parse_human("r500bb",state).amount==49950
    assert parse_human("r500chips",state).amount==450
    assert parse_human("r1000",state).type==ActionType.ALL_IN


def test_pot_raise_includes_the_call():
    state=view()
    assert parse_human("r100%",state).amount==250  # raise-to 3bb
    assert parse_human("r75%",state).amount==200  # raise-to 2.5bb


def test_oversized_bet_is_reported():
    with pytest.raises(ValueError,match="Maximum"):
        parse_human("r500",view(100))
