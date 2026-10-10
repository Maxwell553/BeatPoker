from copy import deepcopy
from pathlib import Path

import pytest

from hu_bot import StackAdaptiveBot
from poker.game import Action, ActionType, GameConfig, NLHGame, Street

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / 'training/slumbot_data/buckets.holdem.13.4.3.hs256.3').exists(),
    reason='Install release model bundle for runtime integration tests')


@pytest.mark.parametrize('stack', [50, 62.5, 75, 100, 137.5, 150, 200, 300, 500, 750, 999.5, 1000])
@pytest.mark.parametrize('button', [0, 1])
def test_selected_policy_returns_legal_actions_on_every_street(stack, button):
    game = NLHGame(GameConfig(num_players=2, stack_bb=stack, button=button, rng_seed=8128))
    game.start_hand()
    bot = StackAdaptiveBot(seed=7)
    try:
        streets = set()
        while not game.finished:
            view = game.view(game.to_act)
            action = bot.act(view)
            trial = deepcopy(game)
            trial.apply(action)
            assert bot.last_decision['checkpoint_selection']['actual_stack_bb'] == pytest.approx(stack)
            streets.add(game.street)
            kinds = {a.type for a in view.legal}
            game.apply(Action(ActionType.CHECK if ActionType.CHECK in kinds else ActionType.CALL))
        assert streets == {Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER}
        bot.reset()
        assert bot.last_decision == {}
    finally:
        bot.close()
