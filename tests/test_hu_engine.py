import random
import pytest
from poker.game import Action, ActionType, GameConfig, NLHGame


def test_reject_underraise_without_mutation():
    g = NLHGame(GameConfig(num_players=2,stack_bb=100))
    g.start_hand()
    before = [(p.stack,p.bet,p.committed) for p in g.players]
    with pytest.raises(ValueError,match="minimum"):
        g.apply(Action(ActionType.RAISE,51))
    assert before == [(p.stack,p.bet,p.committed) for p in g.players]


@pytest.mark.parametrize("stack", [50,51,75,100,137.5,200,500,999,1000])
def test_random_hands_conserve_chips(stack):
    rng = random.Random(719)
    for i in range(100):
        g = NLHGame(GameConfig(num_players=2,stack_bb=stack,rng_seed=i,
                              button=i%2,rake_percent=.05,rake_cap_bb=3))
        g.start_hand()
        total = 2*g.cfg.starting_chips
        steps = 0
        while not g.finished:
            acts = g.legal_actions()
            a = rng.choice(acts)
            g.apply(a)
            steps += 1
            assert steps < 100
            assert all(p.stack >= 0 for p in g.players)
            if not g.finished:
                assert sum(p.stack+p.committed for p in g.players)==total
        assert sum(p.stack for p in g.players)+g.rake_taken==total


def test_uncalled_shove_is_not_raked():
    g = NLHGame(GameConfig(num_players=2,stack_bb=100,rake_percent=.1,rake_cap_bb=3))
    g.start_hand()
    g.apply(Action(ActionType.CALL,50)); g.apply(Action(ActionType.CHECK))
    g.apply(Action(ActionType.ALL_IN,9900)); g.apply(Action(ActionType.FOLD))
    assert g.rake_taken==20
    assert sorted(p.stack for p in g.players)==[9900,10080]
