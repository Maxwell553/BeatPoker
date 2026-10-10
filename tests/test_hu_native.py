import numpy as np
import pytest

from hu_bot.native import NativeSolver, SolverError
from hu_bot.bot import range_string
from hu_bot.preflop import public_flop_ranges
from poker.cards import parse_cards
from poker.equity import combo_index
from poker.game import Action, ActionType, GameConfig, NLHGame


@pytest.fixture
def solver():
    s = NativeSolver(threads=2)
    yield s
    s.close()


def request(**kwargs):
    return dict(cmd="solve",board=["Ks","7h","2c","3d","9s"],
                ranges=["AA,KK,QQ,AKs,A5s","QQ,JJ,TT,AKs,AQs"],
                pot=600,stack=9700,hole=["Ac","Ad"],seconds=3,
                iterations=1000,compact=True,**kwargs)


@pytest.mark.parametrize("compression",[True,False])
def test_river_convergence(solver,compression):
    r = solver.request(request(compression=compression))
    assert r["solve"]["converged"]
    assert r["solve"]["compression"]==compression
    assert r["solve"]["exploitability_chips"] <= 3
    assert abs(sum(r["probabilities"])-1) < 1e-6


def test_exact_offtree_insertion(solver):
    history = [{"kind":"bet","amount":317}]
    req = request(lines=[history],history=history)
    req["hole"] = ["Qc","Qd"]
    r = solver.request(req)
    assert r["player"] == 1
    assert any(a["kind"]=="call" for a in r["actions"])


def test_offtree_is_explicit(solver):
    solver.request(request())
    with pytest.raises(SolverError,match="off_tree"):
        solver.request({"cmd":"query","history":[{"kind":"bet","amount":317}]})


def test_observed_reraise_survives_abstraction_pruning(solver):
    history=[{"kind":"bet","amount":317},{"kind":"raise","amount":1500},
             {"kind":"raise","amount":3300}]
    req=request(lines=[history],history=history)
    req["hole"]=["Qc","Qd"]
    r=solver.request(req)
    assert r["player"]==1 and any(a["kind"]=="call" for a in r["actions"])


def test_required_raises_survive_reduced_memory_tree(solver):
    history=[{"kind":"bet","amount":317},{"kind":"raise","amount":1500},
             {"kind":"raise","amount":3300}]
    req=request(lines=[history],history=history,max_bets=1)
    req["hole"]=["Qc","Qd"]
    r=solver.request(req)
    assert r["player"]==1 and r["solve"]["max_bets"]==1


def test_no_bet_below_big_blind(solver):
    req=request(min_bet=100)
    req.update(pot=100,stack=9700)
    r=solver.request(req)
    assert all(a["amount"] >= 100 for a in r["actions"] if a["kind"]=="bet")


def test_card_mapping_roundtrip(solver):
    w = np.ones(1326)
    req = request()
    req.update(ranges=[range_string(w)]*2,iterations=10)
    r = solver.request(req)
    assert len(r["ranges"][0]["hands"]) == 1081
    assert len(r["ranges"][1]["hands"]) == 1081


def test_tiny_range_weights_are_decimal(solver):
    w=np.full(1326,1e-25)
    w[combo_index(*parse_cards("AcAd"))]=1
    req=request()
    req.update(ranges=[range_string(w)]*2,iterations=1)
    r=solver.request(req)
    assert "probabilities" in r
    assert len(r["ranges"][0]["hands"])==1081


def test_public_ranges_do_not_depend_on_hero_hole():
    g = NLHGame(GameConfig(num_players=2,stack_bb=100,rng_seed=4))
    g.start_hand()
    g.apply(Action(ActionType.RAISE,200))
    g.apply(Action(ActionType.CALL,150))
    v = g.view(g.to_act)
    a,_,_ = public_flop_ranges(v)
    hero_index = combo_index(*v.hole)
    assert a[v.hero,hero_index] > 0
    v.hole = tuple(parse_cards("AcAd"))
    b,_,_ = public_flop_ranges(v)
    np.testing.assert_array_equal(a,b)
