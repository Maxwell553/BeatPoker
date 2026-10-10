"""NLH engine smoke tests: blinds, folds, all-in side pots, HU/6-max."""

from poker.cards import parse_cards
from poker.evaluator import evaluate
from poker.game import Action, ActionType, GameConfig, NLHGame, settle_pots


def test_hu_blinds():
    g = NLHGame(GameConfig(num_players=2, stack_bb=100, button=0, rng_seed=0))
    g.start_hand()
    assert g.players[0].committed == 50  # SB
    assert g.players[1].committed == 100  # BB
    assert g.to_act == 0
    assert g.pot_total() == 150


def test_6max_utg_acts_first():
    g = NLHGame(GameConfig(num_players=6, stack_bb=100, button=0, rng_seed=1))
    g.start_hand()
    assert g.to_act == 3  # UTG


def test_fold_wins_pot():
    g = NLHGame(GameConfig(num_players=2, stack_bb=50, button=0, rng_seed=2))
    g.start_hand()
    g.apply(Action(ActionType.FOLD))
    assert g.finished
    assert g.winners == [1]
    assert g.players[1].stack == 50 * 100 + 50  # started 5000, won SB 50


def test_side_pot():
    # A all-in 300, B 500, C 500, A wins
    contrib = [300, 500, 500]
    folded = [False, False, False]
    # Make A have quads, others trash — inject by using known cards
    board = parse_cards("AhKh9c5d2s")
    hole = [
        tuple(parse_cards("AdAc")),  # set of aces
        tuple(parse_cards("3c3d")),
        tuple(parse_cards("4c4d")),
    ]
    pay, winners = settle_pots(contrib, folded, hole, board)
    assert pay[0] == 900  # wins main 300*3 = 900, no side (B vs C side pot 200*2)
    # Wait: A contributed 300, main pot 900. Side pot 200+200=400 between B and C.
    # 3c3d vs 4c4d on A-high board: 44 wins side.
    assert pay[0] == 900
    assert pay[2] == 400
    assert pay[1] == 0


def test_check_it_down():
    g = NLHGame(GameConfig(num_players=2, stack_bb=100, button=0, rng_seed=3))
    holes = [tuple(parse_cards("AhAd")), tuple(parse_cards("KhKd"))]
    g.start_hand(hole_cards=holes)
    # BTN calls, BB checks
    g.apply(Action(ActionType.CALL, 50))
    g.apply(Action(ActionType.CHECK))
    assert g.street.name == "FLOP"
    for _ in range(6):
        if g.finished:
            break
        v = g.view(g.to_act)
        if v.to_call > 0:
            g.apply(Action(ActionType.CALL, min(v.to_call, v.stacks[v.hero])))
        else:
            g.apply(Action(ActionType.CHECK))
    assert g.finished
    # AA vs KK: AA should win unless board makes a set/straight/flush for KK more... AA favored.
    # We don't assert winner because of random board.


def test_no_flop_no_drop():
    g = NLHGame(
        GameConfig(
            num_players=2,
            stack_bb=50,
            button=0,
            rng_seed=2,
            rake_percent=0.10,
            rake_cap_bb=3.0,
        )
    )
    g.start_hand()
    g.apply(Action(ActionType.FOLD))
    assert g.finished
    assert g.rake_taken == 0
    assert g.players[1].stack == 50 * 100 + 50


def test_flop_uncontested_takes_rake():
    g = NLHGame(
        GameConfig(
            num_players=2,
            stack_bb=100,
            button=0,
            rng_seed=3,
            rake_percent=0.10,
            rake_cap_bb=3.0,
        )
    )
    holes = [tuple(parse_cards("AhAd")), tuple(parse_cards("2c7d"))]
    g.start_hand(hole_cards=holes)
    g.apply(Action(ActionType.CALL, 50))
    g.apply(Action(ActionType.CHECK))
    assert g.street.name == "FLOP"
    # HU postflop: first to act is BB (seat 1)
    assert g.to_act == 1
    g.apply(Action(ActionType.BET, 100))
    g.apply(Action(ActionType.FOLD))
    assert g.finished
    # The uncalled 100 is returned before taking 10% of the contested 200.
    assert g.rake_taken == 20
    assert g.payouts[1] == 180
    assert sum(g.payouts) + g.rake_taken == 200


def run():
    tests = [
        test_hu_blinds,
        test_6max_utg_acts_first,
        test_fold_wins_pot,
        test_side_pot,
        test_check_it_down,
        test_no_flop_no_drop,
        test_flop_uncontested_takes_rake,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    return failed


if __name__ == "__main__":
    raise SystemExit(run())
