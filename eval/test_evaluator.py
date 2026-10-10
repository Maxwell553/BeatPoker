"""Hand-evaluator correctness tests — must all pass before we trust the bot."""

from poker.cards import parse_cards
from poker.evaluator import RANK_CLASS, category, category_name, evaluate, evaluate5


def _v(s: str) -> int:
    return evaluate(parse_cards(s))


def test_categories_order():
    royal = _v("AhKhQhJhTh")
    steel = _v("Ah2h3h4h5h")  # wheel SF
    quads = _v("AhAdAcAsKh")
    boat = _v("AhAdAcKhKd")
    flush = _v("AhKh9h5h2h")
    straight = _v("AhKdQcJsTd")
    trips = _v("AhAdAcKh9d")
    two = _v("AhAdKhKd9c")
    pair = _v("AhAdKh9d8c")
    hc = _v("AhKd9c5s2d")
    assert category(royal) == 8
    assert category(steel) == 8
    assert royal > steel > quads > boat > flush > straight > trips > two > pair > hc


def test_kickers():
    assert _v("AhAdKh9d8c") > _v("AhAdQh9d8c")  # AAK vs AAQ
    assert _v("AhAdKhKd9c") > _v("AhAdKhKd8c")
    assert _v("AhKh9h5h2h") > _v("AhKh8h5h2h")
    assert _v("9h8h7h6h5h") > _v("8h7h6h5h4h")  # SF 9 high vs 8 high


def test_wheel_straight():
    wheel = _v("Ah2d3c4s5h")
    sixhigh = _v("2d3c4s5h6c")
    assert category(wheel) == 4
    assert category(sixhigh) == 4
    assert sixhigh > wheel
    broadway = _v("AhKdQcJsTd")
    assert broadway > sixhigh


def test_seven_card_picks_best():
    # Trips, not a flush (only 3 hearts)
    cards = parse_cards("AhAdAcKh9d7c2s")
    v = evaluate(cards)
    assert category(v) == 3  # trips

    # Boat vs flush: AhKhQhJh9h Ad Kd -> AKQJ9 flush AND pair A, pair K... flush wins vs two pair
    cards = parse_cards("AhKhQhJh9h Ad Kd")
    v = evaluate(cards)
    assert category(v) == 5  # flush (no boat)

    # Actual boat + flush: Ah Ad Ac Kh Kd 2h 3h — boat, not flush
    cards = parse_cards("AhAdAcKhKd2h3h")
    assert category(evaluate(cards)) == 6

    # Quads in 7
    cards = parse_cards("2c2d2h2sAhKd9c")
    assert category(evaluate(cards)) == 7

    # Two trips -> boat
    cards = parse_cards("9c9d9h8c8d8hAc")
    assert category(evaluate(cards)) == 6  # 99988 or 88899 -> 99988


def test_royal():
    assert category(_v("AsKsQsJsTs")) == 8
    # 7-card royal
    assert category(evaluate(parse_cards("AsKsQsJsTs 2c 3d"))) == 8


def test_seven_matches_best_five():
    from itertools import combinations
    import random
    from poker.evaluator import _eval_n

    rng = random.Random(0)
    for _ in range(400):
        cards = rng.sample(range(52), 7)
        fast = _eval_n(cards)
        slow = max(_eval_n(c) for c in combinations(cards, 5))
        assert fast == slow, (cards, fast, slow)


def test_pair_vs_pair():
    assert _v("AhAd5c4s3h") > _v("KhKd5c4s3h")


def run():
    tests = [
        test_categories_order,
        test_kickers,
        test_wheel_straight,
        test_seven_card_picks_best,
        test_royal,
        test_seven_matches_best_five,
        test_pair_vs_pair,
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
    n = run()
    raise SystemExit(n)
