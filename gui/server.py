"""Local HTTP table: heads-up NLH vs BeatPoker.

    PYTHONPATH=. .venv/bin/python play_gui.py
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from hu_bot import HeadsUpBot as FinalBot, SolverConfig
from strategy.rake import RakeConfig
from poker.cards import RANK_CHARS, SUIT_CHARS, card_to_str, rank_of, suit_of
from poker.evaluator import category_name, evaluate
from poker.game import Action, ActionType, GameConfig, NLHGame

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

SUIT_SYM = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
RAKES = {
    "casino": RakeConfig.casino(),
    "online": RakeConfig.online(),
    "none": RakeConfig.none(),
    "heavy": RakeConfig.casino_heavy(),
}


def _card(c: int) -> dict:
    code = card_to_str(c)
    return {
        "code": code,
        "rank": RANK_CHARS[rank_of(c)],
        "suit": code[1],
        "sym": SUIT_SYM[code[1]],
        "red": code[1] in "hd",
    }


def _hand_name(hole, board) -> Optional[str]:
    if not hole or hole[0] < 0 or len(board) < 3:
        return None
    return category_name(evaluate(list(hole) + list(board)))


class Table:
    """One HU cash session. Stacks reset each hand; session bb is tracked."""

    def __init__(self, stack_bb: float = 100.0, rake: RakeConfig | None = None, seconds: float = 30.0, blueprint: str | None = None, full_blueprint: bool = False, resolve_from_street=None,river_gadget=False,native_iterations=1000,native_compression=True):
        if river_gadget and (not full_blueprint or resolve_from_street is not None):
            raise ValueError("River gadget requires a full blueprint without another refinement mode")
        self.solver_config = SolverConfig(seconds=seconds,preflop_blueprint=blueprint,iterations=native_iterations,compression=native_compression)
        self.full_blueprint=full_blueprint
        self.resolve_from_street=resolve_from_street
        self.river_gadget=river_gadget
        self.stack_bb = stack_bb
        self.rake = rake if rake is not None else RakeConfig.none()
        self.hero = 0
        self.bot_seat = 1
        self.hand_no = 0
        self.session_bb = 0.0
        self.log: list[dict] = []
        self.game: Optional[NLHGame] = None
        self.bot: Optional[FinalBot] = None
        self.lock = threading.Lock()
        self.bot = self._make_bot()
        self.new_hand()

    def _make_bot(self):
        if self.full_blueprint:
            if self.rake.percent or self.solver_config.preflop_blueprint:
                raise ValueError("Full blueprint requires no rake and no separate preflop blueprint")
            from hu_bot import StackAdaptiveBot
            return StackAdaptiveBot(seed=1000,seconds=self.solver_config.seconds,rake=self.rake,
                                    native_iterations=self.solver_config.iterations,native_compression=self.solver_config.compression,
                                    resolve_from_street=self.resolve_from_street,river_gadget=self.river_gadget)
        return FinalBot(seat=self.bot_seat,seed=1000,quality="high",rake=self.rake,config=self.solver_config)

    def new_hand(self) -> None:
        if not 50 <= self.stack_bb <= 1000:
            raise ValueError("Starting stack must be 50–1000 bb")
        if self.full_blueprint and self.rake.percent:
            raise ValueError("Full blueprint requires unraked play")
        if self.bot is not None:
            self.bot.rake = self.rake
        cfg = GameConfig(
            num_players=2,
            stack_bb=self.stack_bb,
            button=self.hand_no % 2,
            rng_seed=None,
            rake_percent=self.rake.percent,
            rake_cap_bb=self.rake.cap_bb,
            no_flop_no_drop=self.rake.no_flop_no_drop,
        )
        self.game = NLHGame(cfg)
        self.game.start_hand()
        if self.bot is None:
            self.bot = self._make_bot()
        self.bot.reset()
        self.log = []
        self._note("system", f"Hand {self.hand_no + 1} — you are {self.game.view(self.hero).position}")
        self._pump_bot()

    def next_hand(self) -> None:
        if self.game and self.game.finished:
            start = self.game.cfg.starting_chips
            self.session_bb += (self.game.players[self.hero].stack - start) / self.game.bb
            self.hand_no += 1
        self.new_hand()

    def _note(self, who: str, text: str) -> None:
        self.log.append({"who": who, "text": text})
        if len(self.log) > 40:
            self.log = self.log[-40:]

    def _fmt(self, action: Action, view) -> str:
        bb = view.bb
        if action.type == ActionType.FOLD:
            return "folds"
        if action.type == ActionType.CHECK:
            return "checks"
        if action.type == ActionType.CALL:
            return f"calls {action.amount / bb:.1f}bb"
        if action.type == ActionType.BET and view.current_bet == 0:
            return f"bets {action.amount / bb:.1f}bb"
        if action.type in (ActionType.RAISE,ActionType.BET):
            to = (view.bets[view.hero] + action.amount) / bb
            return f"raises to {to:.1f}bb"
        if action.type == ActionType.ALL_IN:
            return f"all-in {action.amount / bb:.1f}bb"
        return str(action)

    def _pump_bot(self) -> None:
        g = self.game
        n = 0
        while g and (not g.finished) and g.to_act != self.hero and n < 80:
            street = g.street
            view = g.view(g.to_act)
            action = self.bot.act(view)
            self._note("bot", f"BeatPoker {view.position} {self._fmt(action, view)}")
            g.apply(action)
            self.bot.observe_action(view.hero, action, street, g.n)
            n += 1
        if g and g.finished:
            self.bot.end_hand(g.n)
            self._on_end()

    def _on_end(self) -> None:
        g = self.game
        bb = g.bb
        names = []
        if g.winners:
            for w in g.winners:
                who = "You" if w == self.hero else "BeatPoker"
                names.append(who)
        won = " & ".join(names) if names else "—"
        rake = g.rake_taken / bb
        extra = f"  rake {rake:.1f}bb" if g.rake_taken else ""
        self._note("system", f"{won} wins  ({g.payouts[self.hero] / bb:.1f}bb to you){extra}")

    def act(self, kind: str, size_bb: Optional[float] = None) -> None:
        g = self.game
        if not g or g.finished or g.to_act != self.hero:
            raise ValueError("not your turn")
        view = g.view(self.hero)
        stack = view.stacks[self.hero]
        to_call = view.to_call
        kind = kind.lower().strip()
        if kind == "fold":
            action = Action(ActionType.FOLD)
        elif kind == "check":
            action = Action(ActionType.CHECK)
        elif kind == "call":
            action = Action(ActionType.CALL, min(to_call, stack))
        elif kind == "allin":
            action = Action(ActionType.ALL_IN, stack)
        elif kind in ("bet", "raise"):
            if size_bb is None:
                raise ValueError("need a size")
            target = int(round(float(size_bb) * view.bb))
            put = target - view.bets[self.hero]
            put = max(put, 0)
            if to_call <= 0:
                action = Action(ActionType.BET, put)
            else:
                action = Action(ActionType.RAISE, put)
        else:
            raise ValueError(f"unknown action {kind}")
        self._note("you", f"You {self._fmt(action, view)}")
        street = g.street
        g.apply(action)
        self.bot.observe_action(self.hero, action, street, g.n)
        self._pump_bot()

    def snapshot(self) -> dict[str, Any]:
        g = self.game
        assert g is not None
        bb = g.bb
        hero_p = g.players[self.hero]
        bot_p = g.players[self.bot_seat]
        reveal = g.finished
        your_turn = (not g.finished) and g.to_act == self.hero
        to_call = 0
        can_check = False
        min_to_bb = 1.0
        max_to_bb = hero_p.stack / bb
        if your_turn:
            v = g.view(self.hero)
            to_call = v.to_call
            can_check = v.can_check
            min_to_bb = v.min_raise_to / bb
            max_to_bb = v.max_raise_to / bb
        board = [_card(c) for c in g.board]
        result = None
        if g.finished:
            h_delta = (hero_p.stack - g.cfg.starting_chips) / bb
            result = {
                "winners": ["You" if w == self.hero else "BeatPoker" for w in g.winners],
                "hero_delta_bb": round(h_delta, 2),
                "rake_bb": round(g.rake_taken / bb, 2),
                "hero_hand": _hand_name(hero_p.hole, g.board),
                "bot_hand": _hand_name(bot_p.hole, g.board),
            }
        return {
            "hand_no": self.hand_no + 1,
            "street": g.street.name,
            "board": board,
            "pot_bb": round(g.pot_total() / bb, 2),
            "button": g.button,
            "finished": g.finished,
            "your_turn": your_turn,
            "to_call_bb": round(to_call / bb, 2),
            "can_check": can_check,
            "can_fold": your_turn and to_call > 0,
            "can_bet": your_turn and to_call <= 0 and hero_p.stack > 0 and not hero_p.all_in,
            "can_raise": your_turn and to_call > 0 and hero_p.stack > to_call,
            "min_to_bb": round(min_to_bb, 2),
            "max_to_bb": round(max_to_bb, 2),
            "hero_bet_bb": round(hero_p.bet / bb, 2),
            "session_bb": round(self.session_bb, 2),
            "stack_bb": self.stack_bb,
            "rake": "on" if self.rake.active else "off",
            "rake_preset": next((key for key,value in RAKES.items() if value==self.rake),"none"),
            "policy_mode": "experimental_river_gadget" if self.river_gadget else "experimental_full_blueprint" if self.full_blueprint else "native_postflop",
            "you": {
                "name": "You",
                "pos": g.view(self.hero).position if hero_p.hole[0] >= 0 else "—",
                "stack_bb": round(hero_p.stack / bb, 1),
                "bet_bb": round(hero_p.bet / bb, 2),
                "folded": hero_p.folded,
                "allin": hero_p.all_in,
                "hole": [_card(c) for c in hero_p.hole] if hero_p.hole[0] >= 0 else [],
                "to_act": your_turn,
            },
            "villain": {
                "name": "BeatPoker",
                "pos": g.view(self.bot_seat).position if bot_p.hole[0] >= 0 else "—",
                "stack_bb": round(bot_p.stack / bb, 1),
                "bet_bb": round(bot_p.bet / bb, 2),
                "folded": bot_p.folded,
                "allin": bot_p.all_in,
                "hole": [_card(c) for c in bot_p.hole] if reveal and bot_p.hole[0] >= 0 else [],
                "to_act": (not g.finished) and g.to_act == self.bot_seat,
            },
            "log": list(self.log),
            "result": result,
        }


TABLE: Optional[Table] = None
TABLE_LOCK = threading.Lock()


def get_table() -> Table:
    global TABLE
    with TABLE_LOCK:
        if TABLE is None:
            TABLE = Table()
        return TABLE


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        if self.path.startswith("/api"):
            SimpleHTTPRequestHandler.log_message(self, fmt, *args)

    def _json(self, obj: Any, status: int = 200) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8") or "{}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            t = get_table()
            with t.lock:
                self._json(t.snapshot())
            return
        if path == "/":
            self.path = "/index.html"
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        t = get_table()
        try:
            body = self._read_json()
            with t.lock:
                if path == "/api/new":
                    stack=float(body.get("stack_bb",t.stack_bb))
                    rake=RAKES.get(body.get("rake"),t.rake)
                    if not 50<=stack<=1000:
                        raise ValueError("Starting stack must be 50–1000 bb")
                    if t.full_blueprint and rake.percent:
                        raise ValueError("Full blueprint requires unraked play")
                    t.session_bb = 0.0
                    t.hand_no = 0
                    t.stack_bb,t.rake=stack,rake
                    t.new_hand()
                elif path == "/api/next":
                    t.next_hand()
                elif path == "/api/act":
                    t.act(body.get("action", ""), body.get("to_bb"))
                else:
                    self._json({"error": "not found"}, 404)
                    return
                self._json(t.snapshot())
        except Exception as e:
            self._json({"error": str(e)}, 400)


def main() -> None:
    from hu_bot.options import add_policy_arguments,resolve_policy_arguments
    ap = argparse.ArgumentParser(description="Play HU vs BeatPoker in the browser")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--rake", default="none", choices=tuple(RAKES))
    ap.add_argument("--stack", type=float, default=100.0)
    ap.add_argument("--seconds", type=float, default=30.0)
    add_policy_arguments(ap)
    args = ap.parse_args()
    global TABLE
    resolve_policy_arguments(args,ap)
    TABLE = Table(stack_bb=args.stack, rake=RAKES[args.rake], seconds=args.seconds, blueprint=args.blueprint,
                  full_blueprint=args.full_blueprint,resolve_from_street={"turn":2,"river":3}.get(args.resolve_from),river_gadget=args.river_gadget,native_iterations=args.native_iterations,native_compression=args.native_compression)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"BeatPoker GUI  {url}", flush=True)
    print("Heads-up vs BeatPoker. Close this terminal to quit.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
        httpd.shutdown()


if __name__ == "__main__":
    main()
