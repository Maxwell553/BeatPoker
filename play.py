"""Play heads-up No-Limit Hold'em against HeadsUpBot.

Usage:
    python play.py                 # HU 100bb
    python play.py --stack 200
    python play.py --stack 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker.cards import card_to_str, cards_to_str
from poker.evaluator import category_name, evaluate
from poker.game import Action, ActionType, GameConfig, NLHGame
from hu_bot import HeadsUpBot as FinalBot, SolverConfig
from hu_bot.options import add_policy_arguments,resolve_policy_arguments
from strategy.rake import RakeConfig
from Final_Bot.opponents import make_opponent


def parse_human(s: str, view) -> Action:
    s = s.strip().lower()
    if s in ("f", "fold"):
        return Action(ActionType.FOLD)
    if s in ("x", "check"):
        return Action(ActionType.CHECK)
    if s in ("c", "call"):
        return Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero]))
    if s in ("a", "allin", "all-in", "jam"):
        return Action(ActionType.ALL_IN, view.stacks[view.hero])
    if s.startswith("b") or s.startswith("r"):
        rest = s[1:].strip()
        if rest.endswith("%"):
            frac = float(rest[:-1]) / 100.0
            target = view.current_bet + int(round(frac * (view.pot+view.to_call)))
        elif rest.endswith("chips"):
            target = int(round(float(rest[:-5])))
        else:
            target = int(round(float(rest.removesuffix("bb"))*view.bb))
        if target>view.max_raise_to:
            raise ValueError(f"Maximum raise-to is {view.max_raise_to/view.bb:g} bb")
        put = target-view.bets[view.hero]
        if put<=0:
            raise ValueError("Bet size must add chips")
        if put==view.stacks[view.hero]:
            return Action(ActionType.ALL_IN,put)
        return Action(ActionType.RAISE if view.current_bet else ActionType.BET,put)
    raise ValueError("commands: f/x/c/a  |  b<bb>  r<bb>  b50%  e.g. r3, b2.5, b75%")


class Human:
    name = "You"

    def reset(self):
        pass

    def act(self, view):
        print()
        print(self._render(view))
        while True:
            try:
                raw = input("Your action [f/x/c/a/b2.5/r9]: ").strip()
            except EOFError:
                return Action(ActionType.FOLD) if view.to_call > 0 else Action(ActionType.CHECK)
            if not raw:
                continue
            try:
                return parse_human(raw, view)
            except Exception as e:
                print(" ", e)


    def _render(self, view) -> str:
        lines = []
        board = cards_to_str(view.board) if view.board else "—"
        hole = cards_to_str(view.hole)
        lines.append("=" * 56)
        lines.append(f"  {view.street.name:8}  board {board}")
        lines.append(f"  hole {hole}   pot {view.pot} ({view.pot/view.bb:.1f} BB)   SPR {view.spr:.1f}")
        lines.append(f"  to_call {view.to_call} ({view.to_call/view.bb:.2f} BB)   your stack {view.stacks[view.hero]/view.bb:.1f} BB")
        for i in range(view.num_players):
            mark = ">>" if i == view.hero else "  "
            flag = "FOLD" if view.folded[i] else ("ALLIN" if view.all_in[i] else "")
            lines.append(
                f" {mark} {i} {view.positions[i]:4} stack={view.stacks[i]/view.bb:7.1f} BB  bet={view.bets[i]/view.bb:5.1f}  {flag}"
            )
        return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Play heads-up NLH vs HeadsUpBot")
    ap.add_argument("--players", type=int, choices=(2,), default=2)
    ap.add_argument("--seconds", type=float, default=30.0)
    add_policy_arguments(ap)
    ap.add_argument("--stack", type=float, default=100.0, help="stack in BB (50–1000)")
    ap.add_argument("--hero-seat", type=int, default=0)
    ap.add_argument("--fill", default="tag", help="other seats: tag/lag/nit/station/maniac/final")
    ap.add_argument(
        "--rake",
        default="none",
        choices=("casino", "online", "none", "heavy"),
        help="casino=10%% cap 3bb, online=5%% cap 3bb, heavy=10%% cap 2bb, none (default)",
    )
    args = ap.parse_args()
    if not 50 <= args.stack <= 1000:
        ap.error("--stack must be between 50 and 1000 bb")
    resolve_policy_arguments(args,ap)
    n = 2
    hero_seat = args.hero_seat % n
    rake = {
        "casino": RakeConfig.casino(),
        "online": RakeConfig.online(),
        "none": RakeConfig.none(),
        "heavy": RakeConfig.casino_heavy(),
    }[args.rake]

    print("HeadsUpBot — Heads-up No-Limit Hold'em")
    print(f"{n}-max  {args.stack:g} BB starting stacks   blinds 0.5/1")
    if rake.active:
        print(f"Rake {rake.percent:.0%} cap {rake.cap_bb:g}bb" + ("  (no flop no drop)" if rake.no_flop_no_drop else ""))
    else:
        print("Unraked")
    print("Commands: f fold  x check  c call  a all-in  b2.5 bet 2.5bb  r9 raise to 9bb")
    print()

    session_delta = [0] * n
    hand_no = 0
    agents = []
    for i in range(n):
        if i == hero_seat:
            agents.append(Human())
        elif args.full_blueprint:
            from hu_bot import StackAdaptiveBot
            agents.append(StackAdaptiveBot(seed=42,seconds=args.seconds,rake=rake,native_iterations=args.native_iterations,native_compression=args.native_compression,
                resolve_from_street={"turn":2,"river":3}.get(args.resolve_from),river_gadget=args.river_gadget))
        elif args.fill.lower() in ("final", "final_bot", "bot"):
            agents.append(FinalBot(seat=i, seed=1000 + i, quality="high", rake=rake, config=SolverConfig(seconds=args.seconds,preflop_blueprint=args.blueprint,iterations=args.native_iterations,compression=args.native_compression)))
        else:
            if i == (hero_seat + 1) % n:
                agents.append(FinalBot(seat=i, seed=42, quality="high", rake=rake, config=SolverConfig(seconds=args.seconds,preflop_blueprint=args.blueprint,iterations=args.native_iterations,compression=args.native_compression)))
            else:
                agents.append(make_opponent(args.fill, seed=10 + i))
    while True:
        cfg = GameConfig(
            num_players=n,
            stack_bb=args.stack,
            button=hand_no % n,
            rng_seed=None,
            rake_percent=rake.percent,
            rake_cap_bb=rake.cap_bb,
            no_flop_no_drop=rake.no_flop_no_drop,
        )
        game = NLHGame(cfg)
        game.start_hand()
        for ag in agents:
            rst = getattr(ag, "reset", None)
            if rst is not None:
                rst()
        print("\n" + "#" * 56)
        print(f"Hand {hand_no + 1}   button seat {game.button}")
        print(game.pretty_state(hero=hero_seat))
        n_act = 0
        while not game.finished and n_act < 80:
            seat = game.to_act
            street = game.street
            view = game.view(seat)
            action = agents[seat].act(view)
            if seat != hero_seat:
                print(f"  {view.positions[seat]} (seat {seat}, {getattr(agents[seat], 'name', '?')}) -> {action}")
            try:
                game.apply(action)
            except ValueError as exc:
                if seat==hero_seat:
                    print(f"Invalid action: {exc}")
                    continue
                raise RuntimeError(f"Bot produced an invalid action: {action}") from exc
            for ag in agents:
                obs = getattr(ag, "observe_action", None)
                if obs is not None:
                    try:
                        obs(seat, action, street, n)
                    except Exception:
                        pass
            n_act += 1
        if not game.finished:
            raise RuntimeError("Hand exceeded the action limit")
        for ag in agents:
            end = getattr(ag, "end_hand", None)
            if end is not None:
                try:
                    end(n)
                except Exception:
                    pass
        print()
        print(game.pretty_state(hero=None))
        if game.board:
            for i, p in enumerate(game.players):
                if p.hole[0] >= 0 and not p.folded:
                    v = evaluate(list(p.hole) + list(game.board))
                    print(f"  seat {i} {cards_to_str(p.hole)}  {category_name(v)}")
        print(f"Winners: {game.winners}  payouts: {game.payouts}  rake: {game.rake_taken}")
        start = cfg.starting_chips
        for i, p in enumerate(game.players):
            session_delta[i] += p.stack - start
        print("Session bb:", [round(d / cfg.bb, 2) for d in session_delta])
        hand_no += 1
        try:
            nxt = input("Enter for next hand, q to quit: ").strip().lower()
        except EOFError:
            break
        if nxt in ("q", "quit", "exit"):
            break
    for agent in agents:
        close=getattr(agent,"close",None)
        if close:close()


if __name__ == "__main__":
    main()
