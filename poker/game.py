"""No-Limit Texas Hold'em engine.

Supports 2–9 players, any stack depth, blinds, all-ins, side pots, and
standard incomplete-raise (all-in for less) rules.

Chip unit: 1 BB = 100 chips, 1 SB = 50 chips. All amounts are integers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from poker.cards import Deck, card_to_str, cards_to_str
from poker.evaluator import evaluate


class Street(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4
    HAND_OVER = 5


class ActionType(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "allin"


@dataclass(frozen=True)
class Action:
    type: ActionType
    amount: int = 0  # chips PUT IN on this action (not raise-to)

    def __str__(self) -> str:
        if self.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN):
            return f"{self.type.value}:{self.amount}"
        return self.type.value


@dataclass
class GameConfig:
    num_players: int = 6
    stack_bb: float = 100.0
    sb: int = 50
    bb: int = 100
    button: int = 0
    rng_seed: Optional[int] = None
    # Cash rake. `rake_percent` is a fraction (0.10 = 10%). Cap is in big blinds.
    # Defaults unraked so unit tests keep exact chip accounting; play.py / FinalBot
    # use casino rake in production.
    rake_percent: float = 0.0
    rake_cap_bb: float = 3.0
    no_flop_no_drop: bool = True

    def __post_init__(self):
        if not 2 <= self.num_players <= 9:
            raise ValueError("num_players must be 2..9")
        if self.stack_bb < 10:
            raise ValueError("stack_bb too small")

    @property
    def starting_chips(self) -> int:
        return int(round(self.stack_bb * self.bb))

    def rake_chips(self, pot: int, saw_flop: bool) -> int:
        if self.rake_percent <= 1e-12 or pot <= 0:
            return 0
        if self.no_flop_no_drop and not saw_flop:
            return 0
        cap = int(round(self.rake_cap_bb * self.bb))
        return min(cap, int(pot * self.rake_percent))


@dataclass
class PlayerState:
    stack: int
    bet: int = 0  # chips committed this street
    committed: int = 0  # chips committed this hand (for side pots)
    folded: bool = False
    all_in: bool = False
    hole: Tuple[int, int] = field(default_factory=lambda: (-1, -1))
    acted_this_street: bool = False


# Position names by table size, indexed from UTG going clockwise to BB.
# We compute them from button.
_POS_NAMES = {
    2: ["BTN", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["BTN", "SB", "BB", "UTG"],
    5: ["BTN", "SB", "BB", "UTG", "CO"],
    6: ["BTN", "SB", "BB", "UTG", "HJ", "CO"],
    7: ["BTN", "SB", "BB", "UTG", "MP", "HJ", "CO"],
    8: ["BTN", "SB", "BB", "UTG", "UTG1", "MP", "HJ", "CO"],
    9: ["BTN", "SB", "BB", "UTG", "UTG1", "UTG2", "MP", "HJ", "CO"],
}


def position_name(num_players: int, seat: int, button: int) -> str:
    """Name of `seat` given `button` (both 0..N-1)."""
    # offset: 0 = button, 1 = SB (or BB in HU), ...
    offset = (seat - button) % num_players
    names = _POS_NAMES[num_players]
    return names[offset]


def positions_for_table(n: int) -> List[str]:
    return list(_POS_NAMES[n])


@dataclass
class PlayerView:
    """Information-set snapshot for a single acting player (no opponents' hole cards)."""

    hero: int
    num_players: int
    button: int
    street: Street
    board: List[int]
    pot: int
    stacks: List[int]
    bets: List[int]
    folded: List[bool]
    all_in: List[bool]
    hole: Tuple[int, int]
    to_call: int
    min_raise_to: int
    max_raise_to: int
    can_check: bool
    legal: List[Action]
    action_history: List[Tuple[int, Action]]
    sb: int
    bb: int
    position: str
    positions: List[str]
    committed: List[int]
    current_bet: int  # highest bet this street
    players_in_hand: int
    effective_stack: int
    street_history: List[Street] = field(default_factory=list)

    @property
    def stack_bb(self) -> float:
        return self.stacks[self.hero] / self.bb

    @property
    def pot_bb(self) -> float:
        return self.pot / self.bb

    @property
    def spr(self) -> float:
        if self.pot <= 0:
            return 1000.0
        return self.effective_stack / self.pot


class NLHGame:
    def __init__(self, config: GameConfig, rng: Optional[random.Random] = None):
        self.cfg = config
        self.rng = rng or random.Random(config.rng_seed)
        n = config.num_players
        self.players = [PlayerState(stack=config.starting_chips) for _ in range(n)]
        self.button = config.button % n
        self.board: List[int] = []
        self.street = Street.PREFLOP
        self.to_act = 0
        self.last_aggressor = -1
        self.last_full_raise_size = config.bb  # preflop min raise increment is 1 BB
        self.current_bet = 0
        self.action_history: List[Tuple[int, Action]] = []
        self.street_history: List[Street] = []
        self.hand_history: List[str] = []
        self.winners: List[int] = []
        self.payouts: List[int] = [0] * n
        self.deck: Optional[Deck] = None
        self.hand_number = 0
        self.finished = False
        self._betting_reopened = True
        self.rake_taken = 0

    @property
    def n(self) -> int:
        return self.cfg.num_players

    @property
    def bb(self) -> int:
        return self.cfg.bb

    @property
    def sb(self) -> int:
        return self.cfg.sb

    def pot_total(self) -> int:
        return sum(p.committed for p in self.players)

    def live_players(self) -> List[int]:
        return [i for i, p in enumerate(self.players) if not p.folded]

    def players_can_act(self) -> List[int]:
        return [i for i, p in enumerate(self.players) if not p.folded and not p.all_in]

    def start_hand(self, hole_cards: Optional[Sequence[Tuple[int, int]]] = None, board: Optional[Sequence[int]] = None):
        """Deal a new hand. Optionally inject cards for testing."""
        n = self.n
        for p in self.players:
            p.bet = 0
            p.committed = 0
            p.folded = p.stack <= 0
            p.all_in = p.stack <= 0
            p.hole = (-1, -1)
            p.acted_this_street = False
        self.board = []
        self.street = Street.PREFLOP
        self.action_history = []
        self.street_history = []
        self.hand_history = []
        self.winners = []
        self.payouts = [0] * n
        self.rake_taken = 0
        self.finished = False
        self.last_full_raise_size = self.cfg.bb
        self._betting_reopened = True
        self.hand_number += 1

        dead = []
        if hole_cards:
            for h in hole_cards:
                if h[0] >= 0:
                    dead.extend([h[0], h[1]])
        if board:
            dead.extend(board)
        self.deck = Deck(self.rng, dead=dead)

        if hole_cards:
            for i, h in enumerate(hole_cards):
                if h[0] >= 0:
                    self.players[i].hole = (h[0], h[1])
                elif not self.players[i].folded:
                    c = self.deck.draw(2)
                    self.players[i].hole = (c[0], c[1])
        else:
            for i, p in enumerate(self.players):
                if p.stack > 0:
                    c = self.deck.draw(2)
                    p.hole = (c[0], c[1])

        # Blinds
        if n == 2:
            sb_seat = self.button
            bb_seat = (self.button + 1) % n
        else:
            sb_seat = (self.button + 1) % n
            bb_seat = (self.button + 2) % n

        self._post_blind(sb_seat, self.cfg.sb)
        self._post_blind(bb_seat, self.cfg.bb)
        self.current_bet = max(p.bet for p in self.players)

        if n == 2:
            self.to_act = self.button  # SB/BTN first preflop
        else:
            self.to_act = (self.button + 3) % n  # UTG
        self._skip_dead()
        self.last_aggressor = bb_seat

        if board:
            # Pre-set remaining board; still play streets normally but draw is skipped
            self._preset_board = list(board)
        else:
            self._preset_board = None

        if self._hand_should_end_now():
            self._runout_and_showdown()

    def _post_blind(self, seat: int, amount: int) -> None:
        p = self.players[seat]
        pay = min(p.stack, amount)
        p.stack -= pay
        p.bet += pay
        p.committed += pay
        if p.stack == 0:
            p.all_in = True

    def _skip_dead(self) -> None:
        n = self.n
        for _ in range(n):
            p = self.players[self.to_act]
            if not p.folded and not p.all_in and p.stack > 0:
                return
            self.to_act = (self.to_act + 1) % n

    def _hand_should_end_now(self) -> bool:
        live = self.live_players()
        if len(live) <= 1:
            return True
        can = self.players_can_act()
        if len(can) <= 1:
            # At most one player can act: if they have matched the bet (or everyone else all-in)
            if not can:
                return True
            i = can[0]
            if self.players[i].bet >= self.current_bet:
                return True
        return False

    def legal_actions(self, seat: Optional[int] = None) -> List[Action]:
        if self.finished or self.street in (Street.SHOWDOWN, Street.HAND_OVER):
            return []
        i = self.to_act if seat is None else seat
        if i != self.to_act:
            return []
        p = self.players[i]
        if p.folded or p.all_in:
            return []
        to_call = self.current_bet - p.bet
        acts: List[Action] = []
        if to_call > 0:
            acts.append(Action(ActionType.FOLD))
            call_amt = min(to_call, p.stack)
            acts.append(Action(ActionType.CALL, call_amt))
        else:
            acts.append(Action(ActionType.CHECK))

        # Raises / bets
        can_raise = True
        if to_call > 0 and not self._betting_reopened:
            # Facing an incomplete all-in raise: players who already acted cannot re-raise.
            # Players yet to act still may raise if they have chips for a full raise.
            # We treat "already acted" as acted_this_street after a prior full bet.
            if p.acted_this_street:
                can_raise = False

        can_raise = can_raise and any(j != i for j in self.players_can_act())
        if can_raise and p.stack > to_call:
            remaining = p.stack  # chips they can put in
            # Min raise-to: current_bet + last_full_raise_size
            min_raise_to = self.current_bet + self.last_full_raise_size
            # Max raise-to: current_bet + (stack - to_call) = p.bet + p.stack
            max_raise_to = p.bet + p.stack

            if to_call == 0:
                # Bet: min bet is 1 BB (or remaining stack)
                min_bet = min(self.cfg.bb, p.stack)
                # Represent as putting `min_bet` chips in
                if p.stack <= min_bet:
                    acts.append(Action(ActionType.ALL_IN, p.stack))
                else:
                    # Discrete suggestions plus all-in are added by the bot;
                    # engine exposes the min bet and all-in, plus a generic BET
                    # marker with amount=0 meaning "any legal size".
                    acts.append(Action(ActionType.BET, min_bet))
                    if p.stack > min_bet:
                        acts.append(Action(ActionType.ALL_IN, p.stack))
            else:
                # Need to put in at least min(stack, to_call + last_full_raise_size)
                min_put = min(p.stack, to_call + self.last_full_raise_size)
                if p.stack <= to_call:
                    pass  # already handled as call/all-in via CALL
                elif p.stack < to_call + self.last_full_raise_size:
                    # All-in for less than a full raise
                    acts.append(Action(ActionType.ALL_IN, p.stack))
                else:
                    acts.append(Action(ActionType.RAISE, min_put))
                    if p.stack > min_put:
                        acts.append(Action(ActionType.ALL_IN, p.stack))
        return acts

    def min_raise_to(self, seat: Optional[int] = None) -> int:
        i = self.to_act if seat is None else seat
        p = self.players[i]
        return min(p.bet + p.stack, self.current_bet + self.last_full_raise_size)

    def max_raise_to(self, seat: Optional[int] = None) -> int:
        i = self.to_act if seat is None else seat
        p = self.players[i]
        return p.bet + p.stack

    def apply(self, action: Action) -> None:
        if self.finished:
            raise RuntimeError("Hand is over")
        i = self.to_act
        p = self.players[i]
        to_call = self.current_bet - p.bet

        if action.type == ActionType.FOLD:
            if to_call <= 0:
                raise ValueError("Cannot fold when checking is free")
            p.folded = True
            p.acted_this_street = True
            self.action_history.append((i, action))
            self.street_history.append(self.street)
            self._advance_after_action(i, is_aggro=False)
            return

        if action.type == ActionType.CHECK:
            if to_call > 0:
                raise ValueError("Cannot check facing a bet")
            p.acted_this_street = True
            self.action_history.append((i, action))
            self.street_history.append(self.street)
            self._advance_after_action(i, is_aggro=False)
            return

        if action.type == ActionType.CALL:
            if to_call <= 0:
                raise ValueError("Cannot call when checking is free")
            pay = min(to_call, p.stack)
            self._commit(i, pay)
            p.acted_this_street = True
            self.action_history.append((i, Action(ActionType.CALL, pay)))
            self.street_history.append(self.street)
            self._advance_after_action(i, is_aggro=False)
            return

        # BET / RAISE / ALL_IN — `action.amount` is chips put in this action.
        put = action.amount
        if put <= 0 or put > p.stack:
            raise ValueError(f"Invalid put amount {put} (stack {p.stack})")
        if put < to_call and put < p.stack:
            raise ValueError("Must at least call or go all-in")

        new_bet = p.bet + put
        raise_size = new_bet - self.current_bet  # 0 if call, >0 if raise
        if raise_size > 0:
            if not any(j != i for j in self.players_can_act()):
                raise ValueError("Cannot raise an all-in opponent")
            if not self._betting_reopened and p.acted_this_street:
                raise ValueError("Betting has not reopened")
            if raise_size < self.last_full_raise_size and put != p.stack:
                raise ValueError("Raise below the legal minimum")
        was_full_raise = False
        if new_bet > self.current_bet:
            if raise_size >= self.last_full_raise_size:
                was_full_raise = True
                self.last_full_raise_size = raise_size
                self._betting_reopened = True
            else:
                # Incomplete raise (all-in for less)
                self._betting_reopened = False
            self.current_bet = new_bet
            self.last_aggressor = i
            for pl in self.players:
                if pl is not p and not pl.folded and not pl.all_in:
                    pl.acted_this_street = False

        self._commit(i, put)
        p.acted_this_street = True
        kind = action.type
        if put == (p.stack + put) or p.stack == 0:
            kind = ActionType.ALL_IN
        elif to_call == 0:
            kind = ActionType.BET
        else:
            kind = ActionType.RAISE
        self.action_history.append((i, Action(kind, put)))
        self.street_history.append(self.street)
        self._advance_after_action(i, is_aggro=new_bet >= self.current_bet and raise_size > 0)
        _ = was_full_raise

    def apply_raise_to(self, raise_to: int) -> None:
        """Convenience: raise/bet so that this player's street bet equals raise_to."""
        i = self.to_act
        p = self.players[i]
        put = raise_to - p.bet
        if put >= p.stack:
            self.apply(Action(ActionType.ALL_IN, p.stack))
        elif self.current_bet == p.bet:
            self.apply(Action(ActionType.BET, put))
        else:
            self.apply(Action(ActionType.RAISE, put))

    def _commit(self, i: int, amount: int) -> None:
        p = self.players[i]
        amount = min(amount, p.stack)
        p.stack -= amount
        p.bet += amount
        p.committed += amount
        if p.stack == 0:
            p.all_in = True

    def _advance_after_action(self, actor: int, is_aggro: bool) -> None:
        live = self.live_players()
        if len(live) == 1:
            self._award_uncontested(live[0])
            return

        if self._betting_round_closed():
            self._end_street()
            return

        self.to_act = (actor + 1) % self.n
        self._skip_dead()
        # Safety: if we looped everyone and still not closed, check again
        if self.players[self.to_act].folded or self.players[self.to_act].all_in:
            if self._betting_round_closed():
                self._end_street()

    def _betting_round_closed(self) -> bool:
        live = [i for i in self.live_players() if not self.players[i].all_in]
        if len(live) == 0:
            return True
        # Everyone who can act has acted and matched current_bet
        for i, p in enumerate(self.players):
            if p.folded or p.all_in:
                continue
            if not p.acted_this_street:
                return False
            if p.bet < self.current_bet and p.stack > 0:
                return False
        # At least one live player: if only one can still act and they matched, close
        return True

    def _end_street(self) -> None:
        for p in self.players:
            p.bet = 0
            p.acted_this_street = False
        self.current_bet = 0
        self.last_full_raise_size = self.cfg.bb
        self._betting_reopened = True

        live = self.live_players()
        can_act = self.players_can_act()

        if len(live) == 1:
            self._award_uncontested(live[0])
            return

        if self.street == Street.PREFLOP:
            self._deal_board(3)
            self.street = Street.FLOP
        elif self.street == Street.FLOP:
            self._deal_board(1)
            self.street = Street.TURN
        elif self.street == Street.TURN:
            self._deal_board(1)
            self.street = Street.RIVER
        elif self.street == Street.RIVER:
            self._showdown()
            return

        if len(can_act) <= 1:
            # Run out remaining board and showdown
            self._runout_and_showdown()
            return

        # First to act postflop: first live player left of button
        self.to_act = (self.button + 1) % self.n
        self._skip_dead()
        self.last_aggressor = -1

    def _deal_board(self, n: int) -> None:
        if self._preset_board is not None:
            have = len(self.board)
            self.board.extend(self._preset_board[have : have + n])
            return
        self.board.extend(self.deck.draw(n))

    def _runout_and_showdown(self) -> None:
        need = 5 - len(self.board)
        if need > 0:
            self._deal_board(need)
        self.street = Street.RIVER
        live = self.live_players()
        if len(live) == 1:
            self._award_uncontested(live[0])
        else:
            self._showdown()

    def _saw_flop(self) -> bool:
        return len(self.board) >= 3

    def _net_payouts(self, payouts: List[int]) -> List[int]:
        """Take rake from the awarded pot; leftover chips go to the largest shares."""
        pot = sum(payouts)
        rake = self.cfg.rake_chips(pot, self._saw_flop())
        self.rake_taken = rake
        if rake <= 0 or pot <= 0:
            return list(payouts)
        remaining = pot - rake
        out = [0] * len(payouts)
        assigned = 0
        order = sorted(range(len(payouts)), key=lambda i: (-payouts[i], i))
        for i in order:
            if payouts[i] <= 0:
                continue
            share = (payouts[i] * remaining) // pot
            out[i] = share
            assigned += share
        extra = remaining - assigned
        for i in order:
            if extra <= 0:
                break
            if payouts[i] > 0:
                out[i] += 1
                extra -= 1
        return out

    def _award_uncontested(self, winner: int) -> None:
        self._refund_uncalled()
        pot = self.pot_total()
        net = self._net_payouts([pot if i == winner else 0 for i in range(self.n)])
        self.players[winner].stack += net[winner]
        self.payouts = net
        self.winners = [winner]
        self.street = Street.HAND_OVER
        self.finished = True

    def _showdown(self) -> None:
        """Distribute main + side pots, then take rake from the awarded chips."""
        self.street = Street.SHOWDOWN
        self._refund_uncalled()
        contrib = [p.committed for p in self.players]
        folded = [p.folded for p in self.players]
        hole = [p.hole for p in self.players]
        board = list(self.board)
        payouts, winners = settle_pots(contrib, folded, hole, board)
        payouts = self._net_payouts(payouts)
        for i, amt in enumerate(payouts):
            self.players[i].stack += amt
            self.payouts[i] = amt
        self.winners = winners
        self.street = Street.HAND_OVER
        self.finished = True

    def _refund_uncalled(self) -> None:
        """Return unmatched contributions before calculating pot rake."""
        order = sorted(range(self.n), key=lambda i: self.players[i].committed, reverse=True)
        top, second = (self.players[i] for i in order[:2])
        excess = top.committed - second.committed
        if excess > 0:
            top.committed -= excess
            top.bet = max(0, top.bet - excess)
            top.stack += excess

    def view(self, hero: int) -> PlayerView:
        p = self.players[hero]
        to_call = max(0, self.current_bet - p.bet)
        live_stacks = [
            self.players[i].stack
            for i in self.live_players()
            if i != hero or True
        ]
        opp_live = [self.players[i].stack for i in self.live_players() if i != hero]
        eff = min([p.stack] + opp_live) if opp_live else p.stack
        positions = [position_name(self.n, i, self.button) for i in range(self.n)]
        return PlayerView(
            hero=hero,
            num_players=self.n,
            button=self.button,
            street=self.street,
            board=list(self.board),
            pot=self.pot_total(),
            stacks=[pl.stack for pl in self.players],
            bets=[pl.bet for pl in self.players],
            folded=[pl.folded for pl in self.players],
            all_in=[pl.all_in for pl in self.players],
            hole=p.hole,
            to_call=to_call,
            min_raise_to=self.min_raise_to(hero),
            max_raise_to=self.max_raise_to(hero),
            can_check=to_call == 0,
            legal=self.legal_actions(),
            action_history=list(self.action_history),
            street_history=list(self.street_history),
            sb=self.cfg.sb,
            bb=self.cfg.bb,
            position=positions[hero],
            positions=positions,
            committed=[pl.committed for pl in self.players],
            current_bet=self.current_bet,
            players_in_hand=len(self.live_players()),
            effective_stack=eff,
        )

    def rotate_button(self) -> None:
        self.button = (self.button + 1) % self.n

    def pretty_state(self, hero: Optional[int] = None) -> str:
        lines = []
        street = self.street.name
        board = cards_to_str(self.board) if self.board else "(none)"
        lines.append(f"Street={street}  Board={board}  Pot={self.pot_total()} ({self.pot_total()/self.bb:.1f} BB)")
        for i, p in enumerate(self.players):
            pos = position_name(self.n, i, self.button)
            hole = cards_to_str(p.hole) if (hero is None or i == hero) and p.hole[0] >= 0 else "??"
            flags = []
            if p.folded:
                flags.append("FOLDED")
            if p.all_in:
                flags.append("ALLIN")
            if i == self.to_act and not self.finished:
                flags.append("TO ACT")
            extra = " ".join(flags)
            lines.append(
                f"  [{i}] {pos:4} hole={hole:5} stack={p.stack:6} bet={p.bet:5} {extra}"
            )
        return "\n".join(lines)


def settle_pots(
    contrib: Sequence[int],
    folded: Sequence[bool],
    hole: Sequence[Tuple[int, int]],
    board: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """Compute side pots and award them. Returns (payouts, list of all winners)."""
    n = len(contrib)
    payouts = [0] * n
    all_winners = []
    remaining = list(contrib)
    eligible = [i for i in range(n) if remaining[i] > 0]

    while any(remaining[i] > 0 for i in eligible):
        levels = sorted(set(remaining[i] for i in range(n) if remaining[i] > 0))
        level = levels[0]
        pot = 0
        pot_eligible = []
        for i in range(n):
            if remaining[i] >= level:
                pot += level
                remaining[i] -= level
                if not folded[i]:
                    pot_eligible.append(i)
        if not pot_eligible:
            # All folded at this layer (shouldn't happen); dump to last nonfold historically
            break
        if len(pot_eligible) == 1:
            w = pot_eligible[0]
            payouts[w] += pot
            if w not in all_winners:
                all_winners.append(w)
            continue
        best = -1
        winners = []
        for i in pot_eligible:
            v = evaluate(list(hole[i]) + list(board))
            if v > best:
                best = v
                winners = [i]
            elif v == best:
                winners.append(i)
        share = pot // len(winners)
        extra = pot - share * len(winners)
        for k, w in enumerate(winners):
            payouts[w] += share + (1 if k < extra else 0)
            if w not in all_winners:
                all_winners.append(w)
        eligible = [i for i in range(n) if remaining[i] > 0]
    return payouts, all_winners
