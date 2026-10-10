from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np

from poker.cards import card_to_str
from poker.equity import all_combos, combo_index
from poker.game import Action, ActionType, Street
from strategy.rake import RakeConfig
from strategy.range_model import iter_history
from .native import NativeSolver, SolverError
from .preflop import policy, public_flop_ranges


@dataclass(frozen=True)
class SolverConfig:
    seconds: float = 30.0
    iterations: int = 1000
    target_pot_fraction: float = 0.005
    memory_mb: int = 8192
    threads: int = 6
    compact: bool = True
    resolve_later_streets: bool = True
    wide_root: bool = False
    preflop_blueprint: str | None = None
    blueprint_stack_scaling: bool = False
    range_contamination: float = 0.02
    compression: bool = True

    def __post_init__(self):
        if not isinstance(self.compression,bool):raise ValueError('Native compression must be a boolean')
        if isinstance(self.iterations,bool) or not isinstance(self.iterations,int) or not 1<=self.iterations<=2**32-1:
            raise ValueError("Native iterations must be a positive 32-bit integer")
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("Positive time and iteration budgets required")
        if not 0 < self.target_pot_fraction < 1 or self.memory_mb < 32 or self.threads < 1:
            raise ValueError("Invalid solver configuration")
        if not 0 <= self.range_contamination < 1:
            raise ValueError("Range contamination must be in [0,1)")


def range_string(weights):
    parts = []
    for (a,b),w in zip(all_combos(),weights):
        if w <= 0:
            continue
        if (a % 13, a // 13) < (b % 13, b // 13):
            a,b = b,a
        # Native range grammar accepts decimal weights, not scientific notation.
        parts.append(card_to_str(a)+card_to_str(b)+":"+f"{max(float(w),1e-12):.16f}".rstrip("0").rstrip("."))
    return ",".join(parts)


class HeadsUpBot:
    """Native postflop search with public range tracking and mixed actions.

    Search errors are explicit. `last_decision` includes actual convergence,
    timing, and betting-tree scope; it is not a professional-strength claim.
    """
    name = "HeadsUpBot"

    def __init__(self, seat=0, seed=None, quality="high", rake=None, config=None,public_range_provider=None):
        self.seat = seat
        self.rng = random.Random(seed)
        self.rake = rake or RakeConfig.none()
        self.config = config or SolverConfig(seconds=3.0 if quality == "fast" else 30.0)
        self.solver = None
        self.public_range_provider=public_range_provider
        self.blueprint = None
        if self.config.preflop_blueprint:
            from .blueprint import PreflopBlueprint
            self.blueprint = PreflopBlueprint(self.config.preflop_blueprint,self.config.blueprint_stack_scaling)
        self.last_decision = {}
        self.reset()

    def reset(self):
        self.root = None
        self.root_street = Street.FLOP
        self.lines = []
        self.off_tree_resolves = 0
        self.street_resolves = 0
        self.last_decision = {}
        if self.solver is not None:
            self.solver.request({"cmd":"clear"})

    def close(self):
        if self.solver is not None:
            self.solver.close()
            self.solver = None

    def observe_action(self, *args, **kwargs):
        pass  # Public action_history is authoritative, including off-tree sizes.

    def end_hand(self, *args):
        pass

    def _preflop_policy(self, view, rake=None):
        self.preflop_info = {"source":"inherited_preflop_chart"}
        if self.blueprint is not None:
            from .blueprint import BlueprintUnavailable
            try:
                result = self.blueprint.policy(view,rake)
                self.preflop_info = self.blueprint.info.copy()
                return result
            except BlueprintUnavailable as exc:
                self.preflop_info["blueprint_unavailable"] = str(exc)
        return policy(view,rake)

    def _history(self, view):
        hist = []
        bets = [0,0]
        street = self.root_street
        effective = self.root["stack"]
        for seat,a,st in iter_history(view):
            if st.value < self.root_street.value:
                continue
            while street.value < st.value:
                street = Street(street.value+1)
                hist.append({"kind":"chance","card":card_to_str(view.board[street.value+1])})
                effective -= max(bets)
                bets = [0,0]
            if a.type in (ActionType.CHECK, ActionType.FOLD, ActionType.CALL):
                hist.append({"kind":a.type.value})
                if a.type == ActionType.CALL:
                    bets[seat] = min(effective,max(bets))
                continue
            target = min(effective, bets[seat]+a.amount)
            kind = "call" if target <= max(bets) else ("allin" if target == effective else ("bet" if max(bets)==0 else "raise"))
            hist.append({"kind":kind, "amount":target})
            bets[seat] = target
        while street.value < view.street.value:
            street = Street(street.value+1)
            hist.append({"kind":"chance","card":card_to_str(view.board[street.value+1])})
        return hist

    def _solve(self, history, hole):
        c = self.config
        req = {**self.root, "cmd":"solve", "history":history, "lines":self.lines,
               "hole":hole,"seconds":c.seconds,"iterations":c.iterations,
               "target":c.target_pot_fraction,"memory_mb":c.memory_mb,
               "compact":c.compact,"compression":c.compression,"rake_rate":self.rake.percent,
               "wide_root":c.wide_root,"min_bet":self.bb,
               "rake_cap":self.rake.cap_bb*self.bb}
        try:
            return self.solver.request(req, timeout=max(180, c.seconds*3))
        except SolverError as exc:
            if not str(exc).startswith("memory_limit:"):
                raise
            # Keep every observed action exact. Reduce optional raise branches
            # throughout the tree and expose the reduced abstraction in stats.
            req.update(max_bets=1,compact=True,wide_root=False)
            result = self.solver.request(req, timeout=max(180, c.seconds*3))
            result["solve"]["memory_retry"] = str(exc)
            return result

    def _advance_root(self, view):
        """Condition ranges on prior play before starting a smaller street game.

        This is ordinary range-based re-solving, not safe gadget resolving.
        Both ranges use public reaches from the previous strategy.
        """
        history = self._history(view)
        last_chance = max(i for i,a in enumerate(history) if a["kind"]=="chance")
        state = self.solver.request({"cmd":"query","history":history[:last_chance+1]})
        ranges = []
        for player,rg in enumerate(state["ranges"]):
            weights = np.asarray(rg["weights"],dtype=float)
            prior_map={frozenset((hand[:2],hand[2:])):float(weight)
                       for term in self.root["ranges"][player].split(",")
                       for hand,weight in [term.split(":")]}
            board=set(card_to_str(c) for c in view.board)
            prior=np.asarray([prior_map.get(frozenset((h[:2],h[2:])),0)
                              if h[:2] not in board and h[2:] not in board else 0
                              for h in rg["hands"]],dtype=float)
            if prior.sum()<=0:
                raise SolverError("No legal private hands in the public prior")
            prior/=prior.max()
            if weights.sum()>0:
                weights/=weights.max()
                epsilon=self.config.range_contamination
                weights=(1-epsilon)*weights+epsilon*prior
            elif self.config.range_contamination>0:
                weights=prior
            else:
                raise SolverError("Observed history has zero reach under the prior strategy")
            weights[prior==0]=0
            ranges.append(",".join(h+":"+f"{max(float(w),1e-12):.16f}".rstrip("0").rstrip(".")
                                   for h,w in zip(rg["hands"],weights) if w>0))
        self.root = {"board":[card_to_str(c) for c in view.board],
                     "pot":view.pot-sum(view.bets),
                     "stack":min(s+b for s,b in zip(view.stacks,view.bets)),"ranges":ranges}
        self.root_street = view.street
        self.lines = []
        self.street_resolves += 1

    def act(self, view):
        if view.num_players != 2:
            raise ValueError("HeadsUpBot requires exactly two players")
        start = min(s+c for s,c in zip(view.stacks,view.committed))/view.bb
        if not 50 <= start <= 1000:
            raise ValueError("Starting effective stack must be 50–1000 bb")
        self.bb = view.bb
        if view.street == Street.PREFLOP:
            # A chart's call range at 2.5bb must not become a call range vs a
            # 1000bb shove. Price actual equity when there is no future action.
            actions,mix = self._preflop_policy(view,self.rake)
            chart_hand = self.preflop_info["source"] != "experimental_ecfr_preflop" or not self.preflop_info.get("hero_sampled",True)
            if chart_hand and view.to_call > 0 and (view.all_in[1-view.hero] or view.to_call >= view.stacks[view.hero]):
                from .equity import preflop_equity
                from strategy.ranges import Range
                assumption = "QQ+,AKs,AKo" if start <= 200 else "KK+,AKs:0.5"
                weights = Range.from_string(assumption).w
                equity = preflop_equity(view.hole,weights,seed=self.rng.randrange(2**31))
                cost = min(view.to_call,view.stacks[view.hero])
                matched_pot = view.pot-max(0,view.to_call-cost)+cost
                rake = min(self.rake.cap_bb*view.bb,matched_pot*self.rake.percent)
                threshold = cost / max(1,matched_pot-rake)
                self.last_decision = {"source":"preflop_allin_equity","equity":equity,
                    "pot_odds":threshold,"opponent_range_assumption":assumption,
                    "samples":20000,"validated_pro_level":False}
                return Action(ActionType.CALL,cost) if equity >= threshold else Action(ActionType.FOLD)
            weights = mix[combo_index(*view.hole)]
            action = self.rng.choices(actions,weights=weights)[0]
            self.last_decision = {**self.preflop_info, "validated_pro_level":False,
                                  "actions":[str(a) for a in actions],"probabilities":weights.tolist()}
            return action
        if self.solver is None:
            self.solver = NativeSolver(self.config.threads)
        hole = [card_to_str(c) for c in view.hole]
        oop = 1-view.button
        if self.root is None:
            if self.public_range_provider is None:
                ranges,pot,stacks = public_flop_ranges(view,self.rake,self._preflop_policy)
                self.root_street=Street.FLOP
            else:
                ranges,pot,stacks=self.public_range_provider(view)
                self.root_street=view.street
                ranges=np.asarray(ranges,dtype=float)
                if ranges.shape!=(2,1326) or not np.isfinite(ranges).all() or np.any(ranges<0) or np.any(ranges.sum(axis=1)<=0):
                    raise ValueError("Invalid public ranges from provider")
            self.root = {"board":[card_to_str(c) for c in view.board[:self.root_street.value+2]], "pot":pot,
                         "stack":min(stacks),"ranges":[range_string(ranges[oop]),range_string(ranges[view.button])]}
            history = self._history(view)
            self.lines = [history] if history else []
            result = self._solve(history,hole)
        else:
            advanced = self.config.resolve_later_streets and view.street.value > self.root_street.value
            if advanced:
                self._advance_root(view)
            history = self._history(view)
            try:
                if advanced:
                    self.lines = [history] if history else []
                    result = self._solve(history,hole)
                else:
                    result = self.solver.request({"cmd":"query","history":history,"hole":hole})
            except SolverError as exc:
                if not str(exc).startswith("off_tree:"):
                    raise
                # Insert the exact observed amount, re-solve the same public
                # root, and retain all prior inserted lines. No size snapping.
                self.lines.append(history)
                self.off_tree_resolves += 1
                result = self._solve(history,hole)
        expected_player = 0 if view.hero == oop else 1
        if result.get("player") != expected_player:
            raise SolverError("Solver and engine disagree about acting player")
        probabilities = np.asarray(result["probabilities"],dtype=float)
        if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
            raise SolverError("Invalid solver strategy")
        action = self.rng.choices(result["actions"],weights=probabilities)[0]
        self.last_decision = {"source":"native_postflop", **result["solve"],
                              "actions":result["actions"],"probabilities":probabilities.tolist(),
                              "off_tree_resolves":self.off_tree_resolves,
                              "street_resolves":self.street_resolves,"validated_pro_level":False}
        self.last_decision["range_contamination"]=self.config.range_contamination
        self.last_decision["iteration_cap"]=self.config.iterations
        kind = action["kind"]
        if kind == "call":
            return Action(ActionType.CALL,min(view.to_call,view.stacks[view.hero]))
        if kind in ("fold","check"):
            return Action(ActionType(kind))
        put = action["amount"] - view.bets[view.hero]
        if put >= view.stacks[view.hero]:
            return Action(ActionType.ALL_IN,view.stacks[view.hero])
        return Action(ActionType.RAISE if view.to_call else ActionType.BET,put)
