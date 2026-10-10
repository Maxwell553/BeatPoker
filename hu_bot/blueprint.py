"""Optional experimental ECFR preflop policy, with explicit action translation."""
import json
import math
from pathlib import Path

import numpy as np

from poker.cards import RANK_CHARS
from poker.equity import all_combos, combo_index
from poker.game import Action, ActionType, Street
from strategy.range_model import iter_history
from .preflop import policy as chart_policy


class BlueprintUnavailable(ValueError):
    pass


class PreflopBlueprint:
    def __init__(self,path,allow_stack_scaling=False):
        self.path=str(Path(path).resolve())
        self.allow_stack_scaling=allow_stack_scaling
        self.model=json.loads(Path(path).read_text())
        if self.model.get("format")!=1 or len(self.model["hand_buckets"])!=169:
            raise ValueError("Unsupported preflop blueprint")
        self.stack_bb=float(self.model["stack_bb"])
        self.info={}
        indexes=[]
        for a,b in all_combos():
            hi,lo=sorted((a%13,b%13),reverse=True)
            name=RANK_CHARS[hi]+RANK_CHARS[lo]+("" if hi==lo else "s" if a//13==b//13 else "o")
            indexes.append(self.model["hand_buckets"][name])
        self.indexes=np.asarray(indexes,dtype=int)

    def policy(self,view,rake=None):
        start=min(s+c for s,c in zip(view.stacks,view.committed))/view.bb
        if (not self.allow_stack_scaling and abs(start-self.stack_bb)>0.005) or (rake is not None and rake.percent):
            raise BlueprintUnavailable("Blueprint requires its exact starting stack and no rake")
        if not 50 <= start <= 1000:
            raise BlueprintUnavailable("Starting stack must be 50–1000 bb")
        node_path="root"
        bets=[view.sb if p==view.button else view.bb for p in range(2)]
        translations=[]
        for seat,action,street in iter_history(view):
            if street!=Street.PREFLOP:
                break
            node=self.model["nodes"].get(node_path)
            if node is None:
                raise BlueprintUnavailable("Observed preflop line leaves the trained tree")
            expected=1 if seat==view.button else 0
            if node["player"]!=expected:
                raise ValueError("Blueprint actor mismatch")
            previous=max(bets)
            if action.type in (ActionType.CHECK,ActionType.CALL):
                candidates=[a for a in node["actions"] if a["kind"]=="call"]
                bets[seat]=max(bets)
            elif action.type==ActionType.FOLD:
                candidates=[a for a in node["actions"] if a["kind"]=="fold"]
            else:
                bets[seat]+=action.amount
                actual=bets[seat]*2/view.bb
                allin=actual>=start*2-.01
                candidates=[a for a in node["actions"] if a["kind"]=="raise"
                            and (a["target"]>=self.stack_bb*2-.01)==allin]
                candidates.sort(key=lambda a:abs(math.log((a["target"]/node["bet_to"])/(actual/(previous*2/view.bb)))))
                if candidates and abs(candidates[0]["target"]-actual)>.01:
                    translations.append({"actual_bb":actual/2,"abstract_bb":candidates[0]["target"]/2})
            if not candidates:
                raise BlueprintUnavailable("No matching preflop action class")
            node_path += "/"+candidates[0]["name"]
        node=self.model["nodes"].get(node_path)
        if node is None:
            raise BlueprintUnavailable("No decision after observed preflop history")
        if node["player"]!=(1 if view.hero==view.button else 0):
            raise ValueError("Blueprint current actor mismatch")
        counts=np.asarray(node["counts"],dtype=float)
        if counts.shape!=(169,len(node["actions"])) or np.any(counts<0) or not np.isfinite(counts).all():
            raise ValueError("Invalid blueprint probabilities")
        totals=counts.sum(1,keepdims=True)
        weights=counts/np.maximum(totals,1)
        if len(node["actions"])==1:
            weights[:]=1
        actions=[]
        for a in node["actions"]:
            if a["kind"]=="call":
                action=Action(ActionType.CHECK) if view.to_call==0 else Action(ActionType.CALL,min(view.to_call,view.stacks[view.hero]))
            elif a["kind"]=="fold":
                action=Action(ActionType.FOLD)
            else:
                target=view.max_raise_to if a["target"]==self.stack_bb*2 else min(
                    view.max_raise_to,max(view.min_raise_to,round(view.current_bet*a["target"]/node["bet_to"])))
                put=target-view.bets[view.hero]
                if put>=view.stacks[view.hero]:
                    action=Action(ActionType.ALL_IN,view.stacks[view.hero])
                else:
                    action=Action(ActionType.RAISE if view.to_call else ActionType.BET,put)
            actions.append(action)
        mix=weights[self.indexes]
        unsampled=(totals[:,0]==0)[self.indexes] if len(actions)>1 else np.zeros(1326,dtype=bool)
        # Sparse branches need explicit coverage diagnostics and a baseline for
        # unseen hands. Do not invent an equilibrium policy for unvisited nodes.
        merged={}
        for i,a in enumerate(actions):
            merged[a]=merged.get(a,np.zeros(1326))+mix[:,i]
        if unsampled.any():
            chart_actions,chart_mix=chart_policy(view,rake)
            for i,a in enumerate(chart_actions):
                merged[a]=merged.get(a,np.zeros(1326))+np.where(unsampled,chart_mix[:,i],0)
        self.info={"source":"experimental_ecfr_preflop","blueprint":self.path,
                   "training_stack_bb":self.stack_bb,"actual_stack_bb":start,
                   "stack_translation":abs(start-self.stack_bb)>.005,
                   "node":node_path,"batch":self.model["batch"],"translations":translations,
                   "hero_sampled":not bool(unsampled[combo_index(*view.hole)]),
                   "unsampled_physical_hands":int(unsampled.sum())}
        return list(merged),np.column_stack(list(merged.values()))
