"""Fast, experimental full-game checkpoint policy with explicit search fallback."""
from pathlib import Path
import atexit
import json
import math
import random
import select
import subprocess
import tempfile

import numpy as np

from poker.game import Action, ActionType, Street, GameConfig, NLHGame
from poker.equity import combo_index
from strategy.range_model import iter_history
from .bot import HeadsUpBot, SolverConfig
from .card_config import load_card_config
from .betting_config import load_betting_config

ROOT=Path(__file__).resolve().parents[1]


class PolicyServer:
    def __init__(self,stack=200,batch=9,card_config=None,betting_config=None):
        self.stack,self.batch=stack,batch
        self.card_config=load_card_config(card_config)
        self.betting_config=load_betting_config(stack,betting_config)
        self.stderr=tempfile.TemporaryFile(mode="w+t")
        checkpoint=self.card_config.checkpoint(stack,self.betting_config.name)
        args=[ROOT/"vendor/slumbot2019/bin/policy_server",ROOT/"vendor/slumbot2019/runs/holdem_params",
              self.betting_config.path,checkpoint,str(batch),self.card_config.path]
        self.process=subprocess.Popen(list(map(str,args)),cwd=ROOT/"training/slumbot_data",
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,bufsize=1)
        atexit.register(self.close)
        try:
            ready=self._read()
            if ready.get("stack_bb")!=stack or ready.get("card_abstraction")!=self.card_config.name or ready.get("betting_abstraction")!=self.betting_config.name:
                raise RuntimeError("Full-game checkpoint stack, card, or betting abstraction mismatch")
        except Exception:
            self.close()
            raise

    def _read(self):
        if not select.select([self.process.stdout],[],[],30)[0]:
            raise RuntimeError("Policy server timed out")
        line=self.process.stdout.readline()
        if not line:
            self.stderr.seek(0)
            raise RuntimeError("Policy server exited: "+self.stderr.read()[-1000:])
        result=json.loads(line)
        if "error" in result:
            raise ValueError(result["error"])
        return result

    def query(self,view,path):
        convert=lambda c:(c%13)*4+c//13
        cards=[convert(c) for c in (*view.hole,*view.board)]
        line=" ".join(map(str,[view.street.value,*cards,path]))
        self.process.stdin.write(line+"\n")
        self.process.stdin.flush()
        return self._read()

    def query_all(self,view,path):
        cards=[(c%13)*4+c//13 for c in view.board]
        line=" ".join(map(str,["range",view.street.value,*cards,path]))
        self.process.stdin.write(line+"\n")
        self.process.stdin.flush()
        rows=self._read()["rows"]
        node=rows[0]["node"]
        counts=np.zeros((1326,len(node["actions"])))
        possible=np.zeros(1326,dtype=bool)
        for row in rows:
            a,b=[(c//4)+13*(c%4) for c in row["hole"]]
            i=combo_index(a,b)
            counts[i]=row["node"]["counts"]
            possible[i]=True
        return node,counts,possible

    def close(self):
        if getattr(self,"process",None) is not None:
            self.process.terminate()
            try:self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            self.process=None
            self.stderr.close()


class FullBlueprintBot:
    name="FullBlueprintBot"

    def __init__(self,stack=200,batch=9,seed=8128,fallback_seconds=30,allow_stack_scaling=False,resolve_from_street=None,card_config=None,river_gadget=False,river_entry_floor=0.,betting_config=None,fallback_from_current_street=True,native_iterations=1000,native_compression=True):
        self.native_iterations=SolverConfig(iterations=native_iterations,compression=native_compression).iterations
        self.native_compression=native_compression
        if resolve_from_street not in (None,2,3):
            raise ValueError("Blueprint refinement can start on the turn or river")
        if river_gadget and resolve_from_street is not None:
            raise ValueError("Choose either native later-street search or river gadget refinement")
        self.river_gadget=river_gadget
        if not math.isfinite(river_entry_floor) or not 0<=river_entry_floor<1 or (river_entry_floor and not river_gadget):
            raise ValueError("River entry floor requires the gadget and a value in [0,1)")
        self.river_entry_floor=river_entry_floor
        self.river_solution=None
        self.stack,self.batch=stack,batch
        self.resolve_from_street=resolve_from_street
        self.refiner=None
        self.range_info={}
        self.allow_stack_scaling=allow_stack_scaling
        self.rng=random.Random(seed)
        self.seed=seed
        self.card_config=load_card_config(card_config)
        self.betting_config=load_betting_config(stack,betting_config)
        self.server=PolicyServer(stack,batch,self.card_config.path,self.betting_config.path)
        self.fallback_seconds=fallback_seconds
        self.fallback_from_current_street=fallback_from_current_street
        self.fallback_range_info={}
        self.fallback=None
        self.last_decision={}
        self.fallback_count=0

    def reset(self):
        self.last_decision={}
        self.river_solution=None
        self.fallback_count=0
        if self.fallback:self.fallback.reset()
        if self.refiner:self.refiner.reset()
        self.range_info={}
        self.fallback_range_info={}

    def close(self):
        self.server.close()
        if self.fallback:self.fallback.close()
        if self.refiner:self.refiner.close()

    def observe_action(self,*args,**kwargs):pass
    def end_hand(self,*args):pass

    def _action_menu(self,view,node,contributions):
        """Map every model action to its actual legal action, retaining aliases."""
        actions=[];adjustments=[]
        for entry in node['actions']:
            kind=entry['kind']
            if kind=='call':
                action=Action(ActionType.CHECK) if view.to_call==0 else Action(ActionType.CALL,min(view.to_call,view.stacks[view.hero]))
            elif kind=='fold':
                if view.to_call==0:raise ValueError('Abstract fold at a free-check decision')
                action=Action(ActionType.FOLD)
            else:
                if not any(a.type in (ActionType.RAISE,ActionType.BET,ActionType.ALL_IN) for a in view.legal):
                    raise ValueError('Abstract raise when betting is closed')
                if entry['total_target']==self.stack*2:target=view.max_raise_to
                else:
                    cumulative=round(max(contributions)*entry['total_target']/node['bet_to'])
                    target=cumulative-(view.committed[view.hero]-view.bets[view.hero])
                    legal=min(view.max_raise_to,max(view.min_raise_to,target))
                    if legal!=target:adjustments.append({'requested_to':target,'legal_to':legal})
                    target=legal
                put=target-view.bets[view.hero]
                action=Action(ActionType.ALL_IN,view.stacks[view.hero]) if put>=view.stacks[view.hero] else Action(ActionType.RAISE if view.current_bet else ActionType.BET,put)
            actions.append(action)
        return actions,adjustments

    def _replay(self,view,before_street=None,on_action=None):
        if view.num_players!=2:
            raise ValueError("Full blueprint is heads-up only")
        start=min(s+c for s,c in zip(view.stacks,view.committed))
        if not 50 <= start/view.bb <= 1000:
            raise ValueError("Starting stack must be 50–1000 bb")
        if not self.allow_stack_scaling and abs(start/view.bb-self.stack)>.005:
            raise ValueError("Full blueprint requires its exact starting stack")
        contributions=[view.sb if p==view.button else view.bb for p in range(2)]
        public_state=None
        if on_action is not None:
            # Only legality is reconstructed. Placeholder private cards depend
            # on the public board, never either player's actual holding.
            available=[c for c in range(52) if c not in view.board]
            public_state=NLHGame(GameConfig(num_players=2,stack_bb=start/view.bb,
                button=view.button,sb=view.sb,bb=view.bb,rng_seed=0))
            public_state.start_hand(hole_cards=[tuple(available[:2]),tuple(available[2:4])],
                board=list(view.board)+available[4:9-len(view.board)])
        path="root"
        translations=[]
        for seat,action,street in iter_history(view):
            if before_street is not None and street.value>=before_street.value:
                break
            node=self.server.query(view,path)
            expected=1 if seat==view.button else 0
            if node.get("street")!=street.value or node.get("player")!=expected:
                raise ValueError("Observed history leaves the trained tree")
            previous=max(contributions)
            earlier_contributions=contributions.copy()
            if action.type in (ActionType.CHECK,ActionType.CALL):
                contributions[seat]=min(start,previous)
                candidates=[a for a in node["actions"] if a["kind"]=="call"]
            elif action.type==ActionType.FOLD:
                candidates=[a for a in node["actions"] if a["kind"]=="fold"]
            else:
                contributions[seat]=min(start,contributions[seat]+action.amount)
                actual=contributions[seat]
                if actual<=previous:
                    candidates=[a for a in node["actions"] if a["kind"]=="call"]
                else:
                    allin=actual==start
                    candidates=[a for a in node["actions"] if a["kind"]=="raise"
                                and (a["total_target"]==self.stack*2)==allin]
                    candidates.sort(key=lambda a:abs(math.log((a["total_target"]/node["bet_to"])/(actual/previous))))
                    if candidates:
                        mapped=candidates[0]["total_target"]*view.bb/2
                        if abs(mapped-actual)>.01:
                            translations.append({"street":street.name,"actual_total_bb":actual/view.bb,
                                                 "abstract_total_bb":mapped/view.bb})
            if not candidates:raise ValueError("Observed action is outside the trained abstraction")
            if on_action is not None:
                if public_state.to_act!=seat or public_state.street!=street:
                    raise ValueError('Public legality replay disagrees with observed history')
                index=node['actions'].index(candidates[0])
                menu,_=self._action_menu(public_state.view(seat),node,earlier_contributions)
                aliases=[i for i,physical in enumerate(menu) if physical==menu[index]]
                on_action(path,{**node,'observed_action_indexes':aliases},index,seat)
                public_state.apply(action)
            path += "/"+candidates[0]["name"]
        node=self.server.query(view,path)
        return node,path,translations,contributions,start

    def public_ranges(self,view,action_contamination=.02):
        """Public reaches at this street's start under the trained full policy.

        Multiply each player's earlier action probabilities for every possible
        hand; condition only on public cards, never the actual hero's hole cards.
        A 2% uniform action mixture handles sparse/off-policy observations by
        default. Per-seat values allow unsmoothed hero reaches for the gadget.
        """
        weights=np.ones((2,1326))
        sparse=[];merged=[]
        supplied=np.asarray(action_contamination,dtype=float)
        epsilon=np.broadcast_to(supplied,(2,))
        if not np.isfinite(epsilon).all() or np.any((epsilon<0)|(epsilon>1)):
            raise ValueError("Action contamination must be between zero and one")
        def observe(path,node,index,seat):
            metadata,counts,possible=self.server.query_all(view,path)
            if metadata["actions"]!=node["actions"]:
                raise ValueError("Public and private policy trees disagree")
            totals=counts.sum(axis=1)
            probs=np.full(1326,1/counts.shape[1])
            indexes=node.get('observed_action_indexes',[index])
            if len(indexes)>1:merged.append({'node':path,'seat':seat,'model_action_indexes':indexes})
            probs[:]=len(indexes)/counts.shape[1]
            np.divide(counts[:,indexes].sum(axis=1),totals,out=probs,where=totals>0)
            probs=(1-epsilon[seat])*probs+epsilon[seat]*len(indexes)/counts.shape[1]
            weights[seat]*=probs
            weights[:,~possible]=0
            if weights[seat].max()>0:weights[seat]/=weights[seat].max()
            missing=int(np.sum(possible & (totals==0)))
            if missing:sparse.append({"node":path,"unsampled_physical_hands":missing})
        node,path,translations,contributions,_=self._replay(view,before_street=view.street,on_action=observe)
        if node.get("street")!=view.street.value:
            raise ValueError("Trained history does not reach the requested street")
        if min(weights.sum(axis=1))<=0:
            raise ValueError("No public range mass remains")
        self.range_info={"source":"full_blueprint_public_reach","root_node":path,
            "training_stack_bb":self.stack,"batch":self.batch,
            "action_contamination":float(supplied) if supplied.ndim==0 else epsilon.tolist(),
            "translations":translations,"sparse_nodes":sparse,'merged_action_observations':merged,"safe_resolving":False}
        return weights,view.pot-sum(view.bets),[s+b for s,b in zip(view.stacks,view.bets)]

    def _river_policy(self,view,path):
        from .river_refine import RiverTree,RiverPolicySource
        hero=1 if view.hero==view.button else 0
        key=(tuple(view.board),hero)
        if self.river_solution is None or self.river_solution["key"]!=key:
            if self.fallback_count:
                raise ValueError("Earlier search fallback prevents exact blueprint hero reach reconstruction")
            smoothing=[.02,.02];smoothing[view.hero]=0.
            ranges,actual_pot,_=self.public_ranges(view,action_contamination=smoothing)
            if ranges[view.hero,combo_index(*view.hole)]<=0:
                raise ValueError("Actual hand has no unsmoothed blueprint reach")
            ranges=np.asarray([ranges[1-view.button],ranges[view.button]])
            root=self.range_info["root_node"]
            source=RiverPolicySource(self.server,view)
            tree=RiverTree.from_probe(source,root)
            abstract_pot=source.node(root)[0]["bet_to"]
            policy,stats=tree.refine(hero,ranges[hero,tree.indexes],ranges[1-hero,tree.indexes],
                iterations=2000,seconds=self.fallback_seconds,tolerance_bb=.005*abstract_pot,entry_floor=self.river_entry_floor)
            stats.update(abstract_pot_bb=abstract_pot,actual_pot_bb=actual_pot/view.bb,
                hero_reach_mode="blueprint_without_action_smoothing",
                uniform_extension_nodes=source.uniform_extension,public_range_model=self.range_info.copy(),
                baseline="fixed blueprint with uniform extension at unvisited rows")
            indexes={int(index):i for i,index in enumerate(tree.indexes)}
            self.river_solution={"key":key,"root":root,"tree":tree,"policy":policy,"stats":stats,
                                 "hero_range":ranges[hero,tree.indexes],
                                 "nodes":{p:i for i,p in enumerate(tree.paths)},"hands":indexes}
        solution=self.river_solution
        if path not in solution["nodes"]:raise ValueError("River action leaves the refined tree")
        n=solution["nodes"][path];h=solution["hands"][combo_index(*view.hole)]
        if solution["hero_range"][h]<=0:raise ValueError("Actual hand has no unsmoothed blueprint reach")
        return solution["policy"][n,:solution["tree"].counts[n],h].tolist(),solution["stats"]

    def _decision(self,view):
        node,path,translations,contributions,start=self._replay(view)
        if node.get("street")!=view.street.value or node.get("player")!=(1 if view.hero==view.button else 0):
            raise ValueError("Blueprint and engine disagree about the decision")
        counts=node["counts"]
        original_samples=sum(counts)
        refinement=None
        if self.river_gadget and view.street==Street.RIVER:
            counts,refinement=self._river_policy(view,path)
        total=sum(counts)
        if total<=0:raise ValueError("Unvisited full-game information set")
        actions,adjustments=self._action_menu(view,node,contributions)
        chosen=self.rng.choices(actions,weights=counts)[0]
        self.last_decision={"source":"experimental_full_blueprint","stack_bb":self.stack,"batch":self.batch,
            "card_abstraction":self.card_config.name,
            "betting_abstraction":self.betting_config.name,
            "actual_stack_bb":start/view.bb,"stack_translation":abs(start/view.bb-self.stack)>.005,
            "node":path,"bucket":node["bucket"],"sample_count":original_samples,
            "actions":[str(a) for a in actions],"probabilities":[c/total for c in counts],
            "translations":translations,"legal_size_adjustments":adjustments,"validated_pro_level":False}
        if refinement is not None:
            self.last_decision.update(source="experimental_river_gadget",river_gadget=refinement)
        return chosen

    def act(self,view):
        try:
            if self.resolve_from_street is not None and view.street.value>=self.resolve_from_street:
                if self.refiner is None:
                    self.refiner=HeadsUpBot(seed=self.seed,config=SolverConfig(seconds=self.fallback_seconds,wide_root=True,iterations=self.native_iterations,compression=self.native_compression),
                                           public_range_provider=self.public_ranges)
                action=self.refiner.act(view)
                self.last_decision={**self.refiner.last_decision,"blueprint_refinement":self.range_info.copy()}
                return action
            return self._decision(view)
        except ValueError as exc:
            if self.fallback is None:
                model=self.card_config.preflop_export(self.stack,self.batch,self.betting_config.name)
                if not model.exists():
                    from training.export_checkpoint import export
                    export(self.betting_config.path,
                           self.card_config.checkpoint(self.stack,self.betting_config.name),
                           self.batch,model,self.card_config.path)
                self.fallback=HeadsUpBot(seed=self.seed,config=SolverConfig(seconds=self.fallback_seconds,iterations=self.native_iterations,compression=self.native_compression,
                    preflop_blueprint=str(model),blueprint_stack_scaling=self.allow_stack_scaling))
            if view.street!=Street.PREFLOP and self.fallback.root is None:
                # A late unsupported action must not require rebuilding all
                # earlier streets. Their public blueprint reaches define a
                # smaller current-street game with the actual observed sizes.
                # Keep the original path only when earlier history itself is
                # unsupported, or explicitly requested for a comparison.
                self.fallback.public_range_provider=None
                self.fallback_range_info={"source":"preflop_range_flop_root"}
                if self.fallback_from_current_street:
                    try:
                        prepared=self.public_ranges(view)
                    except ValueError as range_error:
                        self.fallback_range_info["current_street_range_unavailable"]=str(range_error)
                    else:
                        self.fallback.public_range_provider=lambda observed:prepared
                        self.fallback_range_info={**self.range_info,"root_street":view.street.value}
            self.fallback_count+=1
            action=self.fallback.act(view)
            self.last_decision={**self.fallback.last_decision,"full_blueprint_fallback":str(exc),
                                "full_blueprint_fallback_count":self.fallback_count,
                                "fallback_public_ranges":self.fallback_range_info.copy()}
            return action
