"""Exact-card river refinement with an opponent opt-out gadget and value gate.

The gate compares every opponent hand's best response in the SAME betting tree
and hero root range. It does not certify off-tree actions or earlier streets.
Resolving construction: Brown & Sandholm, NeurIPS 2017, section 4.2.
https://arxiv.org/abs/1705.02955
"""
from dataclasses import dataclass
import time

import numpy as np
from numba import njit

from poker.equity import all_combos
from poker.evaluator import evaluate


class RiverPolicySource:
    """Public full-policy rows with an explicit uniform sparse-row extension."""
    def __init__(self,server,view):
        self.server,self.view=server,view
        self.cache={};self.uniform_extension={}

    def node(self,path):
        if path not in self.cache:
            node=self.server.query(self.view,path);probs=None
            if not node.get("terminal"):
                if node["street"]!=3:raise ValueError("Refinement contains a non-river node")
                _,counts,possible=self.server.query_all(self.view,path)
                totals=counts.sum(axis=1,keepdims=True)
                probs=np.full(counts.shape,1/counts.shape[1])
                np.divide(counts,totals,out=probs,where=totals>0)
                missing=int(np.count_nonzero(possible & (totals[:,0]==0)))
                if missing:self.uniform_extension[path]=missing
            self.cache[path]=(node,probs)
        return self.cache[path]


@njit(cache=True)
def compatible_mass(hands,weights):
    cards=np.zeros(52)
    total=0.
    for h in range(len(hands)):
        total+=weights[h]
        cards[hands[h,0]]+=weights[h];cards[hands[h,1]]+=weights[h]
    result=np.empty(len(hands))
    for h in range(len(hands)):
        result[h]=total-cards[hands[h,0]]-cards[hands[h,1]]+weights[h]
    return result


@njit(cache=True)
def showdown_values(hands,ranks,order,weights):
    """Weighted wins-minus-losses, removing both shared private cards."""
    cards=np.zeros(52);total=0.
    for h in range(len(hands)):
        total+=weights[h]
        cards[hands[h,0]]+=weights[h];cards[hands[h,1]]+=weights[h]
    less=0.;less_cards=np.zeros(52);tie_cards=np.zeros(52);result=np.empty(len(hands));start=0
    while start<len(order):
        end=start;ties=0.
        while end<len(order) and ranks[order[end]]==ranks[order[start]]:
            h=order[end];ties+=weights[h]
            tie_cards[hands[h,0]]+=weights[h];tie_cards[hands[h,1]]+=weights[h]
            end+=1
        for i in range(start,end):
            h=order[i];a,b=hands[h]
            valid=total-cards[a]-cards[b]+weights[h]
            wins=less-less_cards[a]-less_cards[b]
            equal=ties-tie_cards[a]-tie_cards[b]+weights[h]
            result[h]=2*wins+equal-valid
        less+=ties
        # Only cards present in this rank group can have nonzero tie mass.
        # Transfer each aggregate once, retaining the original addition order
        # within each card's rank-prefix sum and clearing scratch for reuse.
        for i in range(start,end):
            h=order[i];a,b=hands[h]
            less_cards[a]+=tie_cards[a];tie_cards[a]=0.
            less_cards[b]+=tie_cards[b];tie_cards[b]=0.
        start=end
    return result


@njit(cache=True)
def fill_probabilities(regrets,counts,policy):
    for n in range(len(counts)):
        for h in range(regrets.shape[2]):
            total=0.
            for a in range(counts[n]):total+=max(regrets[n,a,h],0.)
            for a in range(counts[n]):
                policy[n,a,h]=max(regrets[n,a,h],0.)/total if total>0 else 1./counts[n]


@njit(cache=True)
def probabilities(regrets,counts):
    policy=np.zeros_like(regrets)
    fill_probabilities(regrets,counts,policy)
    return policy


@njit(cache=True)
def values(kinds,actors,counts,children,payoffs,hands,ranks,order,policy,player,opponent,best):
    reach=np.zeros((len(kinds),len(hands)));reach[0]=opponent
    result=np.zeros_like(reach)
    for n in range(len(kinds)):
        for a in range(counts[n]):
            child=children[n,a]
            if actors[n]==player:reach[child]=reach[n]
            else:reach[child]=reach[n]*policy[n,a]
    for n in range(len(kinds)-1,-1,-1):
        if kinds[n]==2:result[n]=payoffs[n]*showdown_values(hands,ranks,order,reach[n])
        elif kinds[n]==1:
            sign=1 if actors[n]==player else -1
            result[n]=sign*payoffs[n]*compatible_mass(hands,reach[n])
        elif actors[n]==player:
            if best:
                result[n]=result[children[n,0]]
                for a in range(1,counts[n]):result[n]=np.maximum(result[n],result[children[n,a]])
            else:
                for a in range(counts[n]):result[n]+=policy[n,a]*result[children[n,a]]
        else:
            for a in range(counts[n]):result[n]+=result[children[n,a]]
    return result[0]


@njit(cache=True)
def half_iteration(kinds,actors,counts,children,payoffs,hands,ranks,order,regrets,average,
                   player,opponent,own,weight,policy,reach,own_reach,result):
    fill_probabilities(regrets,counts,policy)
    reach[0]=opponent
    own_reach[0]=own
    for n in range(len(kinds)):
        for a in range(counts[n]):
            child=children[n,a]
            if actors[n]==player:
                for h in range(len(hands)):
                    reach[child,h]=reach[n,h]
                    own_reach[child,h]=own_reach[n,h]*policy[n,a,h]
            else:
                for h in range(len(hands)):
                    reach[child,h]=reach[n,h]*policy[n,a,h]
                    own_reach[child,h]=own_reach[n,h]
    for n in range(len(kinds)-1,-1,-1):
        if kinds[n]==2:result[n]=payoffs[n]*showdown_values(hands,ranks,order,reach[n])
        elif kinds[n]==1:
            sign=1 if actors[n]==player else -1
            result[n]=sign*payoffs[n]*compatible_mass(hands,reach[n])
        elif actors[n]==player:
            result[n]=0.
            for a in range(counts[n]):
                child=children[n,a]
                for h in range(len(hands)):result[n,h]+=policy[n,a,h]*result[child,h]
            for a in range(counts[n]):
                child=children[n,a]
                for h in range(len(hands)):
                    regrets[n,a,h]=max(regrets[n,a,h]+result[child,h]-result[n,h],0.)
                    average[n,a,h]+=weight*own_reach[n,h]*policy[n,a,h]
        else:
            result[n]=0.
            for a in range(counts[n]):
                child=children[n,a]
                for h in range(len(hands)):result[n,h]+=result[child,h]
    return result[0]


@njit(cache=True)
def iterate(kinds,actors,counts,children,payoffs,hands,ranks,order,regrets,average,
            gadget_regrets,hero,hero_range,exit_cfvs,entry_floor,first,last):
    uniform=np.ones(len(hands))
    policy=np.zeros_like(regrets)
    reach=np.empty((len(kinds),len(hands)))
    own_reach=np.empty_like(reach);result=np.empty_like(reach)
    for iteration in range(first,last):
        total=gadget_regrets[0]+gadget_regrets[1]
        choice=np.full(len(hands),.5)
        for h in range(len(hands)):
            if total[h]>0:choice[h]=gadget_regrets[0,h]/total[h]
        enter=entry_floor+(1-entry_floor)*choice
        weight=float(max(iteration-100,0))
        half_iteration(kinds,actors,counts,children,payoffs,hands,ranks,order,regrets,average,
                       hero,enter,hero_range,weight,policy,reach,own_reach,result)
        continuing=half_iteration(kinds,actors,counts,children,payoffs,hands,ranks,order,regrets,average,
                                  1-hero,hero_range,uniform*enter,weight,policy,reach,own_reach,result)
        # The two gadget actions enter with probabilities 1 and entry_floor.
        # Their common forced-entry component cancels in the regret difference;
        # a constant (1-entry_floor) scaling of all gadget regrets also cancels.
        current=choice*continuing+(1-choice)*exit_cfvs
        gadget_regrets[0]=np.maximum(gadget_regrets[0]+continuing-current,0.)
        gadget_regrets[1]=np.maximum(gadget_regrets[1]+exit_cfvs-current,0.)


@njit(cache=True)
def iterate_maxmargin(kinds,actors,counts,children,payoffs,hands,ranks,order,regrets,average,
                      selector_regrets,hero,hero_range,mass,exit_cfvs,first,last):
    """Appendix B maxmargin gadget: opponent selects its private infoset.

    Conditional chance then draws hero's compatible hand. Dividing selector
    reach by compatible mass implements that chance distribution. The exit
    value is a constant offset for each selected hand, with no opt-out action.
    """
    policy=np.zeros_like(regrets)
    reach=np.empty((len(kinds),len(hands)))
    own_reach=np.empty_like(reach);result=np.empty_like(reach)
    for iteration in range(first,last):
        total=selector_regrets.sum()
        selection=selector_regrets/total if total>0 else np.full(len(hands),1./len(hands))
        opponent=selection/mass
        weight=float(max(iteration-100,0))
        half_iteration(kinds,actors,counts,children,payoffs,hands,ranks,order,regrets,average,
                       hero,opponent,hero_range,weight,policy,reach,own_reach,result)
        continuing=half_iteration(kinds,actors,counts,children,payoffs,hands,ranks,order,regrets,average,
                                  1-hero,hero_range,opponent,weight,policy,reach,own_reach,result)
        shifted=(continuing-exit_cfvs)/mass
        current=0.
        for h in range(len(hands)):current+=selection[h]*shifted[h]
        for h in range(len(hands)):
            selector_regrets[h]=max(selector_regrets[h]+shifted[h]-current,0.)


@njit(cache=True)
def realization_mix(old,new,actors,counts,children,hero,alpha):
    """Mixture of complete hero strategies, preserving own action correlations."""
    result=old.copy();old_reach=np.ones((len(actors),old.shape[2]));new_reach=old_reach.copy()
    for n in range(len(actors)):
        if actors[n]==hero and counts[n]>0:
            mass=(1-alpha)*old_reach[n]+alpha*new_reach[n]
            for a in range(counts[n]):
                for h in range(old.shape[2]):
                    if mass[h]>0:
                        result[n,a,h]=((1-alpha)*old_reach[n,h]*old[n,a,h]+alpha*new_reach[n,h]*new[n,a,h])/mass[h]
        for a in range(counts[n]):
            child=children[n,a]
            old_reach[child]=old_reach[n];new_reach[child]=new_reach[n]
            if actors[n]==hero:
                old_reach[child]*=old[n,a];new_reach[child]*=new[n,a]
    return result


@dataclass
class RiverTree:
    kinds: np.ndarray
    actors: np.ndarray
    counts: np.ndarray
    children: np.ndarray
    payoffs: np.ndarray
    hands: np.ndarray
    ranks: np.ndarray
    order: np.ndarray
    blueprint: np.ndarray
    indexes: np.ndarray
    paths: list

    @classmethod
    def from_probe(cls,probe,path):
        all_hands=np.asarray(all_combos(),dtype=np.int64)
        indexes=np.flatnonzero([not set(h)&set(probe.view.board) for h in all_hands])
        hands=all_hands[indexes]
        ranks=np.asarray([evaluate(tuple(h)+tuple(probe.view.board)) for h in hands],dtype=np.int64)
        nodes=[];paths=[];descendants=[];mixes=[]
        def visit(path):
            node,mix=probe.node(path);index=len(nodes)
            nodes.append(node);paths.append(path);descendants.append([]);mixes.append(mix)
            for action in node.get("actions",[]):descendants[index].append(visit(path+"/"+action["name"]))
            return index
        visit(path)
        width=max(map(len,descendants))
        children=np.full((len(nodes),width),-1,dtype=np.int64)
        policy=np.zeros((len(nodes),width,len(hands)))
        for n,kids in enumerate(descendants):
            children[n,:len(kids)]=kids
            if kids:policy[n,:len(kids)]=mixes[n][indexes].T
        kinds=np.asarray([2 if n.get("showdown") else 1 if n.get("terminal") else 0 for n in nodes],dtype=np.int64)
        actors=np.asarray([n.get("winner",n.get("player",0)) for n in nodes],dtype=np.int64)
        # Store values in big blinds; the native blueprint uses half-bb units.
        payoffs=np.asarray([n.get("payoff",0)/2 for n in nodes],dtype=float)
        return cls(kinds,actors,np.asarray(list(map(len,descendants)),dtype=np.int64),children,payoffs,
                   hands,ranks,np.argsort(ranks,kind="stable"),policy,indexes,paths)

    def arrays(self):
        return (self.kinds,self.actors,self.counts,self.children,self.payoffs,self.hands,self.ranks,self.order)

    def response(self,policy,player,opponent,best=True):
        return values(*self.arrays(),policy,player,np.asarray(opponent,dtype=float),best)

    def refine(self,hero,hero_range,opponent_range,iterations=2000,seconds=30,tolerance_bb=.01,entry_floor=0.,gadget_exit_cfvs=None,method='resolve'):
        if hero not in (0,1) or iterations<101 or seconds<=0 or tolerance_bb<0 or not 0<=entry_floor<1 or not np.isfinite([seconds,tolerance_bb,entry_floor]).all():
            raise ValueError("Invalid river refinement budget")
        if method not in ('resolve','maxmargin') or (method=='maxmargin' and entry_floor!=0):
            raise ValueError('Maxmargin has no opt-out entry floor; choose a supported method')
        hero_range=np.asarray(hero_range,dtype=float);opponent_range=np.asarray(opponent_range,dtype=float)
        for weights in (hero_range,opponent_range):
            if weights.shape!=(len(self.hands),) or not np.isfinite(weights).all() or np.any(weights<0):
                raise ValueError("Invalid public range")
        mass=compatible_mass(self.hands,hero_range)
        if np.any(mass<=0):raise ValueError("Every opponent hand needs compatible hero support")
        baseline=self.response(self.blueprint,1-hero,hero_range)
        # An expanded opponent menu can make the extended blueprint weaker.
        # Research callers may anchor training to the unexpanded tree's values;
        # acceptance is still measured against this tree's actual baseline.
        exits=baseline if gadget_exit_cfvs is None else np.asarray(gadget_exit_cfvs,dtype=float)
        if exits.shape!=baseline.shape or not np.isfinite(exits).all():
            raise ValueError("Invalid gadget exit counterfactual values")
        regrets=np.zeros_like(self.blueprint);average=np.zeros_like(self.blueprint)
        gadget=np.zeros((2,len(self.hands)))
        selector=np.zeros(len(self.hands))
        start=time.perf_counter();completed=0
        while completed<iterations:
            last=min(completed+100,iterations)
            if method=='resolve':
                iterate(*self.arrays(),regrets,average,gadget,hero,hero_range,exits,entry_floor,completed+1,last+1)
            else:
                iterate_maxmargin(*self.arrays(),regrets,average,selector,hero,hero_range,mass,exits,completed+1,last+1)
            completed=last
            if time.perf_counter()-start>=seconds and completed>100:break
        candidate=probabilities(average,self.counts)
        # A realization mixture is necessary: mixing each local action vector
        # independently does not mix whole strategies and can violate the bound.
        def check(alpha):
            policy=realization_mix(self.blueprint,candidate,self.actors,self.counts,self.children,hero,alpha)
            response=self.response(policy,1-hero,hero_range)
            excess=float(np.max((response-baseline)/mass))
            return policy,response,excess
        policy,response,excess=check(1.)
        alpha=1.
        if excess>tolerance_bb:
            low=0.;high=1.
            for _ in range(16):
                midpoint=(low+high)/2
                mixed,br,error=check(midpoint)
                if error<=tolerance_bb:low=midpoint;policy,response,excess=mixed,br,error
                else:high=midpoint
            alpha=low
            if not low:policy,response,excess=check(0.)
        root_mass=float(opponent_range@mass)
        if root_mass<=0:raise ValueError("Ranges have no compatible pairs")
        improvement=float(opponent_range@(baseline-response)/root_mass)
        if improvement<=0:
            policy=self.blueprint.copy();response=baseline.copy();alpha=0.;excess=0.;improvement=0.
        if excess>tolerance_bb+1e-10:raise ValueError("Refinement violates its best-response gate")
        return policy,{"iterations":completed,"elapsed_seconds":time.perf_counter()-start,"entry_floor":entry_floor,
            'method':method,'minimum_opponent_hand_margin_bb':float(np.min((exits-response)/mass)),
            "gadget_exit_values":"this_tree_baseline" if gadget_exit_cfvs is None else "external_counterfactual_values",
            "accepted":alpha>0,"realization_mixture":alpha,"max_opponent_hand_value_increase_bb":excess,
            "allowed_increase_bb":tolerance_bb,"opponent_br_value_reduction_bb":improvement,
            "scope":"Verified per-opponent-hand best responses in this exact river tree and supplied hero range; not unrestricted or full-game safety.",
            "validated_pro_level":False}
