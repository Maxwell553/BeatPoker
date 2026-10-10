"""Explicit mixed chart baseline and public range replay.

These inherited charts are NOT a solved preflop strategy. The same policy is
used for acting and Bayesian replay, without thresholding mixed frequencies
into pure actions or excluding hero's cards from the public ranges.
"""
from __future__ import annotations

import numpy as np

from poker.equity import all_combos
from poker.game import Action, ActionType, GameConfig, NLHGame, Street
from strategy.preflop_trees import rfi_range, vs_open_action_ranges, vs_3bet_ranges, vs_4bet_ranges
from strategy.range_model import iter_history


def policy(view, rake=None):
    """Return legal actions and one probability row per physical hand."""
    start = min(s + c for s, c in zip(view.stacks, view.committed)) / view.bb
    raises = [(s, a) for s, a, st in iter_history(view)
              if st == Street.PREFLOP and a.type in (ActionType.BET, ActionType.RAISE, ActionType.ALL_IN)]
    n = len(raises)
    zeros = np.zeros(1326)
    check = view.to_call == 0
    if n == 0:
        r = rfi_range("CO" if check else "BTN", 2, start, rake=rake).w
        j, c = zeros, zeros
        size = 3.5 if check else 2.5
    elif n == 1:
        jg, rg, cg = vs_open_action_ranges(view.position, view.positions[raises[0][0]], 2, start, rake=rake)
        j, r, c = jg.w, rg.w, cg.w
        # A 3bet must scale with the actual open size, including off-chart bets.
        size = max(7.5 if start <= 60 else 9.0, view.current_bet / view.bb * (3.0 if view.hero == view.button else 4.0))
    elif n == 2:
        rg, cg = vs_3bet_ranges(view.position, view.positions[raises[-1][0]], start, rake=rake)
        j, r, c = zeros, rg.w, cg.w
        size = max(22.0, 2.5 * view.current_bet / view.bb)
    else:
        jg, cg = vs_4bet_ranges(start)
        j, r, c = jg.w, zeros, cg.w
        size = view.max_raise_to / view.bb
    j, r, c = [np.clip(a, 0, 1) for a in (j, r, c)]
    # Chart continue frequencies are only a baseline. Large off-chart prices
    # require a separate all-in equity decision, implemented by the agent.
    probs = np.column_stack([j, (1-j)*r, (1-j)*(1-r)*c, (1-j)*(1-r)*(1-c)])
    stack = view.stacks[view.hero]
    target = min(view.max_raise_to, max(view.min_raise_to, round(size * view.bb)))
    put = target - view.bets[view.hero]
    passive = Action(ActionType.CHECK) if check else Action(ActionType.FOLD)
    call = Action(ActionType.CHECK) if check else Action(ActionType.CALL, min(stack, view.to_call))
    raise_action = Action(ActionType.ALL_IN, stack) if put >= stack else Action(ActionType.BET if check else ActionType.RAISE, put)
    jam = Action(ActionType.ALL_IN, stack)
    can_raise = any(a.type in (ActionType.RAISE,ActionType.BET,ActionType.ALL_IN) for a in view.legal)
    if stack <= view.to_call or not can_raise:
        jam = raise_action = call
    actions = [jam, raise_action, call, passive]
    merged = {}
    for i, a in enumerate(actions):
        merged[a] = merged.get(a, zeros.copy()) + probs[:, i]
    return list(merged), np.column_stack(list(merged.values()))


def public_flop_ranges(view, rake=None, policy_fn=None):
    """Reconstruct from recorded preflop actions; never inspect private cards."""
    starts = [s+c for s,c in zip(view.stacks, view.committed)]
    game = NLHGame(GameConfig(num_players=2, stack_bb=max(starts)/view.bb,
                              button=view.button, sb=view.sb, bb=view.bb, rng_seed=0))
    for p,s in zip(game.players,starts):
        p.stack = s
    game.start_hand()
    ranges = np.ones((2,1326))
    for seat, action, street in iter_history(view):
        if street != Street.PREFLOP:
            break
        if game.to_act != seat:
            raise ValueError("Invalid preflop action order")
        acts, mix = (policy_fn or policy)(game.view(seat), rake)
        # Multiple trained sizes can represent different ranges. Condition on
        # the exact size when available; otherwise use the nearest legal size.
        indexes = [i for i,a in enumerate(acts) if a == action]
        aggression=(ActionType.ALL_IN,ActionType.RAISE,ActionType.BET)
        if not indexes and action.type in aggression:
            candidates=[i for i,a in enumerate(acts) if a.type in aggression]
            if candidates:
                current=game.players[seat].bet
                target=current+action.amount
                indexes=[min(candidates,key=lambda i:abs(np.log((current+acts[i].amount)/target)))]
        if not indexes:
            indexes = [i for i,a in enumerate(acts) if a.type == action.type]
        likelihood = mix[:,indexes].sum(axis=1) if indexes else np.zeros(1326)
        ranges[seat] *= 0.98 * likelihood + 0.02 / max(len(acts),1)
        ranges[seat] /= max(ranges[seat].max(), 1e-30)
        game.apply(action)
    board = set(view.board[:3])
    for i,h in enumerate(all_combos()):
        if board.intersection(h):
            ranges[:,i] = 0
    return ranges, game.pot_total(), [p.stack for p in game.players]
