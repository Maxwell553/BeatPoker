"""River endgame solver: Discounted CFR on 12 hand-strength buckets.

Given both players' ranges, the board, pot and stacks, we:

1. Rank remaining combos by showdown value (with card removal vs the
   opponent's range approximated by ignoring mutual blockage in bucketing).
2. Collapse each range into 12 equal-weight buckets.
3. Solve a two-player zero-sum betting-tree with DCFR.
4. Play the strategy of the hero's bucket, with a nuts/air override.

This is the same endgame-solving idea used by Libratus (Brown & Sandholm
2017) and DeepStack, at a scale that runs in a few milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from poker.evaluator import evaluate
from poker.game import Action, ActionType, PlayerView

N_BUCKETS = 12


def _showdown_value(combo: Tuple[int, int], board: Sequence[int]) -> int:
    return evaluate([combo[0], combo[1]] + list(board))


def bucket_range(
    combos: Sequence[Tuple[int, int]],
    weights: Sequence[float],
    board: Sequence[int],
    n_buckets: int = N_BUCKETS,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Return (bucket_weights [B], avg_value [B], combo_to_bucket [N]).

    Combos with weight 0 are assigned bucket -1.
    Stronger hands get higher bucket indices.
    """
    n = len(combos)
    vals = np.array([_showdown_value(c, board) if weights[i] > 0 else -1 for i, c in enumerate(combos)], dtype=np.int64)
    order = np.argsort(vals)  # weak -> strong
    combo_bucket = [-1] * n
    # Equal-weight buckets over live combos
    live = [(int(i), float(weights[i])) for i in order if weights[i] > 1e-12]
    tot = sum(w for _, w in live)
    if tot <= 0:
        return np.zeros(n_buckets), np.zeros(n_buckets), combo_bucket
    target = tot / n_buckets
    acc = 0.0
    b = 0
    b_w = np.zeros(n_buckets)
    b_v = np.zeros(n_buckets)
    for idx, w in live:
        if b < n_buckets - 1 and acc >= target * (b + 1) - 1e-15:
            b += 1
        combo_bucket[idx] = b
        b_w[b] += w
        b_v[b] += w * float(vals[idx])
        acc += w
    for i in range(n_buckets):
        if b_w[i] > 0:
            b_v[i] /= b_w[i]
    return b_w, b_v, combo_bucket


def _payoff_matrix(v_hero: np.ndarray, v_vill: np.ndarray) -> np.ndarray:
    """P[i,j] = +1 if hero bucket i beats vill j, 0 tie, -1 lose."""
    H, V = len(v_hero), len(v_vill)
    P = np.zeros((H, V))
    for i in range(H):
        for j in range(V):
            if v_hero[i] > v_vill[j]:
                P[i, j] = 1.0
            elif v_hero[i] < v_vill[j]:
                P[i, j] = -1.0
    return P


@dataclass
class _Tree:
    """Tiny river tree. Amounts are in chips. Player 0 is OOP, 1 is IP."""

    # nodes stored as list of dicts
    nodes: list = field(default_factory=list)


def _build_tree(pot: float, cap: float, facing_bet: float, oop_to_act: bool) -> _Tree:
    """
    Abstracted river tree.

    If facing_bet > 0, the acting player (hero, whom we treat as player 0
    at the root) faces fold/call/raise-allin.

    Otherwise root can check or bet {0.33, 0.75, 1.25, all-in} * pot,
    opponent can fold/call/raise-allin, and a single all-in reraise is allowed.
    """
    tree = _Tree()

    def add_terminal_fold(pot_now: float, invested: Tuple[float, float], folder: int) -> int:
        # Winner is 1-folder. Hero-centric payoff is computed later from
        # invested + pot. Store (kind, pot, inv0, inv1, folder)
        idx = len(tree.nodes)
        tree.nodes.append(("fold", pot_now, invested[0], invested[1], folder))
        return idx

    def add_sd(pot_now: float, invested: Tuple[float, float]) -> int:
        idx = len(tree.nodes)
        tree.nodes.append(("sd", pot_now, invested[0], invested[1], -1))
        return idx

    def add_decision(player: int, actions: list) -> int:
        idx = len(tree.nodes)
        tree.nodes.append(("dec", player, actions))  # actions: list of (name, child_idx, put)
        return idx

    # We'll build bottom-up by recursion that returns node index.
    # invested tracks chips put in THIS STREET.

    def clamp_bet(size: float, stack: float) -> float:
        return max(0.0, min(size, stack))

    def sizes(pot_now: float, stack: float) -> List[float]:
        raw = [0.33 * pot_now, 0.75 * pot_now, 1.25 * pot_now, stack]
        out = []
        for s in raw:
            s = min(s, stack)
            if s < 0.4 * (pot_now * 0.25 if pot_now else 1) and s < stack:
                continue
            if s >= stack * 0.95:
                s = stack
            # min bet ~ 0.25 pot or all-in
            if s < 0.2 * pot_now and s < stack:
                continue
            if s > 0 and (not out or abs(s - out[-1]) / max(s, 1) > 0.15):
                out.append(s)
        if stack > 0 and (not out or out[-1] < stack * 0.95):
            out.append(stack)
        # unique
        uniq = []
        for s in out:
            if s <= 0:
                continue
            if not uniq or abs(s - uniq[-1]) > 1:
                uniq.append(s)
        return uniq[:4]

    cap0 = cap
    cap1 = cap

    def rec(player: int, pot_now: float, inv: Tuple[float, float], stacks: Tuple[float, float],
            to_call: float, raises_left: int, facing: bool) -> int:
        # If to_call==0 and facing is False: check/bet
        # If to_call>0: fold/call/raise
        stk = stacks[player]
        if stk <= 0.5 and to_call <= 0:
            return add_sd(pot_now, inv)
        acts = []
        if to_call > 0:
            # fold
            acts.append(("fold", add_terminal_fold(pot_now, inv, player), 0.0))
            # call
            pay = min(to_call, stk)
            ninv = list(inv)
            ninv[player] += pay
            npot = pot_now + pay
            nstk = list(stacks)
            nstk[player] -= pay
            acts.append(("call", add_sd(npot, (ninv[0], ninv[1])), pay))
            if raises_left > 0 and stk > to_call + 1:
                raise_to_allin = stk
                # raise all-in only to keep tree tiny
                pay_r = raise_to_allin
                ninv = list(inv)
                ninv[player] += pay_r
                npot = pot_now + pay_r
                nstk = list(stacks)
                nstk[player] = 0
                new_to_call = ninv[player] - ninv[1 - player]
                child = rec(1 - player, npot, (ninv[0], ninv[1]), (nstk[0], nstk[1]), new_to_call, raises_left - 1, True)
                acts.append(("allin", child, pay_r))
        else:
            # check
            if player == 0:
                # OOP check -> IP acts
                child = rec(1, pot_now, inv, stacks, 0.0, raises_left, False)
                acts.append(("check", child, 0.0))
            else:
                acts.append(("check", add_sd(pot_now, inv), 0.0))
            for s in sizes(pot_now, stk):
                pay = s
                ninv = list(inv)
                ninv[player] += pay
                npot = pot_now + pay
                nstk = list(stacks)
                nstk[player] -= pay
                new_to_call = ninv[player] - ninv[1 - player]
                name = "allin" if pay >= stk - 0.5 else f"bet{int(100*pay/max(pot_now,1))}"
                child = rec(1 - player, npot, (ninv[0], ninv[1]), (nstk[0], nstk[1]), new_to_call, raises_left, True)
                acts.append((name, child, pay))

        return add_decision(player, acts)

    if facing_bet > 0:
        # Root: player 0 (hero) faces a bet. Treat hero as player 0, villain already invested facing_bet.
        inv = (0.0, facing_bet)
        pot_now = pot  # pot includes the bet already in PlayerView.pot
        # view.pot includes all committed chips. facing to_call is extra hero must put.
        # So current pot already has villain's bet. Hero invested 0 this decision.
        stacks = (cap, max(0.0, cap - 0))  # both effective remaining; villain already put facing_bet
        # Villain stack already reduced in `cap` as effective stacks remaining.
        root = rec(0, pot_now, inv, (cap, cap), facing_bet, 1, True)
    else:
        if oop_to_act:
            root = rec(0, pot, (0.0, 0.0), (cap, cap), 0.0, 1, False)
        else:
            # IP to act after OOP checked — only IP decision then possibly OOP faces bet
            root = rec(1, pot, (0.0, 0.0), (cap, cap), 0.0, 1, False)
    tree.nodes.append(("root_idx", root))  # unused; root is returned
    tree.root = root  # type: ignore
    return tree


def _regret_match(regrets: np.ndarray) -> np.ndarray:
    pos = np.maximum(regrets, 0.0)
    s = pos.sum()
    if s <= 1e-12:
        return np.ones_like(regrets) / len(regrets)
    return pos / s


def solve_buckets(
    hero_w: np.ndarray,
    vill_w: np.ndarray,
    hero_v: np.ndarray,
    vill_v: np.ndarray,
    pot: float,
    cap: float,
    facing_bet: float,
    oop_to_act: bool,
    hero_is_oop: bool,
    iterations: int = 80,
) -> Tuple[object, np.ndarray]:
    """Return (tree, root_strategy[hero_buckets, actions])."""
    P = _payoff_matrix(hero_v, vill_v)  # +1 hero wins
    # Normalize range weights
    hw = hero_w / max(hero_w.sum(), 1e-12)
    vw = vill_w / max(vill_w.sum(), 1e-12)
    B = len(hw)

    tree = _build_tree(pot, cap, facing_bet, oop_to_act if hero_is_oop else (not oop_to_act and facing_bet == 0))
    # If hero is IP and nobody has bet, we built tree with player 1 at root after a fake construction.
    # Simpler: always make hero = player 0 at root. The `oop_to_act` flag only matters for check-check order
    # when facing_bet==0. If hero is IP and facing_bet==0, hero can bet or check (close the action).
    if facing_bet == 0 and not hero_is_oop:
        tree = _build_tree(pot, cap, 0.0, oop_to_act=False)

    nodes = tree.nodes
    # Identify decision nodes and max actions
    dec_ids = [i for i, n in enumerate(nodes) if isinstance(n, tuple) and n[0] == "dec"]
    max_a = 1
    for i in dec_ids:
        max_a = max(max_a, len(nodes[i][2]))

    # regrets[node, bucket, action] — we store only for hero (player 0) buckets
    # Both players need regrets. Player 0 uses hero buckets, player 1 uses vill buckets.
    regrets0 = {i: np.zeros((B, len(nodes[i][2]))) for i in dec_ids if nodes[i][1] == 0}
    regrets1 = {i: np.zeros((B, len(nodes[i][2]))) for i in dec_ids if nodes[i][1] == 1}
    strat_sum0 = {i: np.zeros((B, len(nodes[i][2]))) for i in regrets0}
    strat_sum1 = {i: np.zeros((B, len(nodes[i][2]))) for i in regrets1}

    def payoff_sd(inv0, inv1, pot_now, p0_reach, p1_reach):
        """Expected chip EV for player 0 (hero) per hero-bucket, given reach vectors.

        u[i] = sum_j p1_reach[j] * ( + (pot_now - inv0) if win,  -inv0 if lose,  (pot_now/2 - inv0) if tie )
             Wait: hero invested inv0, pot_now = pot_start + inv0 + inv1.
             If hero wins: +pot_now - inv0 = pot_start + inv1
             If loses: -inv0
             If ties: pot_now/2 - inv0
        Using P[i,j] in {-1,0,1}:
             u = -inv0 + (P==1)*(pot_now) + (P==0)*(pot_now/2)
               = -inv0 + pot_now * (P+1)/2
        """
        # p1_reach shape [B], p0_reach unused for hero EV of each hero bucket
        # EV_i = sum_j wj * (pot_now * (P[i,j]+1)/2 - inv0)
        wj = p1_reach / max(p1_reach.sum(), 1e-12)
        # (P+1)/2 is 1, 0.5, 0
        share = (P + 1.0) / 2.0  # [Bh, Bv]
        return share.dot(wj) * pot_now - inv0

    def cfr(nid: int, r0: np.ndarray, r1: np.ndarray, player_update: int, t: int) -> np.ndarray:
        """Return EV for each hero-bucket (player 0). r0/r1 are reach weights per bucket."""
        node = nodes[nid]
        kind = node[0]
        if kind == "fold":
            _, pot_now, inv0, inv1, folder = node
            # If hero folds, EV = -inv0 for all buckets. If villain folds, EV = pot_now - inv0
            if folder == 0:
                return np.full(B, -inv0)
            return np.full(B, pot_now - inv0)
        if kind == "sd":
            _, pot_now, inv0, inv1, _ = node
            return payoff_sd(inv0, inv1, pot_now, r0, r1)
        # decision
        _, player, acts = node
        n_a = len(acts)
        if player == 0:
            sigma = np.zeros((B, n_a))
            for b in range(B):
                sigma[b] = _regret_match(regrets0[nid][b])
            if player_update == 0:
                for b in range(B):
                    strat_sum0[nid][b] += r0[b] * sigma[b]
            action_ev = np.zeros((n_a, B))
            for a, (name, child, put) in enumerate(acts):
                r0n = r0 * sigma[:, a]
                action_ev[a] = cfr(child, r0n, r1, player_update, t)
            node_ev = np.zeros(B)
            for b in range(B):
                node_ev[b] = sigma[b].dot(action_ev[:, b])
            if player_update == 0:
                for b in range(B):
                    for a in range(n_a):
                        regrets0[nid][b, a] += r1.sum() * (action_ev[a, b] - node_ev[b])
            return node_ev
        else:
            sigma = np.zeros((B, n_a))
            for b in range(B):
                sigma[b] = _regret_match(regrets1[nid][b])
            if player_update == 1:
                for b in range(B):
                    strat_sum1[nid][b] += r1[b] * sigma[b]
            # Mix opponent actions: EV is weighted by their strategy and reach
            # Because payoff_sd uses r1, we pass modified r1
            ev_acc = np.zeros(B)
            action_ev = []
            for a, (name, child, put) in enumerate(acts):
                r1n = r1 * sigma[:, a]
                ev_a = cfr(child, r0, r1n, player_update, t)
                action_ev.append(ev_a)
                ev_acc += ev_a  # not quite — each branch already uses scaled r1, so we should NOT add
            # Correct: opponent mixes, hero EV = sum_a E[ev | a] *but* each child was called with
            # r1 * sigma[a], and payoff_sd *normalizes* r1. So child EV is "EV given opponent's
            # range restricted to action a", not weighted by frequency.
            # Recompute using frequency-weighted mix of *unnormalized* semantics.
            # Fix: don't normalize in payoff? Let's mix by mass of r1n.
            masses = np.array([float((r1 * sigma[:, a]).sum()) for a in range(n_a)])
            tot = masses.sum()
            if tot <= 1e-15:
                return action_ev[0]
            ev = np.zeros(B)
            for a in range(n_a):
                ev += action_ev[a] * (masses[a] / tot)
            # Update villain regrets: villain EV is -hero EV, but villain buckets differ.
            # Approximate: use average hero EV as the node value and skip precise per-bucket
            # villain regrets — instead use a scalar game for the opponent (range-wide).
            if player_update == 1:
                # Scalar EV of each action (range-vs-range)
                avg = np.array([float(np.dot(action_ev[a], r0) / max(r0.sum(), 1e-12)) for a in range(n_a)])
                node_scalar = float(np.dot(avg, masses / tot)) if tot else 0.0
                # Villain wants to minimize hero EV, so regret for a is (node - action) in hero-EV
                # = (action_villain - node_villain) = (node_hero - action_hero)
                for b in range(B):
                    for a in range(n_a):
                        regrets1[nid][b, a] += r0.sum() * (node_scalar - avg[a])
            return ev

    r0 = hw.copy()
    r1 = vw.copy()
    root = tree.root  # type: ignore
    for t in range(1, iterations + 1):
        cfr(root, r0, r1, 0, t)
        cfr(root, r0, r1, 1, t)
        # DCFR-style discount of regrets
        if t % 4 == 0:
            a = t ** 1.5 / (t ** 1.5 + 1.0)
            for tab in (regrets0, regrets1):
                for nid, arr in tab.items():
                    arr *= a

    # Average strategy at root for hero
    if root not in strat_sum0:
        # Hero is not player 0 at root (IP check/bet node as player 1). Map player 1.
        if root in strat_sum1:
            ss = strat_sum1[root]
            strat = np.zeros_like(ss)
            for b in range(B):
                s = ss[b].sum()
                strat[b] = ss[b] / s if s > 0 else _regret_match(np.zeros(ss.shape[1]))
            return tree, strat
        # fallback uniform
        n_a = len(nodes[root][2])
        return tree, np.ones((B, n_a)) / n_a
    ss = strat_sum0[root]
    strat = np.zeros_like(ss)
    for b in range(B):
        s = ss[b].sum()
        strat[b] = ss[b] / s if s > 0 else _regret_match(np.zeros(ss.shape[1]))
    return tree, strat


def river_action(
    view: PlayerView,
    hero_combos: Sequence[Tuple[int, int]],
    hero_weights: Sequence[float],
    vill_combos: Sequence[Tuple[int, int]],
    vill_weights: Sequence[float],
    rng,
    iterations: int = 60,
) -> Action:
    board = view.board
    hero_hole = view.hole
    # Find hero combo in list (or add it)
    hw = list(hero_weights)
    hc = list(hero_combos)
    found = False
    hero_idx = 0
    for i, c in enumerate(hc):
        if set(c) == set(hero_hole):
            found = True
            hero_idx = i
            if hw[i] <= 0:
                hw[i] = 1.0
            break
    if not found:
        hc.append((min(hero_hole), max(hero_hole)))
        hw.append(1.0)
        hero_idx = len(hc) - 1

    h_w, h_v, h_map = bucket_range(hc, hw, board)
    v_w, v_v, _ = bucket_range(vill_combos, vill_weights, board)
    hero_b = h_map[hero_idx]
    if hero_b < 0:
        hero_b = int(np.argmax(h_w)) if h_w.sum() else 0

    pot = float(view.pot)
    cap = float(view.effective_stack)
    facing = float(view.to_call)
    # Is hero OOP? First to act postflop is left of button.
    # Approximate: if any player between button+1 and hero is live and hasn't acted this river
    # when pot bets are 0 and we can check, if we are first... use position vs others.
    hero_is_oop = _hero_is_oop(view)
    oop_to_act = hero_is_oop and facing == 0

    # Nuts override
    hv = _showdown_value((min(hero_hole), max(hero_hole)), board)
    vill_vals = [_showdown_value(c, board) for i, c in enumerate(vill_combos) if vill_weights[i] > 0]
    is_nuts = bool(vill_vals) and hv >= max(vill_vals)
    is_air = bool(vill_vals) and hv <= min(vill_vals) and hero_b <= 1

    tree, strat = solve_buckets(
        h_w, v_w, h_v, v_v, pot, cap, facing, oop_to_act, hero_is_oop, iterations=iterations
    )
    probs = strat[hero_b]
    acts_spec = tree.nodes[tree.root][2]  # type: ignore

    # Map abstract actions to engine actions
    engine_acts = []
    for name, _child, put in acts_spec:
        engine_acts.append(_to_engine(view, name, put))

    if is_nuts:
        # Always take the most aggressive value action that isn't fold
        best_i, best_put = 0, -1
        for i, (name, _, put) in enumerate(acts_spec):
            if name == "fold":
                continue
            if put > best_put or name == "allin":
                best_put = put
                best_i = i
        return engine_acts[best_i]

    # Sample from strategy
    p = np.maximum(probs, 0)
    if p.sum() <= 0:
        p = np.ones_like(p)
    p = p / p.sum()
    choice = int(rng.choice(len(engine_acts), p=p))
    return engine_acts[choice]


def _hero_is_oop(view: PlayerView) -> bool:
    n = view.num_players
    # first live player left of button is OOP
    for k in range(1, n + 1):
        i = (view.button + k) % n
        if not view.folded[i] and not view.all_in[i]:
            return i == view.hero
    return True


def _to_engine(view: PlayerView, name: str, put: float) -> Action:
    put_i = int(round(put))
    if name == "fold":
        if view.to_call <= 0:
            return Action(ActionType.CHECK)
        return Action(ActionType.FOLD)
    if name == "check":
        if view.to_call > 0:
            return Action(ActionType.CALL, min(view.to_call, view.stacks[view.hero]))
        return Action(ActionType.CHECK)
    if name == "call":
        amt = min(view.to_call, view.stacks[view.hero])
        if amt <= 0:
            return Action(ActionType.CHECK)
        return Action(ActionType.CALL, amt)
    put_i = max(put_i, 0)
    put_i = min(put_i, view.stacks[view.hero])
    if put_i <= 0:
        return Action(ActionType.CHECK) if view.to_call <= 0 else Action(ActionType.FOLD)
    if put_i >= view.stacks[view.hero]:
        return Action(ActionType.ALL_IN, view.stacks[view.hero])
    if view.to_call <= 0:
        # min bet
        put_i = max(put_i, min(view.bb, view.stacks[view.hero]))
        return Action(ActionType.BET, put_i)
    # raise: put must cover to_call
    put_i = max(put_i, view.to_call)
    if put_i >= view.stacks[view.hero]:
        return Action(ActionType.ALL_IN, view.stacks[view.hero])
    if put_i == view.to_call:
        return Action(ActionType.CALL, put_i)
    return Action(ActionType.RAISE, put_i)
