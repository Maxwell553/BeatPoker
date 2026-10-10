"""Seeded Monte Carlo preflop equity, used only when no betting remains."""
import numpy as np
from numba import njit
from poker.equity import all_combos
from strategy.cfr_core import _eval_nb


@njit(cache=True)
def _sample(hero, combos, weights, samples, seed):
    np.random.seed(seed)
    cdf = np.empty(len(weights))
    total = 0.0
    for i in range(len(weights)):
        a,b = combos[i]
        if a != hero[0] and a != hero[1] and b != hero[0] and b != hero[1]:
            total += weights[i]
        cdf[i] = total
    if total <= 0:
        raise ValueError("Opponent range is fully blocked")
    score = 0.0
    board = np.empty(5,dtype=np.int64)
    for _ in range(samples):
        j = np.searchsorted(cdf,np.random.random()*total,side='right')
        a,b = combos[j]
        for k in range(5):
            while True:
                c = np.random.randint(52)
                if c==a or c==b or c==hero[0] or c==hero[1]:
                    continue
                duplicate = False
                for t in range(k):
                    if board[t] == c:
                        duplicate = True
                if duplicate:
                    continue
                board[k]=c
                break
        h = _eval_nb(hero[0],hero[1],board[0],board[1],board[2],board[3],board[4],5)
        v = _eval_nb(a,b,board[0],board[1],board[2],board[3],board[4],5)
        score += 1.0 if h>v else (0.5 if h==v else 0.0)
    return score/samples


def preflop_equity(hole,weights,samples=20000,seed=41):
    if samples < 1:
        raise ValueError("Positive sample count required")
    return float(_sample(np.array(hole),np.array(all_combos()),np.asarray(weights,dtype=float),samples,seed))
