"""Numba-compiled CFR inner kernels.

The recursive Python tree walk stays in live_solve / mccfr. These kernels
are the overnight loop: regret-matching, linear averaging, and a 6-max
infoset update that is not 8 spots × 169.
"""

from __future__ import annotations

import numpy as np

try:
    import numba as _nb

    @_nb.njit(cache=True)
    def regret_match(regrets: np.ndarray) -> np.ndarray:
        n = regrets.shape[0]
        out = np.empty(n, dtype=np.float64)
        s = 0.0
        for i in range(n):
            v = regrets[i]
            if v < 0.0:
                v = 0.0
            out[i] = v
            s += v
        if s <= 1e-15:
            inv = 1.0 / max(n, 1)
            for i in range(n):
                out[i] = inv
            return out
        for i in range(n):
            out[i] /= s
        return out

    @_nb.njit(cache=True)
    def linear_update(regrets, ssum, sig, u_a, u, t, sign):
        n = regrets.shape[0]
        for i in range(n):
            ssum[i] += t * sig[i]
            regrets[i] += sign * (u_a[i] - u)

    @_nb.njit(cache=True)
    def discount_regrets(regrets, a):
        for i in range(regrets.shape[0]):
            for j in range(regrets.shape[1]):
                regrets[i, j] *= a

    COMPILED = True
except Exception:  # pragma: no cover
    COMPILED = False

    def regret_match(regrets: np.ndarray) -> np.ndarray:
        pos = np.maximum(regrets, 0.0)
        s = pos.sum()
        if s <= 1e-12:
            return np.ones_like(regrets) / max(len(regrets), 1)
        return pos / s

    def linear_update(regrets, ssum, sig, u_a, u, t, sign):
        ssum += t * sig
        regrets += sign * (u_a - u)

    def discount_regrets(regrets, a):
        regrets *= a
