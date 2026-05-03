"""CHMM and ALERGIA wrappers for the HMM forecasting comparison.

Both expose the same interface as the existing OOMForecaster /
GDCForecaster:

    fit(sequences)
    horizon_emission(prefix_obs, h)  -> shape (nA,)

`horizon_emission` returns the model's predicted distribution over
the next-symbol-at-step-(t+h) given prefix `prefix_obs`.
"""

from __future__ import annotations
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "chmm_tests", "naturecomm_cscg"))

from chmm_actions import CHMM, forward as chmm_forward  # noqa: E402

try:
    from aalpy.learning_algs import run_Alergia
    HAS_AALPY = True
except Exception:
    HAS_AALPY = False


# --------------------------------------------------------------------
# CHMM
# --------------------------------------------------------------------
class CHMMForecaster:
    def __init__(self, nA, K=4, n_em_iters=50, pseudocount=1e-3, seed=0):
        self.nA = nA
        self.K = K
        self.n_em = n_em_iters
        self.pseudo = pseudocount
        self.seed = seed

    def fit(self, sequences):
        x_train = np.concatenate(
            [np.asarray(s, dtype=np.int64) for s in sequences if len(s) > 0]
        ).astype(np.int64)
        a_train = np.zeros_like(x_train)
        n_clones = np.full(self.nA, self.K, dtype=np.int64)
        self.model = CHMM(n_clones=n_clones, x=x_train, a=a_train,
                          pseudocount=self.pseudo, seed=self.seed)
        self.model.learn_em_T(x_train, a_train, n_iter=self.n_em,
                              term_early=True)
        # Cache transition matrix and clone-block lookups
        self.T = self.model.T[0].astype(np.float64)   # (n_total, n_total)
        self.state_loc = np.hstack(([0], n_clones)).cumsum().astype(np.int64)
        self.n_total = int(n_clones.sum())
        return self

    def horizon_emission(self, prefix_obs, h):
        prefix = np.asarray(prefix_obs, dtype=np.int64)
        if len(prefix) == 0:
            alpha_full = self.model.Pi_x.astype(np.float64).copy()
        else:
            a_prefix = np.zeros_like(prefix)
            _, mess_fwd = chmm_forward(
                self.model.T.transpose(0, 2, 1), self.model.Pi_x,
                self.model.n_clones, prefix, a_prefix,
                store_messages=True)
            mess_loc = np.hstack(
                ([0], self.model.n_clones[prefix])).cumsum().astype(np.int64)
            t_last = len(prefix) - 1
            ms, me = int(mess_loc[t_last]), int(mess_loc[t_last + 1])
            xt = int(prefix[-1])
            gs, ge = int(self.state_loc[xt]), int(self.state_loc[xt + 1])
            alpha_full = np.zeros(self.n_total, dtype=np.float64)
            alpha_full[gs:ge] = mess_fwd[ms:me].astype(np.float64)
        # Apply transition h times to get state-dist at t+h
        for _ in range(h):
            alpha_full = alpha_full @ self.T
        out = np.zeros(self.nA, dtype=np.float64)
        for a in range(self.nA):
            gs, ge = int(self.state_loc[a]), int(self.state_loc[a + 1])
            out[a] = alpha_full[gs:ge].sum()
        s = out.sum()
        if s <= 0:
            return np.full(self.nA, 1.0 / self.nA)
        return out / s


# --------------------------------------------------------------------
# ALERGIA via AALpy
# --------------------------------------------------------------------
class AlergiaForecaster:
    def __init__(self, nA, eps=0.05):
        self.nA = nA
        self.eps = eps

    def fit(self, sequences):
        if not HAS_AALPY:
            raise RuntimeError("AALpy not installed")
        # AALpy MC requires every sequence to start with the same symbol;
        # prepend a sentinel START token.
        START = self.nA   # one past the alphabet
        self.START = START
        data = [[START] + [int(t) for t in s] for s in sequences
                if len(s) > 0]
        self.mc = run_Alergia(data, automaton_type='mc', eps=self.eps,
                              print_info=False)
        states = list(self.mc.states)
        self.state_idx = {id(s): i for i, s in enumerate(states)}
        self.n_states = len(states)
        outputs = np.array([s.output for s in states], dtype=int)
        self.outputs = outputs
        # Transition matrix (sums across emissions to get state-to-state)
        T = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        for s in states:
            i = self.state_idx[id(s)]
            for target, prob in s.transitions:
                j = self.state_idx[id(target)]
                T[i, j] += float(prob)
        # Numerical: each row should already sum to ~1; tolerate drift
        rs = T.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        self.T = T / rs
        self.initial_idx = self.state_idx[id(self.mc.initial_state)]
        return self

    def _state_after_prefix(self, prefix):
        """Walk the MC deterministically following the symbols in `prefix`.
        Returns either a state index (int) or None if the path is broken
        (no compatible transition found at some step)."""
        st = self.mc.initial_state  # start state has output == START
        for sym in prefix:
            ns = None
            best_p = -1.0
            for target, prob in st.transitions:
                if target.output == int(sym) and float(prob) > best_p:
                    ns = target; best_p = float(prob)
            if ns is None:
                return None
            st = ns
        return self.state_idx[id(st)]

    def horizon_emission(self, prefix_obs, h):
        idx = self._state_after_prefix(list(prefix_obs))
        if idx is None:
            # Path broken — fall back to uniform
            return np.full(self.nA, 1.0 / self.nA)
        state_dist = np.zeros(self.n_states, dtype=np.float64)
        state_dist[idx] = 1.0
        for _ in range(h):
            state_dist = state_dist @ self.T
        out = np.zeros(self.nA, dtype=np.float64)
        for a in range(self.nA):
            mask = self.outputs == a
            out[a] = state_dist[mask].sum()
        s = out.sum()
        if s <= 0:
            return np.full(self.nA, 1.0 / self.nA)
        return out / s


def fit_chmm(sequences, alphabet_size, K=4, n_em_iters=50):
    return CHMMForecaster(alphabet_size, K=K,
                          n_em_iters=n_em_iters).fit(sequences)


def fit_alergia(sequences, alphabet_size, eps=0.05):
    return AlergiaForecaster(alphabet_size, eps=eps).fit(sequences)
