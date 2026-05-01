"""
Random discrete-emission HMM utilities for the forecasting comparison
experiment.

Conventions (all numpy):
    pi : shape (nS,)              initial state distribution
    T  : shape (nS, nS)           T[i, j] = P(state_{t+1}=j | state_t=i)
    E  : shape (nS, nA)           E[i, a] = P(emit=a | state=i)

Given a prefix of observations o_1..o_t we can compute:
    alpha_t[i]        = P(state_t=i | o_1..o_t)             (forward filter)
    P(o_{t+h}=a | prefix) = (alpha_t @ T^h @ E)[a]          (horizon-h emission)

RandomHMM exposes:
    .sample(length, rng)         -> (states, obs)
    .sample_many(N, length, rng) -> list of obs arrays
    .filter(obs)                 -> alpha_t (posterior over states)
    .horizon_emission(alpha, h)  -> shape (nA,) predicted next-symbol at t+h
"""

from __future__ import annotations

import numpy as np


class RandomHMM:
    def __init__(self, T: np.ndarray, E: np.ndarray, pi: np.ndarray):
        assert T.ndim == 2 and T.shape[0] == T.shape[1]
        assert E.ndim == 2 and E.shape[0] == T.shape[0]
        assert pi.shape == (T.shape[0],)
        if not np.allclose(T.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("T rows must sum to 1")
        if not np.allclose(E.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("E rows must sum to 1")
        if not np.isclose(pi.sum(), 1.0, atol=1e-8):
            raise ValueError("pi must sum to 1")
        self.T = np.asarray(T, dtype=np.float64)
        self.E = np.asarray(E, dtype=np.float64)
        self.pi = np.asarray(pi, dtype=np.float64)
        self.nS = T.shape[0]
        self.nA = E.shape[1]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(self, length: int, rng: np.random.Generator):
        states = np.empty(length, dtype=np.int64)
        obs = np.empty(length, dtype=np.int64)
        s = rng.choice(self.nS, p=self.pi)
        for t in range(length):
            states[t] = s
            obs[t] = rng.choice(self.nA, p=self.E[s])
            s = rng.choice(self.nS, p=self.T[s])
        return states, obs

    def sample_many(self, n_seq: int, length: int, rng: np.random.Generator):
        return [self.sample(length, rng)[1] for _ in range(n_seq)]

    # ------------------------------------------------------------------
    # Exact forecasting
    # ------------------------------------------------------------------
    def filter(self, obs: np.ndarray) -> np.ndarray:
        """Forward filter; returns posterior over states after observing obs."""
        a = self.pi * self.E[:, obs[0]]
        s = a.sum()
        a = a / s if s > 0 else np.full(self.nS, 1.0 / self.nS)
        for o in obs[1:]:
            a = (a @ self.T) * self.E[:, o]
            s = a.sum()
            a = a / s if s > 0 else np.full(self.nS, 1.0 / self.nS)
        return a

    def horizon_emission(self, alpha: np.ndarray, h: int) -> np.ndarray:
        """P(o_{t+h} = a | prefix) for horizon h >= 1."""
        if h < 1:
            raise ValueError("h must be >= 1")
        # Propagate state distribution h-1 full transitions, then emit.
        # alpha_{t+h-1} = alpha_t @ T^{h-1}; then emission is alpha_{t+h-1} @ T @ E
        #              -- one final transition because we want the state at t+h,
        # equivalent: alpha_t @ T^h @ E
        state_dist = alpha
        for _ in range(h):
            state_dist = state_dist @ self.T
        return state_dist @ self.E


# ----------------------------------------------------------------------
# Random HMM constructors
# ----------------------------------------------------------------------
def random_dense_hmm(nS: int, nA: int, rng: np.random.Generator,
                     T_concentration: float = 1.0,
                     E_concentration: float = 1.0) -> RandomHMM:
    """Rows of T and E drawn iid from Dirichlet(concentration)."""
    T = rng.dirichlet(np.full(nS, T_concentration), size=nS)
    E = rng.dirichlet(np.full(nA, E_concentration), size=nS)
    pi = rng.dirichlet(np.full(nS, 1.0))
    return RandomHMM(T, E, pi)


def random_sparse_topology_hmm(nS: int, nA: int, rng: np.random.Generator,
                               fanout: int = 2,
                               E_concentration: float = 1.0) -> RandomHMM:
    """Each row of T supports exactly `fanout` successors (uniformly random)."""
    fanout = min(fanout, nS)
    T = np.zeros((nS, nS))
    for i in range(nS):
        targets = rng.choice(nS, size=fanout, replace=False)
        w = rng.dirichlet(np.full(fanout, 1.0))
        T[i, targets] = w
    E = rng.dirichlet(np.full(nA, E_concentration), size=nS)
    pi = rng.dirichlet(np.full(nS, 1.0))
    return RandomHMM(T, E, pi)


def random_lowrank_hmm(nS: int, nA: int, rank: int,
                       rng: np.random.Generator,
                       E_concentration: float = 1.0) -> RandomHMM:
    """Construct a rank-`rank` transition matrix and a random emission matrix.

    T = row_normalize(U @ V) where U is (nS, rank) and V is (rank, nS), each
    built to yield a proper stochastic matrix. Specifically:
        V[r, :] ~ Dirichlet(1)              (each 'factor' is a distribution)
        mix[i, :] ~ Dirichlet(1) over rank  (each state mixes factors)
        T[i, :] = mix[i, :] @ V
    This ensures rank(T) <= rank exactly.
    """
    assert 1 <= rank <= nS
    V = rng.dirichlet(np.full(nS, 1.0), size=rank)        # (rank, nS)
    mix = rng.dirichlet(np.full(rank, 1.0), size=nS)       # (nS, rank)
    T = mix @ V                                            # (nS, nS)
    # Numerical row-normalise
    T = T / T.sum(axis=1, keepdims=True)
    E = rng.dirichlet(np.full(nA, E_concentration), size=nS)
    pi = rng.dirichlet(np.full(nS, 1.0))
    return RandomHMM(T, E, pi)


__all__ = [
    "RandomHMM",
    "random_dense_hmm",
    "random_sparse_topology_hmm",
    "random_lowrank_hmm",
]
