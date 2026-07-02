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


def random_bimodal_hmm(nS: int, nA: int, rng: np.random.Generator,
                       sticky_prob: float = 0.95,
                       E_concentration: float = 0.1) -> RandomHMM:
    """Two equal-size state clusters with disjoint emission supports.
    Cluster A (states 0..nS/2-1) emits symbols [0, nA/2);
    cluster B (states nS/2..nS-1) emits symbols [nA/2, nA).

    Within-cluster transitions are dense Dirichlet; cross-cluster
    transitions take (1 - sticky_prob) of the row mass.

    Stationary emission marginal ≈ uniform across nA symbols (under
    balanced cluster visit), but conditional given prefix concentrates
    on one cluster's nA/2 symbols. So a unigram-frequency baseline
    sits near uniform while structure-aware models recover cluster
    membership and concentrate on the right half.
    """
    assert nS % 2 == 0 and nA % 2 == 0, "nS and nA must be even"
    half_S = nS // 2
    half_A = nA // 2

    T = np.zeros((nS, nS))
    for i in range(nS):
        in_A = i < half_S
        if in_A:
            T[i, :half_S] = sticky_prob * rng.dirichlet(
                np.full(half_S, 1.0))
            T[i, half_S:] = (1 - sticky_prob) * rng.dirichlet(
                np.full(half_S, 1.0))
        else:
            T[i, half_S:] = sticky_prob * rng.dirichlet(
                np.full(half_S, 1.0))
            T[i, :half_S] = (1 - sticky_prob) * rng.dirichlet(
                np.full(half_S, 1.0))

    E = np.zeros((nS, nA))
    for i in range(nS):
        if i < half_S:
            E[i, :half_A] = rng.dirichlet(
                np.full(half_A, E_concentration))
        else:
            E[i, half_A:] = rng.dirichlet(
                np.full(half_A, E_concentration))

    pi = rng.dirichlet(np.full(nS, 1.0))
    return RandomHMM(T, E, pi)


def random_cyclic_hmm(nS: int, nA: int, rng: np.random.Generator,
                      advance_prob: float = 0.95,
                      E_concentration: float = 0.1) -> RandomHMM:
    """States arranged in a directed cycle of length nS. State i
    transitions to (i+1) mod nS with `advance_prob`; the rest is
    uniform diffuse over the other nS-1 states. Each state emits
    one symbol (i mod nA) with strong concentration.

    If nS > nA, the cycle "wraps" the alphabet — multiple states
    emit the same symbol, but at different cycle positions, so the
    prefix can disambiguate which cycle position you're at.

    Stationary marginal ≈ uniform across nA symbols; conditional
    given a sufficiently long prefix is near-deterministic.
    """
    diffuse_rate = (1.0 - advance_prob) / (nS - 1)
    T = np.full((nS, nS), diffuse_rate)
    for i in range(nS):
        T[i, i] = 0.0
        T[i, (i + 1) % nS] = advance_prob
    T = T / T.sum(axis=1, keepdims=True)

    E = np.zeros((nS, nA))
    for i in range(nS):
        target = i % nA
        alpha_emit = np.full(nA, E_concentration)
        alpha_emit[target] = 10.0  # strong concentration on target symbol
        E[i] = rng.dirichlet(alpha_emit)

    pi = rng.dirichlet(np.full(nS, 1.0))
    return RandomHMM(T, E, pi)


def random_binary_deep_hmm(nS: int, rng: np.random.Generator,
                           fanout: int = 2,
                           E_concentration: float = 0.1) -> RandomHMM:
    """Binary alphabet (nA = 2) sparse-topology HMM. Specialization
    of random_sparse_topology_hmm with nA fixed to 2 and a roughly
    balanced 0/1 emission frequency across states. Designed so the
    stationary marginal is ~50/50 — a unigram baseline guesses 0.5,
    while methods that track state can resolve the next bit much
    more sharply."""
    return random_sparse_topology_hmm(nS, 2, rng, fanout=fanout,
                                      E_concentration=E_concentration)


def random_reset_chain_hmm(nS: int, nA: int, rng: np.random.Generator,
                           advance_prob: float = 0.90,
                           reset_prob: float = 0.05,
                           E_concentration: float = 0.1) -> RandomHMM:
    """Linear chain with occasional resets. State i advances to
    state i+1 with `advance_prob`, jumps back to state 0 (reset)
    with `reset_prob`, and a small uniform diffuse covers the
    remaining mass. The last state advances back to state 0 (the
    chain is wrap-closed).

    Each state emits one symbol (i mod nA) with strong concentration.
    Stationary marginal ≈ uniform across symbols; conditional given
    prefix can localize the chain position via the recent emission
    pattern, then predict near-deterministically.

    Different from random_cyclic_hmm: the reset events break the
    long-range correlation, so prefix-based localization needs a
    moderate-length window (not too long, otherwise everything is
    "after the most recent reset").
    """
    diffuse_rate = (1.0 - advance_prob - reset_prob) / max(nS - 2, 1)
    T = np.zeros((nS, nS))
    for i in range(nS):
        # Diffuse to all but i and i+1 / 0 (we'll fill those in)
        T[i, :] = diffuse_rate
        T[i, i] = 0.0
        # Advance
        if i < nS - 1:
            T[i, i + 1] = advance_prob
        else:
            T[i, 0] = advance_prob  # last state wraps
        # Reset (state 0 doesn't reset; advance from 0 covers it)
        if i > 0:
            T[i, 0] = T[i, 0] + reset_prob
    T = T / T.sum(axis=1, keepdims=True)

    E = np.zeros((nS, nA))
    for i in range(nS):
        target = i % nA
        alpha_emit = np.full(nA, E_concentration)
        alpha_emit[target] = 10.0
        E[i] = rng.dirichlet(alpha_emit)

    pi = rng.dirichlet(np.full(nS, 1.0))
    return RandomHMM(T, E, pi)


__all__ = [
    "RandomHMM",
    "random_dense_hmm",
    "random_sparse_topology_hmm",
    "random_lowrank_hmm",
    "random_bimodal_hmm",
    "random_cyclic_hmm",
    "random_binary_deep_hmm",
    "random_reset_chain_hmm",
]
