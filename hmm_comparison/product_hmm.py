"""Product-space HMM utilities.

Given N independent component HMMs (each with their own state space and
binary emissions), the *product HMM* is a single HMM whose:
  - state = (s1, s2, ..., sN), encoded as an integer base-(nS_i product)
  - emission = (e1, e2, ..., eN), encoded as an integer base-(nA_i product)

The transition matrix is the Kronecker product of the components, the
emission matrix is the Kronecker product of the component E matrices,
and the initial distribution is the Kronecker product of the component
pi vectors.

The "observed output is the product space of observations" the user
described corresponds to this 1-to-1 encoding of (e1, ..., eN) into a
single integer in [0, prod(nA_i)).

For binary-emission components: encoding is e1 + 2 * e2 + 4 * e3 + ...
(so (0,0,0)→0, (1,0,0)→1, (0,1,0)→2, ..., (1,1,1)→7).
The user's 1-indexed convention is just this + 1; we use the 0-indexed
form internally because every other piece of code already does.
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from random_hmm import RandomHMM, random_sparse_topology_hmm  # noqa: E402


def kronecker_chain(matrices):
    """Kronecker product of a list of 2-D arrays."""
    out = matrices[0]
    for M in matrices[1:]:
        out = np.kron(out, M)
    return out


def build_product_hmm(component_hmms):
    """Construct the product HMM of N component HMMs.

    Returns a RandomHMM whose state space and emission space are the
    products of the components. The encoding of (s1, ..., sN) into a
    single state index is the row-major (np.kron) order:
        idx = s1 * prod(nS[1:]) + s2 * prod(nS[2:]) + ... + sN
    Same for emissions.
    """
    Ts = [h.T for h in component_hmms]
    Es = [h.E for h in component_hmms]
    pis = [h.pi for h in component_hmms]
    T_prod = kronecker_chain(Ts)
    E_prod = kronecker_chain(Es)
    pi_prod = kronecker_chain([pi.reshape(1, -1) for pi in pis]).ravel()
    return RandomHMM(T_prod, E_prod, pi_prod)


def encode_components(component_obs):
    """Convert a list of (T,) component-observation arrays to a single
    (T,) product-encoded integer array. Encoding matches np.kron order:
    e1 * prod(nA_2..N) + e2 * prod(nA_3..N) + ... + eN.

    For binary components: idx = sum_i e_i * 2^(N-1-i).
    """
    arrs = [np.asarray(o, dtype=np.int64) for o in component_obs]
    N = len(arrs)
    T = arrs[0].shape[0]
    # Determine each component's nA from observation maxima (assumes 0..nA-1).
    # For exact correctness we need the user to supply nA per component.
    # Default to binary (max(o)+1 = 2 typically).
    nAs = [int(o.max()) + 1 if o.size > 0 else 1 for o in arrs]
    idx = np.zeros(T, dtype=np.int64)
    stride = 1
    for i in reversed(range(N)):
        idx += stride * arrs[i]
        stride *= nAs[i]
    return idx, nAs


def make_independent_components(n_components, nS, nA, fanout, rng,
                                  E_concentration=0.1):
    """Sample n_components independent sparse-topology HMMs (each with
    same nS, nA, fanout, E_concentration). Use distinct rngs derived
    from the master rng so the components are independent."""
    components = []
    for i in range(n_components):
        sub_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        components.append(
            random_sparse_topology_hmm(nS, nA, sub_rng, fanout=fanout,
                                         E_concentration=E_concentration))
    return components


def random_state_preferred_hmm(nS: int, nA: int, rng,
                                  fanout: int = 2,
                                  min_pref_prob: float = 0.5,
                                  min_self_prob: float = 0.0,
                                  t_concentration: float = None):
    """Sparse-topology HMM where state i has a 'preferred' emission
    symbol (state i ↔ symbol i % nA) that appears with at least
    `min_pref_prob` probability when in that state.

    The remaining 1 - p_pref emission mass is split uniformly random
    (Dirichlet(1)) across the other (nA - 1) symbols.

    Transitions:
      - If `t_concentration` is given: dense Dirichlet — each row of T
        is independently drawn from Dirichlet(t_concentration · 𝟙_nS).
        Values < 1 give peaked rows (concentrated on one state); values
        > 1 give uniform-like rows. Use t_concentration=0.1 for
        sparse-but-fully-connected transitions.
      - Else if `min_self_prob == 0`: each state's transition is
        supported on a random `fanout`-sized subset of all states.
      - Else: T[i, i] uniform in [min_self_prob, 1]; remaining
        (1 - p_self) split among (fanout - 1) other states. Slows
        chain mixing.
    """
    fanout = min(fanout, nS)
    T = np.zeros((nS, nS))
    if t_concentration is not None:
        for i in range(nS):
            T[i] = rng.dirichlet(np.full(nS, t_concentration))
    elif min_self_prob <= 0.0:
        for i in range(nS):
            targets = rng.choice(nS, size=fanout, replace=False)
            w = rng.dirichlet(np.full(fanout, 1.0))
            T[i, targets] = w
    else:
        for i in range(nS):
            p_self = float(rng.uniform(min_self_prob, 1.0))
            T[i, i] = p_self
            n_other = min(fanout - 1, nS - 1)
            if n_other > 0:
                others = [j for j in range(nS) if j != i]
                targets = rng.choice(others, size=n_other, replace=False)
                w = rng.dirichlet(np.full(n_other, 1.0))
                T[i, targets] = (1.0 - p_self) * w

    E = np.zeros((nS, nA))
    for i in range(nS):
        pref = i % nA
        # Preferred mass uniform in [min_pref_prob, 1].
        p_pref = float(rng.uniform(min_pref_prob, 1.0))
        E[i, pref] = p_pref
        # Remaining (1 - p_pref) split across other (nA - 1) symbols
        # via Dirichlet(1).
        if nA > 1:
            other_idx = [j for j in range(nA) if j != pref]
            other_w = rng.dirichlet(np.full(nA - 1, 1.0))
            E[i, other_idx] = (1.0 - p_pref) * other_w
    pi = rng.dirichlet(np.full(nS, 1.0))
    return RandomHMM(T, E, pi)


def make_state_preferred_components(n_components, nS, nA, fanout, rng,
                                       min_pref_prob: float = 0.5,
                                       min_self_prob: float = 0.0,
                                       t_concentration: float = None):
    """Same as `make_independent_components` but each component HMM
    has state-preferred emissions (E[i, i % nA] >= min_pref_prob).
    Pass `t_concentration` for Dirichlet transitions, else
    `min_self_prob` for self-loop bias."""
    components = []
    for i in range(n_components):
        sub_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        components.append(
            random_state_preferred_hmm(nS, nA, sub_rng, fanout=fanout,
                                          min_pref_prob=min_pref_prob,
                                          min_self_prob=min_self_prob,
                                          t_concentration=t_concentration))
    return components


def sample_product(component_hmms, length, rng):
    """Sample a sequence of product-encoded observations by sampling
    each component independently. Returns (component_obs_list, prod_obs)."""
    component_obs = []
    for h in component_hmms:
        sub_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        _, obs = h.sample(length, sub_rng)
        component_obs.append(obs)
    prod_obs, _ = encode_components(component_obs)
    return component_obs, prod_obs
