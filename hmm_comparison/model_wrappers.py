"""
Thin adapters that expose a common API for next-symbol horizon forecasting
on scalar-emission sequences:

    m = fit_oom(sequences, alphabet_size, L=3)
    m = fit_gdc(sequences, alphabet_size, alpha=0.95, theta=0.05, beta=0.02)

    dist = m.horizon_emission(prefix_obs, h)    # shape (alphabet_size,)
"""

from __future__ import annotations

import sys
import os
import numpy as np

# Make the worktree root importable so we can load spectral_oom and GDC.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from spectral_oom import SpectralOOM            # noqa: E402
from generative_dense_chain import GenerativeDenseChain  # noqa: E402


# ---------------------------------------------------------------------------
# OOM wrapper
# ---------------------------------------------------------------------------
class OOMForecaster:
    """Spectral OOM trained on scalar-token sequences.

    alphabet : fixed 0..nA-1 so that unseen-in-training symbols still produce
               a valid (uniform fallback) prediction.
    """
    def __init__(self, nA: int, max_basis_length: int = 3,
                 rank: int | None = None, prob_mode: str = 'clip'):
        self.nA = nA
        self.prob_mode = prob_mode
        self.oom = SpectralOOM(max_basis_length=max_basis_length,
                               rank=rank, verbose=False)
        self._fitted = False

    def fit(self, sequences):
        seqs = [[int(x) for x in s] for s in sequences]
        self.oom.fit(seqs)
        self._fitted = True
        # Map OOM's alphabet (sorted, possibly a strict subset of 0..nA-1)
        # to indices in the full 0..nA-1 vector.
        self._oom_to_full = np.array([self.oom.tok2id.get(a, -1)
                                      for a in range(self.nA)], dtype=np.int64)
        return self

    def horizon_emission(self, prefix_obs, h: int) -> np.ndarray:
        toks = [int(x) for x in prefix_obs]
        # Filter to tokens OOM has seen; unknown tokens get skipped so state
        # simply doesn't advance (same behaviour as the Turing test).
        state, _ = self.oom.forward_pass(toks, return_history=True)
        probs_oom = self.oom.predict_next_probs(
            state, horizon=h, mode=self.prob_mode)  # over OOM alphabet
        # Scatter into full alphabet.
        out = np.zeros(self.nA)
        for full_a, oom_idx in enumerate(self._oom_to_full):
            if oom_idx >= 0:
                out[full_a] = probs_oom[oom_idx]
        s = out.sum()
        if s <= 0:
            return np.full(self.nA, 1.0 / self.nA)
        return out / s


# ---------------------------------------------------------------------------
# GDC wrapper
# ---------------------------------------------------------------------------
class GDCForecaster:
    """GDC trained on scalar-token sequences.

    Observations are passed as (T, 1) int arrays.
    horizon_emission marginalises the state distribution at t+h back to a
    symbol distribution using the fact that each GDC hidden state is tied to
    a specific 1-d observation.
    """
    def __init__(self, nA: int, alpha: float = 0.95, theta: float = 0.0,
                 gamma: float = 0.0, beta: float = 0.02,
                 transition_type: str = 'self_loop',
                 initial_dist: str = 'sequence_starts',
                 terminal_behavior: str = 'diffuse'):
        self.nA = nA
        self.alpha = alpha
        self.theta = theta
        self.gamma = gamma
        self.beta = beta
        self.transition_type = transition_type
        self.initial_dist = initial_dist
        self.terminal_behavior = terminal_behavior
        self.gdc = None
        self._symbol_of_state = None

    def fit(self, sequences):
        seq_arrays = [np.asarray(s, dtype=np.int64).reshape(-1, 1) for s in sequences]
        self.gdc = GenerativeDenseChain(
            seq_arrays,
            alpha=self.alpha, theta=self.theta, gamma=self.gamma,
            beta=self.beta,
            transition_type=self.transition_type,
            initial_dist=self.initial_dist,
            terminal_behavior=self.terminal_behavior,
        )
        # Each GDC hidden state corresponds to exactly one observed scalar.
        self._symbol_of_state = self.gdc.states[:, 0].astype(np.int64)
        return self

    def horizon_emission(self, prefix_obs, h: int) -> np.ndarray:
        obs = np.asarray(prefix_obs, dtype=np.int64).reshape(-1, 1)
        final_dist = self.gdc.forward_pass(obs, return_history=False)
        forecast = self.gdc.forecast(final_dist, n_steps=h)
        # Marginalise out hidden state: sum probability by associated symbol.
        out = np.zeros(self.nA)
        syms = self._symbol_of_state
        np.add.at(out, syms, forecast)
        s = out.sum()
        if s <= 0:
            return np.full(self.nA, 1.0 / self.nA)
        return out / s


def fit_oom(sequences, alphabet_size, max_basis_length=3, rank=None,
            prob_mode='clip'):
    return OOMForecaster(alphabet_size, max_basis_length, rank,
                         prob_mode=prob_mode).fit(sequences)


def fit_gdc(sequences, alphabet_size, **kwargs):
    return GDCForecaster(alphabet_size, **kwargs).fit(sequences)
