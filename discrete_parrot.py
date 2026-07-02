"""Discrete top-K context parroting.

Companion to skolr_bench/forecast/parrot_torch.py for *discrete* sequence
prediction (HMM, Turing-machine traces, Dyck-1).

Core idea — given a corpus of historical token sequences, the predictor
maintains all length-L sliding windows of the corpus together with their
1-step-ahead continuations. To forecast the distribution over the next
token after some `prefix`, find the K corpus L-windows closest (Hamming)
to `prefix[-L:]`, count the K observed next-tokens, and Laplace-smooth.

The same mechanism extends to multi-step horizons: instead of looking at
`corpus[s+1]`, look at `corpus[s+h]`.

API:

    pool = DiscreteParrotPool(corpora, alphabet_size, L)
        # corpora : list of 1-D int arrays (sequences)
    p = pool.predict_distribution(prefix, h=1, K=5, alpha=1.0)
        # prefix : 1-D int array (length >= L)
        # returns : (alphabet_size,) probability vector
    a = pool.predict_argmax(prefix, h=1, K=5, mask=None)
        # mask  : optional bool mask restricting candidates to a subset
        # returns : int — argmax of (possibly masked) distribution

When K is None the predictor falls back to K=1 hard nearest-neighbour
(no smoothing); for K=1 with Laplace smoothing pass alpha>0.
"""
from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np


class DiscreteParrotPool:
    """Pool of sliding L-windows over a list of training sequences.

    Stored as a single (n_w, L) int64 matrix `W` plus a (n_w,) array
    `next_idx` giving the absolute position in the concatenated corpus of
    the token *immediately after* each window. Multi-step lookups
    (corpus[s + h]) are computed on the fly from `next_idx` by indexing
    into the original concatenation.
    """

    def __init__(self, corpora: Sequence[np.ndarray], alphabet_size: int,
                 L: int):
        self.alphabet_size = int(alphabet_size)
        self.L = int(L)
        self._build(corpora)

    def _build(self, corpora):
        L = self.L
        Ws = []
        next_idxs = []
        # We also keep the concatenated corpus to support h-step lookups.
        concat = []
        offset = 0
        for seq in corpora:
            seq = np.asarray(seq, dtype=np.int64).ravel()
            if len(seq) < L + 1:
                # Sequence too short for any (window, next) pair
                concat.append(seq); offset += len(seq); continue
            # n_w = len(seq) - L windows that have at least 1 future token
            n_w = len(seq) - L
            starts = np.arange(n_w)
            W = seq[starts[:, None] + np.arange(L)[None, :]]
            # absolute index in concatenation:  offset + (start + L)
            next_abs = offset + starts + L
            Ws.append(W); next_idxs.append(next_abs)
            concat.append(seq); offset += len(seq)
        self.concat = np.concatenate(concat) if concat else np.empty(0, dtype=np.int64)
        if Ws:
            self.W = np.concatenate(Ws, axis=0)
            self.next_idx = np.concatenate(next_idxs, axis=0)
        else:
            self.W = np.empty((0, L), dtype=np.int64)
            self.next_idx = np.empty(0, dtype=np.int64)
        # Marginal token distribution from the concatenated corpus, for fallback
        if len(self.concat) > 0:
            counts = np.bincount(self.concat, minlength=self.alphabet_size).astype(np.float64)
            self.marginal = (counts + 1.0) / (counts.sum() + self.alphabet_size)
        else:
            self.marginal = np.full(self.alphabet_size, 1.0 / self.alphabet_size)

    def _topk_indices(self, prime: np.ndarray, K: int) -> np.ndarray:
        """Hamming-distance top-K window indices."""
        if self.W.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        # Hamming distance per window
        d = (self.W != prime[None, :]).sum(axis=1)
        K = min(K, self.W.shape[0])
        if K == self.W.shape[0]:
            return np.argsort(d)
        # argpartition for top-K, then optional sort within top-K (we only
        # need the set; ordering matters only for ties when K > pool size,
        # which is handled above).
        return np.argpartition(d, K - 1)[:K]

    def _continuations_at_horizon(self, neighbour_idx: np.ndarray, h: int):
        """Return the corpus tokens at offset h-1 past each neighbour's
        next-position. Skips neighbours that fall off the end."""
        # For neighbour s, predicted-token absolute index is `next_idx[s] + (h-1)`.
        abs_idx = self.next_idx[neighbour_idx] + (h - 1)
        ok = abs_idx < len(self.concat)
        return self.concat[abs_idx[ok]] if ok.any() else np.empty(0, dtype=np.int64)

    def predict_distribution(self, prefix: np.ndarray, h: int = 1,
                              K: int = 5, alpha_prior: float = 1.0,
                              fall_back_to_marginal: bool = True
                              ) -> np.ndarray:
        """Return a length-`alphabet_size` probability vector.

        alpha_prior : Laplace smoothing parameter applied to the empirical
                      count vector before normalisation. alpha_prior=0
                      gives the raw empirical distribution (which can have
                      hard zeros and break log-likelihood scoring).
        fall_back_to_marginal : if no neighbour is valid (all fall off the
                      end of the corpus), return the corpus marginal
                      distribution instead of uniform.
        """
        prefix = np.asarray(prefix, dtype=np.int64).ravel()
        if len(prefix) < self.L or self.W.shape[0] == 0:
            return (self.marginal.copy() if fall_back_to_marginal
                    else np.full(self.alphabet_size, 1.0 / self.alphabet_size))
        prime = prefix[-self.L:]
        nb = self._topk_indices(prime, K)
        cont = self._continuations_at_horizon(nb, h)
        if cont.size == 0:
            return (self.marginal.copy() if fall_back_to_marginal
                    else np.full(self.alphabet_size, 1.0 / self.alphabet_size))
        counts = np.bincount(cont, minlength=self.alphabet_size).astype(np.float64)
        smoothed = counts + alpha_prior
        return smoothed / smoothed.sum()

    def predict_argmax(self, prefix: np.ndarray, h: int = 1, K: int = 5,
                       mask: Optional[np.ndarray] = None,
                       alpha_prior: float = 1.0) -> int:
        """Argmax of distribution, optionally restricted to symbols where
        `mask[a]` is True. Falls back to the marginal among the masked
        symbols if no neighbours land on a masked symbol."""
        p = self.predict_distribution(prefix, h=h, K=K,
                                       alpha_prior=alpha_prior)
        if mask is not None:
            p = p.copy()
            p[~mask] = -1.0
        return int(np.argmax(p))


__all__ = ['DiscreteParrotPool']
