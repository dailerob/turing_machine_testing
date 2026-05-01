"""
Spectral Observable Operator Model (OOM) / Weighted Finite Automaton (WFA).

Learns a rank-d linear dynamical system over discrete tokens from a set of
training sequences, via SVD of the empirical Hankel matrix of substring counts.

Core formulation (Balle & Mohri, Hsu-Kakade-Zhang style):

    H  [u, v] = # occurrences of u v as a substring of training data
    H_a[u, v] = # occurrences of u a v as a substring
    H = U Σ Vᵀ   (truncated to rank d)

    A_a      = Uᵀ H_a V diag(1/σ)       # (d x d) operator for token a
    α_0      = U[ε, :]                    # initial state (row of U for empty prefix)
    α_∞      = diag(σ) V[ε, :]            # final state (column of ΣVᵀ for empty suffix)

Probability-like score of sequence w = a_1 ... a_n:

    f(w) ≈ α_0ᵀ A_{a_1} ... A_{a_n} α_∞

For prediction at step t with prefix context state ω_t:
    score(a | context) = ω_tᵀ A_a α_∞
    P(a | context) ∝ max(score, 0)
"""

from __future__ import annotations

import time
import numpy as np
from collections import Counter


class SpectralOOM:
    def __init__(self, max_basis_length: int = 2, rank: int | None = None,
                 sv_rel_threshold: float = 1e-8, renormalize: bool = True,
                 verbose: bool = True):
        """
        Parameters
        ----------
        max_basis_length
            Max substring length L used for prefix/suffix basis. Basis contains
            the empty string plus all substrings of length 1..L in training.
        rank
            Target rank for SVD truncation. None = use all singular values above
            sv_rel_threshold * σ_max.
        sv_rel_threshold
            Relative threshold for dropping tiny singular values.
        renormalize
            If True, rescale the state vector to unit L2 norm after each step
            during forward passes to avoid numerical under/overflow. Does not
            affect predictions since predict_next normalises too.
        """
        self.max_basis_length = max_basis_length
        self.rank = rank
        self.sv_rel_threshold = sv_rel_threshold
        self.renormalize = renormalize
        self.verbose = verbose

    def _log(self, *args):
        if self.verbose:
            print("[OOM]", *args, flush=True)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, sequences):
        """
        sequences: iterable of iterables of hashable tokens.
        """
        t0 = time.time()

        # Alphabet from all tokens that appear in training
        symbols = set()
        for seq in sequences:
            for tok in seq:
                symbols.add(tok)
        self.alphabet = sorted(symbols, key=str)
        self.tok2id = {t: i for i, t in enumerate(self.alphabet)}
        nA = len(self.alphabet)
        self._log(f"alphabet size = {nA}")

        # Convert to int32 arrays
        seqs = [np.asarray([self.tok2id[t] for t in seq], dtype=np.int32)
                for seq in sequences]

        # Build basis: all substrings up to length L that actually appear.
        L = self.max_basis_length
        basis_set = {()}  # always include empty string
        for seq in seqs:
            T = len(seq)
            for start in range(T):
                upto = min(L, T - start)
                for length in range(1, upto + 1):
                    basis_set.add(tuple(int(x) for x in seq[start:start + length]))
        basis = sorted(basis_set, key=lambda s: (len(s), s))
        self.prefix_basis = basis
        self.suffix_basis = basis
        self.P_idx = {u: i for i, u in enumerate(basis)}
        self.S_idx = self.P_idx
        nP = len(basis)
        self._log(f"basis size = {nP} (shared prefix/suffix)")

        # Build Hankel tensors via substring counts.
        #   H  [u, v] = count of substring  u·v
        #   H_a[u, v] = count of substring  u·a·v
        H = np.zeros((nP, nP), dtype=np.float64)
        H_a = np.zeros((nA, nP, nP), dtype=np.float64)

        t1 = time.time()
        for seq in seqs:
            T = int(len(seq))
            seq_list = seq.tolist()
            for pos in range(T):
                max_u = min(L, T - pos)
                for ul in range(0, max_u + 1):
                    u = tuple(seq_list[pos:pos + ul])
                    ui = self.P_idx.get(u)
                    if ui is None:
                        continue
                    max_v = min(L, T - pos - ul)
                    for vl in range(0, max_v + 1):
                        v = tuple(seq_list[pos + ul:pos + ul + vl])
                        vi = self.S_idx.get(v)
                        if vi is None:
                            continue
                        H[ui, vi] += 1.0
                    # H_a entries: token at position pos + ul
                    if pos + ul < T:
                        a_id = seq_list[pos + ul]
                        max_v2 = min(L, T - pos - ul - 1)
                        Ha_slice = H_a[a_id]
                        base = pos + ul + 1
                        for vl in range(0, max_v2 + 1):
                            v = tuple(seq_list[base:base + vl])
                            vi = self.S_idx.get(v)
                            if vi is None:
                                continue
                            Ha_slice[ui, vi] += 1.0
        self._log(f"Hankel built in {time.time() - t1:.1f}s, "
                  f"H nnz ~ {int(np.count_nonzero(H))}/{nP * nP}")

        # SVD of H
        t1 = time.time()
        U, s, Vt = np.linalg.svd(H, full_matrices=False)
        self._log(f"SVD done in {time.time() - t1:.1f}s; "
                  f"sigma[:10] = {np.round(s[:10], 2)}")

        # Determine rank
        rmax = int(np.sum(s > self.sv_rel_threshold * s[0]))
        r = min(rmax, self.rank) if self.rank else rmax
        self._rank_used = r
        self._log(f"using rank = {r} (max usable = {rmax})")

        Ud = U[:, :r]               # (nP, r)
        sd = s[:r]                  # (r,)
        Vd = Vt[:r, :].T            # (nP, r)
        inv_sd = 1.0 / sd

        # Operators  A_a = Uᵀ H_a V diag(1/σ)     — shape (r, r)
        self.A = np.empty((nA, r, r), dtype=np.float64)
        for a in range(nA):
            self.A[a] = (Ud.T @ H_a[a] @ Vd) * inv_sd[None, :]

        # α_0 = U[ε, :]     (row),  α_∞ = Σ V[ε, :]  (column-like)
        eps_i = self.P_idx[()]
        self.alpha0 = Ud[eps_i].copy()
        self.alpha_inf = sd * Vd[eps_i]

        self._log(f"fit complete in {time.time() - t0:.1f}s. "
                  f"operators shape = {self.A.shape}")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _maybe_renorm(self, state):
        if not self.renormalize:
            return state
        n = np.linalg.norm(state)
        if n > 0 and (n > 1e50 or n < 1e-50):
            return state / n
        return state

    def forward_pass(self, sequence, return_history=True):
        """
        Run forward pass. Unknown tokens are skipped (state unchanged).

        state_history[t] = state after processing tokens 0..t-1.
        state_history[0] = α_0, state_history[T] = final state.
        """
        hist = [self.alpha0.copy()]
        state = self.alpha0.copy()
        for tok in sequence:
            idx = self.tok2id.get(tok)
            if idx is not None:
                state = state @ self.A[idx]
                state = self._maybe_renorm(state)
            hist.append(state.copy())
        if return_history:
            return state, hist
        return state

    def predict_next_scores(self, state, horizon=1):
        """
        Return raw scores for each token at forecasting horizon h.

        score_h[a] = state · A_total^(h-1) · A_a · alpha_inf
        where A_total = sum_a A_a is the unconditional one-step operator
        (marginalizing over intermediate symbols).
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        s = state
        if horizon > 1:
            A_total = self.A.sum(axis=0)
            # Apply (A_total)^(h-1) from the left
            for _ in range(horizon - 1):
                s = s @ A_total
        return np.einsum('i,aij,j->a', s, self.A, self.alpha_inf)

    def predict_next_probs(self, state, horizon=1, mode='clip'):
        """Project OOM scores to a valid distribution over the alphabet.

        mode:
          'clip' — replace negatives with 0, renormalise (standard).
          'abs'  — take |score|, renormalise (preserves magnitude ranking).
          'softmax' — softmax(score / |score|_max), stable temperature=1 on
                      normalised scores.
          'simplex' — L2 projection onto the probability simplex (Euclidean).
        """
        s = self.predict_next_scores(state, horizon=horizon)
        nA = len(self.alphabet)
        if mode == 'clip':
            s = np.maximum(s, 0.0)
            total = s.sum()
            if total <= 0:
                return np.full(nA, 1.0 / nA)
            return s / total
        if mode == 'abs':
            s = np.abs(s)
            total = s.sum()
            if total <= 0:
                return np.full(nA, 1.0 / nA)
            return s / total
        if mode == 'softmax':
            m = np.max(np.abs(s))
            if m > 0:
                s = s / m
            s = s - s.max()
            e = np.exp(s)
            return e / e.sum()
        if mode == 'simplex':
            # Duchi et al. (2008) Euclidean projection onto the simplex.
            u = np.sort(s)[::-1]
            cssv = np.cumsum(u) - 1
            rho_candidates = u - cssv / (np.arange(nA) + 1)
            rho = np.max(np.where(rho_candidates > 0)[0]) if (rho_candidates > 0).any() else 0
            theta = cssv[rho] / (rho + 1)
            proj = np.maximum(s - theta, 0)
            return proj
        raise ValueError(f"unknown mode: {mode}")


if __name__ == "__main__":
    # Tiny sanity check: train on a deterministic sequence and check it predicts
    # the next symbol correctly.
    seqs = [tuple("abcabcabcabc")] * 5
    oom = SpectralOOM(max_basis_length=2, rank=None, verbose=True)
    oom.fit(seqs)
    state, _ = oom.forward_pass("ab", return_history=True)
    probs = oom.predict_next_probs(state)
    print("After 'ab', P over alphabet:", dict(zip(oom.alphabet, probs.round(3))))
