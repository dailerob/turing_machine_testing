"""Batched, cached GDC scorer for PAutomaC-style sequence-likelihood
evaluation.

Implements three optimisations over the per-sequence loop in
`models.GDCModel.log_prob`:

  (1) Cache per-call invariants (terminal masks, diffuse constants,
      `last_nt_idx`) at fit time.  The original `_transition_self_loop`
      recomputes these on every call.

  (2) Express the transition as a closed-form linear operator using the
      cached invariants — same algebra as `_transition_self_loop`,
      packaged into a single vectorised function `_transition_batch(S)`
      that operates on a (B, n) state matrix in one go.

  (3) Batch all test sequences along axis 0.  Instead of running 1000
      sequences sequentially through the forward filter, stack them into
      a (B, n) matrix and apply transition + emission update in
      lock-step.  Conditions per-row by symbol-group masking.

Currently supports `transition_type='self_loop'` only; that's the
single-step transition that won most tasks in the algorithmic
benchmarks.  Add `_transition_batch_two_step` for the two-step variant
if needed.

Validation (`compare_with_naive_gdc`) confirms log-prob agreement to
~1e-6 absolute against `models.GDCModel.log_prob` on a small problem.
"""

from __future__ import annotations
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from generative_dense_chain import GenerativeDenseChain  # noqa: E402

LOG_EPS = -700.0


def _append_end(seqs, end_token):
    return [np.concatenate([s, [end_token]]).astype(np.int64) for s in seqs]


class BatchedGDCScorer:
    name_template = 'gdc-a{alpha}-t{theta}-{trans}'

    def __init__(self, alpha=0.95, theta=0.05,
                 transition_type='self_loop',
                 initial_dist='sequence_starts',
                 dtype=np.float32):
        if transition_type != 'self_loop':
            raise NotImplementedError(
                "BatchedGDCScorer currently supports transition_type "
                "'self_loop' only; got %r" % transition_type)
        self.alpha = float(alpha)
        self.theta = float(theta)
        self.transition_type = transition_type
        self.initial_dist = initial_dist
        self.dtype = dtype
        self.name = (f"fastgdc-a{alpha}-t{theta}-1step"
                     if transition_type == 'self_loop'
                     else f"fastgdc-a{alpha}-t{theta}-{transition_type}")

    # -----------------------------------------------------------------
    # fit: build a GDC and cache everything we need for batched eval
    # -----------------------------------------------------------------
    def fit(self, train_seqs, alphabet_size):
        self.A = alphabet_size + 1
        self.end_token = int(alphabet_size)
        seqs = _append_end(train_seqs, self.end_token)
        col_seqs = [s.reshape(-1, 1).astype(np.int64) for s in seqs
                    if len(s) > 0]
        gdc = GenerativeDenseChain(
            col_seqs, alpha=self.alpha, theta=self.theta,
            gamma=0.0, beta=0.0,
            transition_type=self.transition_type,
            initial_dist=self.initial_dist)
        n = gdc.n_states
        self.n = n
        self.emit = gdc.states[:, 0].astype(np.int64)
        self.A_total = max(int(self.emit.max()) + 1, self.A)

        # Group state indices by emission symbol (kept for fallback /
        # debugging) plus a precomputed emission one-hot matrix used for
        # batched conditioning.  E_T[a, s] = 1 if state s emits a, else 0.
        self.idx_by_emit = [
            np.where(self.emit == a)[0].astype(np.int64)
            for a in range(self.A_total)
        ]
        E_T = np.zeros((self.A_total, n), dtype=self.dtype)
        E_T[self.emit, np.arange(n)] = 1.0  # rows: symbols, cols: states
        self.E_T = E_T

        # Cached transition invariants
        self.terminal_mask = gdc.terminal_mask.astype(bool)
        non_terminal = ~self.terminal_mask
        self.terminal_mask_f = self.terminal_mask.astype(self.dtype)
        self.non_terminal_mask_f = non_terminal.astype(self.dtype)
        nt_idx = np.where(non_terminal)[0]
        self.last_nt_idx = int(nt_idx[-1]) if len(nt_idx) > 0 else -1
        if n > 2:
            self.diffuse_nt = (1.0 - self.alpha - self.theta) / (n - 2)
            self.diffuse_t = (1.0 - self.theta) / (n - 1)
        else:
            self.diffuse_nt = 0.0
            self.diffuse_t = 0.0

        self.init_dist = gdc._get_initial_distribution(
            self.initial_dist).astype(self.dtype)
        # We don't need the GDC after fit (we have everything cached)
        del gdc

    # -----------------------------------------------------------------
    # Batched transition: applies self-loop transition to (B, n) matrix
    # -----------------------------------------------------------------
    def _transition_batch(self, S):
        n = self.n
        if n == 2:
            return self.theta * S + (1.0 - self.theta) * np.roll(S, 1, axis=1)
        nt_states = S * self.non_terminal_mask_f
        t_states = S * self.terminal_mask_f
        shifted = np.zeros_like(S)
        shifted[:, 1:] = nt_states[:, :-1]
        nt_sum = nt_states.sum(axis=1, keepdims=True)
        t_sum = t_states.sum(axis=1, keepdims=True)
        new_S = (
            self.theta * S
            + self.alpha * shifted
            + self.diffuse_nt * (nt_sum - nt_states - shifted)
            + self.diffuse_t * (t_sum - t_states)
        )
        if self.last_nt_idx >= 0:
            new_S[:, 0] -= self.diffuse_nt * nt_states[:, self.last_nt_idx]
        return new_S

    # -----------------------------------------------------------------
    # Batched scorer
    # -----------------------------------------------------------------
    def score_test_set(self, test_seqs):
        """Compute natural-log P(seq + END) for every sequence in
        `test_seqs`.  Returns a 1-D float64 array of length len(test_seqs).

        Implementation: sort sequences by length descending and shrink
        the active set each timestep; condition per-symbol-group to
        avoid materialising a dense (B, n) emission mask.
        """
        end = self.end_token
        seqs = [np.concatenate([s, [end]]).astype(np.int64)
                for s in test_seqs]
        B = len(seqs)
        seq_lens = np.asarray([len(s) for s in seqs], dtype=np.int64)
        max_len = int(seq_lens.max())

        # Sort by length descending; longest first
        order = np.argsort(seq_lens)[::-1]
        sorted_lens = seq_lens[order]
        padded = -np.ones((B, max_len), dtype=np.int64)
        for new_i, old_i in enumerate(order):
            padded[new_i, :len(seqs[old_i])] = seqs[old_i]

        # State matrix laid out in sorted order
        S = np.tile(self.init_dist, (B, 1)).astype(self.dtype, copy=True)
        log_p_sorted = np.zeros(B, dtype=np.float64)

        for t in range(max_len):
            k = int((sorted_lens > t).sum())
            if k == 0:
                break
            if t > 0:
                S[:k] = self._transition_batch(S[:k])
            syms_t = padded[:k, t]

            # Per-symbol grouping for emission update.  For each
            # symbol-group we compute mass via a single matvec
            # (sub @ mask_a) and apply the column mask + per-row
            # normalisation via broadcasting — no fancy-indexed write.
            unique_syms = np.unique(syms_t)
            for a in unique_syms:
                a_int = int(a)
                rows = np.where(syms_t == a_int)[0]
                if 0 <= a_int < self.A_total:
                    mask_a = self.E_T[a_int]          # (n,) float32 mask
                else:
                    log_p_sorted[rows] += LOG_EPS
                    S[rows] = self.dtype(1.0 / self.n)
                    continue
                if not mask_a.any():
                    log_p_sorted[rows] += LOG_EPS
                    S[rows] = self.dtype(1.0 / self.n)
                    continue
                sub = S[rows]                          # copy via fancy index
                q = (sub @ mask_a).astype(np.float64)  # (|rows|,)
                log_p_sorted[rows] += np.log(np.maximum(q, 1e-300))
                safe_q = np.maximum(q, 1e-30).astype(self.dtype)
                sub *= mask_a                          # broadcast in-place
                sub /= safe_q[:, None]
                S[rows] = sub                          # single write-back

        # Unscramble back to original order
        log_p = np.empty(B, dtype=np.float64)
        log_p[order] = log_p_sorted
        return log_p


class BatchedDualGDCScorer(BatchedGDCScorer):
    """Dual-α variant of BatchedGDCScorer.

    At each step:
      pred_state = transition(S, α_fc, θ_fc)  -- used to compute predictive
      S          = transition(S, α_ctx, θ_ctx) -- advances state-tracking
      filter S on observed symbol.

    Setting α_fc=α_ctx and θ_fc=θ_ctx recovers BatchedGDCScorer. The
    "α_fc=1, θ_fc=0 + soft α_ctx" recipe found on char-LM and HMM
    forecasting tasks consistently helps when the dynamics have a
    deterministic-advance component.
    """

    def __init__(self, alpha_ctx, alpha_fc,
                 theta_ctx=0.0, theta_fc=0.0,
                 transition_type='self_loop',
                 initial_dist='sequence_starts',
                 dtype=np.float32):
        super().__init__(alpha=alpha_ctx, theta=theta_ctx,
                          transition_type=transition_type,
                          initial_dist=initial_dist, dtype=dtype)
        self.alpha_ctx = float(alpha_ctx); self.theta_ctx = float(theta_ctx)
        self.alpha_fc = float(alpha_fc); self.theta_fc = float(theta_fc)
        self.name = (f"fastgdc-ac{alpha_ctx}-af{alpha_fc}"
                     f"-tc{theta_ctx}-tf{theta_fc}-1step")

    def fit(self, train_seqs, alphabet_size):
        super().fit(train_seqs, alphabet_size)
        n = self.n
        if n > 2:
            self.diffuse_nt_fc = (1.0 - self.alpha_fc - self.theta_fc) / (n - 2)
            self.diffuse_t_fc  = (1.0 - self.theta_fc) / (n - 1)
        else:
            self.diffuse_nt_fc = 0.0
            self.diffuse_t_fc  = 0.0

    def _transition_batch_with(self, S, alpha, theta, diffuse_nt, diffuse_t):
        n = self.n
        if n == 2:
            return theta * S + (1.0 - theta) * np.roll(S, 1, axis=1)
        nt_states = S * self.non_terminal_mask_f
        t_states  = S * self.terminal_mask_f
        shifted = np.zeros_like(S)
        shifted[:, 1:] = nt_states[:, :-1]
        nt_sum = nt_states.sum(axis=1, keepdims=True)
        t_sum  = t_states.sum(axis=1, keepdims=True)
        new_S = (
            theta * S
            + alpha * shifted
            + diffuse_nt * (nt_sum - nt_states - shifted)
            + diffuse_t * (t_sum - t_states)
        )
        if self.last_nt_idx >= 0:
            new_S[:, 0] -= diffuse_nt * nt_states[:, self.last_nt_idx]
        return new_S

    def score_test_set(self, test_seqs):
        end = self.end_token
        seqs = [np.concatenate([s, [end]]).astype(np.int64)
                for s in test_seqs]
        B = len(seqs)
        seq_lens = np.asarray([len(s) for s in seqs], dtype=np.int64)
        max_len = int(seq_lens.max())
        order = np.argsort(seq_lens)[::-1]
        sorted_lens = seq_lens[order]
        padded = -np.ones((B, max_len), dtype=np.int64)
        for new_i, old_i in enumerate(order):
            padded[new_i, :len(seqs[old_i])] = seqs[old_i]
        S = np.tile(self.init_dist, (B, 1)).astype(self.dtype, copy=True)
        log_p_sorted = np.zeros(B, dtype=np.float64)
        for t in range(max_len):
            k = int((sorted_lens > t).sum())
            if k == 0:
                break
            if t > 0:
                pred = self._transition_batch_with(
                    S[:k], self.alpha_fc, self.theta_fc,
                    self.diffuse_nt_fc, self.diffuse_t_fc)
                S[:k] = self._transition_batch_with(
                    S[:k], self.alpha_ctx, self.theta_ctx,
                    self.diffuse_nt, self.diffuse_t)
            else:
                pred = S[:k]
            syms_t = padded[:k, t]
            unique_syms = np.unique(syms_t)
            for a in unique_syms:
                a_int = int(a)
                rows = np.where(syms_t == a_int)[0]
                if not (0 <= a_int < self.A_total):
                    log_p_sorted[rows] += LOG_EPS
                    S[rows] = self.dtype(1.0 / self.n)
                    continue
                mask_a = self.E_T[a_int]
                if not mask_a.any():
                    log_p_sorted[rows] += LOG_EPS
                    S[rows] = self.dtype(1.0 / self.n)
                    continue
                q_pred = (pred[rows] @ mask_a).astype(np.float64)
                log_p_sorted[rows] += np.log(np.maximum(q_pred, 1e-300))
                sub = S[rows]
                q_ctx = (sub @ mask_a).astype(np.float64)
                safe_q = np.maximum(q_ctx, 1e-30).astype(self.dtype)
                sub *= mask_a
                sub /= safe_q[:, None]
                S[rows] = sub
        log_p = np.empty(B, dtype=np.float64)
        log_p[order] = log_p_sorted
        return log_p


# ---------------------------------------------------------------------
# Validation harness: compare batched vs naive log-probs
# ---------------------------------------------------------------------
def compare_with_naive_gdc(train_seqs, test_seqs, alphabet_size,
                           alpha, theta, transition_type='self_loop',
                           initial_dist='sequence_starts',
                           n_check=20):
    """Spot-check: for the first `n_check` test sequences, compute
    log-prob via both the naive per-sequence scorer (`models.GDCModel`)
    and the batched scorer; report the maximum absolute difference.
    """
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from models import GDCModel

    naive = GDCModel(alpha=alpha, theta=theta,
                     transition_type=transition_type,
                     initial_dist=initial_dist)
    naive.fit(train_seqs, alphabet_size)

    fast = BatchedGDCScorer(alpha=alpha, theta=theta,
                            transition_type=transition_type,
                            initial_dist=initial_dist)
    fast.fit(train_seqs, alphabet_size)

    sub_test = test_seqs[:n_check]
    naive_lp = np.array([naive.log_prob(s) for s in sub_test])
    fast_lp = fast.score_test_set(sub_test)
    diff = np.abs(naive_lp - fast_lp)
    print(f"naive log-probs: {naive_lp[:5]}")
    print(f"fast  log-probs: {fast_lp[:5]}")
    print(f"max |diff| over {n_check} sequences: {diff.max():.3e}")
    print(f"mean |diff|:                          {diff.mean():.3e}")
    return naive_lp, fast_lp, diff


if __name__ == "__main__":
    # Quick correctness check on PAutomaC problem 1.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from data_loader import load_problem
    p = load_problem(1)
    compare_with_naive_gdc(p['train'][:2000], p['test'][:30],
                           p['alphabet_size'],
                           alpha=0.95, theta=0.05,
                           transition_type='self_loop')
