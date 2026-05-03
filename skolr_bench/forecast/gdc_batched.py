"""Batched GDC-TS forecast — process B primes simultaneously.

Wraps a built GenerativeDenseChainTimeSeries and exposes:
  forecast_gdc_style_batched(primes, n_steps) -> (B, n_steps, k) forecasts

This shares all Python interpreter overhead across the B primes:
  - Forward-pass: B parallel filtering runs sharing the same emission
    and transition kernels (operations become (B, N) instead of (N,))
  - Forecast loop: one transition op per step over (B, N) state distributions

Currently supports: self_loop transition with absorb terminal behavior,
1-D states, uniform initial distribution. (The configs we use everywhere.)
"""
from __future__ import annotations
import numpy as np


def _logsumexp(arr, axis=None, keepdims=False):
    m = arr.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(arr - m).sum(axis=axis, keepdims=True))).squeeze(axis=axis if not keepdims else None) if not keepdims else (m + np.log(np.exp(arr - m).sum(axis=axis, keepdims=True)))


def _logsumexp_axis1(arr_2d):
    """logsumexp over axis=1 of a 2D array, returns (B,) result."""
    m = arr_2d.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(arr_2d - m).sum(axis=1, keepdims=True))).squeeze(axis=1)


class BatchedGDC:
    """Wraps a GenerativeDenseChainTimeSeries instance for batched ops.

    Caches: states, terminal_mask, beta, log_norm_const, last_nt_idx, alpha, theta.
    Assumes: transition_type='self_loop', terminal_behavior='absorb',
             initial_dist='uniform', 1-D states (k=1).
    """

    def __init__(self, gdc):
        assert gdc.transition_type == 'self_loop', \
            f"BatchedGDC only supports self_loop, got {gdc.transition_type}"
        assert gdc.terminal_behavior == 'absorb', \
            f"BatchedGDC only supports absorb, got {gdc.terminal_behavior}"
        self.states = np.asarray(gdc.states, dtype=np.float64)  # (N, k)
        self.k = int(self.states.shape[1])
        self.n = int(self.states.shape[0])
        self.terminal_mask = np.asarray(gdc.terminal_mask, dtype=bool)
        self.nt_mask = ~self.terminal_mask
        nt_idx = np.where(self.nt_mask)[0]
        self.last_nt_idx = int(nt_idx[-1]) if len(nt_idx) else self.n - 1
        self.beta = float(gdc.beta)
        self.alpha = float(gdc.alpha)
        self.theta = float(gdc.theta)
        self.log_norm_const = -0.5 * self.k * np.log(2.0 * np.pi * self.beta)
        # Precompute beta_nt
        if self.n >= 3:
            self.beta_nt = (1 - self.alpha - self.theta) / (self.n - 2)
        else:
            self.beta_nt = 0.0
        # Precompute states for emission distance (squeeze to (N,) if k=1)
        if self.k == 1:
            self.states_1d = self.states[:, 0]
        else:
            self.states_1d = None

    # --- Batched emission ---
    def _emis_batch(self, obs_batch):
        """obs_batch: (B, k) → log emis (B, N)."""
        if self.k == 1:
            # Cheaper: (B, 1) - (N,) → (B, N) sq dist
            obs = obs_batch[:, 0]  # (B,)
            sq = (self.states_1d[None, :] - obs[:, None]) ** 2  # (B, N)
        else:
            # (B, 1, k) - (1, N, k) → (B, N, k)
            d = obs_batch[:, None, :] - self.states[None, :, :]
            sq = (d ** 2).sum(axis=2)
        return -0.5 * sq / self.beta + self.log_norm_const

    # --- Batched transition (self_loop, absorb) ---
    def _trans_batch(self, dist_batch):
        """dist_batch: (B, N) → (B, N) after one transition."""
        n = self.n; theta = self.theta; alpha = self.alpha
        beta_nt = self.beta_nt
        # Mask out terminal contributions
        nt_mask_f = self.nt_mask.astype(np.float64)
        non_terminal = dist_batch * nt_mask_f[None, :]      # (B, N)
        non_terminal_sum = non_terminal.sum(axis=1, keepdims=True)  # (B, 1)
        # No-wrap shift right by 1 over non-terminal
        shifted = np.zeros_like(dist_batch)
        shifted[:, 1:n] = non_terminal[:, :n - 1]
        # self loop + sequential + nt_diffusion (absorb mode skips t_diffusion)
        out = theta * dist_batch + alpha * shifted
        # nt_diffusion = beta_nt * (non_terminal_sum - non_terminal - shifted)
        out += beta_nt * (non_terminal_sum - non_terminal - shifted)
        # subtract wrap-around component: nt_diffusion[0] -= beta_nt * non_terminal[last_nt_idx]
        out[:, 0] -= beta_nt * non_terminal[:, self.last_nt_idx]
        return out

    # --- Batched forward pass over L observations ---
    def forward_pass_batch(self, obs_seqs):
        """obs_seqs: (B, L, k) → end log_dist (B, N)."""
        B, L, k = obs_seqs.shape
        assert k == self.k
        n = self.n
        tiny = np.finfo(np.float64).tiny
        # Initial uniform distribution
        log_dist = np.full((B, n), -np.log(n), dtype=np.float64)
        for t in range(L):
            if t > 0:
                # Convert log -> linear (with safe normalization), apply transition, back to log
                m = log_dist.max(axis=1, keepdims=True)
                lin = np.exp(log_dist - m)
                lin /= lin.sum(axis=1, keepdims=True)
                lin = self._trans_batch(lin)
                log_dist = np.log(lin + tiny)
            log_dist += self._emis_batch(obs_seqs[:, t, :])
            # log_normalize per row
            lse = _logsumexp_axis1(log_dist)
            log_dist -= lse[:, None]
        return log_dist  # (B, N)

    def forecast_gdc_style_batch(self, obs_seqs, n_steps):
        """obs_seqs: (B, L, k); returns forecasts (B, n_steps, k)."""
        log_end = self.forward_pass_batch(obs_seqs)
        end = np.exp(log_end)
        end[:, -1] = 0.0
        s = end.sum(axis=1, keepdims=True)
        # Fall back to uniform for any rows that sum to 0
        bad = (s.squeeze(axis=1) <= 0)
        if bad.any():
            end[bad, :] = 1.0 / self.n
            s = end.sum(axis=1, keepdims=True)
        end = end / s
        cur = end.copy()
        # Track non-terminal mass distribution per step (for renormalized expected value)
        forecasts = np.empty((obs_seqs.shape[0], n_steps, self.k), dtype=np.float64)
        nt_mask_f = self.nt_mask.astype(np.float64)
        for t in range(n_steps):
            cur = self._trans_batch(cur)
            # Extract non-terminal-renormalized expected value (matches eval_v3/v5 protocol)
            sd_nt = cur * nt_mask_f[None, :]
            sd_sum = sd_nt.sum(axis=1, keepdims=True)
            safe = np.where(sd_sum > 1e-12, sd_sum, 1.0)
            forecasts[:, t, :] = (sd_nt / safe) @ self.states
            # Re-zero terminal for next step
            cur[:, -1] = 0.0
            s2 = cur.sum(axis=1, keepdims=True)
            mask2 = (s2.squeeze(axis=1) > 0)
            if mask2.any():
                cur[mask2] /= s2[mask2]
        return forecasts


def smoke_test():
    """Compare batched vs single-prime forecasts; verify they match."""
    import sys, os, time
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(HERE))
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    from generative_dense_chain_timeseries import GenerativeDenseChainTimeSeries

    rng = np.random.default_rng(0)
    N = 12000
    series = np.cumsum(rng.standard_normal(N)).reshape(-1, 1)
    gdc = GenerativeDenseChainTimeSeries(
        series, beta=4.0, alpha=1.0, theta=0.0,
        transition_type='self_loop',
        terminal_behavior='absorb',
        initial_dist='uniform')
    bgdc = BatchedGDC(gdc)
    L_match = 192; T = 192; B = 256
    primes = np.stack([series[i:i+L_match, :] for i in range(B)], axis=0)
    # Reference (per-prime)
    t0 = time.time()
    ref = []
    nt_mask_f = (~gdc.terminal_mask).astype(float)
    for i in range(B):
        _, sd = gdc.forecast_gdc_style(primes[i], n_steps=T)
        sd_nt = sd * nt_mask_f[None, :]
        sd_sum = sd_nt.sum(axis=1, keepdims=True)
        safe = np.where(sd_sum > 1e-12, sd_sum, 1.0)
        f = (sd_nt / safe) @ gdc.states
        ref.append(f)
    ref = np.stack(ref, axis=0)  # (B, T, k)
    t_ref = time.time() - t0
    # Batched
    t0 = time.time()
    out = bgdc.forecast_gdc_style_batch(primes, T)
    t_batch = time.time() - t0
    diff = np.abs(out - ref).max()
    print(f"Smoke test: B={B}, L={L_match}, T={T}, N={gdc.n_states}")
    print(f"  per-prime: {t_ref:.3f}s ({t_ref/B*1000:.1f}ms each)")
    print(f"  batched:   {t_batch:.3f}s ({t_batch/B*1000:.1f}ms each)")
    print(f"  speedup:   {t_ref/t_batch:.1f}x")
    print(f"  max abs diff: {diff:.2e}")


if __name__ == "__main__":
    smoke_test()
