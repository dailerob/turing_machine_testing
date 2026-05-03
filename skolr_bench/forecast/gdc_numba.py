"""Numba-JIT GDC-TS forecast for self_loop + absorb + 1-D states.

Single tight JIT'd kernel does the full forecast_gdc_style:
  - Forward pass over L observations (filter)
  - Forecast loop over T steps (with terminal re-zero each step)
  - Returns (T,) forecast (non-terminal-renormalized expected value)

All inner ops are scalar operations inside @njit, fused into the loop.
No NumPy temporary array allocation. Should beat the Python+NumPy version
by ~10-50× when N is large, since:
  - No Python interpreter overhead per inner op
  - No NumPy allocation per op
  - Single sequential scan over the (N,) state arrays per loop iteration
    (cache-friendly)
"""
from __future__ import annotations
import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def _forecast_one_kernel(states_1d, terminal_idx, beta, alpha, theta,
                          obs, n_steps):
    """JIT'd full forecast for a single prime.

    Parameters
    ----------
    states_1d : (N,) float64 — the state space values (1-D).
    terminal_idx : int — index of the terminal state (typically N-1).
    beta : float — emission variance.
    alpha, theta : float — self-loop transition coefficients.
    obs : (L,) float64 — prime observations.
    n_steps : int — forecast horizon T.

    Returns
    -------
    forecasts : (n_steps,) float64 — expected value per step,
        renormalized over non-terminal mass (matches v3/v5 protocol).
    """
    N = states_1d.shape[0]
    L = obs.shape[0]
    log_norm_const = -0.5 * np.log(2.0 * np.pi * beta)
    inv_2beta = 1.0 / (2.0 * beta)
    tiny = 1e-300

    if N < 3:
        # degenerate; just return last obs repeated
        out = np.empty(n_steps, dtype=np.float64)
        for t in range(n_steps):
            out[t] = obs[L - 1] if L > 0 else 0.0
        return out

    beta_nt = (1.0 - alpha - theta) / (N - 2)
    last_nt_idx = terminal_idx - 1  # second-to-last (since terminal is N-1)

    # log-space forward pass
    log_dist = np.empty(N, dtype=np.float64)
    init_lp = -np.log(N)
    for i in range(N):
        log_dist[i] = init_lp
    cur = np.empty(N, dtype=np.float64)
    nxt = np.empty(N, dtype=np.float64)

    for t in range(L):
        if t > 0:
            # log -> linear (with safe normalization)
            mx = log_dist[0]
            for i in range(1, N):
                if log_dist[i] > mx:
                    mx = log_dist[i]
            s = 0.0
            for i in range(N):
                cur[i] = np.exp(log_dist[i] - mx)
                s += cur[i]
            inv_s = 1.0 / s
            for i in range(N):
                cur[i] *= inv_s

            # transition: self_loop + absorb
            # non_terminal_sum and last_nt_val
            non_terminal_sum = 0.0
            for i in range(N):
                if i != terminal_idx:
                    non_terminal_sum += cur[i]
            last_nt_val = cur[last_nt_idx]
            # Build nxt
            # state 0: self_loop + 0 (no shift inflow) + diffusion
            #   nt_diffusion[0] = beta_nt*non_terminal_sum - beta_nt*nt[0] - beta_nt*shifted[0]
            #                    - beta_nt*nt[last_nt_idx]
            # shifted[0] = 0, nt[0] = cur[0] if 0 != terminal_idx else 0
            # General formula:
            #   non_terminal[i] = cur[i] if i != terminal_idx else 0
            #   shifted[i] = non_terminal[i-1] if i >= 1 else 0
            #   self_loop[i] = theta * cur[i]
            #   sequential[i] = alpha * shifted[i]
            #   nt_diffusion[i] = beta_nt * (non_terminal_sum - non_terminal[i] - shifted[i])
            #   nt_diffusion[0] -= beta_nt * non_terminal[last_nt_idx]
            for i in range(N):
                nt_i = cur[i] if i != terminal_idx else 0.0
                shifted_i = (cur[i - 1] if (i - 1) != terminal_idx else 0.0) if i >= 1 else 0.0
                nxt[i] = theta * cur[i] + alpha * shifted_i + \
                          beta_nt * (non_terminal_sum - nt_i - shifted_i)
            nxt[0] -= beta_nt * last_nt_val

            # log of nxt + tiny
            for i in range(N):
                v = nxt[i]
                if v < tiny:
                    v = tiny
                log_dist[i] = np.log(v)

        # add emission log-likelihood
        oi = obs[t]
        for i in range(N):
            d = states_1d[i] - oi
            log_dist[i] += -d * d * inv_2beta + log_norm_const

        # log-normalize
        mx = log_dist[0]
        for i in range(1, N):
            if log_dist[i] > mx:
                mx = log_dist[i]
        s = 0.0
        for i in range(N):
            s += np.exp(log_dist[i] - mx)
        lse = mx + np.log(s)
        for i in range(N):
            log_dist[i] -= lse

    # End of forward pass; convert to linear, zero terminal, normalize
    for i in range(N):
        cur[i] = np.exp(log_dist[i])
    cur[terminal_idx] = 0.0
    s = 0.0
    for i in range(N):
        s += cur[i]
    if s <= 0:
        for i in range(N):
            cur[i] = 1.0 / N
    else:
        inv_s = 1.0 / s
        for i in range(N):
            cur[i] *= inv_s

    # Forecast loop with terminal re-zero each step
    forecasts = np.empty(n_steps, dtype=np.float64)
    for step in range(n_steps):
        non_terminal_sum = 0.0
        for i in range(N):
            if i != terminal_idx:
                non_terminal_sum += cur[i]
        last_nt_val = cur[last_nt_idx]
        for i in range(N):
            nt_i = cur[i] if i != terminal_idx else 0.0
            shifted_i = (cur[i - 1] if (i - 1) != terminal_idx else 0.0) if i >= 1 else 0.0
            nxt[i] = theta * cur[i] + alpha * shifted_i + \
                      beta_nt * (non_terminal_sum - nt_i - shifted_i)
        nxt[0] -= beta_nt * last_nt_val

        # Compute non-terminal-renormalized expected value at this step
        nt_sum_post = 0.0
        for i in range(N):
            if i != terminal_idx:
                nt_sum_post += nxt[i]
        if nt_sum_post > 1e-12:
            ev = 0.0
            for i in range(N):
                if i != terminal_idx:
                    ev += nxt[i] * states_1d[i]
            forecasts[step] = ev / nt_sum_post
        else:
            forecasts[step] = states_1d[last_nt_idx]

        # Re-zero terminal, normalize for next iter
        nxt[terminal_idx] = 0.0
        s2 = 0.0
        for i in range(N):
            s2 += nxt[i]
        if s2 > 0:
            inv_s2 = 1.0 / s2
            for i in range(N):
                cur[i] = nxt[i] * inv_s2
        else:
            for i in range(N):
                cur[i] = nxt[i]

    return forecasts


@njit(cache=True, fastmath=True, parallel=True)
def forecast_many(states_1d, terminal_idx, beta, alpha, theta,
                   obs_batch, n_steps):
    """Forecast for B primes in parallel (numba prange across primes).

    obs_batch : (B, L) float64.
    Returns : (B, n_steps) float64.
    """
    B = obs_batch.shape[0]
    N = states_1d.shape[0]
    out = np.empty((B, n_steps), dtype=np.float64)
    for b in prange(B):
        f = _forecast_one_kernel(states_1d, terminal_idx, beta, alpha, theta,
                                  obs_batch[b], n_steps)
        for t in range(n_steps):
            out[b, t] = f[t]
    return out


def smoke_test():
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

    L_match = 192; T = 192; B = 256
    primes = np.stack([series[i:i+L_match, 0] for i in range(B)], axis=0)
    states_1d = gdc.states[:, 0]
    terminal_idx = int(np.where(gdc.terminal_mask)[0][-1])

    # Warm-up JIT
    _ = forecast_many(states_1d, terminal_idx, gdc.beta, gdc.alpha, gdc.theta,
                       primes[:1], T)

    # Reference (per-prime, original numpy path)
    nt_mask_f = (~gdc.terminal_mask).astype(float)
    t0 = time.time()
    ref = np.empty((B, T), dtype=np.float64)
    for i in range(B):
        _, sd = gdc.forecast_gdc_style(primes[i].reshape(-1, 1), n_steps=T)
        sd_nt = sd * nt_mask_f[None, :]
        sd_sum = sd_nt.sum(axis=1, keepdims=True)
        safe = np.where(sd_sum > 1e-12, sd_sum, 1.0)
        ref[i] = ((sd_nt / safe) @ gdc.states)[:, 0]
    t_ref = time.time() - t0

    # Numba batched
    t0 = time.time()
    out = forecast_many(states_1d, terminal_idx, gdc.beta, gdc.alpha, gdc.theta,
                         primes, T)
    t_numba = time.time() - t0

    diff = np.abs(out - ref).max()
    rdiff = np.abs(out - ref).max() / max(np.abs(ref).max(), 1e-9)
    print(f"Smoke test: B={B}, L={L_match}, T={T}, N={N}")
    print(f"  per-prime numpy: {t_ref:.3f}s ({t_ref/B*1000:.2f}ms each)")
    print(f"  numba parallel:  {t_numba:.3f}s ({t_numba/B*1000:.2f}ms each)")
    print(f"  speedup:         {t_ref/t_numba:.1f}x")
    print(f"  max abs diff:    {diff:.2e}  (rel {rdiff:.2e})")


if __name__ == "__main__":
    smoke_test()
