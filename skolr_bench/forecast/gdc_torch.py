"""PyTorch / GPU version of the GDC-TS batched forecast kernel.

Mirrors `_forecast_one_kernel` and `forecast_many` from gdc_numba.py:
  - self_loop transition with absorb terminal behavior
  - 1-D state space (univariate)
  - uniform initial distribution
  - non-terminal-renormalized expected-value forecast extraction

API:
  forecast_many_torch(states_1d, beta, alpha, theta, primes, T,
                      device='cuda', dtype=torch.float64) -> (B, T) tensor

Internally: vectorized over the B prime dimension throughout. Forward
pass is L iterations on (B, N) state-distribution tensors, forecast
is T more iterations.

Goal: should produce numerics matching the numba kernel to ~1e-6
in float32, ~1e-12 in float64.
"""
from __future__ import annotations
import math
import torch


def _trans_self_loop_absorb(dist, alpha, theta, beta_nt, last_nt_idx, terminal_idx):
    """Batched self_loop+absorb transition.
    dist: (B, N) tensor.  Returns (B, N) tensor."""
    N = dist.shape[1]
    # non_terminal mask: 1 except at terminal_idx
    # Mask out terminal contributions
    nt = dist.clone()
    nt[:, terminal_idx] = 0.0
    non_terminal_sum = nt.sum(dim=1, keepdim=True)  # (B, 1)
    last_nt_val = nt[:, last_nt_idx]                # (B,)
    # No-wrap shift right by 1: shifted[:, i] = nt[:, i-1] for i >= 1, else 0
    shifted = torch.zeros_like(dist)
    shifted[:, 1:] = nt[:, :N - 1]
    # self_loop + sequential + nt_diffusion (absorb mode skips t_diffusion)
    out = theta * dist + alpha * shifted
    out = out + beta_nt * (non_terminal_sum - nt - shifted)
    out[:, 0] = out[:, 0] - beta_nt * last_nt_val
    return out


def forecast_many_torch(states_1d, beta, alpha, theta, primes, T,
                        device='cuda', dtype=torch.float64):
    """Batched forecast for B primes.

    Parameters
    ----------
    states_1d : (N,) array-like — state space values (1-D).
    beta : float — emission variance.
    alpha, theta : float — self-loop coefficients.
    primes : (B, L) array-like — prime observations per batch.
    T : int — forecast horizon.

    Returns
    -------
    forecasts : (B, T) torch tensor on `device`.
    """
    states = torch.as_tensor(states_1d, dtype=dtype, device=device)  # (N,)
    primes = torch.as_tensor(primes, dtype=dtype, device=device)     # (B, L)
    N = states.shape[0]
    B, L = primes.shape
    if N < 3:
        return primes[:, -1:].expand(B, T).clone()

    terminal_idx = N - 1
    last_nt_idx = N - 2
    beta_nt = (1.0 - alpha - theta) / (N - 2)
    log_norm_const = -0.5 * math.log(2.0 * math.pi * beta)
    inv_2beta = 1.0 / (2.0 * beta)
    tiny = torch.finfo(dtype).tiny

    # Initial uniform log distribution
    log_dist = torch.full((B, N), -math.log(N), dtype=dtype, device=device)

    # --- Forward pass over L observations ---
    for t in range(L):
        if t > 0:
            # log -> linear with safe normalization
            mx = log_dist.max(dim=1, keepdim=True).values
            lin = torch.exp(log_dist - mx)
            lin = lin / lin.sum(dim=1, keepdim=True)
            lin = _trans_self_loop_absorb(lin, alpha, theta, beta_nt,
                                           last_nt_idx, terminal_idx)
            log_dist = torch.log(lin + tiny)
        # Emission: -0.5 * (state - obs)^2 / beta + log_norm_const
        obs = primes[:, t:t+1]  # (B, 1)
        sq = (states[None, :] - obs) ** 2
        log_dist = log_dist + (-sq * inv_2beta + log_norm_const)
        # log-normalize per row (logsumexp)
        m = log_dist.max(dim=1, keepdim=True).values
        lse = m + torch.log(torch.exp(log_dist - m).sum(dim=1, keepdim=True))
        log_dist = log_dist - lse

    # End of forward pass; convert to linear, zero terminal, normalize
    cur = torch.exp(log_dist)
    cur[:, terminal_idx] = 0.0
    s = cur.sum(dim=1, keepdim=True)
    bad = (s.squeeze(1) <= 0)
    if bool(bad.any()):
        cur[bad] = 1.0 / N
        s = cur.sum(dim=1, keepdim=True)
    cur = cur / s

    # --- Forecast loop ---
    forecasts = torch.empty((B, T), dtype=dtype, device=device)
    for step in range(T):
        nxt = _trans_self_loop_absorb(cur, alpha, theta, beta_nt,
                                       last_nt_idx, terminal_idx)
        # Non-terminal-renormalized expected value at this step
        nt = nxt.clone()
        nt[:, terminal_idx] = 0.0
        nt_sum = nt.sum(dim=1, keepdim=True)  # (B, 1)
        safe = torch.where(nt_sum > 1e-12, nt_sum, torch.ones_like(nt_sum))
        forecasts[:, step] = (nt / safe @ states)
        # Re-zero terminal, normalize for next iter
        nxt[:, terminal_idx] = 0.0
        s2 = nxt.sum(dim=1, keepdim=True)
        mask2 = (s2.squeeze(1) > 0)
        if bool(mask2.any()):
            nxt = torch.where(s2 > 0, nxt / s2, nxt)
        cur = nxt

    return forecasts


def forecast_many_torch_dual(states_1d, beta, alpha, theta,
                                alpha_fc, theta_fc, primes, T,
                                device='cuda', dtype=torch.float64):
    """Dual-α variant of `forecast_many_torch`.

    The prefix forward pass uses `(alpha, theta)`; the forecast roll-out
    uses `(alpha_fc, theta_fc)`. Setting alpha_fc=alpha and theta_fc=theta
    recovers the single-α scorer exactly.
    """
    states = torch.as_tensor(states_1d, dtype=dtype, device=device)
    primes = torch.as_tensor(primes, dtype=dtype, device=device)
    N = states.shape[0]
    B, L = primes.shape
    if N < 3:
        return primes[:, -1:].expand(B, T).clone()

    terminal_idx = N - 1
    last_nt_idx = N - 2
    beta_nt_ctx = (1.0 - alpha    - theta   ) / (N - 2)
    beta_nt_fc  = (1.0 - alpha_fc - theta_fc) / (N - 2)
    log_norm_const = -0.5 * math.log(2.0 * math.pi * beta)
    inv_2beta = 1.0 / (2.0 * beta)
    tiny = torch.finfo(dtype).tiny

    log_dist = torch.full((B, N), -math.log(N), dtype=dtype, device=device)

    # Forward pass uses (alpha, theta)
    for t in range(L):
        if t > 0:
            mx = log_dist.max(dim=1, keepdim=True).values
            lin = torch.exp(log_dist - mx)
            lin = lin / lin.sum(dim=1, keepdim=True)
            lin = _trans_self_loop_absorb(lin, alpha, theta, beta_nt_ctx,
                                           last_nt_idx, terminal_idx)
            log_dist = torch.log(lin + tiny)
        obs = primes[:, t:t+1]
        sq = (states[None, :] - obs) ** 2
        log_dist = log_dist + (-sq * inv_2beta + log_norm_const)
        m = log_dist.max(dim=1, keepdim=True).values
        lse = m + torch.log(torch.exp(log_dist - m).sum(dim=1, keepdim=True))
        log_dist = log_dist - lse

    cur = torch.exp(log_dist)
    cur[:, terminal_idx] = 0.0
    s = cur.sum(dim=1, keepdim=True)
    bad = (s.squeeze(1) <= 0)
    if bool(bad.any()):
        cur[bad] = 1.0 / N
        s = cur.sum(dim=1, keepdim=True)
    cur = cur / s

    # Forecast loop uses (alpha_fc, theta_fc)
    forecasts = torch.empty((B, T), dtype=dtype, device=device)
    for step in range(T):
        nxt = _trans_self_loop_absorb(cur, alpha_fc, theta_fc, beta_nt_fc,
                                       last_nt_idx, terminal_idx)
        nt = nxt.clone()
        nt[:, terminal_idx] = 0.0
        nt_sum = nt.sum(dim=1, keepdim=True)
        safe = torch.where(nt_sum > 1e-12, nt_sum, torch.ones_like(nt_sum))
        forecasts[:, step] = (nt / safe @ states)
        nxt[:, terminal_idx] = 0.0
        s2 = nxt.sum(dim=1, keepdim=True)
        mask2 = (s2.squeeze(1) > 0)
        if bool(mask2.any()):
            nxt = torch.where(s2 > 0, nxt / s2, nxt)
        cur = nxt

    return forecasts


def forecast_many_torch_dual_batched(
        states_1d, beta, alphas, thetas, alphas_fc, thetas_fc,
        primes, T, device='cuda', dtype=torch.float64):
    """Same kernel as forecast_many_torch_dual, but batches over multiple
    (α, θ, α_fc, θ_fc) configs that share the same state space and prime.

    Parameters
    ----------
    states_1d : (N,) state values.
    beta : float — emission variance (shared across batch).
    alphas, thetas, alphas_fc, thetas_fc : (B,) tensors or lists.
    primes : (B, L) — one prime per config. (Pass the same prime
        broadcast across B configs.)
    T : forecast horizon.

    Returns (B, T) forecasts.
    """
    states = torch.as_tensor(states_1d, dtype=dtype, device=device)
    primes_t = torch.as_tensor(primes, dtype=dtype, device=device)
    a    = torch.as_tensor(alphas,    dtype=dtype, device=device).view(-1, 1)
    th   = torch.as_tensor(thetas,    dtype=dtype, device=device).view(-1, 1)
    a_fc = torch.as_tensor(alphas_fc, dtype=dtype, device=device).view(-1, 1)
    th_fc= torch.as_tensor(thetas_fc, dtype=dtype, device=device).view(-1, 1)
    N = states.shape[0]
    B, L = primes_t.shape
    if N < 3:
        return primes_t[:, -1:].expand(B, T).clone()
    terminal_idx = N - 1
    last_nt_idx = N - 2
    beta_nt_ctx = (1.0 - a    - th   ) / (N - 2)
    beta_nt_fc  = (1.0 - a_fc - th_fc) / (N - 2)
    log_norm_const = -0.5 * math.log(2.0 * math.pi * beta)
    inv_2beta = 1.0 / (2.0 * beta)
    tiny = torch.finfo(dtype).tiny

    def trans(dist, alpha_, theta_, beta_nt_):
        nt = dist.clone()
        nt[:, terminal_idx] = 0.0
        non_terminal_sum = nt.sum(dim=1, keepdim=True)
        last_nt_val = nt[:, last_nt_idx]                # (B,)
        shifted = torch.zeros_like(dist)
        shifted[:, 1:] = nt[:, :N - 1]
        out = theta_ * dist + alpha_ * shifted
        out = out + beta_nt_ * (non_terminal_sum - nt - shifted)
        out[:, 0] = out[:, 0] - beta_nt_.squeeze(-1) * last_nt_val
        return out

    log_dist = torch.full((B, N), -math.log(N), dtype=dtype, device=device)
    for t in range(L):
        if t > 0:
            mx = log_dist.max(dim=1, keepdim=True).values
            lin = torch.exp(log_dist - mx)
            lin = lin / lin.sum(dim=1, keepdim=True)
            lin = trans(lin, a, th, beta_nt_ctx)
            log_dist = torch.log(lin + tiny)
        obs = primes_t[:, t:t+1]
        sq = (states[None, :] - obs) ** 2
        log_dist = log_dist + (-sq * inv_2beta + log_norm_const)
        m = log_dist.max(dim=1, keepdim=True).values
        lse = m + torch.log(torch.exp(log_dist - m).sum(dim=1, keepdim=True))
        log_dist = log_dist - lse

    cur = torch.exp(log_dist)
    cur[:, terminal_idx] = 0.0
    s = cur.sum(dim=1, keepdim=True)
    bad = (s.squeeze(1) <= 0)
    if bool(bad.any()):
        cur[bad] = 1.0 / N
        s = cur.sum(dim=1, keepdim=True)
    cur = cur / s

    forecasts = torch.empty((B, T), dtype=dtype, device=device)
    for step in range(T):
        nxt = trans(cur, a_fc, th_fc, beta_nt_fc)
        nt = nxt.clone()
        nt[:, terminal_idx] = 0.0
        nt_sum = nt.sum(dim=1, keepdim=True)
        safe = torch.where(nt_sum > 1e-12, nt_sum, torch.ones_like(nt_sum))
        forecasts[:, step] = (nt / safe @ states)
        nxt[:, terminal_idx] = 0.0
        s2 = nxt.sum(dim=1, keepdim=True)
        mask2 = (s2.squeeze(1) > 0)
        if bool(mask2.any()):
            nxt = torch.where(s2 > 0, nxt / s2, nxt)
        cur = nxt
    return forecasts


@torch.no_grad()
def forecast_dual_xseries(states_padded, terminal_idx, betas,
                            alpha, theta, alpha_fc, theta_fc,
                            primes, T, device='cuda', dtype=torch.float64):
    """Cross-series batched dual-α GDC forecast.

    Runs B series with possibly-different state spaces in a single torch
    call, all sharing one (alpha, theta, alpha_fc, theta_fc) config. Per
    series, beta (emission variance) may vary. State spaces are padded
    to N_max with masking; the absorbing terminal is at `terminal_idx[b]`
    for series b. Primes are length-L (same L for all batch elements;
    pad short primes to L externally if needed).

    Parameters
    ----------
    states_padded : (B, N_max) float — per-series state values, padded
        with arbitrary values past position `terminal_idx[b]` (those are
        masked out and never receive mass).
    terminal_idx : (B,) long — position of the absorbing terminal in
        each series. The non-terminal positions are [0, terminal_idx[b]).
    betas : (B,) float — per-series emission variance.
    alpha, theta, alpha_fc, theta_fc : floats (shared across batch).
    primes : (B, L) float.
    T : int — forecast horizon.

    Returns (B, T) forecasts.
    """
    import math
    sp = torch.as_tensor(states_padded, dtype=dtype, device=device)
    tid = torch.as_tensor(terminal_idx, dtype=torch.long, device=device)
    bts = torch.as_tensor(betas, dtype=dtype, device=device)
    pr = torch.as_tensor(primes, dtype=dtype, device=device)
    B, N_max = sp.shape
    L = pr.shape[1]
    real_lengths = (tid + 1).to(dtype)               # (B,)
    real_lengths_int = (tid + 1).to(torch.long)

    pos = torch.arange(N_max, device=device).unsqueeze(0)    # (1, N_max)
    valid_mask = (pos < real_lengths_int.unsqueeze(1)).to(dtype)
    terminal_mask_bool = (pos == tid.unsqueeze(1))
    non_terminal_mask = valid_mask * (1.0 - terminal_mask_bool.to(dtype))
    last_nt_idx = (tid - 1).clamp(min=0)                       # (B,)

    beta_nt_ctx = ((1.0 - alpha    - theta   ) / (real_lengths - 2)).unsqueeze(1)
    beta_nt_fc  = ((1.0 - alpha_fc - theta_fc) / (real_lengths - 2)).unsqueeze(1)

    log_norm_const = -0.5 * torch.log(2.0 * math.pi * bts)     # (B,)
    inv_2beta = (1.0 / (2.0 * bts)).unsqueeze(1)               # (B, 1)
    tiny = torch.finfo(dtype).tiny
    SENT = -1e30  # sentinel for padding in log space

    def trans(dist, a_, t_, b_nt_):
        nt = dist * non_terminal_mask
        sum_nt = nt.sum(dim=1, keepdim=True)
        last_nt_val = torch.gather(nt, 1, last_nt_idx.unsqueeze(1)).squeeze(1)
        shifted = torch.zeros_like(dist)
        shifted[:, 1:] = nt[:, :N_max - 1]
        out = t_ * dist + a_ * shifted
        out = out + b_nt_ * (sum_nt - nt - shifted)
        out[:, 0] = out[:, 0] - b_nt_.squeeze(-1) * last_nt_val
        out = out * valid_mask
        return out

    # Init log_dist: uniform over valid positions, sentinel in padding
    log_dist = -torch.log(real_lengths.unsqueeze(1)).expand(B, N_max).clone()
    log_dist = torch.where(valid_mask > 0, log_dist, torch.full_like(log_dist, SENT))

    # Forward pass
    for t in range(L):
        if t > 0:
            mx = log_dist.max(dim=1, keepdim=True).values
            lin = torch.exp(log_dist - mx)
            lin = lin / lin.sum(dim=1, keepdim=True)
            lin = trans(lin, alpha, theta, beta_nt_ctx)
            log_dist = torch.log(lin + tiny)
        obs = pr[:, t:t+1]
        sq = (sp - obs) ** 2
        log_dist = log_dist + (-sq * inv_2beta + log_norm_const.unsqueeze(1))
        log_dist = torch.where(valid_mask > 0, log_dist, torch.full_like(log_dist, SENT))
        m = log_dist.max(dim=1, keepdim=True).values
        lse = m + torch.log(torch.exp(log_dist - m).sum(dim=1, keepdim=True))
        log_dist = log_dist - lse

    cur = torch.exp(log_dist)
    cur = cur.scatter(1, tid.unsqueeze(1), 0.0)  # zero terminal
    cur = cur * valid_mask
    s = cur.sum(dim=1, keepdim=True)
    bad = (s.squeeze(1) <= 0)
    if bool(bad.any()):
        cur[bad] = valid_mask[bad] / real_lengths[bad].unsqueeze(1)
        s = cur.sum(dim=1, keepdim=True)
    cur = cur / s

    forecasts = torch.empty((B, T), dtype=dtype, device=device)
    for step in range(T):
        nxt = trans(cur, alpha_fc, theta_fc, beta_nt_fc)
        nt = nxt * non_terminal_mask
        nt_sum = nt.sum(dim=1, keepdim=True)
        safe = torch.where(nt_sum > 1e-12, nt_sum, torch.ones_like(nt_sum))
        forecasts[:, step] = ((nt / safe) * sp).sum(dim=1)
        nxt = nxt.scatter(1, tid.unsqueeze(1), 0.0)
        nxt = nxt * valid_mask
        s2 = nxt.sum(dim=1, keepdim=True)
        nxt = torch.where(s2 > 0, nxt / s2, nxt)
        cur = nxt
    return forecasts


def smoke_test():
    """Compare torch vs numba forecasts at realistic sizes."""
    import os, sys, time
    import numpy as np
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.dirname(HERE))
    sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
    from generative_dense_chain_timeseries import GenerativeDenseChainTimeSeries
    from gdc_numba import forecast_many

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available; running on CPU"); device = 'cpu'
    else:
        device = 'cuda'
        print(f"Device: {torch.cuda.get_device_name(0)}")

    rng = np.random.default_rng(0)
    for N, B, L, T in [(2000, 64, 96, 96),
                        (12000, 256, 192, 192),
                        (12000, 2857, 720, 24)]:
        print(f"\n--- N={N} B={B} L={L} T={T} ---")
        series = np.cumsum(rng.standard_normal(N))
        gdc = GenerativeDenseChainTimeSeries(
            series.reshape(-1, 1), beta=4.0, alpha=1.0, theta=0.0,
            transition_type='self_loop',
            terminal_behavior='absorb',
            initial_dist='uniform')
        states_1d = gdc.states[:, 0]
        terminal_idx = int((gdc.terminal_mask).nonzero()[0][-1])
        primes = np.stack([series[i % (N - L - T)+np.arange(L)]
                            for i in range(B)], axis=0)
        # Numba reference
        _ = forecast_many(states_1d, terminal_idx, gdc.beta, gdc.alpha,
                           gdc.theta, primes[:1], T)  # warm-up
        t0 = time.time()
        ref = forecast_many(states_1d, terminal_idx, gdc.beta, gdc.alpha,
                             gdc.theta, primes, T)
        t_numba = time.time() - t0
        # Torch fp64
        # Warm-up GPU
        _ = forecast_many_torch(states_1d, gdc.beta, gdc.alpha, gdc.theta,
                                 primes[:1], T, device=device, dtype=torch.float64)
        if device == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        out64 = forecast_many_torch(states_1d, gdc.beta, gdc.alpha, gdc.theta,
                                     primes, T, device=device, dtype=torch.float64)
        if device == 'cuda': torch.cuda.synchronize()
        t_torch64 = time.time() - t0
        # Torch fp32
        if device == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        out32 = forecast_many_torch(states_1d, gdc.beta, gdc.alpha, gdc.theta,
                                     primes, T, device=device, dtype=torch.float32)
        if device == 'cuda': torch.cuda.synchronize()
        t_torch32 = time.time() - t0
        diff64 = float((out64.cpu().numpy() - ref).max() if out64.is_cuda
                       else (out64.numpy() - ref).max())
        rdiff64 = abs(diff64) / max(abs(ref).max(), 1e-9)
        diff32 = float((out32.cpu().numpy().astype('float64') - ref).max()
                       if out32.is_cuda else (out32.numpy().astype('float64') - ref).max())
        rdiff32 = abs(diff32) / max(abs(ref).max(), 1e-9)
        print(f"  numba:    {t_numba:.3f}s  ({t_numba/B*1000:.2f}ms/prime)")
        print(f"  torch64:  {t_torch64:.3f}s  ({t_torch64/B*1000:.2f}ms/prime) "
              f"speedup={t_numba/t_torch64:.1f}x  max abs diff={diff64:.2e}  rel={rdiff64:.2e}")
        print(f"  torch32:  {t_torch32:.3f}s  ({t_torch32/B*1000:.2f}ms/prime) "
              f"speedup={t_numba/t_torch32:.1f}x  max abs diff={diff32:.2e}  rel={rdiff32:.2e}")


if __name__ == "__main__":
    smoke_test()
