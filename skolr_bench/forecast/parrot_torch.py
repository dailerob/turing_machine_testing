"""PyTorch / GPU context-parroting predictor.

The "context-parroting" baseline of Zhang & Gilpin (2025,
arXiv:2505.11349) is the simplest possible prefix-memorising forecaster:
given a recent lookback window, find the most-similar L-window in the
historical pool and copy the values that came after it.

This is mechanistically very close to GDC's `raw σ→0, α=1.0` limit
(hard nearest-prefix lookup vs. soft Gaussian-weighted lookup), so it's
the natural baseline against which to characterise the GDC kernel.

API mirrors `gdc_torch.forecast_many_torch`:

    forecast_many_parrot(state_series, primes, T,
                         k=1, device='cuda', dtype=torch.float32)
        -> (B, T) tensor

Inputs:
  state_series : (N,) array — historical pool (e.g., train+val
                 concatenated, exactly as GDC uses).
  primes       : (B, L) array — lookback windows for the B test
                 instances. Same convention as GDC: prime[:, -1] is the
                 most recent observation.
  T            : int — forecast horizon.
  k            : int — top-k neighbours to mean over (k=1 = pure NN).

Algorithm (vectorised, single-matmul distance):
  1. Build all sliding L-windows of `state_series` that have at least
     T points after them:  W[i, :] = state_series[i:i+L]
                           C[i, :] = state_series[i+L:i+L+T]
     for i in 0 .. N - L - T  (so n_w = N - L - T + 1 windows).
  2. Squared-Euclidean distance from each prime to each window via the
     identity ||p - w||² = ||p||² + ||w||² - 2 p·w.
  3. Take topk smallest distances per prime; mean the k continuations
     to get the forecast.

This file does ONLY the predictor.  Dataset loading + sliding-window
test eval lives in `parrot_sweep.py` (mirroring `gdc_etth1_full_sweep`
etc.), which lets us reuse `informer_loaders` for the historical pool.
"""
from __future__ import annotations
import torch
import numpy as np


def _make_windows_and_continuations(state_1d, L, T):
    """Slide length-L windows over state_1d, paired with length-T continuations.

    Returns
    -------
    W : (n_w, L) tensor on the same device/dtype as state_1d
    C : (n_w, T) tensor on the same device/dtype as state_1d
    where n_w = N - L - T + 1 (zero if the series is too short).
    """
    N = state_1d.shape[0]
    n_w = max(0, N - L - T + 1)
    if n_w == 0:
        return (state_1d.new_empty(0, L), state_1d.new_empty(0, T))
    # Use as_strided for a zero-copy view when possible
    starts = torch.arange(n_w, device=state_1d.device)
    w_idx = starts[:, None] + torch.arange(L, device=state_1d.device)[None, :]
    c_idx = starts[:, None] + L + torch.arange(T, device=state_1d.device)[None, :]
    W = state_1d[w_idx]
    C = state_1d[c_idx]
    return W, C


def forecast_many_parrot(state_1d, primes, T, k=1,
                         device='cuda', dtype=torch.float32,
                         window_chunk=8192):
    """Top-k nearest-neighbour parroting forecast.

    Parameters
    ----------
    state_1d : (N,) array-like — historical pool to search over.
    primes   : (B, L) array-like — test lookback windows.
    T        : int — forecast horizon.
    k        : int — top-k neighbours to average. k=1 is pure NN.
    device   : torch device for the heavy matmul.
    dtype    : torch.float32 (recommended) or torch.float64.
    window_chunk : if the prime batch is huge, search the historical
                   windows in chunks of this size to bound memory.

    Returns
    -------
    forecasts : (B, T) torch tensor on `device`.
    """
    state = torch.as_tensor(state_1d, dtype=dtype, device=device)
    primes = torch.as_tensor(primes, dtype=dtype, device=device)
    B, L = primes.shape
    N = state.shape[0]
    if B == 0:
        return primes.new_empty(0, T)
    if N < L + T:
        # Not enough history for any valid window; fall back to last-value persistence
        last = primes[:, -1:]
        return last.expand(B, T).clone()

    W, C = _make_windows_and_continuations(state, L, T)
    n_w = W.shape[0]
    if n_w < k:
        k = n_w  # only as many neighbours as exist

    # Squared Euclidean: ||p||² + ||w||² - 2 p w^T (the ||p||² term is constant per row,
    # so we can drop it for argmin / topk; we keep it for clarity, the cost is trivial).
    p_sq = (primes ** 2).sum(dim=1, keepdim=True)        # (B, 1)
    w_sq = (W ** 2).sum(dim=1, keepdim=True).T           # (1, n_w)

    # Chunk over historical windows so memory stays bounded for large datasets.
    # We do (B, chunk) at a time, accumulating top-k over chunks via merging.
    if n_w <= window_chunk:
        # Single shot
        d = p_sq + w_sq - 2.0 * (primes @ W.T)             # (B, n_w)
        topk_vals, topk_idx = torch.topk(d, k, dim=1, largest=False)
        forecasts = C[topk_idx].mean(dim=1)                 # (B, T)
        return forecasts

    # Chunked path: keep a running top-k buffer
    best_vals = primes.new_full((B, k), float('inf'))
    best_idx = torch.zeros((B, k), dtype=torch.long, device=device)
    for c0 in range(0, n_w, window_chunk):
        c1 = min(n_w, c0 + window_chunk)
        d_chunk = p_sq + w_sq[:, c0:c1] - 2.0 * (primes @ W[c0:c1].T)
        # Merge with running top-k: concat existing best with new chunk and re-topk
        cand_vals = torch.cat([best_vals, d_chunk], dim=1)
        # Adjust indices: best_idx are absolute; chunk indices need offset c0
        chunk_global_idx = (torch.arange(c1 - c0, device=device) + c0
                            )[None, :].expand(B, -1)
        cand_idx = torch.cat([best_idx, chunk_global_idx], dim=1)
        new_vals, sel = torch.topk(cand_vals, k, dim=1, largest=False)
        best_vals = new_vals
        best_idx = torch.gather(cand_idx, 1, sel)
    forecasts = C[best_idx].mean(dim=1)
    return forecasts


def forecast_many_persistence(primes, T, device='cuda', dtype=torch.float32):
    """Trivial persistence baseline: forecast the last observed value
    repeated T times. Useful as a sanity floor.
    """
    primes = torch.as_tensor(primes, dtype=dtype, device=device)
    return primes[:, -1:].expand(primes.shape[0], T).clone()


def forecast_many_seasonal_naive(primes, T, season, device='cuda',
                                  dtype=torch.float32):
    """Seasonal naïve: repeat the last observed period.

    forecast[b, t] = primes[b, L - season + (t mod season)]
    Works whenever season <= L. Useful as a stronger structural floor.
    """
    primes = torch.as_tensor(primes, dtype=dtype, device=device)
    B, L = primes.shape
    if season > L:
        # Fall back to persistence
        return forecast_many_persistence(primes, T, device=device, dtype=dtype)
    # Pull the last `season` values, tile to length T
    period = primes[:, L - season:]                  # (B, season)
    reps = (T + season - 1) // season
    tiled = period.repeat(1, reps)[:, :T]            # (B, T)
    return tiled


# -----------------------------------------------------------------------------
# Diff-mode wrapper: mirrors GDC's `kind='diff'` recipe so the parrot vs GDC
# comparison can be done in matched recipes.
#   - forecast 1-step diffs by NN over diff-windows
#   - cumsum onto the last raw observation
# -----------------------------------------------------------------------------

def forecast_many_parrot_diff(state_1d, primes_raw, T, k=1,
                               device='cuda', dtype=torch.float32,
                               window_chunk=8192):
    """Diff-mode parrot.

    Parameters
    ----------
    state_1d : (N,) raw historical series.
    primes_raw : (B, L+1) raw lookback windows.  Note: this is one
                 element LONGER than the raw-mode L because we need L
                 diffs per prime.
    T : int — forecast horizon (number of diffs to predict).
    k, device, dtype, window_chunk : as in `forecast_many_parrot`.

    Returns
    -------
    forecasts : (B, T) torch tensor of raw-space forecasts.
    """
    state = torch.as_tensor(state_1d, dtype=dtype, device=device)
    primes_raw = torch.as_tensor(primes_raw, dtype=dtype, device=device)
    B, Lp1 = primes_raw.shape
    L = Lp1 - 1
    d_state = state[1:] - state[:-1]                 # (N-1,)
    d_primes = primes_raw[:, 1:] - primes_raw[:, :-1]  # (B, L)
    anchors = primes_raw[:, -1:]                      # (B, 1)
    # NN search in diff-space; the "T continuations" are T diffs of state.
    forecast_d = forecast_many_parrot(d_state, d_primes, T,
                                      k=k, device=device, dtype=dtype,
                                      window_chunk=window_chunk)
    return anchors + torch.cumsum(forecast_d, dim=1)


if __name__ == "__main__":
    # Self-test: a deterministic periodic series should produce zero MSE
    # because the NN finds the matching prefix exactly.
    torch.manual_seed(0)
    period = 24
    series = torch.sin(torch.arange(2000, dtype=torch.float64) * 2 * torch.pi / period)
    L, T = 48, 24
    # Form a single test prime from the tail
    prime = series[-(L + T):-T].unsqueeze(0)
    truth = series[-T:].unsqueeze(0)
    # State pool: everything before the test region
    state = series[:-T - L]

    fc = forecast_many_parrot(state, prime, T, k=1,
                              device='cuda' if torch.cuda.is_available() else 'cpu',
                              dtype=torch.float32)
    mse = ((fc - truth.to(fc.device, fc.dtype)) ** 2).mean().item()
    print(f"Periodic-sine self-test: MSE={mse:.3e} (should be near zero)")
