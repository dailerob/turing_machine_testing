"""Smoke test for Informer-style univariate eval at min T.

ETTh1, target='OT', T=24, L=2T=48 lookback.
State space = train+val (fixed); prime = lookback (last L points before pred).
Uses Numba kernel for batched forecast.

Goals:
  1. Confirm pipeline correctness (sanity-check predictions look reasonable)
  2. Time it; estimate runtime for full ETTh1 sweep across all horizons.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SKOLR_BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(SKOLR_BENCH)
sys.path.insert(0, HERE); sys.path.insert(0, SKOLR_BENCH); sys.path.insert(0, ROOT)
from generative_dense_chain_timeseries import GenerativeDenseChainTimeSeries  # noqa: E402
from informer_loaders import load_univariate  # noqa: E402
from gdc_numba import forecast_many  # noqa: E402


def primes_truths_from(series, L, T, stride=1):
    """Build (B, L) primes and (B, T) truths sliding stride=1."""
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    n_w = max(0, n - L - T + 1)
    if n_w == 0: return np.empty((0, L)), np.empty((0, T))
    starts = np.arange(0, n_w, stride)
    p_idx = np.arange(L)[None, :] + starts[:, None]
    t_idx = np.arange(L, L + T)[None, :] + starts[:, None]
    return series[p_idx], series[t_idx]


def build_gdc_1d(state_series, window_len, sigma_frac, alpha, theta=0.0):
    sigma_per_step = float(np.std(state_series)) * sigma_frac
    sigma_per_step = max(sigma_per_step, 1e-9)
    sigma_gdc = sigma_per_step * np.sqrt(window_len)
    beta = max(sigma_gdc ** 2, 1e-9)
    states = state_series.reshape(-1, 1)
    gdc = GenerativeDenseChainTimeSeries(
        states, beta=beta, alpha=alpha, theta=theta,
        transition_type='self_loop',
        terminal_behavior='absorb',
        initial_dist='uniform')
    return gdc


def run(dataset, T, L, sigma_frac=0.10, alpha=1.0, kind='raw', verbose=True):
    train, val, test, mu, sd = load_univariate(dataset)
    if verbose:
        print(f"  {dataset}: train={len(train)}, val={len(val)}, test={len(test)}")
    # State space = train + val (fixed)
    state_series = np.concatenate([train, val])
    n_state = len(state_series)
    if verbose:
        print(f"  State space: {n_state} points (train+val)")
    # Build GDC once
    if kind == 'diff':
        d_state = np.diff(state_series)
        gdc = build_gdc_1d(d_state, L, sigma_frac, alpha)
    else:
        gdc = build_gdc_1d(state_series, L, sigma_frac, alpha)
    states_1d = gdc.states[:, 0]
    terminal_idx = int(np.where(gdc.terminal_mask)[0][-1])
    # Primes & truths from test set (each test sample needs L lookback BEFORE
    # its prediction; the very first lookback overlaps the end of val).
    test_with_lookback = np.concatenate([val[-L:], test])
    primes, truths = primes_truths_from(test_with_lookback, L, T, stride=1)
    if verbose:
        print(f"  Test windows: {primes.shape[0]} (each: L={L} lookback, T={T} target)")
    if kind == 'diff':
        # Convert primes to diffs of length L; predict T diffs; cumsum onto anchor.
        # For window starting at i in test_with_lookback:
        #   prime = test_with_lookback[i:i+L]
        #   prime_d = diff(prime), length L-1   (need L+1 for length-L diff window)
        # To match length, use longer prime: take L+1 lookback then diff to L.
        ext_with_lookback = np.concatenate([val[-L-1:], test])
        ext_primes, _ = primes_truths_from(ext_with_lookback, L + 1, T, stride=1)
        diffed_primes = np.diff(ext_primes, axis=1)  # (B, L)
        anchors = ext_primes[:, -1]
        t0 = time.time()
        forecast_d = forecast_many(states_1d, terminal_idx, gdc.beta,
                                    gdc.alpha, gdc.theta, diffed_primes, T)
        cum = np.cumsum(forecast_d, axis=1)
        forecasts = anchors[:, None] + cum
        elapsed = time.time() - t0
        truths = truths[:len(forecasts)]
    else:
        t0 = time.time()
        forecasts = forecast_many(states_1d, terminal_idx, gdc.beta,
                                   gdc.alpha, gdc.theta, primes, T)
        elapsed = time.time() - t0
    diff = truths - forecasts
    mse = float((diff ** 2).mean())
    mae = float(np.abs(diff).mean())
    if verbose:
        print(f"  Forecast time: {elapsed:.2f}s  ({elapsed/primes.shape[0]*1000:.2f}ms/window)")
    return mse, mae, elapsed, primes.shape[0]


def main():
    print("=== Smoke test: ETTh1 univariate, T=24, L=48 ===")
    print("State space = train+val (~11520 states)")
    print()

    # First call: warm up Numba JIT
    print("Warming up Numba JIT (first call compiles)...")
    t_warm = time.time()
    _ = run('ETTh1', T=24, L=48, sigma_frac=0.10, alpha=1.0,
             kind='raw', verbose=False)
    print(f"  warm-up: {time.time()-t_warm:.1f}s\n")

    # Real timing
    print("--- Single config: raw, sigma=0.10, alpha=1.0 ---")
    mse, mae, t_, n_w = run('ETTh1', T=24, L=48, sigma_frac=0.10, alpha=1.0,
                             kind='raw')
    print(f"  Test MSE={mse:.4f}  MAE={mae:.4f}")
    print(f"  Informer ARIMA reference: MSE=0.108, MAE=0.284")
    print(f"  Informer Informer reference: MSE=0.098, MAE=0.247")
    print()

    # Try a few configs to get a sense of variance
    print("--- Quick mini-sweep ---")
    configs = [
        ('raw', 0.05, 1.0),
        ('raw', 0.10, 1.0),
        ('raw', 0.25, 1.0),
        ('raw', 0.10, 0.95),
        ('diff', 0.25, 1.0),
        ('diff', 0.50, 1.0),
    ]
    total_t = 0.0
    for kind, s, a in configs:
        mse, mae, t_, _ = run('ETTh1', T=24, L=48,
                               sigma_frac=s, alpha=a, kind=kind, verbose=False)
        total_t += t_
        print(f"    {kind} sigma={s} alpha={a}: MSE={mse:.4f}  MAE={mae:.4f}  ({t_:.2f}s)")
    print(f"  Total {len(configs)} configs: {total_t:.1f}s ({total_t/len(configs):.2f}s each)")
    print()

    # Estimate full sweep
    print("--- Runtime estimate ---")
    print("Per-config time at T=24 (smallest horizon):")
    print(f"  ~{total_t/len(configs):.1f}s/config")
    print()
    print("Scaling with T (forecast cost ~ T*N + L*N where L=2T):")
    for T in [24, 48, 168, 336, 720]:
        # rough scaling: cost ~ (L+T)*N = 3T*N
        scale = (3 * T) / (3 * 24)
        est = (total_t / len(configs)) * scale
        print(f"  T={T:>3d}: ~{est:.1f}s/config")
    print()
    cfgs_per_run = 27
    horizons = [24, 48, 168, 336, 720]
    total_est = sum((total_t / len(configs)) * (3*T) / (3*24) * cfgs_per_run for T in horizons)
    print(f"Full ETTh1 univariate sweep ({cfgs_per_run} configs × 5 horizons):")
    print(f"  Estimate: ~{total_est:.0f}s = ~{total_est/60:.1f} min")


if __name__ == "__main__":
    main()
