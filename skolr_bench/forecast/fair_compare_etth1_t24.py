"""Three apples-to-apples GDC vs ARIMA comparisons on ETTh1 T=24.

GDC variants:
  GDC-A: state space = train+val (~11520 pts);  prime length L=48
  GDC-B: state space = train+val (~11520 pts);  prime length L=720
  GDC-C: state space = lookback only (L=720 pts); prime length L=720
         (matches ARIMA's "input length only" setup exactly)
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
from gdc_numba import forecast_many, _forecast_one_kernel  # noqa: E402


def build_gdc_1d(state_series, window_len, sigma_frac, alpha, theta=0.0):
    sigma = max(float(np.std(state_series)) * sigma_frac, 1e-9)
    sigma_gdc = sigma * np.sqrt(window_len)
    beta = max(sigma_gdc ** 2, 1e-9)
    states = np.asarray(state_series, dtype=np.float64).reshape(-1, 1)
    return GenerativeDenseChainTimeSeries(
        states, beta=beta, alpha=alpha, theta=theta,
        transition_type='self_loop',
        terminal_behavior='absorb',
        initial_dist='uniform')


def primes_truths(series, L, T, anchor_offset=0):
    s = np.asarray(series, dtype=np.float64)
    n = len(s)
    n_w = max(0, n - L - T + 1)
    starts = np.arange(0, n_w, 1)
    p_idx = np.arange(L)[None, :] + starts[:, None]
    t_idx = np.arange(L, L + T)[None, :] + starts[:, None]
    return s[p_idx], s[t_idx], starts


def eval_gdc_a_b(state_series, full_test, L, T, sigma, alpha, kind):
    """GDC-A or GDC-B: fixed state_space = state_series, prime sliding."""
    if kind == 'diff':
        d_state = np.diff(state_series)
        gdc = build_gdc_1d(d_state, L, sigma, alpha)
        ext, truths, starts = primes_truths(full_test, L+1, T)
        diffed_primes = np.diff(ext, axis=1)  # (B, L)
        anchors = ext[:, -1]
        forecast_d = forecast_many(gdc.states[:, 0],
                                    int(np.where(gdc.terminal_mask)[0][-1]),
                                    gdc.beta, gdc.alpha, gdc.theta,
                                    diffed_primes, T)
        cum = np.cumsum(forecast_d, axis=1)
        forecasts = anchors[:, None] + cum
    else:
        gdc = build_gdc_1d(state_series, L, sigma, alpha)
        primes, truths, starts = primes_truths(full_test, L, T)
        forecasts = forecast_many(gdc.states[:, 0],
                                   int(np.where(gdc.terminal_mask)[0][-1]),
                                   gdc.beta, gdc.alpha, gdc.theta,
                                   primes, T)
    diff = truths - forecasts
    return float((diff**2).mean()), float(np.abs(diff).mean()), forecasts.shape[0]


def eval_gdc_c(full_test, L, T, sigma, alpha, kind):
    """GDC-C: state space = the L-point lookback itself (matches ARIMA setup).
    Per window we have to build a fresh GDC. Use Numba kernel for speed."""
    s = np.asarray(full_test, dtype=np.float64)
    n = len(s); n_w = max(0, n - L - T + 1)
    starts = np.arange(0, n_w, 1)
    truths = np.empty((len(starts), T), dtype=np.float64)
    forecasts = np.empty((len(starts), T), dtype=np.float64)
    for k, i in enumerate(starts):
        if kind == 'diff':
            window = s[i:i+L+1]
            if len(window) < L + 1: continue
            d_state = np.diff(window)  # (L,)
            sigma_v = max(float(np.std(d_state)) * sigma, 1e-9)
            beta = max((sigma_v * np.sqrt(L))**2, 1e-9)
            anchor = s[i+L]
            forecast_d = _forecast_one_kernel(d_state, len(d_state)-1,
                                                beta, alpha, 0.0,
                                                d_state, T)
            forecasts[k] = anchor + np.cumsum(forecast_d)
        else:
            window = s[i:i+L]
            if len(window) < L: continue
            sigma_v = max(float(np.std(window)) * sigma, 1e-9)
            beta = max((sigma_v * np.sqrt(L))**2, 1e-9)
            forecasts[k] = _forecast_one_kernel(window, len(window)-1,
                                                 beta, alpha, 0.0,
                                                 window, T)
        truths[k] = s[i+L:i+L+T]
    diff = truths - forecasts
    return float((diff**2).mean()), float(np.abs(diff).mean()), forecasts.shape[0]


def main():
    train, val, test, mu, sd = load_univariate('ETTh1')
    T = 24
    state_train_val = np.concatenate([train, val])
    print(f"=== ETTh1 T=24 univariate apples-to-apples ===\n")
    # Use the val-picked config from prior smoke test
    kind, sigma, alpha = 'diff', 0.25, 1.0
    print(f"GDC config: {kind}, sigma={sigma}, alpha={alpha} (val-picked at L=48)")
    print()

    # GDC-A: state=train+val, L=48
    test_with48 = np.concatenate([val[-49:], test])  # need L+1=49 for diff
    t0 = time.time()
    mse_a, mae_a, n_a = eval_gdc_a_b(state_train_val, test_with48, 48, T,
                                      sigma, alpha, kind)
    print(f"GDC-A: state=train+val ({len(state_train_val)} pts), prime L=48")
    print(f"  MSE={mse_a:.4f}  MAE={mae_a:.4f}  n_windows={n_a}  ({time.time()-t0:.1f}s)\n")

    # GDC-B: state=train+val, L=720
    test_with720 = np.concatenate([val[-721:], test])
    t0 = time.time()
    mse_b, mae_b, n_b = eval_gdc_a_b(state_train_val, test_with720, 720, T,
                                      sigma, alpha, kind)
    print(f"GDC-B: state=train+val ({len(state_train_val)} pts), prime L=720")
    print(f"  MSE={mse_b:.4f}  MAE={mae_b:.4f}  n_windows={n_b}  ({time.time()-t0:.1f}s)\n")

    # GDC-C: state = lookback only (L=720 pts), most apples-to-apples vs ARIMA
    t0 = time.time()
    mse_c, mae_c, n_c = eval_gdc_c(test_with720, 720, T, sigma, alpha, kind)
    print(f"GDC-C: state=L=720 lookback only, prime L=720 (~ARIMA setup)")
    print(f"  MSE={mse_c:.4f}  MAE={mae_c:.4f}  n_windows={n_c}  ({time.time()-t0:.1f}s)\n")

    print("=== Summary ===")
    print(f"{'method':<55s}  {'MSE':>8s}  {'MAE':>8s}")
    print(f"{'Published ARIMA (Informer Tab.1)':<55s}  {0.108:>8.4f}  {0.284:>8.4f}")
    print(f"{'Published Informer':<55s}  {0.098:>8.4f}  {0.247:>8.4f}")
    print(f"{'Our auto_arima L=48':<55s}  {0.046:>8.4f}  {0.150:>8.4f}")
    print(f"{'Our auto_arima L=720':<55s}  {0.034:>8.4f}  {0.139:>8.4f}")
    print(f"{'GDC-A (state=train+val, L=48 prime)':<55s}  {mse_a:>8.4f}  {mae_a:>8.4f}")
    print(f"{'GDC-B (state=train+val, L=720 prime)':<55s}  {mse_b:>8.4f}  {mae_b:>8.4f}")
    print(f"{'GDC-C (state=lookback only L=720, ~ARIMA)':<55s}  {mse_c:>8.4f}  {mae_c:>8.4f}")


if __name__ == "__main__":
    main()
