"""Run the context-parroting baseline across the same datasets/horizons
on which GDC was evaluated (paper/tables.tex).

Datasets covered (matching `informer_loaders.DATASETS`):

  Informer protocol  (Table: ETTh1 in main results):
    ETTh1   L = 720,  T in {24, 48, 168, 336, 720}

  Autoformer protocol (paper/tables.tex Tables 1, 4):
    ETTm2     L = 96,  T in {96, 192, 336, 720}
    Exchange  L = 96,  T in {96, 192, 336, 720}
    ECL_AF    L = 96,  T in {96, 192, 336, 720}
    Traffic_AF L = 96, T in {96, 192, 336, 720}
    ILI_AF    L = 36,  T in {24, 36, 48, 60}

For each (dataset, T) we report test MSE / MAE for:
  - Persistence    (predict last value, sanity floor)
  - Seasonal naive (predict last period; M4-Naive-1 cousin)
  - Parrot raw  k=1, k=5
  - Parrot diff k=1, k=5

Protocol mirrors `gdc_etth1_full_sweep` / `gdc_ettm2_autoformer`:
  - State pool for test eval = train + val (no test-prefix leakage)
  - Test lookbacks span the val tail into the test region (so every
    test point has L observations of context, just like GDC does)
  - StandardScaler is already applied by the loader (mu, sd from train)
"""
from __future__ import annotations
import os, sys, time, csv
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SKOLR_BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(SKOLR_BENCH)
sys.path.insert(0, HERE); sys.path.insert(0, SKOLR_BENCH); sys.path.insert(0, ROOT)
from informer_loaders import load_univariate
from parrot_torch import (
    forecast_many_parrot,
    forecast_many_parrot_diff,
    forecast_many_persistence,
    forecast_many_seasonal_naive,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

# (name in DATASETS, lookback L, list of horizons, optional season for seasonal-naive)
SWEEPS = [
    ('ETTh1',      720, [24, 48, 168, 336, 720], 24),     # Informer protocol; hourly seasonality
    ('ETTm2',       96, [96, 192, 336, 720],     96),     # Autoformer; 15-min seasonality (24*4=96/day)
    ('Exchange',    96, [96, 192, 336, 720],     None),   # daily exchange rate; no clean seasonality
    ('ECL_AF',      96, [96, 192, 336, 720],     24),     # hourly electricity, daily season
    ('Traffic_AF',  96, [96, 192, 336, 720],     24),     # hourly traffic, daily season
    ('ILI_AF',      36, [24, 36, 48, 60],        None),   # weekly ILI; no clean season at this scale
]


def make_primes_truths(series_1d, L, T):
    """Slide (L, T) windows over a 1-D series. Returns (primes, truths)."""
    s = np.asarray(series_1d, dtype=np.float64)
    n = len(s)
    n_w = max(0, n - L - T + 1)
    if n_w == 0:
        return np.empty((0, L)), np.empty((0, T))
    starts = np.arange(n_w)
    p_idx = starts[:, None] + np.arange(L)[None, :]
    t_idx = starts[:, None] + L + np.arange(T)[None, :]
    return s[p_idx], s[t_idx]


def make_primes_truths_diff(series_1d, L, T):
    """As above, but primes are length L+1 so we can take L diffs.
    Returns (primes_raw_Lp1, truths_T) — same n_w as raw mode."""
    s = np.asarray(series_1d, dtype=np.float64)
    n = len(s)
    n_w = max(0, n - L - T)
    if n_w == 0:
        return np.empty((0, L + 1)), np.empty((0, T))
    starts = np.arange(n_w)
    p_idx = starts[:, None] + np.arange(L + 1)[None, :]
    t_idx = starts[:, None] + (L + 1) + np.arange(T)[None, :]
    return s[p_idx], s[t_idx]


def mse_mae(forecasts, truths):
    if torch.is_tensor(forecasts):
        forecasts = forecasts.detach().cpu().numpy().astype(np.float64)
    diff = truths - forecasts
    return float((diff ** 2).mean()), float(np.abs(diff).mean())


def eval_dataset(name, L, horizons, season):
    print(f"\n=== {name}  (L={L}, season={season}) ===")
    train, val, test, mu, sd = load_univariate(name)
    state_train_val = np.concatenate([train, val])
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}  "
          f"state_pool={len(state_train_val)}  mu={mu:.3f}  sd={sd:.3f}")

    rows = []
    for T in horizons:
        # Test windows: last L of val joined with full test (so first prime
        # uses val tail as context). Matches gdc_*_autoformer exactly.
        full_test = np.concatenate([val[-L:], test])
        primes, truths = make_primes_truths(full_test, L, T)
        if primes.shape[0] == 0:
            print(f"  T={T}: not enough test data for any window — skipping")
            continue

        # diff-mode primes need one more raw observation
        full_test_d = np.concatenate([val[-(L + 1):], test])
        primes_d, truths_d = make_primes_truths_diff(full_test_d, L, T)
        # Sanity: same number of windows + same truths
        assert primes_d.shape[0] == primes.shape[0]
        assert np.allclose(truths_d, truths)

        variants = []
        # Persistence
        t0 = time.time()
        f = forecast_many_persistence(primes, T, device=DEVICE, dtype=DTYPE)
        m, a = mse_mae(f, truths)
        variants.append(('persistence', m, a, time.time() - t0))

        # Seasonal naive (if applicable)
        if season is not None and season <= L:
            t0 = time.time()
            f = forecast_many_seasonal_naive(primes, T, season,
                                              device=DEVICE, dtype=DTYPE)
            m, a = mse_mae(f, truths)
            variants.append((f'seasonal_naive_s{season}', m, a, time.time() - t0))

        # Parrot raw, k=1 and k=5
        for k in (1, 5):
            t0 = time.time()
            f = forecast_many_parrot(state_train_val, primes, T, k=k,
                                     device=DEVICE, dtype=DTYPE)
            m, a = mse_mae(f, truths)
            variants.append((f'parrot_raw_k{k}', m, a, time.time() - t0))

        # Parrot diff, k=1 and k=5
        for k in (1, 5):
            t0 = time.time()
            f = forecast_many_parrot_diff(state_train_val, primes_d, T, k=k,
                                          device=DEVICE, dtype=DTYPE)
            m, a = mse_mae(f, truths)
            variants.append((f'parrot_diff_k{k}', m, a, time.time() - t0))

        # Print mini-table for this T
        best_mse = min(v[1] for v in variants)
        print(f"  T={T:>3d}  n_windows={primes.shape[0]:>5d}")
        for vname, vmse, vmae, vsec in variants:
            tag = '  *' if vmse == best_mse else '   '
            print(f"    {tag} {vname:<22s}  MSE={vmse:.4f}  MAE={vmae:.4f}  ({vsec:.2f}s)")
            rows.append((name, T, vname, vmse, vmae, vsec))
    return rows


def main():
    print(f"=== Parroting baseline sweep ===")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})  dtype={DTYPE}")
    t_total = time.time()
    all_rows = []
    for name, L, horizons, season in SWEEPS:
        all_rows += eval_dataset(name, L, horizons, season)
    print(f"\nTotal: {time.time() - t_total:.1f}s\n")

    out = os.path.join(HERE, 'results', 'parrot_sweep.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'T', 'variant', 'mse', 'mae', 'time_s'])
        for r in all_rows: w.writerow(r)
    print(f"Wrote {out}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
