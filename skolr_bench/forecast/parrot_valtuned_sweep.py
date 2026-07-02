"""Val-tuned context-parroting sweep, leakage-free.

For each (dataset, T):
  - VAL tuning  : state_pool = train         lookback = train tail → val
                  Compute val MSE for each variant in:
                    parrot_raw_k1, parrot_raw_k5, parrot_diff_k1, parrot_diff_k5
                  Pick the variant with lowest val MSE.
  - TEST eval   : state_pool = train + val   lookback = val tail → test
                  Run the val-picked variant; report test MSE / MAE.

Same protocol as `gdc_etth1_full_sweep` / `gdc_ettm2_autoformer`.

Output: results/parrot_valtuned.csv with one row per (dataset, T).
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
from parrot_torch import forecast_many_parrot, forecast_many_parrot_diff

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float32

# (loader name, lookback L, list of horizons)
SWEEPS = [
    ('ETTh1',      720, [24, 48, 168, 336, 720]),
    ('ETTm2',       96, [96, 192, 336, 720]),
    ('Exchange',    96, [96, 192, 336, 720]),
    ('ECL_AF',      96, [96, 192, 336, 720]),
    ('Traffic_AF',  96, [96, 192, 336, 720]),
    ('ILI_AF',      36, [24, 36, 48, 60]),
]

VARIANTS = [
    ('parrot_raw_k1',  'raw',  1),
    ('parrot_raw_k5',  'raw',  5),
    ('parrot_diff_k1', 'diff', 1),
    ('parrot_diff_k5', 'diff', 5),
]


def make_primes_truths(series_1d, L, T):
    s = np.asarray(series_1d, dtype=np.float64)
    n_w = max(0, len(s) - L - T + 1)
    if n_w == 0:
        return np.empty((0, L)), np.empty((0, T))
    starts = np.arange(n_w)
    p_idx = starts[:, None] + np.arange(L)[None, :]
    t_idx = starts[:, None] + L + np.arange(T)[None, :]
    return s[p_idx], s[t_idx]


def make_primes_truths_diff(series_1d, L, T):
    """primes are length L+1 so we can take L diffs."""
    s = np.asarray(series_1d, dtype=np.float64)
    n_w = max(0, len(s) - L - T)
    if n_w == 0:
        return np.empty((0, L + 1)), np.empty((0, T))
    starts = np.arange(n_w)
    p_idx = starts[:, None] + np.arange(L + 1)[None, :]
    t_idx = starts[:, None] + (L + 1) + np.arange(T)[None, :]
    return s[p_idx], s[t_idx]


def eval_one(state_pool, lookback_src, target_series, L, T, mode, k):
    """Run one (mode, k) variant and return (mse, mae) on the given target.

    state_pool, lookback_src, target_series follow GDC's convention:
      - state_pool: 1-D series we search over (the historical pool)
      - lookback_src: tail of this provides the start of each prime
      - target_series: predictions are made over windows of this region
    """
    if mode == 'raw':
        full = np.concatenate([lookback_src[-L:], target_series])
        primes, truths = make_primes_truths(full, L, T)
        if primes.shape[0] == 0:
            return float('nan'), float('nan')
        fc = forecast_many_parrot(state_pool, primes, T, k=k,
                                   device=DEVICE, dtype=DTYPE)
    else:
        full = np.concatenate([lookback_src[-(L + 1):], target_series])
        primes, truths = make_primes_truths_diff(full, L, T)
        if primes.shape[0] == 0:
            return float('nan'), float('nan')
        fc = forecast_many_parrot_diff(state_pool, primes, T, k=k,
                                        device=DEVICE, dtype=DTYPE)
    if torch.is_tensor(fc):
        fc = fc.detach().cpu().numpy().astype(np.float64)
    diff = truths - fc
    return float((diff ** 2).mean()), float(np.abs(diff).mean())


def sweep_dataset(name, L, horizons):
    print(f"\n=== {name} (L={L}) ===")
    train, val, test, mu, sd = load_univariate(name)
    state_train     = train
    state_train_val = np.concatenate([train, val])
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")

    rows = []
    for T in horizons:
        # --- val tuning: search over train, score on val ---
        val_results = []
        for vname, mode, k in VARIANTS:
            v_mse, v_mae = eval_one(state_train, train, val, L, T, mode, k)
            val_results.append((v_mse, v_mae, vname, mode, k))
        val_results.sort(key=lambda r: r[0])
        v_mse, v_mae, vname, mode, k = val_results[0]

        # --- test eval with val-picked variant ---
        t_mse, t_mae = eval_one(state_train_val, val, test, L, T, mode, k)
        rows.append((name, T, vname, v_mse, t_mse, t_mae))
        # diagnostic
        ranking = '  '.join(f"{r[2]}={r[0]:.3f}" for r in val_results)
        print(f"  T={T:>3d}  val ranks: {ranking}")
        print(f"           PICK {vname:<16s}  val MSE={v_mse:.3f}  "
              f"test MSE={t_mse:.3f}  MAE={t_mae:.3f}")
    return rows


def main():
    print(f"=== Parroting val-tuned sweep ===")
    print(f"Device: {DEVICE}  dtype={DTYPE}")
    t0 = time.time()
    all_rows = []
    for name, L, horizons in SWEEPS:
        all_rows += sweep_dataset(name, L, horizons)
    print(f"\nTotal: {time.time() - t0:.1f}s")

    out = os.path.join(HERE, 'results', 'parrot_valtuned.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'T', 'val_pick', 'val_mse', 'test_mse', 'test_mae'])
        for r in all_rows: w.writerow(r)
    print(f"Wrote {out}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
