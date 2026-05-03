"""Validation-tuned per-series benchmark.

For each series:
  1. Hold out the last 48 hours of training as a validation forecast.
  2. Sweep a grid of (window_len, sigma_frac) configs by training on
     train[:-48] and forecasting 48 steps; score against train[-48:].
  3. Select the config with the lowest validation sMAPE.
  4. Refit on the full training and forecast for the actual test.

This is the honest analog of the per-series oracle from the full
benchmark — no test-set leakage.
"""
from __future__ import annotations
import os
import sys
import time
import csv
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M4_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(M4_ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, M4_ROOT)
sys.path.insert(0, ROOT)

import data_loader as dl
from v0_basic_gdc import smape, naive_seasonal_forecast, H_HORIZON

# Wider grid since GDC-proper is cheap
CONFIGS = []
for L in [24, 48, 72, 168]:
    for s in [0.02, 0.05, 0.10, 0.20]:
        CONFIGS.append(dict(window_len=L, sigma_frac=s))

OUT_CSV = os.path.join(HERE, 'val_tuned_results.csv')


def run_series(args):
    sid, train, test = args
    sys.path.insert(0, HERE)
    sys.path.insert(0, ROOT)
    from v5_gdc_proper import gdc_proper_forecast

    rows = []
    # Hold out last H_HORIZON of training for validation
    if len(train) <= 2 * H_HORIZON:
        # Series too short for validation split — use single best config
        # heuristic (L=48, σ=0.10)
        best_cfg = dict(window_len=48, sigma_frac=0.10)
        val_smape = float('nan')
        pred_test, _ = gdc_proper_forecast(train, **best_cfg, h=H_HORIZON)
        rows.append(dict(sid=sid, **best_cfg,
                         val_smape=val_smape,
                         test_smape=smape(test, pred_test),
                         smape_naive=smape(test,
                            naive_seasonal_forecast(train, H_HORIZON, 24)),
                         note='too-short-for-val'))
        return rows

    train_train = train[:-H_HORIZON]
    train_val = train[-H_HORIZON:]

    # Validation sweep
    val_results = []
    for cfg in CONFIGS:
        try:
            pred_val, _ = gdc_proper_forecast(train_train, **cfg, h=H_HORIZON)
            sm = smape(train_val, pred_val)
        except Exception:
            sm = float('inf')
        val_results.append((sm, cfg))

    # Pick best config by validation sMAPE
    best_val_sm, best_cfg = min(val_results, key=lambda x: x[0])

    # Refit on full training, forecast test
    pred_test, _ = gdc_proper_forecast(train, **best_cfg, h=H_HORIZON)
    test_sm = smape(test, pred_test)
    naive_sm = smape(test, naive_seasonal_forecast(train, H_HORIZON, 24))

    rows.append(dict(sid=sid, window_len=best_cfg['window_len'],
                     sigma_frac=best_cfg['sigma_frac'],
                     val_smape=best_val_sm, test_smape=test_sm,
                     smape_naive=naive_sm, note='val-tuned'))
    return rows


def main():
    train_d = dl.load_train()
    test_d = dl.load_test()
    ids = sorted(train_d.keys(), key=lambda s: int(s[1:]))
    tasks = [(sid, train_d[sid], test_d[sid]) for sid in ids]

    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    print(f"Validation-tuned benchmark on {len(tasks)} series, "
          f"{len(CONFIGS)} configs each, {n_workers} workers", flush=True)

    all_rows = []
    t0 = time.time()
    done = 0
    with mp.Pool(processes=n_workers) as pool:
        for series_rows in pool.imap_unordered(run_series, tasks, chunksize=4):
            all_rows.extend(series_rows)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} [{time.time()-t0:.0f}s]",
                      flush=True)

    fields = ['sid', 'window_len', 'sigma_frac', 'val_smape',
              'test_smape', 'smape_naive', 'note']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"Wrote {OUT_CSV}", flush=True)

    # Aggregate
    test_sm = np.array([r['test_smape'] for r in all_rows])
    naive_sm = np.array([r['smape_naive'] for r in all_rows])
    print(f"\n=== Validation-tuned GDC vs naive (all {len(all_rows)} series) ===")
    print(f"  GDC-tuned  mean={np.mean(test_sm):>6.2f}%  "
          f"median={np.median(test_sm):>6.2f}%  "
          f"p25={np.percentile(test_sm, 25):>5.2f}%  "
          f"p75={np.percentile(test_sm, 75):>5.2f}%")
    print(f"  naive      mean={np.mean(naive_sm):>6.2f}%  "
          f"median={np.median(naive_sm):>6.2f}%  "
          f"p25={np.percentile(naive_sm, 25):>5.2f}%  "
          f"p75={np.percentile(naive_sm, 75):>5.2f}%")
    print(f"  GDC beats naive: {int((test_sm < naive_sm).sum())}/{len(test_sm)}")
    print(f"  GDC ties naive (within 0.5pp): "
          f"{int(np.abs(test_sm - naive_sm).mean() < 0.5)}/{len(test_sm)}")

    # Per-config selection breakdown
    from collections import Counter
    sel = Counter((r['window_len'], r['sigma_frac']) for r in all_rows)
    print(f"\n=== Config selection frequency (val-tuned) ===")
    for cfg, count in sorted(sel.items(), key=lambda x: -x[1]):
        print(f"  L={cfg[0]:>3d}, s%={cfg[1]:.2f}: {count:>3d} series "
              f"({100*count/len(all_rows):>5.1f}%)")


if __name__ == "__main__":
    main()
