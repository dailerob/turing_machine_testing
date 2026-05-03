"""Run GDC-proper on all 414 M4 hourly series and compute the
benchmark sMAPE metric.

Reference points from the M4 competition (hourly subset):
  Naive (last value)          ~ 21%
  Naive seasonal (last 24h)   ~ 14%
  Statistical ensembles       ~ 11-12%
  Competition winner          ~ 9-10%

We run a small grid of (window_len, sigma_frac) at fixed config so we
can tell whether one config is universally good or per-series tuning
is needed.
"""
from __future__ import annotations
import os
import sys
import time
import multiprocessing as mp
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M4_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(M4_ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, M4_ROOT)
sys.path.insert(0, ROOT)

import data_loader as dl
from v0_basic_gdc import smape, naive_seasonal_forecast, H_HORIZON

# Import these inside worker to avoid pickling issues on Windows
CONFIGS = [
    dict(window_len=24,  sigma_frac=0.10),
    dict(window_len=48,  sigma_frac=0.10),
    dict(window_len=168, sigma_frac=0.10),
    dict(window_len=24,  sigma_frac=0.05),
    dict(window_len=168, sigma_frac=0.05),
]

OUT_CSV = os.path.join(HERE, 'full_benchmark_results.csv')


def run_series(args):
    sid, train, test = args
    sys.path.insert(0, HERE)
    sys.path.insert(0, ROOT)
    from v5_gdc_proper import gdc_proper_forecast
    rows = []
    naive = naive_seasonal_forecast(train, H_HORIZON, period=24)
    smape_naive = smape(test, naive)
    for cfg in CONFIGS:
        try:
            pred, _ = gdc_proper_forecast(train, **cfg, h=H_HORIZON)
            sm = smape(test, pred)
        except Exception as e:
            sys.stderr.write(f"{sid} cfg {cfg}: {e}\n")
            sm = float('nan')
        rows.append(dict(sid=sid, window_len=cfg['window_len'],
                         sigma_frac=cfg['sigma_frac'],
                         smape_gdc=sm, smape_naive=smape_naive,
                         len_train=len(train)))
    return rows


def main():
    train_d = dl.load_train()
    test_d = dl.load_test()
    ids = sorted(train_d.keys(), key=lambda s: int(s[1:]))
    tasks = [(sid, train_d[sid], test_d[sid]) for sid in ids]

    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    print(f"Running on {len(tasks)} series with {len(CONFIGS)} configs "
          f"on {n_workers} workers...", flush=True)
    all_rows = []
    t0 = time.time()
    done = 0
    with mp.Pool(processes=n_workers) as pool:
        for series_rows in pool.imap_unordered(run_series, tasks, chunksize=4):
            all_rows.extend(series_rows)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} series  "
                      f"[{time.time()-t0:.0f}s]", flush=True)

    fields = ['sid', 'window_len', 'sigma_frac', 'smape_gdc',
              'smape_naive', 'len_train']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {OUT_CSV}  [{time.time()-t0:.1f}s, {len(all_rows)} rows]",
          flush=True)

    # Aggregate
    print("\n=== Mean / median sMAPE by config (414 series) ===")
    print(f"{'config':>30s}  {'mean':>7s}  {'median':>7s}  "
          f"{'p25':>7s}  {'p75':>7s}  {'beats naive':>12s}")
    by_cfg = {}
    for r in all_rows:
        key = (r['window_len'], r['sigma_frac'])
        by_cfg.setdefault(key, []).append(r)
    for cfg in CONFIGS:
        key = (cfg['window_len'], cfg['sigma_frac'])
        sub = by_cfg[key]
        gdc = np.array([r['smape_gdc'] for r in sub])
        naive = np.array([r['smape_naive'] for r in sub])
        beats = int((gdc < naive).sum())
        tag = f"L={cfg['window_len']}, s%={cfg['sigma_frac']:.2f}"
        print(f"{tag:>30s}  "
              f"{np.nanmean(gdc):>6.2f}%  {np.nanmedian(gdc):>6.2f}%  "
              f"{np.nanpercentile(gdc, 25):>6.2f}%  "
              f"{np.nanpercentile(gdc, 75):>6.2f}%  "
              f"{beats:>4d}/414")
    # Naive baseline single number
    naive_all = np.array([by_cfg[(CONFIGS[0]['window_len'], CONFIGS[0]['sigma_frac'])][i]
                          ['smape_naive'] for i in range(len(by_cfg[next(iter(by_cfg))]))])
    print(f"\n  naive seasonal mean = {np.nanmean(naive_all):.2f}%, "
          f"median = {np.nanmedian(naive_all):.2f}%")

    # Per-series best
    by_sid = {}
    for r in all_rows:
        by_sid.setdefault(r['sid'], []).append(r)
    best_smape = []
    for sid, rs in by_sid.items():
        best = min(rs, key=lambda x: x['smape_gdc'])
        best_smape.append(best['smape_gdc'])
    best_smape = np.array(best_smape)
    print(f"\n=== Per-series oracle (best of {len(CONFIGS)} configs per series) ===")
    print(f"  mean = {np.nanmean(best_smape):.2f}%, "
          f"median = {np.nanmedian(best_smape):.2f}%, "
          f"p75 = {np.nanpercentile(best_smape, 75):.2f}%")


if __name__ == "__main__":
    main()
