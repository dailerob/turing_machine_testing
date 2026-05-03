"""Compute the per-series oracle on the same 16-config grid as the
val_tuned benchmark, for an apples-to-apples comparison."""
from __future__ import annotations
import os, sys, csv, time, multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M4_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(M4_ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, M4_ROOT)
sys.path.insert(0, ROOT)

import data_loader as dl
from v0_basic_gdc import smape, naive_seasonal_forecast, H_HORIZON

CONFIGS = []
for L in [24, 48, 72, 168]:
    for s in [0.02, 0.05, 0.10, 0.20]:
        CONFIGS.append(dict(window_len=L, sigma_frac=s))


def run_series(args):
    sid, train, test = args
    sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
    from v5_gdc_proper import gdc_proper_forecast
    test_smapes = []
    for cfg in CONFIGS:
        try:
            pred, _ = gdc_proper_forecast(train, **cfg, h=H_HORIZON)
            test_smapes.append(smape(test, pred))
        except Exception:
            test_smapes.append(float('inf'))
    return sid, test_smapes


def main():
    train_d = dl.load_train()
    test_d = dl.load_test()
    ids = sorted(train_d.keys(), key=lambda s: int(s[1:]))
    tasks = [(sid, train_d[sid], test_d[sid]) for sid in ids]

    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    t0 = time.time()
    all_results = {}
    with mp.Pool(processes=n_workers) as pool:
        for sid, sm_list in pool.imap_unordered(run_series, tasks, chunksize=4):
            all_results[sid] = sm_list
    print(f"Computed {len(all_results)} series in {time.time()-t0:.1f}s")

    oracle = np.array([min(all_results[sid]) for sid in ids])
    print(f"\n=== 16-config oracle (best test sMAPE per series) ===")
    print(f"  mean   = {oracle.mean():.2f}%")
    print(f"  median = {np.median(oracle):.2f}%")
    print(f"  p25    = {np.percentile(oracle, 25):.2f}%")
    print(f"  p75    = {np.percentile(oracle, 75):.2f}%")

    # Compare to val-tuned numbers
    val_lookup = {}
    with open(os.path.join(HERE, 'val_tuned_results.csv')) as f:
        for r in csv.DictReader(f):
            val_lookup[r['sid']] = float(r['test_smape'])
    val_arr = np.array([val_lookup[sid] for sid in ids])

    print(f"\n=== Val-tuned vs 16-config-oracle gap ===")
    print(f"  val-tuned mean       = {val_arr.mean():.2f}%")
    print(f"  oracle mean          = {oracle.mean():.2f}%")
    print(f"  gap (val to oracle)  = {val_arr.mean() - oracle.mean():.2f}pp")
    n_optimum = int((val_arr <= oracle + 1e-9).sum())
    print(f"  validation picked optimum config: {n_optimum}/{len(ids)} series "
          f"({100*n_optimum/len(ids):.1f}%)")
    # Also show per-series ranking
    avg_gap = (val_arr - oracle)
    print(f"  median per-series gap = {np.median(avg_gap):.2f}pp")
    print(f"  max per-series gap    = {avg_gap.max():.2f}pp")


if __name__ == "__main__":
    main()
