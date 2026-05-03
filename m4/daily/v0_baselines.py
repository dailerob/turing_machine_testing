"""Naive baselines for M4 daily forecasting.

Per the M4 paper, daily series have weak periodicity (most aren't
daily-cycle data — they're financial/macro series with trends).  The
official "Naive2" baseline is "last value" (random walk forecast),
not a seasonal naive.

Baselines tested:
  naive_last:       repeat the last training value 14 times
  naive_last7:      repeat the mean of the last 7 days
  naive_seasonal7:  repeat the last 7 days, twice (h=14, period=7)
  naive_seasonal_decomposed:
                    detrend via OLS, predict trend continuation +
                    last-week residual
  drift:            linear extrapolation from first/last training values
"""
from __future__ import annotations
import os
import sys
import numpy as np
import multiprocessing as mp
import csv
import time

HERE = os.path.dirname(os.path.abspath(__file__))
M4_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(M4_ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, M4_ROOT)
sys.path.insert(0, ROOT)

import data_loader as dl


HORIZON = 14


def smape(actual, forecast):
    actual = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return 100.0 * np.mean(np.abs(actual - forecast) / denom)


def naive_last(train, h=HORIZON):
    return np.full(h, train[-1])


def naive_last7_mean(train, h=HORIZON):
    last7 = train[-7:] if len(train) >= 7 else train
    return np.full(h, last7.mean())


def naive_seasonal7(train, h=HORIZON):
    last7 = train[-7:] if len(train) >= 7 else np.array([train[-1]] * 7)
    return np.tile(last7, int(np.ceil(h / 7)))[:h]


def drift(train, h=HORIZON):
    """Naive drift: linear extrapolation of (first, last) line."""
    if len(train) < 2:
        return np.full(h, train[-1])
    slope = (train[-1] - train[0]) / (len(train) - 1)
    return train[-1] + slope * np.arange(1, h + 1)


BASELINES = {
    "naive_last": naive_last,
    "naive_last7_mean": naive_last7_mean,
    "naive_seasonal7": naive_seasonal7,
    "drift": drift,
}


def run_series(args):
    sid, train, test = args
    out = []
    for name, fn in BASELINES.items():
        pred = fn(train)
        out.append(dict(sid=sid, model=name, smape=smape(test, pred)))
    return out


def main():
    train = dl.load_train("Daily")
    test = dl.load_test("Daily")
    ids = sorted(train.keys(), key=lambda s: int(s[1:]))
    tasks = [(sid, train[sid], test[sid]) for sid in ids]
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    print(f"Running {len(BASELINES)} baselines on {len(tasks)} series, "
          f"{n_workers} workers", flush=True)
    all_rows = []
    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(run_series, tasks, chunksize=32):
            all_rows.extend(r)
    print(f"Done in {time.time()-t0:.1f}s, {len(all_rows)} rows", flush=True)

    # Aggregate
    print(f"\n=== Per-baseline mean / median sMAPE on {len(tasks)} daily series ===")
    print(f"{'baseline':>20s}  {'mean':>7s}  {'median':>7s}  "
          f"{'p25':>7s}  {'p75':>7s}")
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in all_rows:
        by_model[r['model']].append(r['smape'])
    for name in BASELINES:
        v = np.array(by_model[name])
        print(f"{name:>20s}  {v.mean():>6.2f}%  {np.median(v):>6.2f}%  "
              f"{np.percentile(v, 25):>6.2f}%  "
              f"{np.percentile(v, 75):>6.2f}%")

    out_csv = os.path.join(HERE, "baselines_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sid", "model", "smape"])
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
