"""Naive baselines for M4 monthly forecasting (h=18, season=12)."""
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
sys.path.insert(0, HERE); sys.path.insert(0, M4_ROOT); sys.path.insert(0, ROOT)
import data_loader as dl


HORIZON = 18
SEASON = 12


def smape(actual, forecast):
    actual = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return 100.0 * np.mean(np.abs(actual - forecast) / denom)


def naive_last(train, h=HORIZON):
    return np.full(h, train[-1])


def drift(train, h=HORIZON):
    if len(train) < 2:
        return np.full(h, train[-1])
    slope = (train[-1] - train[0]) / (len(train) - 1)
    return train[-1] + slope * np.arange(1, h + 1)


def naive_seasonal(train, h=HORIZON, period=SEASON):
    """M4's "Naive2" for monthly: repeat the last `period` values."""
    if len(train) < period:
        return naive_last(train, h)
    last = train[-period:]
    return np.tile(last, int(np.ceil(h / period)))[:h]


def naive_seasonal_drift(train, h=HORIZON, period=SEASON):
    """Seasonal naive + linear drift estimated over (last - period - first)."""
    if len(train) < 2 * period:
        return naive_seasonal(train, h, period)
    seas = naive_seasonal(train, h, period)
    # Estimate drift over (n - period) to anchor at 'last full period before'
    n = len(train)
    slope = (train[-1] - train[-period - 1]) / period
    return seas + slope * np.arange(1, h + 1)


BASELINES = {
    "naive_last": naive_last,
    "drift": drift,
    "naive_seasonal12": naive_seasonal,
    "naive_seasonal12_drift": naive_seasonal_drift,
}


def run_series(args):
    sid, train, test = args
    return [dict(sid=sid, model=name, smape=smape(test, fn(train)))
            for name, fn in BASELINES.items()]


def main():
    train = dl.load_train("Monthly"); test = dl.load_test("Monthly")
    ids = sorted(train.keys(), key=lambda s: int(s[1:]))
    tasks = [(sid, train[sid], test[sid]) for sid in ids]
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    print(f"{len(BASELINES)} baselines on {len(tasks)} series, "
          f"{n_workers} workers", flush=True)
    all_rows = []
    t0 = time.time()
    done = 0
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(run_series, tasks, chunksize=64):
            all_rows.extend(r); done += 1
            if done % 5000 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} [{time.time()-t0:.0f}s]",
                      flush=True)
    print(f"Done in {time.time()-t0:.1f}s", flush=True)

    print(f"\n=== Per-baseline sMAPE on {len(tasks)} monthly series ===")
    print(f"{'baseline':>25s}  {'mean':>7s}  {'median':>7s}  "
          f"{'p25':>7s}  {'p75':>7s}")
    from collections import defaultdict
    by_m = defaultdict(list)
    for r in all_rows:
        by_m[r['model']].append(r['smape'])
    for name in BASELINES:
        v = np.array(by_m[name])
        print(f"{name:>25s}  {v.mean():>6.2f}%  {np.median(v):>6.2f}%  "
              f"{np.percentile(v, 25):>6.2f}%  "
              f"{np.percentile(v, 75):>6.2f}%")

    out_csv = os.path.join(HERE, "baselines_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sid", "model", "smape"])
        w.writeheader(); w.writerows(all_rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
