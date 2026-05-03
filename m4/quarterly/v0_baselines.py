"""Naive baselines for M4 quarterly forecasting (h=8, season=4)."""
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


HORIZON = 8
SEASON = 4


def smape(actual, forecast):
    a = np.asarray(actual, dtype=np.float64); f = np.asarray(forecast, dtype=np.float64)
    denom = (np.abs(a) + np.abs(f)) / 2.0
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return 100.0 * np.mean(np.abs(a - f) / denom)


def naive_last(train, h=HORIZON):
    return np.full(h, train[-1])


def drift(train, h=HORIZON):
    if len(train) < 2: return np.full(h, train[-1])
    slope = (train[-1] - train[0]) / (len(train) - 1)
    return train[-1] + slope * np.arange(1, h + 1)


def naive_seasonal(train, h=HORIZON, period=SEASON):
    if len(train) < period: return naive_last(train, h)
    last = train[-period:]
    return np.tile(last, int(np.ceil(h / period)))[:h]


BASELINES = {"naive_last": naive_last, "drift": drift,
             "naive_seasonal4": naive_seasonal}


def run_series(args):
    sid, train, test = args
    return [dict(sid=sid, model=n, smape=smape(test, fn(train)))
            for n, fn in BASELINES.items()]


def main():
    train = dl.load_train("Quarterly"); test = dl.load_test("Quarterly")
    ids = sorted(train.keys(), key=lambda s: int(s[1:]))
    tasks = [(sid, train[sid], test[sid]) for sid in ids]
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    print(f"{len(BASELINES)} baselines on {len(tasks)} series, {n_workers} workers")
    all_rows = []
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(run_series, tasks, chunksize=64):
            all_rows.extend(r)

    print(f"\n=== Per-baseline sMAPE on {len(tasks)} quarterly series ===")
    print(f"{'baseline':>20s}  {'mean':>7s}  {'median':>7s}")
    from collections import defaultdict
    by_m = defaultdict(list)
    for r in all_rows: by_m[r['model']].append(r['smape'])
    for n in BASELINES:
        v = np.array(by_m[n])
        print(f"{n:>20s}  {v.mean():>6.2f}%  {np.median(v):>6.2f}%")
    out_csv = os.path.join(HERE, "baselines_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sid","model","smape"])
        w.writeheader(); w.writerows(all_rows)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
