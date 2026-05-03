"""V3: wider sigma sweep on NN-diff + val-tuned per-series selection.

Adds:
  - higher sigmas (1.0, 2.0, inf-equivalent)
  - 'flat' baseline (just the mean of historical diffs - random walk
    with drift, where drift = mean(d))
  - val-tune: hold out last h, pick best config per series, score on test
"""
from __future__ import annotations
import os
import sys
import csv
import time
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M4_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(M4_ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, M4_ROOT)
sys.path.insert(0, ROOT)

import data_loader as dl
from v0_baselines import smape, naive_last, drift, HORIZON
from v2_nn_diff import nn_diff_forecast


def mean_diff_forecast(train, h=HORIZON):
    """train[-1] + cumsum of mean(diff(train)) — random walk with drift."""
    if len(train) < 2:
        return np.full(h, train[-1])
    mu = float(np.mean(np.diff(train)))
    return train[-1] + mu * np.arange(1, h + 1)


CONFIGS = [
    ('naive_last', dict()),
    ('mean_diff',  dict()),
    ('drift',      dict()),
    ('nn_L14_s0.50', dict(window_len=14, sigma_frac=0.50)),
    ('nn_L28_s0.50', dict(window_len=28, sigma_frac=0.50)),
    ('nn_L14_s1.00', dict(window_len=14, sigma_frac=1.00)),
    ('nn_L28_s1.00', dict(window_len=28, sigma_frac=1.00)),
    ('nn_L14_s2.00', dict(window_len=14, sigma_frac=2.00)),
    ('nn_L7_s0.50',  dict(window_len=7,  sigma_frac=0.50)),
]


def predict(name, train, cfg, h=HORIZON):
    if name == 'naive_last':
        return naive_last(train, h)
    if name == 'mean_diff':
        return mean_diff_forecast(train, h)
    if name == 'drift':
        return drift(train, h)
    return nn_diff_forecast(train, h=h, **cfg)


def run_series(args):
    sid, train, test, h = args
    rows = []
    # Held-out validation: split train -> tr, val (last h)
    if len(train) >= 3 * h + 56:
        tr = train[:-h]; val = train[-h:]
        val_scores = {}
        for name, cfg in CONFIGS:
            p = predict(name, tr, cfg, h=h)
            val_scores[name] = smape(val, p)
        best_name = min(val_scores, key=val_scores.get)
    else:
        best_name = 'naive_last'
        val_scores = {n: np.nan for n, _ in CONFIGS}
    # Test predictions for every config plus the val-tuned pick
    for name, cfg in CONFIGS:
        p = predict(name, train, cfg, h=h)
        rows.append(dict(sid=sid, model=name, smape=smape(test, p),
                         val=val_scores.get(name, np.nan),
                         val_picked=int(name == best_name)))
    # Tuned model = the val-best one's test prediction
    cfg = dict(CONFIGS)[best_name]
    p = predict(best_name, train, cfg, h=h)
    rows.append(dict(sid=sid, model='val_tuned', smape=smape(test, p),
                     val=val_scores.get(best_name, np.nan), val_picked=1))
    return rows


def main():
    train = dl.load_train("Daily")
    test = dl.load_test("Daily")
    ids = sorted(train.keys(), key=lambda s: int(s[1:]))
    tasks = [(sid, train[sid], test[sid], HORIZON) for sid in ids]
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    print(f"Sweep + val-tune on {len(tasks)} series, {n_workers} workers",
          flush=True)
    all_rows = []
    t0 = time.time()
    done = 0
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(run_series, tasks, chunksize=16):
            all_rows.extend(r); done += 1
            if done % 1000 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} [{time.time()-t0:.0f}s]",
                      flush=True)

    out_csv = os.path.join(HERE, "v3_results.csv")
    fields = ["sid", "model", "smape", "val", "val_picked"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"Wrote {out_csv}")

    print(f"\n=== Per-model test sMAPE on M4 daily ({len(ids)} series) ===")
    print(f"{'model':>20s}  {'mean':>7s}  {'median':>7s}  {'p25':>7s}  {'p75':>7s}  {'picked':>7s}")
    from collections import defaultdict
    by_m = defaultdict(list)
    pick_counts = defaultdict(int)
    for r in all_rows:
        by_m[r['model']].append(r['smape'])
        if r['val_picked'] and r['model'] != 'val_tuned':
            pick_counts[r['model']] += 1
    for name in [n for n, _ in CONFIGS] + ['val_tuned']:
        v = np.array(by_m[name])
        pc = pick_counts.get(name, '')
        print(f"{name:>20s}  {v.mean():>6.2f}%  {np.median(v):>6.2f}%  "
              f"{np.percentile(v, 25):>6.2f}%  "
              f"{np.percentile(v, 75):>6.2f}%  {str(pc):>7s}")


if __name__ == "__main__":
    main()
