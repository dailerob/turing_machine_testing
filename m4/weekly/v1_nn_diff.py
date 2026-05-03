"""V1: NN-matching on 1-step diffs for M4 weekly (daily's winning recipe)."""
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
from v0_baselines import smape, naive_last, HORIZON


def nn_diff_forecast(train, window_len=13, sigma_frac=0.50, h=HORIZON):
    n = len(train)
    if n < window_len + h + 2:
        return naive_last(train, h)
    d = np.diff(train)
    nd = len(d)
    n_windows = nd - window_len - h + 1
    if n_windows <= 0:
        return naive_last(train, h)
    sigma = float(np.std(d)) * sigma_frac
    sigma2 = max(sigma ** 2, 1e-9)
    indices = np.arange(window_len)[None, :] + np.arange(n_windows)[:, None]
    W = d[indices]
    q = d[-window_len:]
    diff = W - q[None, :]
    dist2 = np.sum(diff ** 2, axis=1) / window_len
    log_w = -0.5 * dist2 / sigma2
    log_w -= log_w.max()
    w = np.exp(log_w); s = w.sum()
    w = w / s if s > 0 else np.ones(n_windows) / n_windows
    cont_idx = (np.arange(window_len, window_len + h)[None, :]
                + np.arange(n_windows)[:, None])
    cont = d[cont_idx]
    forecast_d = w @ cont
    return train[-1] + np.cumsum(forecast_d)


CONFIGS = []
for L in [4, 8, 13, 26, 52]:
    for s in [0.10, 0.25, 0.50, 1.00, 2.00]:
        CONFIGS.append(dict(window_len=L, sigma_frac=s))


def run_series(args):
    sid, train, test = args
    rows = []
    for cfg in CONFIGS:
        pred = nn_diff_forecast(train, **cfg)
        rows.append(dict(sid=sid, **cfg, smape=smape(test, pred)))
    return rows


def main():
    train = dl.load_train("Weekly"); test = dl.load_test("Weekly")
    ids = sorted(train.keys(), key=lambda s: int(s[1:]))
    tasks = [(sid, train[sid], test[sid]) for sid in ids]
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    print(f"NN-diff on {len(tasks)} series, {len(CONFIGS)} configs, "
          f"{n_workers} workers", flush=True)
    all_rows = []
    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(run_series, tasks, chunksize=8):
            all_rows.extend(r)
    print(f"Done in {time.time()-t0:.1f}s", flush=True)

    out_csv = os.path.join(HERE, "v1_nn_diff_results.csv")
    fields = ["sid", "window_len", "sigma_frac", "smape"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"Wrote {out_csv}")

    print(f"\n=== NN-diff on M4 weekly ({len(ids)} series) ===")
    print(f"{'config':>30s}  {'mean':>7s}  {'median':>7s}  {'p25':>7s}  {'p75':>7s}")
    from collections import defaultdict
    by_cfg = defaultdict(list)
    for r in all_rows:
        by_cfg[(r['window_len'], r['sigma_frac'])].append(r['smape'])
    for cfg in CONFIGS:
        key = (cfg['window_len'], cfg['sigma_frac'])
        v = np.array(by_cfg[key])
        tag = f"L={cfg['window_len']:>3d}, s%={cfg['sigma_frac']:.2f}"
        print(f"{tag:>30s}  {v.mean():>6.2f}%  {np.median(v):>6.2f}%  "
              f"{np.percentile(v, 25):>6.2f}%  "
              f"{np.percentile(v, 75):>6.2f}%")

    naive_v = np.array([smape(test[sid], naive_last(train[sid])) for sid in ids])
    print(f"\n  naive_last reference: mean={naive_v.mean():.2f}%, "
          f"median={np.median(naive_v):.2f}%")


if __name__ == "__main__":
    main()
