"""V1: NN-matching adapted for M4 daily.

Same recipe as hourly v4: take the last L observations as the prime,
find similar L-windows in training history, average their next-h
continuations weighted by Gaussian similarity.

Key challenge for daily (vs hourly):
  - Many series are trending/drifting (no stationary cycle)
  - No strong periodicity (weekly cycle is weak in most series)
  - Series can be very long (median 2940 days vs 960 hourly)

To handle trends, we offer two prediction modes:
  - 'absolute': forecast = sum_i w_i * train[i+L+h]  (raw values)
  - 'detrended': subtract a local trend estimated from prime,
    forecast residuals via NN, add trend back

The detrended mode should help on series with strong drift.
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
from v0_baselines import smape, naive_last, HORIZON


def nn_forecast(train, window_len=14, sigma_frac=0.10, h=HORIZON,
                detrend=False):
    n = len(train)
    if n < window_len + h:
        return naive_last(train, h)
    sigma = float(np.std(train)) * sigma_frac
    sigma2 = max(sigma ** 2, 1e-9)
    n_windows = n - window_len - h + 1
    indices = np.arange(window_len)[None, :] + np.arange(n_windows)[:, None]
    W = train[indices]
    q = train[-window_len:]
    # Optionally detrend each window so matching is shape-based
    if detrend:
        # Subtract per-window linear trend
        x = np.arange(window_len, dtype=np.float64)
        x_mean = x.mean()
        x_var = float(((x - x_mean) ** 2).sum())
        # Per-window slope and intercept
        W_mean = W.mean(axis=1, keepdims=True)
        slopes = ((W - W_mean) * (x - x_mean)).sum(axis=1) / x_var
        intercepts = W_mean.squeeze() - slopes * x_mean
        W_resid = W - (slopes[:, None] * x[None, :] + intercepts[:, None])
        q_mean = q.mean()
        q_slope = float(((q - q_mean) * (x - x_mean)).sum() / x_var)
        q_intercept = q_mean - q_slope * x_mean
        q_resid = q - (q_slope * x + q_intercept)
        # Distance in residual space
        diff = W_resid - q_resid[None, :]
    else:
        diff = W - q[None, :]
    dist2 = np.sum(diff ** 2, axis=1) / window_len
    log_w = -0.5 * dist2 / sigma2
    log_w -= log_w.max()
    w = np.exp(log_w)
    w_sum = w.sum()
    if w_sum <= 0:
        w = np.ones(n_windows) / n_windows
    else:
        w = w / w_sum
    cont_idx = (np.arange(window_len, window_len + h)[None, :]
                + np.arange(n_windows)[:, None])
    cont = train[cont_idx]
    if detrend:
        # Continuations need their own trend extrapolated
        # Trend = (slope_i, intercept_i) at window i extrapolated to x in [L, L+h)
        x_fut = np.arange(window_len, window_len + h, dtype=np.float64)
        cont_trend = (slopes[:, None] * x_fut[None, :]
                      + intercepts[:, None])
        cont_resid = cont - cont_trend
        # Extrapolate query trend
        q_trend_fut = q_slope * x_fut + q_intercept
        forecast_resid = w @ cont_resid
        forecast = q_trend_fut + forecast_resid
    else:
        forecast = w @ cont
    return forecast


def run_series(args):
    sid, train, test, configs = args
    rows = []
    for cfg in configs:
        pred = nn_forecast(train, **cfg)
        rows.append(dict(sid=sid, **cfg,
                         smape=smape(test, pred)))
    return rows


def main():
    train = dl.load_train("Daily")
    test = dl.load_test("Daily")
    ids = sorted(train.keys(), key=lambda s: int(s[1:]))

    # Try a small grid
    configs = [
        dict(window_len=14, sigma_frac=0.10, detrend=False),
        dict(window_len=28, sigma_frac=0.10, detrend=False),
        dict(window_len=56, sigma_frac=0.10, detrend=False),
        dict(window_len=14, sigma_frac=0.10, detrend=True),
        dict(window_len=28, sigma_frac=0.10, detrend=True),
        dict(window_len=56, sigma_frac=0.10, detrend=True),
    ]

    tasks = [(sid, train[sid], test[sid], configs) for sid in ids]
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    print(f"NN-matching on {len(tasks)} series, {len(configs)} configs, "
          f"{n_workers} workers", flush=True)
    all_rows = []
    t0 = time.time()
    done = 0
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(run_series, tasks, chunksize=16):
            all_rows.extend(r); done += 1
            if done % 500 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} [{time.time()-t0:.0f}s]",
                      flush=True)

    out_csv = os.path.join(HERE, "v1_nn_results.csv")
    fields = ["sid", "window_len", "sigma_frac", "detrend", "smape"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"Wrote {out_csv}")

    # Aggregate
    print(f"\n=== NN-matching on M4 daily ({len(ids)} series) ===")
    print(f"{'config':>40s}  {'mean':>7s}  {'median':>7s}  {'p25':>7s}  {'p75':>7s}")
    from collections import defaultdict
    by_cfg = defaultdict(list)
    for r in all_rows:
        key = (r['window_len'], r['sigma_frac'], r['detrend'])
        by_cfg[key].append(r['smape'])
    for cfg in configs:
        key = (cfg['window_len'], cfg['sigma_frac'], cfg['detrend'])
        v = np.array(by_cfg[key])
        tag = f"L={cfg['window_len']:>3d}, s%={cfg['sigma_frac']:.2f}, detrend={cfg['detrend']}"
        print(f"{tag:>40s}  {v.mean():>6.2f}%  {np.median(v):>6.2f}%  "
              f"{np.percentile(v, 25):>6.2f}%  "
              f"{np.percentile(v, 75):>6.2f}%")

    # Compare to naive_last as reference
    naive_v = np.array([smape(test[sid], naive_last(train[sid])) for sid in ids])
    print(f"\n  naive_last reference: mean={naive_v.mean():.2f}%, "
          f"median={np.median(naive_v):.2f}%")


if __name__ == "__main__":
    main()
