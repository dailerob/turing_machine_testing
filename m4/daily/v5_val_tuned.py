"""V5: per-series val-tuned GDC-TS-on-diffs for M4 daily.

Holds out last HORIZON of train as validation, picks best config per
series, scores on test. Candidate set spans:
  - naive_last (random walk fallback)
  - GDC-diff with alpha in {1.0, 0.99, 0.95, 0.9}, theta in {0.0, 0.05},
    window_len in {7, 14, 28, 56}, sigma_frac in {0.25, 0.50, 1.00}
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
from generative_dense_chain_timeseries import GenerativeDenseChainTimeSeries  # noqa: E402
from v0_baselines import smape, naive_last, HORIZON


def gdc_diff_forecast(train, window_len=14, sigma_frac=0.50,
                      alpha=1.0, theta=0.0, h=HORIZON):
    n = len(train)
    if n < window_len + h + 2:
        return naive_last(train, h)
    d = np.diff(train)
    if len(d) < window_len + h:
        return naive_last(train, h)
    sigma_per_step = float(np.std(d)) * sigma_frac
    sigma_gdc = sigma_per_step * np.sqrt(window_len)
    beta = max(sigma_gdc ** 2, 1e-9)
    states = d.reshape(-1, 1)
    gdc = GenerativeDenseChainTimeSeries(
        states, beta=beta, alpha=alpha, theta=theta,
        transition_type='self_loop',
        terminal_behavior='absorb',
        initial_dist='uniform',
    )
    prime = d[-window_len:].reshape(-1, 1)
    _, state_dists = gdc.forecast_gdc_style(prime, n_steps=h)
    nt_mask = (~gdc.terminal_mask).astype(float)
    sd_nt = state_dists * nt_mask[None, :]
    sd_nt_sum = sd_nt.sum(axis=1, keepdims=True)
    safe = np.where(sd_nt_sum > 1e-12, sd_nt_sum, 1.0)
    sd_nt_norm = sd_nt / safe
    forecast_d = (sd_nt_norm @ gdc.states)[:, 0]
    return train[-1] + np.cumsum(forecast_d)


# Build a focused candidate config list: small set to avoid val-overfitting
CONFIGS = [
    ('naive_last',                    dict()),
    ('gdc_L14_s0.50_a1.0_t0.0',       dict(window_len=14, sigma_frac=0.50,
                                           alpha=1.0, theta=0.0)),
    ('gdc_L28_s0.50_a1.0_t0.0',       dict(window_len=28, sigma_frac=0.50,
                                           alpha=1.0, theta=0.0)),
    ('gdc_L56_s0.50_a1.0_t0.0',       dict(window_len=56, sigma_frac=0.50,
                                           alpha=1.0, theta=0.0)),
    ('gdc_L14_s1.00_a1.0_t0.0',       dict(window_len=14, sigma_frac=1.00,
                                           alpha=1.0, theta=0.0)),
]


def predict(name, cfg, train, h=HORIZON):
    if name == 'naive_last':
        return naive_last(train, h)
    return gdc_diff_forecast(train, h=h, **cfg)


def run_series(args):
    sid, train, test = args
    h = HORIZON
    rows = []
    if len(train) < 3 * h + 56:
        # too short to validate; just use naive
        pred = naive_last(train, h)
        rows.append(dict(sid=sid, model='val_tuned', best_cfg='naive_last',
                         val_smape=float('nan'),
                         test_smape=smape(test, pred),
                         naive_smape=smape(test, naive_last(train, h))))
        return rows
    tr = train[:-h]; val = train[-h:]
    val_scores = {}
    for name, cfg in CONFIGS:
        try:
            p = predict(name, cfg, tr, h=h)
            val_scores[name] = smape(val, p)
        except Exception:
            val_scores[name] = float('inf')
    best_name = min(val_scores, key=val_scores.get)
    best_cfg = dict(CONFIGS)[best_name]
    pred = predict(best_name, best_cfg, train, h=h)
    naive_pred = naive_last(train, h)
    rows.append(dict(sid=sid, model='val_tuned', best_cfg=best_name,
                     val_smape=val_scores[best_name],
                     test_smape=smape(test, pred),
                     naive_smape=smape(test, naive_pred)))
    return rows


def main():
    train = dl.load_train("Daily")
    test = dl.load_test("Daily")
    ids = sorted(train.keys(), key=lambda s: int(s[1:]))
    tasks = [(sid, train[sid], test[sid]) for sid in ids]
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    print(f"Val-tuned GDC-diff on {len(tasks)} series, {len(CONFIGS)} configs, "
          f"{n_workers} workers", flush=True)
    all_rows = []
    t0 = time.time()
    done = 0
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(run_series, tasks, chunksize=4):
            all_rows.extend(r); done += 1
            if done % 200 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} [{time.time()-t0:.0f}s]",
                      flush=True)

    out_csv = os.path.join(HERE, "v5_val_tuned_results.csv")
    fields = list(all_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"Wrote {out_csv}")

    test_v = np.array([r['test_smape'] for r in all_rows])
    naive_v = np.array([r['naive_smape'] for r in all_rows])
    print(f"\n=== V5 val-tuned vs naive_last on M4 daily ({len(ids)} series) ===")
    print(f"  V5 val-tuned  mean={test_v.mean():.2f}%  "
          f"median={np.median(test_v):.2f}%  "
          f"p25={np.percentile(test_v, 25):.2f}%  "
          f"p75={np.percentile(test_v, 75):.2f}%")
    print(f"  naive_last    mean={naive_v.mean():.2f}%  "
          f"median={np.median(naive_v):.2f}%")
    print(f"  V5 beats naive: "
          f"{int((test_v < naive_v).sum())}/{len(test_v)}")

    from collections import Counter
    pick_counts = Counter(r['best_cfg'] for r in all_rows)
    print(f"\n=== Top picked configs ===")
    for name, c in pick_counts.most_common(15):
        print(f"  {name:>30s}: {c}")


if __name__ == "__main__":
    main()
