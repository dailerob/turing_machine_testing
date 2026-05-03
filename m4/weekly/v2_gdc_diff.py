"""V2: GDC-TS (absorb mode) on 1-step diffs for M4 weekly."""
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


def gdc_diff_forecast(train, window_len=26, sigma_frac=0.25,
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


CONFIGS = []
for L in [8, 13, 26, 52]:
    for s in [0.10, 0.25, 0.50]:
        for a in [1.0, 0.99, 0.95]:
            for th in [0.0, 0.05]:
                if a == 1.0 and th != 0.0:
                    continue
                CONFIGS.append(dict(window_len=L, sigma_frac=s,
                                    alpha=a, theta=th))


def run_series(args):
    sid, train, test = args
    rows = []
    for cfg in CONFIGS:
        try:
            pred = gdc_diff_forecast(train, **cfg)
            sm = smape(test, pred)
        except Exception:
            sm = float('nan')
        rows.append(dict(sid=sid, **cfg, smape=sm))
    return rows


def main():
    train = dl.load_train("Weekly"); test = dl.load_test("Weekly")
    ids = sorted(train.keys(), key=lambda s: int(s[1:]))
    tasks = [(sid, train[sid], test[sid]) for sid in ids]
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    print(f"GDC-diff on {len(tasks)} series, {len(CONFIGS)} configs, "
          f"{n_workers} workers", flush=True)
    all_rows = []
    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(run_series, tasks, chunksize=4):
            all_rows.extend(r)
    print(f"Done in {time.time()-t0:.1f}s", flush=True)

    out_csv = os.path.join(HERE, "v2_gdc_diff_results.csv")
    fields = ["sid", "window_len", "sigma_frac", "alpha", "theta", "smape"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"Wrote {out_csv}")

    print(f"\n=== GDC-TS on diffs, M4 weekly ({len(ids)} series) ===")
    print(f"{'config':>40s}  {'mean':>7s}  {'median':>7s}")
    from collections import defaultdict
    by_cfg = defaultdict(list)
    for r in all_rows:
        if not np.isnan(r['smape']):
            by_cfg[(r['window_len'], r['sigma_frac'], r['alpha'], r['theta'])].append(r['smape'])
    # Print sorted by mean
    out = []
    for cfg in CONFIGS:
        key = (cfg['window_len'], cfg['sigma_frac'], cfg['alpha'], cfg['theta'])
        v = np.array(by_cfg[key])
        out.append((v.mean(), np.median(v), key))
    out.sort()
    for m, md, key in out[:10]:
        L, s, a, th = key
        tag = f"L={L}, s%={s:.2f}, a={a}, th={th}"
        print(f"{tag:>40s}  {m:>6.2f}%  {md:>6.2f}%")

    naive_v = np.array([smape(test[sid], naive_last(train[sid])) for sid in ids])
    print(f"\n  naive_last reference: mean={naive_v.mean():.2f}%, "
          f"median={np.median(naive_v):.2f}%")


if __name__ == "__main__":
    main()
