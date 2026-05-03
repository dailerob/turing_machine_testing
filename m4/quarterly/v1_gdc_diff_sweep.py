"""V1: GDC-TS-on-diffs sweep for M4 quarterly. h=8, season=4."""
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
sys.path.insert(0, HERE); sys.path.insert(0, M4_ROOT); sys.path.insert(0, ROOT)
import data_loader as dl
from generative_dense_chain_timeseries import GenerativeDenseChainTimeSeries  # noqa: E402
from v0_baselines import smape, naive_last, HORIZON


def gdc_diff_forecast(train, window_len=8, sigma_frac=0.25,
                      alpha=0.95, theta=0.0, h=HORIZON):
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
    forecast_d = ((sd_nt / safe) @ gdc.states)[:, 0]
    return train[-1] + np.cumsum(forecast_d)


def nn_diff_forecast(train, window_len=8, sigma_frac=0.50, h=HORIZON):
    n = len(train)
    if n < window_len + h + 2: return naive_last(train, h)
    d = np.diff(train)
    nd = len(d); n_windows = nd - window_len - h + 1
    if n_windows <= 0: return naive_last(train, h)
    sigma = float(np.std(d)) * sigma_frac
    sigma2 = max(sigma ** 2, 1e-9)
    indices = np.arange(window_len)[None, :] + np.arange(n_windows)[:, None]
    W = d[indices]; q = d[-window_len:]
    diff = W - q[None, :]
    dist2 = np.sum(diff ** 2, axis=1) / window_len
    log_w = -0.5 * dist2 / sigma2; log_w -= log_w.max()
    w = np.exp(log_w); s = w.sum()
    w = w / s if s > 0 else np.ones(n_windows) / n_windows
    cont_idx = np.arange(window_len, window_len + h)[None, :] + np.arange(n_windows)[:, None]
    forecast_d = w @ d[cont_idx]
    return train[-1] + np.cumsum(forecast_d)


CONFIGS = []
# NN-diff
for L in [4, 6, 8, 12]:
    for s in [0.25, 0.50, 1.00]:
        CONFIGS.append(('nn', dict(window_len=L, sigma_frac=s)))
# GDC-diff
for L in [4, 6, 8, 12]:
    for s in [0.10, 0.25, 0.50]:
        for a in [1.0, 0.99, 0.95, 0.9]:
            for th in [0.0]:
                CONFIGS.append(('gdc', dict(window_len=L, sigma_frac=s,
                                            alpha=a, theta=th)))


def predict(kind, cfg, train, h=HORIZON):
    if kind == 'nn': return nn_diff_forecast(train, h=h, **cfg)
    if kind == 'gdc': return gdc_diff_forecast(train, h=h, **cfg)


def run_series(args):
    sid, train, test = args
    rows = []
    for kind, cfg in CONFIGS:
        try: pred = predict(kind, cfg, train)
        except Exception: pred = naive_last(train, HORIZON)
        rows.append(dict(sid=sid, kind=kind, **cfg, smape=smape(test, pred)))
    return rows


def main():
    train = dl.load_train("Quarterly"); test = dl.load_test("Quarterly")
    ids = sorted(train.keys(), key=lambda s: int(s[1:]))
    tasks = [(sid, train[sid], test[sid]) for sid in ids]
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    print(f"Sweep on {len(tasks)} series, {len(CONFIGS)} configs, {n_workers} workers", flush=True)
    all_rows = []
    t0 = time.time(); done = 0
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(run_series, tasks, chunksize=16):
            all_rows.extend(r); done += 1
            if done % 5000 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} [{time.time()-t0:.0f}s]", flush=True)

    out_csv = os.path.join(HERE, "v1_results.csv")
    fields = sorted({k for r in all_rows for k in r.keys()})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    print(f"Wrote {out_csv}")

    print(f"\n=== Top configs by mean ===")
    print(f"{'config':>50s}  {'mean':>7s}  {'median':>7s}")
    from collections import defaultdict
    by_cfg = defaultdict(list)
    for r in all_rows:
        key = (r['kind'], r['window_len'], r['sigma_frac'],
               r.get('alpha', None), r.get('theta', None))
        by_cfg[key].append(r['smape'])
    out = []
    for k, v in by_cfg.items():
        v = np.array(v); out.append((v.mean(), np.median(v), k))
    out.sort()
    for m, md, k in out[:15]:
        kind, L, s, a, th = k
        if kind == 'nn':
            tag = f"nn  L={L}, s%={s:.2f}"
        else:
            tag = f"gdc L={L}, s%={s:.2f}, a={a}, th={th}"
        print(f"{tag:>50s}  {m:>6.2f}%  {md:>6.2f}%")


if __name__ == "__main__":
    main()
