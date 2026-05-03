"""V6: use the new GDC-TS absorb mode + forecast_gdc_style.

Replaces v5's manual-lookahead workaround with the proper GDC-TS
framework call:

    GenerativeDenseChainTimeSeries(states, beta, alpha=1.0, theta=0.0,
        terminal_behavior='absorb', initial_dist='uniform')
        .forecast_gdc_style(prime, n_steps=H_HORIZON)

Goal: validate that this produces equivalent (or better) M4 hourly
results to v5's manual approach.

We also modify forecast_gdc_style's prediction-extraction step
to renormalize over surviving (non-terminal) mass — the "honest"
absorbing-state expected-value extraction.
"""
from __future__ import annotations
import os
import sys
import time
import csv
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
from v0_basic_gdc import smape, naive_seasonal_forecast, H_HORIZON


def gdc_absorb_forecast(train, window_len=48, sigma_frac=0.10,
                        h=H_HORIZON):
    """Use GDC-TS absorb mode + forecast_gdc_style."""
    n = len(train)
    if n < window_len + h + 1:
        return naive_seasonal_forecast(train, h, period=24)
    sigma_per_step = float(np.std(train)) * sigma_frac
    sigma_gdc = sigma_per_step * np.sqrt(window_len)
    beta = max(sigma_gdc ** 2, 1e-9)
    states = train.reshape(-1, 1)
    gdc = GenerativeDenseChainTimeSeries(
        states, beta=beta, alpha=1.0, theta=0.0,
        transition_type='self_loop',
        terminal_behavior='absorb',
        initial_dist='uniform',
    )
    prime = train[-window_len:].reshape(-1, 1)
    forecasts, state_dists = gdc.forecast_gdc_style(prime, n_steps=h)
    # forecast_gdc_style returns dot(state_dists, states) without
    # renormalization.  In absorb mode, mass can sit at terminal
    # before being zeroed.  Renormalize over non-terminal mass for
    # correct expected-value extraction:
    nt_mask = (~gdc.terminal_mask).astype(float)
    sd_nt = state_dists * nt_mask[None, :]
    sd_nt_sum = sd_nt.sum(axis=1, keepdims=True)
    safe = np.where(sd_nt_sum > 1e-12, sd_nt_sum, 1.0)
    sd_nt_norm = sd_nt / safe
    forecasts_norm = (sd_nt_norm @ gdc.states)[:, 0]
    return forecasts_norm


# ----- batch driver -----
CONFIGS = []
for L in [24, 48, 72, 168]:
    for s in [0.02, 0.05, 0.10, 0.20]:
        CONFIGS.append(dict(window_len=L, sigma_frac=s))


def run_series(args):
    sid, train, test = args
    sys.path.insert(0, HERE)
    sys.path.insert(0, ROOT)
    rows = []
    if len(train) <= 2 * H_HORIZON:
        best_cfg = dict(window_len=48, sigma_frac=0.10)
        pred_test = gdc_absorb_forecast(train, **best_cfg, h=H_HORIZON)
        rows.append(dict(sid=sid, **best_cfg,
                         val_smape=float('nan'),
                         test_smape=smape(test, pred_test),
                         smape_naive=smape(test,
                            naive_seasonal_forecast(train, H_HORIZON, 24))))
        return rows
    train_train = train[:-H_HORIZON]
    train_val = train[-H_HORIZON:]
    val_results = []
    for cfg in CONFIGS:
        try:
            pred_val = gdc_absorb_forecast(train_train, **cfg, h=H_HORIZON)
            sm = smape(train_val, pred_val)
        except Exception:
            sm = float('inf')
        val_results.append((sm, cfg))
    best_val_sm, best_cfg = min(val_results, key=lambda x: x[0])
    pred_test = gdc_absorb_forecast(train, **best_cfg, h=H_HORIZON)
    test_sm = smape(test, pred_test)
    naive_sm = smape(test, naive_seasonal_forecast(train, H_HORIZON, 24))
    rows.append(dict(sid=sid, window_len=best_cfg['window_len'],
                     sigma_frac=best_cfg['sigma_frac'],
                     val_smape=best_val_sm, test_smape=test_sm,
                     smape_naive=naive_sm))
    return rows


def main():
    train_d = dl.load_train()
    test_d = dl.load_test()
    ids = sorted(train_d.keys(), key=lambda s: int(s[1:]))
    tasks = [(sid, train_d[sid], test_d[sid]) for sid in ids]

    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    print(f"V6 absorb-mode val-tuned on {len(tasks)} series "
          f"({len(CONFIGS)} configs, {n_workers} workers)", flush=True)

    all_rows = []
    t0 = time.time()
    done = 0
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(run_series, tasks, chunksize=4):
            all_rows.extend(r); done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} [{time.time()-t0:.0f}s]",
                      flush=True)

    out_csv = os.path.join(HERE, 'v6_absorb_results.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"Wrote {out_csv}", flush=True)

    test_sm = np.array([r['test_smape'] for r in all_rows])
    naive_sm = np.array([r['smape_naive'] for r in all_rows])
    print(f"\n=== V6 absorb-mode (val-tuned) vs naive (414 series) ===")
    print(f"  V6 absorb  mean={np.mean(test_sm):.2f}%  "
          f"median={np.median(test_sm):.2f}%  "
          f"p25={np.percentile(test_sm, 25):.2f}%  "
          f"p75={np.percentile(test_sm, 75):.2f}%")
    print(f"  naive      mean={np.mean(naive_sm):.2f}%  "
          f"median={np.median(naive_sm):.2f}%")
    print(f"  V6 beats naive: "
          f"{int((test_sm < naive_sm).sum())}/{len(test_sm)}")

    # Compare to v5 result
    try:
        v5_rows = []
        with open(os.path.join(HERE, 'val_tuned_results.csv')) as f:
            for r in csv.DictReader(f):
                v5_rows.append(float(r['test_smape']))
        v5_sm = np.array(v5_rows)
        print(f"\n=== Compared to v5 manual-lookahead (val-tuned) ===")
        print(f"  v5 manual  mean={np.mean(v5_sm):.2f}%  "
              f"median={np.median(v5_sm):.2f}%")
        print(f"  V6 absorb  mean={np.mean(test_sm):.2f}%  "
              f"median={np.median(test_sm):.2f}%")
        print(f"  Per-series correlation: "
              f"{np.corrcoef(v5_sm, test_sm)[0, 1]:.4f}")
        # Series where they differ a lot
        diff = test_sm - v5_sm
        print(f"  Per-series sMAPE diff (V6 - v5): "
              f"mean={diff.mean():+.3f}, median={np.median(diff):+.3f}, "
              f"max_abs={np.max(np.abs(diff)):.3f}")
    except FileNotFoundError:
        print("(val_tuned_results.csv not found; skipping v5 comparison)")


if __name__ == "__main__":
    main()
