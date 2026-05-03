"""V3: prime with a recent window, then forecast.

Key change from v2: instead of forward-passing the full training (which
concentrates mass at the last position, then `forecast_gdc_style`
zeroes it and predictions decay to mean), we forward-pass only the
last K observations starting from a uniform initial distribution.

The forward filter then places mass at *all* training positions whose
recent K-step context matches the prefix. For periodic data this
naturally puts mass at periodically-matching positions (24h ago, 48h
ago, etc.), and the subsequent forecast averages their continuations.

We then `forecast` (not `forecast_gdc_style`) for H_HORIZON steps and
return E[obs] = state_dist @ states_continuous.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
M4_ROOT = os.path.dirname(HERE)
ROOT = os.path.dirname(M4_ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, M4_ROOT)
sys.path.insert(0, ROOT)

import data_loader as dl
from generative_dense_chain_timeseries import GenerativeDenseChainTimeSeries  # noqa: E402
from v0_basic_gdc import smape, naive_seasonal_forecast, H_HORIZON


def forecast_prime_window(train, beta_frac=0.02, alpha=0.95, theta=0.005,
                          prime_len=24, transition_type='self_loop'):
    sigma = float(np.std(train)) * beta_frac
    beta = max(sigma ** 2, 1e-9)
    states = train.reshape(-1, 1)
    gdc = GenerativeDenseChainTimeSeries(
        states, beta=beta, alpha=alpha, theta=theta,
        transition_type=transition_type,
        initial_dist='uniform',
    )
    # Use last `prime_len` observations as the priming window
    prime = train[-prime_len:].reshape(-1, 1)
    state_after_prime = gdc.forward_pass(prime)
    out = np.zeros(H_HORIZON, dtype=np.float64)
    dist = state_after_prime.copy()
    for t in range(H_HORIZON):
        dist = gdc._transition(dist)
        out[t] = float(np.dot(dist, gdc.states[:, 0]))
    return out, sigma


def run_one(sid, train, test, **kwargs):
    pred, sigma = forecast_prime_window(train, **kwargs)
    naive = naive_seasonal_forecast(train, H_HORIZON, period=24)
    return dict(sid=sid, train=train, test=test, pred=pred, naive=naive,
                smape_gdc=smape(test, pred),
                smape_naive=smape(test, naive),
                sigma=sigma, **kwargs)


def main():
    train_d = dl.load_train()
    test_d = dl.load_test()
    ids = ["H1", "H50", "H150", "H300"]

    # Sweep prime_len: 24 (one cycle), 48 (two cycles), 168 (one week)
    configs = [
        dict(prime_len=24,  alpha=0.99, theta=0.005, beta_frac=0.02),
        dict(prime_len=48,  alpha=0.99, theta=0.005, beta_frac=0.02),
        dict(prime_len=168, alpha=0.99, theta=0.005, beta_frac=0.02),
    ]
    print(f"{'sid':>4s}  {'config':>40s}  {'GDC':>7s}  {'naive':>7s}")
    fig, axes = plt.subplots(len(ids), 1, figsize=(13, 3 * len(ids)))
    if len(ids) == 1: axes = [axes]
    colors = ['salmon', 'orange', 'crimson']
    naive_recorded = {}
    all_results = {}
    for cfg in configs:
        for sid in ids:
            r = run_one(sid, train_d[sid], test_d[sid], **cfg)
            tag = f"prime={cfg['prime_len']}, a={cfg['alpha']}, b%={cfg['beta_frac']}"
            print(f"{sid:>4s}  {tag:>40s}  {r['smape_gdc']:>7.2f}%  "
                  f"{r['smape_naive']:>7.2f}%", flush=True)
            naive_recorded[sid] = r['smape_naive']
            all_results.setdefault(sid, {})[cfg['prime_len']] = r
    # Plot — for each series, overlay the three priming-length forecasts
    for ax, sid in zip(axes, ids):
        train = train_d[sid]; test = test_d[sid]
        n_train = len(train)
        zoom = 6 * 24
        start = max(0, n_train - zoom)
        ax.plot(np.arange(start, n_train), train[start:],
                color='steelblue', linewidth=0.7, label='train')
        ax.plot(np.arange(n_train, n_train + H_HORIZON), test,
                color='black', linewidth=1.5, label='test (actual)')
        for color, prime_len in zip(colors, [24, 48, 168]):
            r = all_results[sid][prime_len]
            ax.plot(np.arange(n_train, n_train + H_HORIZON), r['pred'],
                    color=color, linewidth=1.1,
                    label=f'GDC prime={prime_len} (sMAPE={r["smape_gdc"]:.1f}%)')
        naive = naive_seasonal_forecast(train, H_HORIZON, period=24)
        ax.plot(np.arange(n_train, n_train + H_HORIZON), naive,
                color='green', linewidth=0.9, linestyle='--',
                label=f'naive (sMAPE={naive_recorded[sid]:.1f}%)')
        ax.axvline(n_train, color='black', linestyle=':', linewidth=0.8)
        ax.set_title(f'{sid} — last 6d + 48h forecast')
        ax.set_xlabel('hour'); ax.set_ylabel('value')
        ax.legend(loc='upper left', fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(HERE, 'fig_v3_prime_window.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
