"""V2: continuous-Gaussian GDC for M4 hourly using the existing
`GenerativeDenseChainTimeSeries` (GDC-TS) class from the repo root.

GDC-TS:
  - Each hidden state stores the continuous training value.
  - Emission likelihood is N(obs; state, beta * I).
  - Same transition kernel as GDC (self_loop, two_step, sequential).
  - `forecast_gdc_style` runs forward pass, zeros the last state to
    avoid forecast getting stuck at the end of training, then propagates
    via transitions and returns E[obs] per future step.

Compared to v0 (binned argmax) and naive seasonal.
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


def run_one(sid, train, test,
            alpha=0.95, theta=0.005, beta_frac=0.02,
            transition_type='self_loop'):
    """beta_frac : sigma as a fraction of training std (variance = sigma^2)."""
    sigma = float(np.std(train)) * beta_frac
    beta = sigma ** 2
    states = train.reshape(-1, 1)
    gdc = GenerativeDenseChainTimeSeries(
        states, beta=beta, alpha=alpha, theta=theta,
        transition_type=transition_type,
        initial_dist='sequence_starts',
    )
    obs = train.reshape(-1, 1)
    forecasts, _ = gdc.forecast_gdc_style(obs, n_steps=H_HORIZON)
    pred = forecasts[:, 0]
    naive = naive_seasonal_forecast(train, H_HORIZON, period=24)
    return dict(sid=sid, train=train, test=test, pred=pred, naive=naive,
                smape_gdc=smape(test, pred),
                smape_naive=smape(test, naive),
                sigma=sigma, beta=beta, alpha=alpha, theta=theta)


def main():
    train_d = dl.load_train()
    test_d = dl.load_test()
    ids = ["H1", "H50", "H150", "H300"]
    results = []
    for sid in ids:
        r = run_one(sid, train_d[sid], test_d[sid])
        results.append(r)
        print(f"{sid}  alpha={r['alpha']} theta={r['theta']} sigma={r['sigma']:.3f}  "
              f"sMAPE GDC-TS={r['smape_gdc']:>6.2f}%  "
              f"naive={r['smape_naive']:>6.2f}%", flush=True)

    fig, axes = plt.subplots(len(ids), 1, figsize=(12, 3 * len(ids)))
    if len(ids) == 1: axes = [axes]
    for ax, r in zip(axes, results):
        zoom = 4 * 24
        s_train = r['train']; n_train = len(s_train)
        start = max(0, n_train - zoom)
        ax.plot(np.arange(start, n_train), s_train[start:],
                color='steelblue', linewidth=0.8, label='train')
        ax.plot(np.arange(n_train, n_train + H_HORIZON), r['test'],
                color='black', linewidth=1.5, label='test (actual)')
        ax.plot(np.arange(n_train, n_train + H_HORIZON), r['pred'],
                color='salmon', linewidth=1.2,
                label=f'GDC-TS sMAPE={r["smape_gdc"]:.2f}%')
        ax.plot(np.arange(n_train, n_train + H_HORIZON), r['naive'],
                color='green', linewidth=1.0, linestyle='--',
                label=f'naive seasonal sMAPE={r["smape_naive"]:.2f}%')
        ax.axvline(n_train, color='black', linestyle=':', linewidth=0.8)
        ax.set_title(f'{r["sid"]} — last 4d + 48h forecast')
        ax.set_xlabel('hour'); ax.set_ylabel('value')
        ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(HERE, 'fig_v2_continuous_gdc.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
