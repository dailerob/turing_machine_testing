"""V1: improved GDC pipeline.

Changes from v0:
  1. More bins (64 instead of 16) — finer discretization.
  2. Expected-value extraction: predict E[bin_center] under the
     predictive distribution instead of argmax bin midpoint. Yields
     continuous predictions (smaller quantization error).
  3. Sharper prefix-matching: alpha=0.99, beta=0 — make the GDC commit
     to the most-similar training position instead of smoothing.
  4. Compare against two GDC modes:
       'expected' : E_p[bin_center]
       'argmax'   : argmax bin (v0 mode)
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
from generative_dense_chain import GenerativeDenseChain  # noqa: E402
from v0_basic_gdc import (quantile_bins, discretize, smape,
                          naive_seasonal_forecast, H_HORIZON)


def forecast_gdc_expected(gdc, train_int, n_bins, centers, h):
    """Forecast `h` continuous values via expected bin-center under the
    one-step predictive at each step.  State is advanced by conditioning
    on the *expected* bin (rounded to nearest), which approximates the
    Bayesian filter for continuous outputs."""
    obs = train_int.reshape(-1, 1)
    final_state, _ = gdc.forward_pass(obs, return_history=True)
    state = final_state
    emit = gdc.states[:, 0].astype(np.int64)
    out = np.zeros(h, dtype=np.float64)
    for t in range(h):
        next_state = gdc.forecast(state, n_steps=1)
        marg = np.zeros(n_bins, dtype=np.float64)
        np.add.at(marg, emit, next_state)
        s = marg.sum()
        if s <= 0:
            ev = float(centers.mean())
            chosen_bin = int(np.argmin(np.abs(centers - ev)))
        else:
            marg = marg / s
            ev = float(np.sum(marg * centers))
            chosen_bin = int(np.argmin(np.abs(centers - ev)))
        out[t] = ev
        # Advance state by conditioning on the chosen bin
        match = (emit == chosen_bin).astype(np.float64)
        unnorm = next_state * match
        z = unnorm.sum()
        state = unnorm / z if z > 0 else next_state
    return out


def run_one(sid, train, test, n_bins=64,
            alpha=0.99, theta=0.005, beta=0.0):
    edges, centers = quantile_bins(train, n_bins)
    train_int = discretize(train, edges)
    gdc = GenerativeDenseChain(
        train_int.reshape(-1, 1),
        alpha=alpha, theta=theta, gamma=0.0, beta=beta,
        transition_type='self_loop',
        initial_dist='sequence_starts',
    )
    pred = forecast_gdc_expected(gdc, train_int, n_bins, centers, H_HORIZON)
    naive = naive_seasonal_forecast(train, H_HORIZON, period=24)
    return dict(sid=sid, train=train, test=test, pred=pred, naive=naive,
                centers=centers,
                smape_gdc=smape(test, pred),
                smape_naive=smape(test, naive),
                n_bins=n_bins, n_states=gdc.n_states)


def main():
    train = dl.load_train()
    test = dl.load_test()
    ids = ["H1", "H50", "H150", "H300"]
    results = []
    for sid in ids:
        r = run_one(sid, train[sid], test[sid])
        results.append(r)
        print(f"{sid}  n_bins={r['n_bins']}  states={r['n_states']:>5d}  "
              f"sMAPE GDC-expected={r['smape_gdc']:>6.2f}%  "
              f"naive_seasonal={r['smape_naive']:>6.2f}%", flush=True)

    fig, axes = plt.subplots(len(ids), 1, figsize=(12, 3 * len(ids)))
    if len(ids) == 1: axes = [axes]
    for ax, r in zip(axes, results):
        zoom = 4 * 24
        s_train = r['train']
        n_train = len(s_train)
        start = max(0, n_train - zoom)
        ax.plot(np.arange(start, n_train), s_train[start:],
                color='steelblue', linewidth=0.8, label='train')
        ax.plot(np.arange(n_train, n_train + H_HORIZON), r['test'],
                color='black', linewidth=1.5, label='test (actual)')
        ax.plot(np.arange(n_train, n_train + H_HORIZON), r['pred'],
                color='salmon', linewidth=1.2,
                label=f'GDC expected (sMAPE={r["smape_gdc"]:.2f}%)')
        ax.plot(np.arange(n_train, n_train + H_HORIZON), r['naive'],
                color='green', linewidth=1.0, linestyle='--',
                label=f'naive seasonal (sMAPE={r["smape_naive"]:.2f}%)')
        ax.axvline(n_train, color='black', linestyle=':', linewidth=0.8)
        ax.set_title(f'{r["sid"]} — last 4d + 48h forecast')
        ax.set_xlabel('hour'); ax.set_ylabel('value')
        ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(HERE, 'fig_v1_better_gdc.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
