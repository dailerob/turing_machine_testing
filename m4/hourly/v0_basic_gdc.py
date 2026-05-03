"""V0: simplest possible GDC pipeline on M4 hourly.

Pipeline:
  1. Discretize series via quantile bins (K bins per series).
  2. Train GDC on the integer sequence (one big sequence).
  3. Forecast 48 steps via greedy sampling from the predicted next-symbol
     distribution, advancing the GDC state at each step.
  4. De-discretize using bin midpoints.
  5. Plot predictions vs actual; compute sMAPE.

Tested on H1, H50, H150 — three series of varying difficulty.
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

H_HORIZON = 48


# ---------------------------------------------------------------
# Discretization helpers
# ---------------------------------------------------------------
def quantile_bins(values, n_bins):
    """Return (bin_edges, bin_centers) using empirical quantiles of `values`."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, qs)
    # Make edges strictly increasing (handle constant runs)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def discretize(values, edges):
    """Map continuous values to bin index in [0, n_bins)."""
    n_bins = len(edges) - 1
    idx = np.searchsorted(edges, values, side='right') - 1
    return np.clip(idx, 0, n_bins - 1).astype(np.int64)


# ---------------------------------------------------------------
# GDC forecast
# ---------------------------------------------------------------
def forecast_gdc(gdc, train_int, n_bins, h, mode='argmax', rng=None):
    """Run forward through training history then forecast `h` steps.

    `mode='argmax'` picks the most-likely next bin at each step;
    `mode='sample'` samples from the predictive distribution (rng required).

    Returns array of bin indices length `h`.
    """
    obs = train_int.reshape(-1, 1)
    final_state, _ = gdc.forward_pass(obs, return_history=True)
    state = final_state
    emit = gdc.states[:, 0].astype(np.int64)
    out = np.zeros(h, dtype=np.int64)
    for t in range(h):
        next_state = gdc.forecast(state, n_steps=1)
        # Aggregate by emission
        marg = np.zeros(n_bins, dtype=np.float64)
        np.add.at(marg, emit, next_state)
        s = marg.sum()
        if s <= 0:
            sym = 0
            marg = np.full(n_bins, 1.0 / n_bins)
        else:
            marg = marg / s
            if mode == 'argmax':
                sym = int(np.argmax(marg))
            else:
                sym = int(rng.choice(n_bins, p=marg))
        out[t] = sym
        # Advance state by conditioning on the chosen symbol
        match = (emit == sym).astype(np.float64)
        unnorm = next_state * match
        z = unnorm.sum()
        if z > 0:
            state = unnorm / z
        else:
            state = next_state
    return out


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
def smape(actual, forecast):
    """Symmetric MAPE in percent; standard M4 definition."""
    actual = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return 100.0 * np.mean(np.abs(actual - forecast) / denom)


def naive_seasonal_forecast(train, h, period=24):
    """Repeat the last `period` values to fill horizon `h`."""
    last = train[-period:]
    out = np.tile(last, int(np.ceil(h / period)))[:h]
    return out


# ---------------------------------------------------------------
# Per-series experiment
# ---------------------------------------------------------------
def run_one(sid, train, test, n_bins=16,
            alpha=0.95, theta=0.005, beta=0.05):
    edges, centers = quantile_bins(train, n_bins)
    train_int = discretize(train, edges)
    gdc = GenerativeDenseChain(
        train_int.reshape(-1, 1),
        alpha=alpha, theta=theta, gamma=0.0, beta=beta,
        transition_type='self_loop',
        initial_dist='sequence_starts',
    )
    pred_int = forecast_gdc(gdc, train_int, n_bins, H_HORIZON, mode='argmax')
    pred = centers[pred_int]
    naive = naive_seasonal_forecast(train, H_HORIZON, period=24)
    return dict(sid=sid, train=train, test=test, pred=pred, naive=naive,
                edges=edges, centers=centers,
                smape_gdc=smape(test, pred),
                smape_naive=smape(test, naive),
                n_bins=n_bins, n_states=gdc.n_states)


def main():
    train = dl.load_train()
    test = dl.load_test()
    ids = ["H1", "H50", "H150"]
    results = []
    for sid in ids:
        r = run_one(sid, train[sid], test[sid])
        results.append(r)
        print(f"{sid}  n_bins={r['n_bins']}  GDC_states={r['n_states']:>6d}  "
              f"sMAPE GDC={r['smape_gdc']:>6.2f}%  "
              f"sMAPE naive_seasonal={r['smape_naive']:>6.2f}%", flush=True)

    # Plot
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
                color='salmon', linewidth=1.2, label=f'GDC argmax (sMAPE={r["smape_gdc"]:.2f}%)')
        ax.plot(np.arange(n_train, n_train + H_HORIZON), r['naive'],
                color='green', linewidth=1.0, linestyle='--',
                label=f'naive seasonal (sMAPE={r["smape_naive"]:.2f}%)')
        ax.axvline(n_train, color='black', linestyle=':', linewidth=0.8)
        ax.set_title(f'{r["sid"]} — last 4d + 48h forecast')
        ax.set_xlabel('hour'); ax.set_ylabel('value')
        ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(HERE, 'fig_v0_basic_gdc.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
