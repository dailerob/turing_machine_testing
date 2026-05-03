"""V5: connect to GDC-TS properly.

Approach: use GDC-TS for the matching step (forward_pass on a prime
window from uniform initial distribution at alpha=1, theta=0 — pure
+1 shift), but for forecasting do not rely on GDC-TS's transition
kernel (which has a terminal-diffusion fix that smears predictions).
Instead, read the h-step lookahead values directly from training.

After forward_pass on the last L observations:
    state_dist[p] ∝ exp(-||train[p-L+1:p+1] - prime||^2 / (2 * beta))
    for p in [L-1, n-1] (with edge zeros for p < L-1).

For valid h-step lookahead, restrict to p such that p + h < n.
Forecast at horizon h:
    forecast[h] = sum_p state_dist[p] * train[p + h]
                  / sum_p state_dist[p]   (over valid p)

This is mathematically equivalent to v4 NN matching when the GDC-TS
transition is alpha=1, theta=0 (pure +1 shift between emissions).
The GDC-TS sigma should be sqrt(L) * v4_sigma to match v4's per-step
likelihood weighting (since GDC-TS multiplies likelihoods at each
emission step rather than dividing the squared distance by L).
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
M4_ROOT = os.path.dirname(HERE); ROOT = os.path.dirname(M4_ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, M4_ROOT)
sys.path.insert(0, ROOT)

import data_loader as dl
from generative_dense_chain_timeseries import GenerativeDenseChainTimeSeries  # noqa: E402
from v0_basic_gdc import smape, naive_seasonal_forecast, H_HORIZON
from v4_nn_matching import nn_forecast


def gdc_proper_forecast(train, window_len=48, sigma_frac=0.10,
                        h=H_HORIZON, alpha=1.0, theta=0.0):
    """GDC-TS used for matching only; lookahead values read from training."""
    n = len(train)
    if n < window_len + h + 1:
        return naive_seasonal_forecast(train, h, period=24), None
    # GDC-TS sigma scales with sqrt(L) relative to v4's "average dist"
    # parameterisation.  This gives matching weights that exactly equal
    # v4's NN weights.
    sigma_per_step = float(np.std(train)) * sigma_frac
    sigma_gdc = sigma_per_step * np.sqrt(window_len)
    beta = max(sigma_gdc ** 2, 1e-9)
    states = train.reshape(-1, 1)
    gdc = GenerativeDenseChainTimeSeries(
        states, beta=beta, alpha=alpha, theta=theta,
        transition_type='self_loop',
        initial_dist='uniform',
    )
    prime = train[-window_len:].reshape(-1, 1)
    state_dist = gdc.forward_pass(prime)
    # state_dist is over training positions [0, n-1].  After matching the
    # last L observations, mass should peak at p = n-1 (trivially matches
    # itself) plus periodic earlier positions.  Forecasting at horizon h
    # = direct lookup train[p + h], with p restricted to p + h < n.
    out = np.zeros(h, dtype=np.float64)
    for h_idx in range(h):
        # h_idx=0 corresponds to "one step past the matched end" = p+1
        offset = h_idx + 1
        valid = np.arange(n - offset)
        w = state_dist[valid]
        z = w.sum()
        if z <= 0:
            out[h_idx] = float(train.mean())
            continue
        out[h_idx] = float(np.dot(w, train[valid + offset]) / z)
    return out, sigma_per_step


def run_one(sid, train, test, window_len=48, sigma_frac=0.10):
    pred_gdc, sigma = gdc_proper_forecast(train, window_len, sigma_frac)
    pred_nn = nn_forecast(train, window_len=window_len,
                          sigma_frac=sigma_frac, h=H_HORIZON)
    naive = naive_seasonal_forecast(train, H_HORIZON, period=24)
    return dict(sid=sid, train=train, test=test,
                pred_gdc=pred_gdc, pred_nn=pred_nn, naive=naive,
                smape_gdc=smape(test, pred_gdc),
                smape_nn=smape(test, pred_nn),
                smape_naive=smape(test, naive),
                sigma=sigma, window_len=window_len)


def main():
    train_d = dl.load_train()
    test_d = dl.load_test()
    ids = ["H1", "H50", "H150", "H300"]

    configs = [
        dict(window_len=24,  sigma_frac=0.10),
        dict(window_len=48,  sigma_frac=0.10),
        dict(window_len=168, sigma_frac=0.10),
    ]
    print(f"{'sid':>4s}  {'config':>22s}  "
          f"{'GDC-proper':>11s}  {'NN-direct':>10s}  {'naive':>7s}")
    all_results = {}
    for cfg in configs:
        for sid in ids:
            r = run_one(sid, train_d[sid], test_d[sid], **cfg)
            tag = f"L={cfg['window_len']:>3d}, s%={cfg['sigma_frac']:.2f}"
            print(f"{sid:>4s}  {tag:>22s}  {r['smape_gdc']:>10.2f}%  "
                  f"{r['smape_nn']:>9.2f}%  {r['smape_naive']:>6.2f}%",
                  flush=True)
            all_results.setdefault(sid, []).append((tag, r))

    fig, axes = plt.subplots(len(ids), 1, figsize=(13, 3 * len(ids)))
    if len(ids) == 1: axes = [axes]
    for ax, sid in zip(axes, ids):
        train = train_d[sid]; test = test_d[sid]
        n_train = len(train)
        zoom = 4 * 24
        start = max(0, n_train - zoom)
        ax.plot(np.arange(start, n_train), train[start:],
                color='steelblue', linewidth=0.7, label='train')
        ax.plot(np.arange(n_train, n_train + H_HORIZON), test,
                color='black', linewidth=1.5, label='test (actual)')
        for color, (tag, r) in zip(
                ['salmon', 'orange', 'crimson'], all_results[sid]):
            ax.plot(np.arange(n_train, n_train + H_HORIZON), r['pred_gdc'],
                    color=color, linewidth=1.0,
                    label=f'GDC {tag} (sMAPE={r["smape_gdc"]:.1f}%)')
        best_nn = min(all_results[sid], key=lambda x: x[1]['smape_nn'])
        ax.plot(np.arange(n_train, n_train + H_HORIZON), best_nn[1]['pred_nn'],
                color='blue', linewidth=0.8, linestyle=':',
                label=f'NN best ({best_nn[0]}, sMAPE={best_nn[1]["smape_nn"]:.1f}%)')
        naive = naive_seasonal_forecast(train, H_HORIZON, period=24)
        ax.plot(np.arange(n_train, n_train + H_HORIZON), naive,
                color='green', linewidth=0.9, linestyle='--',
                label=f'naive (sMAPE={smape(test, naive):.1f}%)')
        ax.axvline(n_train, color='black', linestyle=':', linewidth=0.8)
        ax.set_title(f'{sid}')
        ax.set_xlabel('hour'); ax.set_ylabel('value')
        ax.legend(loc='upper left', fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(HERE, 'fig_v5_gdc_proper.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
