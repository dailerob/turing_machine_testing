"""V4: direct nearest-neighbor matching with Gaussian similarities.

Equivalent to GDC at alpha=1, no diffusion, no theta. The cleanest
test of whether the "find similar history, play forward" idea works.

Algorithm:
  1. Build windows: W[i] = train[i:i+L] for i in 0..n-L-H.
  2. Query: q = train[-L:].
  3. similarities[i] = exp(-||W[i] - q||^2 / (2 * sigma^2 * L))
  4. weights = similarities / sum(similarities)
  5. forecast[h] = sum_i weights[i] * train[i + L + h]
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import data_loader as dl
from v0_basic_gdc import smape, naive_seasonal_forecast, H_HORIZON


def nn_forecast(train, window_len=48, sigma_frac=0.05, h=H_HORIZON):
    n = len(train)
    if n < window_len + h:
        # Fall back to naive seasonal
        return naive_seasonal_forecast(train, h, period=24)
    sigma = float(np.std(train)) * sigma_frac
    sigma2 = max(sigma ** 2, 1e-9)
    # Build window matrix
    n_windows = n - window_len - h + 1
    # W[i] = train[i:i+window_len]
    indices = np.arange(window_len)[None, :] + np.arange(n_windows)[:, None]
    W = train[indices]                         # (n_windows, window_len)
    # Query: last `window_len` of training
    q = train[-window_len:]
    # Squared distance per window
    diff = W - q[None, :]
    dist2 = np.sum(diff ** 2, axis=1) / window_len
    # Gaussian weights (log-softmax for stability)
    log_w = -0.5 * dist2 / sigma2
    log_w -= log_w.max()
    w = np.exp(log_w)
    w_sum = w.sum()
    if w_sum <= 0:
        w = np.ones(n_windows) / n_windows
    else:
        w = w / w_sum
    # Continuations: for each window i, the h values train[i+L : i+L+h]
    cont_idx = (np.arange(window_len, window_len + h)[None, :]
                + np.arange(n_windows)[:, None])
    cont = train[cont_idx]                     # (n_windows, h)
    forecast = np.dot(w, cont)
    return forecast


def run_one(sid, train, test, window_len=48, sigma_frac=0.05):
    pred = nn_forecast(train, window_len=window_len, sigma_frac=sigma_frac,
                       h=H_HORIZON)
    naive = naive_seasonal_forecast(train, H_HORIZON, period=24)
    return dict(sid=sid, train=train, test=test, pred=pred, naive=naive,
                smape_gdc=smape(test, pred),
                smape_naive=smape(test, naive),
                window_len=window_len, sigma_frac=sigma_frac)


def main():
    train_d = dl.load_train()
    test_d = dl.load_test()
    ids = ["H1", "H50", "H150", "H300"]

    configs = [
        dict(window_len=24,  sigma_frac=0.10),
        dict(window_len=48,  sigma_frac=0.10),
        dict(window_len=168, sigma_frac=0.10),
        dict(window_len=48,  sigma_frac=0.02),
    ]
    print(f"{'sid':>4s}  {'config':>30s}  {'GDC-NN':>7s}  {'naive':>7s}")
    all_results = {}
    for cfg in configs:
        for sid in ids:
            r = run_one(sid, train_d[sid], test_d[sid], **cfg)
            tag = f"L={cfg['window_len']:>3d}, s%={cfg['sigma_frac']:.2f}"
            print(f"{sid:>4s}  {tag:>30s}  {r['smape_gdc']:>7.2f}%  "
                  f"{r['smape_naive']:>7.2f}%", flush=True)
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
                ['salmon', 'orange', 'crimson', 'mediumvioletred'],
                all_results[sid]):
            ax.plot(np.arange(n_train, n_train + H_HORIZON), r['pred'],
                    color=color, linewidth=1.0,
                    label=f'NN {tag} sMAPE={r["smape_gdc"]:.1f}%')
        naive = naive_seasonal_forecast(train, H_HORIZON, period=24)
        ax.plot(np.arange(n_train, n_train + H_HORIZON), naive,
                color='green', linewidth=0.9, linestyle='--',
                label=f'naive sMAPE={smape(test, naive):.1f}%')
        ax.axvline(n_train, color='black', linestyle=':', linewidth=0.8)
        ax.set_title(f'{sid}')
        ax.set_xlabel('hour'); ax.set_ylabel('value')
        ax.legend(loc='upper left', fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(HERE, 'fig_v4_nn_matching.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
