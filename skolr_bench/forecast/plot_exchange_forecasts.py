"""Generate many sanity-check / publication-style forecast plots for the
Exchange dataset using the val-tuned GDC config from
gdc_exchange_autoformer.py.

Output: skolr_bench/forecast/plots/exchange/*.png
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SKOLR_BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(SKOLR_BENCH)
sys.path.insert(0, HERE); sys.path.insert(0, SKOLR_BENCH); sys.path.insert(0, ROOT)
from informer_loaders import load_univariate
from gdc_torch import forecast_many_torch


L = 96  # Autoformer lookback
DTYPE = torch.float32
DEVICE = 'cuda'

# Val-tuned configs (from gdc_exchange_autoformer.py)
CONFIGS = {
    96:  ('diff', 0.25, 1.00),
    192: ('diff', 0.25, 1.00),
    336: ('diff', 0.25, 1.00),
    720: ('diff', 0.10, 1.00),  # the overfit val pick (kept honest)
}

OUT_DIR = os.path.join(HERE, 'plots', 'exchange')
os.makedirs(OUT_DIR, exist_ok=True)


def make_primes_truths(series, L_match, T):
    s = np.asarray(series, dtype=np.float64)
    n = len(s); n_w = max(0, n - L_match - T + 1)
    if n_w == 0:
        return np.empty((0, L_match)), np.empty((0, T))
    starts = np.arange(n_w)
    p_idx = np.arange(L_match)[None, :] + starts[:, None]
    t_idx = np.arange(L_match, L_match + T)[None, :] + starts[:, None]
    return s[p_idx], s[t_idx]


def get_forecasts(state_series, eval_lookback, eval_target, T, kind, sigma_frac, alpha):
    """Return (primes, truths, forecasts) where each row is one window."""
    if kind == 'diff':
        d_state = np.diff(state_series)
        sigma = max(float(np.std(d_state)) * sigma_frac, 1e-9)
        beta = max((sigma * np.sqrt(L)) ** 2, 1e-9)
        full = np.concatenate([eval_lookback[-(L+1):], eval_target])
        ext_primes, _ = make_primes_truths(full, L+1, T)
        diffed_primes = np.diff(ext_primes, axis=1)
        anchors = ext_primes[:, -1]
        truths_idx = np.arange(L+1, L+1+T)[None, :] + np.arange(diffed_primes.shape[0])[:, None]
        truths = full[truths_idx]
        forecast_d = forecast_many_torch(d_state, beta, alpha, 0.0,
                                          diffed_primes, T,
                                          device=DEVICE, dtype=DTYPE).cpu().numpy().astype(np.float64)
        forecasts = anchors[:, None] + np.cumsum(forecast_d, axis=1)
        primes = ext_primes[:, 1:]  # the L raw values aligned with diffs
    else:
        sigma = max(float(np.std(state_series)) * sigma_frac, 1e-9)
        beta = max((sigma * np.sqrt(L)) ** 2, 1e-9)
        full = np.concatenate([eval_lookback[-L:], eval_target])
        primes, truths = make_primes_truths(full, L, T)
        forecasts = forecast_many_torch(state_series, beta, alpha, 0.0,
                                         primes, T,
                                         device=DEVICE, dtype=DTYPE).cpu().numpy().astype(np.float64)
    return primes, truths, forecasts


def main():
    train, val, test, mu, sd = load_univariate('Exchange')
    print(f"Exchange: train={len(train)}, val={len(val)}, test={len(test)}")
    state = np.concatenate([train, val])  # state space for test eval
    rng = np.random.default_rng(0)

    # Per-horizon forecasts
    by_T = {}
    for T, (kind, s, a) in CONFIGS.items():
        primes, truths, forecasts = get_forecasts(state, val, test, T, kind, s, a)
        per_window_mse = ((forecasts - truths) ** 2).mean(axis=1)
        per_window_mae = np.abs(forecasts - truths).mean(axis=1)
        by_T[T] = dict(primes=primes, truths=truths, forecasts=forecasts,
                       mse=per_window_mse, mae=per_window_mae,
                       cfg=(kind, s, a))
        print(f"T={T}: {forecasts.shape[0]} windows, "
              f"MSE mean={per_window_mse.mean():.4f}  median={np.median(per_window_mse):.4f}")

    # ----- Plot 1: MSE distribution across the 4 horizons -----
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.5))
    for ax, T in zip(axes, [96, 192, 336, 720]):
        d = by_T[T]
        ax.hist(d['mse'], bins=40, color='steelblue', edgecolor='black', alpha=0.8)
        ax.axvline(d['mse'].mean(), color='red', linestyle='--',
                   label=f"mean={d['mse'].mean():.3f}")
        ax.axvline(np.median(d['mse']), color='green', linestyle=':',
                   label=f"median={np.median(d['mse']):.3f}")
        cfg = d['cfg']
        ax.set_title(f"T={T}  ({cfg[0]} σ={cfg[1]} α={cfg[2]})", fontsize=11)
        ax.set_xlabel('per-window MSE'); ax.set_ylabel('count')
        ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle('Exchange — per-window MSE distribution per horizon', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'mse_distributions.png'), dpi=120,
                bbox_inches='tight'); plt.close()

    # ----- Plot 2: best/median/worst forecasts per horizon -----
    fig, axes = plt.subplots(4, 3, figsize=(15, 11))
    for row, T in enumerate([96, 192, 336, 720]):
        d = by_T[T]
        idx = {
            'best':   int(np.argmin(d['mse'])),
            'median': int(np.argsort(d['mse'])[len(d['mse']) // 2]),
            'worst':  int(np.argmax(d['mse'])),
        }
        for col, (label, i) in zip([0, 1, 2], idx.items()):
            ax = axes[row, col]
            ax.plot(np.arange(L), d['primes'][i], color='gray', linewidth=1.0,
                    label='lookback')
            ax.plot(np.arange(L, L+T), d['truths'][i], color='black',
                    linewidth=1.5, label='truth')
            ax.plot(np.arange(L, L+T), d['forecasts'][i], color='salmon',
                    linewidth=1.3, linestyle='--', label='forecast')
            ax.axvline(L, color='black', linestyle=':', linewidth=0.7)
            mse_v = d['mse'][i]
            ax.set_title(f"T={T} {label.upper()} (window #{i}, MSE={mse_v:.4f})",
                         fontsize=10)
            if row == 0 and col == 0:
                ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
    fig.suptitle('Exchange GDC forecasts — best / median / worst window per horizon',
                 fontsize=13, y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'best_median_worst.png'), dpi=120,
                bbox_inches='tight'); plt.close()

    # ----- Plot 3: 6 random windows per horizon (zoomed) -----
    rng2 = np.random.default_rng(0)
    for T in [96, 192, 336, 720]:
        d = by_T[T]
        n_w = d['forecasts'].shape[0]
        idxs = rng2.choice(n_w, size=6, replace=False)
        fig, axes = plt.subplots(2, 3, figsize=(15, 6))
        for ax, i in zip(axes.flatten(), idxs):
            ax.plot(np.arange(L), d['primes'][i], color='gray', linewidth=0.9,
                    label='lookback')
            ax.plot(np.arange(L, L+T), d['truths'][i], color='black',
                    linewidth=1.4, label='truth')
            ax.plot(np.arange(L, L+T), d['forecasts'][i], color='salmon',
                    linewidth=1.2, linestyle='--', label='forecast')
            ax.axvline(L, color='black', linestyle=':', linewidth=0.6)
            ax.set_title(f"window #{i} (MSE={d['mse'][i]:.4f})", fontsize=9)
            ax.grid(True, alpha=0.3)
        axes[0, 0].legend(loc='upper left', fontsize=8)
        cfg = d['cfg']
        fig.suptitle(f"Exchange GDC forecasts — T={T} ({cfg[0]} σ={cfg[1]} α={cfg[2]}), 6 random test windows",
                     fontsize=12, y=1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'random_windows_T{T}.png'),
                    dpi=120, bbox_inches='tight'); plt.close()

    # ----- Plot 4: 1-step-ahead rolling overlay (first 500 windows) -----
    fig, axes = plt.subplots(4, 1, figsize=(14, 9))
    for ax, T in zip(axes, [96, 192, 336, 720]):
        d = by_T[T]
        n_show = min(500, d['forecasts'].shape[0])
        ax.plot(d['truths'][:n_show, 0], color='black', linewidth=0.8,
                label='truth (1-step ahead)')
        ax.plot(d['forecasts'][:n_show, 0], color='salmon', linewidth=0.8,
                label='forecast (1-step ahead)')
        ax.set_title(f'T={T} prediction setting — first {n_show} test windows',
                     fontsize=10)
        ax.set_ylabel('OT (standardized)')
        if T == 96: ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('test window index')
    fig.suptitle('Exchange — 1-step-ahead overlay across horizons', fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'rolling_overlay.png'), dpi=120,
                bbox_inches='tight'); plt.close()

    # ----- Plot 5: full series view (train+val+test) with sample test forecasts -----
    full_series = np.concatenate([train, val, test])
    n_train, n_val = len(train), len(val)
    fig, ax = plt.subplots(1, 1, figsize=(15, 4.5))
    x = np.arange(len(full_series))
    ax.plot(x, full_series, color='#444', linewidth=0.6, label='Exchange OT (full series)')
    ax.axvline(n_train, color='blue', linestyle=':', linewidth=0.8,
               label='train | val')
    ax.axvline(n_train + n_val, color='red', linestyle=':', linewidth=0.8,
               label='val | test')
    # Overlay 4 sample T=96 forecasts spaced through test
    d = by_T[96]
    for i in [50, 400, 800, 1300]:
        if i >= d['forecasts'].shape[0]:
            continue
        # Window i in test means start index i in [val[-L:] ++ test], so test index = i - L (but our windows start at 0 of test_with_lookback)
        test_start = i  # after our slicing, primes[i] aligns with test_with_lookback[i:i+L]
        truth_global_start = n_train + n_val + (test_start + 1 - 0)  # rough
        # Use raw mapping: primes start at val[-L+i:..], truths at test[i:i+96]
        # Simpler: plot truth in test-coords, just shift by n_train+n_val
        ax.plot(n_train + n_val + np.arange(test_start, test_start + 96) - L,
                d['truths'][i], color='black', linewidth=1.0)
        ax.plot(n_train + n_val + np.arange(test_start, test_start + 96) - L,
                d['forecasts'][i], color='salmon', linewidth=1.0, linestyle='--')
    ax.set_title('Exchange OT — full series (gray); 4 example T=96 GDC forecasts (salmon dashed) over the test region')
    ax.set_xlabel('time index'); ax.set_ylabel('OT (standardized)')
    ax.legend(loc='upper left', fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'full_series_with_forecasts.png'),
                dpi=120, bbox_inches='tight'); plt.close()

    # ----- Plot 6: same lookback, predicting at all 4 horizons (multi-horizon view) -----
    # Pick 3 well-distributed test starting points (early, middle, late)
    test_with_lookback = np.concatenate([val[-(L+1):], test])
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    sample_starts = [50, 600, 1200]
    for ax, start in zip(axes, sample_starts):
        # Get T=720 truth (longest), use prefix for shorter horizons
        if start + L + 720 > len(test_with_lookback): continue
        lookback = test_with_lookback[start:start+L]
        truth = test_with_lookback[start+L:start+L+720]
        # Also get all 4 forecasts at this exact starting point
        ax.plot(np.arange(-L, 0), lookback, color='gray', linewidth=1.0,
                label='lookback')
        ax.plot(np.arange(0, 720), truth, color='black', linewidth=1.3,
                label='truth (720 ahead)')
        colors = {96: '#1f77b4', 192: '#2ca02c', 336: '#d62728', 720: '#9467bd'}
        for T in [96, 192, 336, 720]:
            d = by_T[T]
            if start < d['forecasts'].shape[0]:
                ax.plot(np.arange(0, T), d['forecasts'][start], linestyle='--',
                        linewidth=1.0, color=colors[T],
                        label=f"GDC T={T}")
        ax.axvline(0, color='black', linestyle=':', linewidth=0.6)
        ax.set_title(f"Forecasts from test window starting at index {start}",
                     fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle('Exchange — multi-horizon GDC forecasts from same lookback',
                 fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'multi_horizon_views.png'), dpi=120,
                bbox_inches='tight'); plt.close()

    # ----- Plot 7: scatter forecast vs truth (all points across all 4 horizons) -----
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, T in zip(axes, [96, 192, 336, 720]):
        d = by_T[T]
        truths = d['truths'].flatten()
        forecasts = d['forecasts'].flatten()
        # Subsample for plot clarity
        if len(truths) > 50000:
            idx = np.random.default_rng(0).choice(len(truths), 50000, replace=False)
            truths = truths[idx]; forecasts = forecasts[idx]
        ax.scatter(truths, forecasts, s=3, alpha=0.25, color='steelblue', edgecolors='none')
        lo = min(truths.min(), forecasts.min())
        hi = max(truths.max(), forecasts.max())
        ax.plot([lo, hi], [lo, hi], color='red', linewidth=1.0, linestyle='--',
                label='y = x')
        ax.set_title(f"T={T} forecast vs truth (MSE={d['mse'].mean():.3f})", fontsize=10)
        ax.set_xlabel('truth'); ax.set_ylabel('forecast')
        ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.3); ax.set_aspect('equal', 'datalim')
    fig.suptitle('Exchange — forecast vs truth scatter per horizon', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'forecast_vs_truth_scatter.png'), dpi=120,
                bbox_inches='tight'); plt.close()

    # ----- Plot 8: per-horizon error growth (avg |error| at each step ahead) -----
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    colors = {96: '#1f77b4', 192: '#2ca02c', 336: '#d62728', 720: '#9467bd'}
    for T in [96, 192, 336, 720]:
        d = by_T[T]
        err = np.abs(d['forecasts'] - d['truths'])
        avg_err_per_step = err.mean(axis=0)  # (T,)
        ax.plot(np.arange(1, T+1), avg_err_per_step, color=colors[T],
                linewidth=1.5, label=f'T={T}')
    ax.set_xlabel('step ahead'); ax.set_ylabel('mean |forecast - truth|')
    ax.set_title('Exchange — error growth with prediction step (averaged over windows)')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'error_vs_step.png'), dpi=120,
                bbox_inches='tight'); plt.close()

    # ----- Plot 9: 12-window grid for T=96 (the headline horizon) -----
    d = by_T[96]
    n_w = d['forecasts'].shape[0]
    # Sort by MSE and pick: 4 best, 4 median, 4 worst
    order = np.argsort(d['mse'])
    best4   = order[:4]
    worst4  = order[-4:][::-1]
    mid_start = max(0, len(order)//2 - 2)
    median4 = order[mid_start:mid_start+4]
    fig, axes = plt.subplots(3, 4, figsize=(18, 8))
    for row, (label, idxs) in enumerate([('BEST', best4), ('MEDIAN', median4), ('WORST', worst4)]):
        for col, i in enumerate(idxs):
            ax = axes[row, col]
            ax.plot(np.arange(L), d['primes'][i], color='gray', linewidth=0.9)
            ax.plot(np.arange(L, L+96), d['truths'][i], color='black', linewidth=1.3)
            ax.plot(np.arange(L, L+96), d['forecasts'][i], color='salmon',
                    linestyle='--', linewidth=1.2)
            ax.axvline(L, color='black', linestyle=':', linewidth=0.6)
            ax.set_title(f"{label} #{i} (MSE={d['mse'][i]:.4f})", fontsize=9)
            ax.grid(True, alpha=0.3)
    axes[0, 0].plot([], [], color='gray', label='lookback')
    axes[0, 0].plot([], [], color='black', label='truth')
    axes[0, 0].plot([], [], color='salmon', linestyle='--', label='forecast')
    axes[0, 0].legend(loc='upper left', fontsize=8)
    fig.suptitle('Exchange T=96 GDC forecasts — best 4 / median 4 / worst 4 windows',
                 fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'T96_grid_12windows.png'), dpi=120,
                bbox_inches='tight'); plt.close()

    # ----- Plot 10: per-horizon comparison of single window forecasts (small multiples) -----
    # For 3 well-chosen test starts, show the forecast at each horizon side by side
    fig, axes = plt.subplots(3, 4, figsize=(18, 8))
    starts = [100, 700, 1100]
    for row, start in enumerate(starts):
        for col, T in enumerate([96, 192, 336, 720]):
            ax = axes[row, col]
            d = by_T[T]
            if start >= d['forecasts'].shape[0]:
                ax.set_visible(False); continue
            ax.plot(np.arange(L), d['primes'][start], color='gray', linewidth=0.9)
            ax.plot(np.arange(L, L+T), d['truths'][start], color='black',
                    linewidth=1.3, label='truth')
            ax.plot(np.arange(L, L+T), d['forecasts'][start], color='salmon',
                    linestyle='--', linewidth=1.2, label='forecast')
            ax.axvline(L, color='black', linestyle=':', linewidth=0.6)
            mse = ((d['truths'][start] - d['forecasts'][start])**2).mean()
            ax.set_title(f"start={start}, T={T} (MSE={mse:.4f})", fontsize=9)
            ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc='upper left', fontsize=8)
    fig.suptitle('Exchange — same starting points across all 4 horizons',
                 fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'multi_start_horizons.png'), dpi=120,
                bbox_inches='tight'); plt.close()

    print(f"\nWrote 10 plots to {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith('.png'):
            print(f"  {f}")


if __name__ == "__main__":
    main()
