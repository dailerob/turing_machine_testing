"""Leakage-free val-tuned eval for ETTh1 univariate T=24.

Protocol:
  Val sweep:
    state_space = train ONLY
    lookback for each val window = (train+val tail) sliding through val
    pick best (kind, sigma, alpha) by val MSE
  Test eval:
    state_space = train + val
    lookback for each test window = (val+test tail) sliding through test
    report test MSE/MAE for the val-picked config

This avoids any leakage: val tuning sees only train; test eval is
allowed train+val (everything before test) as memory + the L-point
lookback at inference time.

Also produces sanity-check plots:
  - Best/median/worst test windows: forecast vs truth
  - Histogram of per-window MSE on test
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SKOLR_BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(SKOLR_BENCH)
sys.path.insert(0, HERE); sys.path.insert(0, SKOLR_BENCH); sys.path.insert(0, ROOT)
from generative_dense_chain_timeseries import GenerativeDenseChainTimeSeries  # noqa: E402
from informer_loaders import load_univariate  # noqa: E402
from gdc_numba import forecast_many  # noqa: E402


def make_primes_truths(series_with_lookback, L, T, stride=1):
    """series_with_lookback length n; primes shape (B, L), truths (B, T)."""
    s = np.asarray(series_with_lookback, dtype=np.float64)
    n = len(s)
    n_w = max(0, n - L - T + 1)
    if n_w == 0: return np.empty((0, L)), np.empty((0, T))
    starts = np.arange(0, n_w, stride)
    p_idx = np.arange(L)[None, :] + starts[:, None]
    t_idx = np.arange(L, L + T)[None, :] + starts[:, None]
    return s[p_idx], s[t_idx]


def build_gdc_1d(state_series, window_len, sigma_frac, alpha, theta=0.0):
    sigma = max(float(np.std(state_series)) * sigma_frac, 1e-9)
    sigma_gdc = sigma * np.sqrt(window_len)
    beta = max(sigma_gdc ** 2, 1e-9)
    states = np.asarray(state_series, dtype=np.float64).reshape(-1, 1)
    return GenerativeDenseChainTimeSeries(
        states, beta=beta, alpha=alpha, theta=theta,
        transition_type='self_loop',
        terminal_behavior='absorb',
        initial_dist='uniform')


def predict(state_series, primes_for_eval, T, L, sigma, alpha, kind, anchors_for_diff=None):
    """Returns forecasts shape (B, T)."""
    if kind == 'raw':
        gdc = build_gdc_1d(state_series, L, sigma, alpha)
        return forecast_many(gdc.states[:, 0],
                              int(np.where(gdc.terminal_mask)[0][-1]),
                              gdc.beta, gdc.alpha, gdc.theta,
                              primes_for_eval, T)
    elif kind == 'diff':
        d_state = np.diff(state_series)
        gdc = build_gdc_1d(d_state, L, sigma, alpha)
        # primes_for_eval here are RAW lookback values of length L+1; we diff
        diffed_primes = np.diff(primes_for_eval, axis=1)  # (B, L)
        anchors = primes_for_eval[:, -1]
        forecast_d = forecast_many(gdc.states[:, 0],
                                    int(np.where(gdc.terminal_mask)[0][-1]),
                                    gdc.beta, gdc.alpha, gdc.theta,
                                    diffed_primes, T)
        cum = np.cumsum(forecast_d, axis=1)
        return anchors[:, None] + cum
    else:
        raise ValueError(kind)


def eval_split(state_series, lookback_series, eval_series, L, T, sigma, alpha, kind):
    """state_series: GDC state space.
       lookback_series + eval_series concatenated form the rolling lookback.
       Returns (mse, mae, forecasts, truths).
    """
    if kind == 'diff':
        # Need L+1 lookback to form L diffs
        full = np.concatenate([lookback_series[-(L+1):], eval_series])
        primes, truths = make_primes_truths(full, L + 1, T, stride=1)
        forecasts = predict(state_series, primes, T, L, sigma, alpha, 'diff')
    else:
        full = np.concatenate([lookback_series[-L:], eval_series])
        primes, truths = make_primes_truths(full, L, T, stride=1)
        forecasts = predict(state_series, primes, T, L, sigma, alpha, 'raw')
    diff = truths - forecasts
    mse = float((diff ** 2).mean())
    mae = float(np.abs(diff).mean())
    return mse, mae, forecasts, truths


def build_configs():
    """Reasonable small grid for val tuning."""
    configs = []
    for s in [0.02, 0.05, 0.10, 0.25, 0.50]:
        for a in [1.0, 0.99]:
            configs.append(('raw', s, a))
    for s in [0.10, 0.25, 0.50, 1.00]:
        for a in [1.0, 0.99, 0.95]:
            configs.append(('diff', s, a))
    return configs


def main():
    print("=== Leakage-free val-tuned ETTh1 T=24 univariate (OT) ===\n")

    train, val, test, mu, sd = load_univariate('ETTh1')
    print(f"train={len(train)}, val={len(val)}, test={len(test)}")
    L = 48; T = 24
    print(f"L={L}, T={T}\n")

    # Warm up Numba
    _ = predict(train, np.zeros((1, L+1)), T, L, 0.10, 1.0, 'diff')
    _ = predict(train, np.zeros((1, L)),   T, L, 0.10, 1.0, 'raw')

    # === Val sweep: state space = train only, lookback from train+val ===
    configs = build_configs()
    print(f"Sweeping {len(configs)} configs on VAL "
          f"(state_space = train only, {len(train)} pts)...")
    t0 = time.time()
    val_results = []
    for kind, s, a in configs:
        v_mse, v_mae, _, _ = eval_split(train, train, val, L, T, s, a, kind)
        val_results.append((v_mse, kind, s, a))
    val_results.sort(key=lambda x: x[0])
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Top 5 by val MSE:")
    for vm, k, s, a in val_results[:5]:
        print(f"    {k:>4s} sigma={s:.2f} alpha={a:.2f}  -> val MSE={vm:.4f}")
    print(f"  Bottom 3:")
    for vm, k, s, a in val_results[-3:]:
        print(f"    {k:>4s} sigma={s:.2f} alpha={a:.2f}  -> val MSE={vm:.4f}")

    val_best = val_results[0]
    print(f"\n  *** Picked by val: {val_best[1]} sigma={val_best[2]} alpha={val_best[3]} "
          f"(val MSE={val_best[0]:.4f}) ***")

    # === Test eval: state space = train + val ===
    state_test = np.concatenate([train, val])
    print(f"\n--- Test eval (state_space = train+val, "
          f"{len(state_test)} pts) ---")
    kind, s, a = val_best[1], val_best[2], val_best[3]
    t0 = time.time()
    t_mse, t_mae, forecasts, truths = eval_split(
        state_test, val, test, L, T, s, a, kind)
    elapsed = time.time() - t0
    print(f"  Test MSE={t_mse:.4f}  MAE={t_mae:.4f}  "
          f"({forecasts.shape[0]} windows in {elapsed:.1f}s)")

    # Reference
    print(f"\n=== Comparison ===")
    print(f"  ARIMA (Informer Tab. 1): MSE=0.108  MAE=0.284")
    print(f"  Informer (Tab. 1):       MSE=0.098  MAE=0.247")
    print(f"  GDC val-tuned:           MSE={t_mse:.3f}  MAE={t_mae:.3f}")
    print(f"  GDC vs ARIMA: {t_mse/0.108:.2f}x MSE  ({(1-t_mse/0.108)*100:+.0f}% better)")
    print(f"  GDC vs Informer: {t_mse/0.098:.2f}x MSE  ({(1-t_mse/0.098)*100:+.0f}% better)")

    # Also report what the test-oracle best would have been
    print(f"\n  (For sanity: re-evaluating top-3 val-best on test...)")
    for vm, k, ss, aa in val_results[:3]:
        tm, tma, _, _ = eval_split(state_test, val, test, L, T, ss, aa, k)
        print(f"    val-rank {ss}/{aa}/{k}: val={vm:.4f}, test={tm:.4f}")

    # === Sanity plots ===
    out_dir = os.path.join(HERE, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    per_window_mse = ((forecasts - truths) ** 2).mean(axis=1)
    rng = np.random.default_rng(0)

    # Plot 1: histogram of per-window MSE
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(per_window_mse, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(t_mse, color='red', linestyle='--',
               label=f'mean MSE = {t_mse:.3f}')
    ax.axvline(np.median(per_window_mse), color='green', linestyle=':',
               label=f'median MSE = {np.median(per_window_mse):.3f}')
    ax.set_xlabel('per-window MSE'); ax.set_ylabel('count')
    ax.set_title(f'ETTh1 T={T} test per-window MSE distribution\n'
                 f'(val-tuned {kind}, sigma={s}, alpha={a})')
    ax.legend(); ax.grid(True, alpha=0.3)
    out1 = os.path.join(out_dir, 'etth1_t24_mse_hist.png')
    plt.tight_layout(); plt.savefig(out1, dpi=120); plt.close()

    # Plot 2: best, median, worst windows
    best_i = int(np.argmin(per_window_mse))
    worst_i = int(np.argmax(per_window_mse))
    median_i = int(np.argsort(per_window_mse)[len(per_window_mse) // 2])
    rand_idx = rng.choice(len(forecasts), size=3, replace=False)
    pick_idx = [best_i, median_i, worst_i] + rand_idx.tolist()
    pick_labels = ['BEST', 'MEDIAN', 'WORST'] + [f'rand #{i}' for i in rand_idx]
    fig, axes = plt.subplots(len(pick_idx), 1, figsize=(10, 2.0 * len(pick_idx)))
    for ax, i, lab in zip(axes, pick_idx, pick_labels):
        ax.plot(np.arange(T), truths[i], color='black', linewidth=1.5,
                label='truth', marker='o', markersize=3)
        ax.plot(np.arange(T), forecasts[i], color='salmon', linewidth=1.5,
                label='forecast', marker='x', markersize=3)
        wmse = float(((forecasts[i] - truths[i]) ** 2).mean())
        ax.set_title(f'{lab} (window #{i}, MSE={wmse:.4f})', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'ETTh1 T=24 forecast samples '
                 f'(val-tuned {kind} sigma={s} alpha={a})', fontsize=11)
    plt.tight_layout()
    out2 = os.path.join(out_dir, 'etth1_t24_windows.png')
    plt.savefig(out2, dpi=120); plt.close()

    # Plot 3: a longer rolling overlay - 200 consecutive windows' last-step prediction
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    n_show = min(500, len(forecasts))
    ax.plot(truths[:n_show, 0], color='black', linewidth=0.8,
            label='truth (1-step ahead)')
    ax.plot(forecasts[:n_show, 0], color='salmon', linewidth=0.8,
            label='forecast (1-step ahead)')
    ax.set_xlabel('test window index')
    ax.set_ylabel('OT (standardized)')
    ax.set_title(f'ETTh1 T=24: 1-step ahead predictions, first {n_show} windows')
    ax.legend(); ax.grid(True, alpha=0.3)
    out3 = os.path.join(out_dir, 'etth1_t24_rolling.png')
    plt.tight_layout(); plt.savefig(out3, dpi=120); plt.close()

    print(f"\nSaved plots:")
    print(f"  {out1}")
    print(f"  {out2}")
    print(f"  {out3}")


if __name__ == "__main__":
    main()
