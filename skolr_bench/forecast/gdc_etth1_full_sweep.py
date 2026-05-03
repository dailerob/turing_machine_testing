"""Full GDC val-tuned sweep on ETTh1 univariate at Informer's 5 horizons.

Uses the GPU PyTorch kernel.

Protocol (leakage-free, matches Informer Table 1):
  - Univariate target = OT
  - Splits: 12 / 4 / 4 months
  - Lookback L = 720 (Informer's choice for univariate)
  - Horizons T in {24, 48, 168, 336, 720}
  - Val tuning:  state_space = train,         lookback from train+val tail
  - Test eval:   state_space = train + val,   lookback from val+test tail
  - StandardScaler fit on train only
  - MSE/MAE on standardized data
"""
from __future__ import annotations
import os, sys, time, csv
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SKOLR_BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(SKOLR_BENCH)
sys.path.insert(0, HERE); sys.path.insert(0, SKOLR_BENCH); sys.path.insert(0, ROOT)
from informer_loaders import load_univariate
from gdc_torch import forecast_many_torch


HORIZONS = [24, 48, 168, 336, 720]
L = 720  # Informer's univariate lookback for ETTh1
DEVICE = 'cuda'
DTYPE = torch.float64  # use fp64 to match ARIMA/Informer eval precision


def build_configs():
    """22-config grid: raw + diff variants."""
    configs = []
    for sigma in [0.02, 0.05, 0.10, 0.25, 0.50]:
        for alpha in [1.0, 0.99]:
            configs.append(('raw', sigma, alpha))
    for sigma in [0.10, 0.25, 0.50, 1.00]:
        for alpha in [1.0, 0.99, 0.95]:
            configs.append(('diff', sigma, alpha))
    return configs


def make_primes(series, L_match, T):
    s = np.asarray(series, dtype=np.float64)
    n = len(s); n_w = max(0, n - L_match - T + 1)
    if n_w == 0: return np.empty((0, L_match)), np.empty((0, T))
    starts = np.arange(n_w)
    p_idx = np.arange(L_match)[None, :] + starts[:, None]
    t_idx = np.arange(L_match, L_match + T)[None, :] + starts[:, None]
    return s[p_idx], s[t_idx]


def eval_one(state_series, eval_lookback, eval_target_series, L_match, T,
             kind, sigma_frac, alpha):
    """Build state space + forecast; return MSE, MAE.

    `eval_lookback` is the head segment used to form lookback windows
    that overlap with `eval_target_series`. Concretely we form
    sliding windows of length L_match+T over [eval_lookback[-L_match:] ++
    eval_target_series], take prime = first L_match, truth = last T.
    """
    if kind == 'diff':
        d_state = np.diff(state_series)
        sigma = max(float(np.std(d_state)) * sigma_frac, 1e-9)
        beta = max((sigma * np.sqrt(L_match)) ** 2, 1e-9)
        # We need length-(L+1) windows so we can take L diffs of them
        full = np.concatenate([eval_lookback[-(L_match+1):], eval_target_series])
        ext_primes, _ = make_primes(full, L_match + 1, T)
        diffed_primes = np.diff(ext_primes, axis=1)
        anchors = ext_primes[:, -1]
        truths_idx = np.arange(L_match + 1, L_match + 1 + T)[None, :] + np.arange(diffed_primes.shape[0])[:, None]
        truths = full[truths_idx]
        forecast_d = forecast_many_torch(d_state, beta, alpha, 0.0,
                                          diffed_primes, T,
                                          device=DEVICE, dtype=DTYPE)
        if torch.is_tensor(forecast_d): forecast_d = forecast_d.cpu().numpy()
        forecasts = anchors[:, None] + np.cumsum(forecast_d, axis=1)
    else:
        sigma = max(float(np.std(state_series)) * sigma_frac, 1e-9)
        beta = max((sigma * np.sqrt(L_match)) ** 2, 1e-9)
        full = np.concatenate([eval_lookback[-L_match:], eval_target_series])
        primes, truths = make_primes(full, L_match, T)
        forecasts = forecast_many_torch(state_series, beta, alpha, 0.0,
                                         primes, T,
                                         device=DEVICE, dtype=DTYPE)
        if torch.is_tensor(forecasts): forecasts = forecasts.cpu().numpy()
    diff = truths - forecasts
    return float((diff**2).mean()), float(np.abs(diff).mean())


def main():
    print(f"=== ETTh1 univariate full sweep: GDC val-tuned, L={L}, GPU ===\n")
    train, val, test, mu, sd = load_univariate('ETTh1')
    print(f"train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Device: {torch.cuda.get_device_name(0)}, dtype: {DTYPE}\n")

    state_train_only = train  # for val tuning
    state_train_val = np.concatenate([train, val])  # for test eval

    configs = build_configs()
    print(f"Sweeping {len(configs)} configs per horizon\n")
    rows = []
    t_total0 = time.time()
    for T in HORIZONS:
        print(f"--- T={T} ---")
        # Val sweep
        t0 = time.time()
        val_results = []
        for kind, sigma, alpha in configs:
            v_mse, v_mae = eval_one(state_train_only, train, val,
                                     L, T, kind, sigma, alpha)
            val_results.append((v_mse, kind, sigma, alpha))
        val_results.sort(key=lambda x: x[0])
        best = val_results[0]
        # Test eval with val pick
        t_mse, t_mae = eval_one(state_train_val, val, test,
                                 L, T, best[1], best[2], best[3])
        elapsed = time.time() - t0
        rows.append((T, best[1], best[2], best[3], best[0], t_mse, t_mae, elapsed))
        # Top 3 picks for diagnostic
        print(f"  Top 3 by val: ", end='')
        for v, k, s, a in val_results[:3]:
            print(f"{k}/sigma={s}/alpha={a}->{v:.4f}", end='  ')
        print()
        print(f"  PICK {best[1]} sigma={best[2]} alpha={best[3]}: "
              f"val MSE={best[0]:.4f}  test MSE={t_mse:.4f}  MAE={t_mae:.4f}  "
              f"({elapsed:.1f}s)\n")

    print(f"Total: {time.time()-t_total0:.1f}s\n")

    # === Comparison vs published ===
    pub = {
        24:  dict(arima=(0.108,0.284), prophet=(0.115,0.275), lstma=(0.114,0.272),
                  deepar=(0.107,0.280), reformer=(0.222,0.389), logtrans=(0.103,0.259),
                  informer=(0.098,0.247), informer_d=(0.092,0.246)),
        48:  dict(arima=(0.175,0.424), prophet=(0.168,0.330), lstma=(0.193,0.358),
                  deepar=(0.162,0.327), reformer=(0.284,0.445), logtrans=(0.167,0.328),
                  informer=(0.158,0.319), informer_d=(0.161,0.322)),
        168: dict(arima=(0.396,0.504), prophet=(1.224,0.763), lstma=(0.236,0.392),
                  deepar=(0.239,0.422), reformer=(1.522,1.191), logtrans=(0.207,0.375),
                  informer=(0.183,0.346), informer_d=(0.187,0.355)),
        336: dict(arima=(0.468,0.593), prophet=(1.549,1.820), lstma=(0.590,0.698),
                  deepar=(0.445,0.552), reformer=(1.860,1.124), logtrans=(0.230,0.398),
                  informer=(0.222,0.387), informer_d=(0.215,0.369)),
        720: dict(arima=(0.659,0.766), prophet=(2.735,3.253), lstma=(0.683,0.768),
                  deepar=(0.658,0.707), reformer=(2.112,1.436), logtrans=(0.273,0.463),
                  informer=(0.269,0.435), informer_d=(0.257,0.421)),
    }
    print(f"=== ETTh1 Univariate Long-Horizon Forecasting ===")
    print(f"GDC val-tuned vs Informer Tab.1 baselines (MSE, MAE)\n")
    methods = ['arima', 'prophet', 'lstma', 'deepar', 'reformer', 'logtrans',
               'informer', 'informer_d']
    header = f"{'T':>4s}  {'GDC':>14s}  " + "  ".join(f"{m:>13s}" for m in methods)
    print(header)
    print("-" * len(header))
    for T, kind, sigma, alpha, v_mse, t_mse, t_mae, _ in rows:
        gdc_str = f"{t_mse:.3f}/{t_mae:.3f}"
        cells = [f"{gdc_str:>14s}"]
        for m in methods:
            mse, mae = pub[T][m]
            cells.append(f"{mse:.3f}/{mae:.3f}")
        print(f"{T:>4d}  " + "  ".join(c.rjust(13) for c in cells))
    print()

    # GDC vs best published per horizon
    print(f"=== GDC vs best non-GDC per horizon ===")
    print(f"{'T':>4s}  {'GDC MSE':>9s}  {'best other':>30s}  {'GDC ratio':>10s}")
    for T, kind, sigma, alpha, v_mse, t_mse, t_mae, _ in rows:
        all_mses = [(pub[T][m][0], m) for m in methods]
        all_mses.sort()
        best_mse, best_method = all_mses[0]
        ratio = t_mse / best_mse
        rel = (1 - ratio) * 100
        print(f"{T:>4d}  {t_mse:>9.4f}  {best_method:>20s}={best_mse:.3f}  {ratio:>5.2f}x ({rel:+.0f}%)")

    # Save CSV
    out = os.path.join(HERE, 'results', 'gdc_etth1_full_sweep.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['T', 'kind', 'sigma', 'alpha', 'val_mse', 'test_mse', 'test_mae', 'time_s'])
        for r in rows: w.writerow(r)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
