"""GDC val-tuned sweep on Exchange univariate, Autoformer convention.

Protocol (matches Autoformer Table 2 univariate Exchange setting):
  - Univariate target = OT
  - Splits: 7 / 1 / 2 ratio (5311 / 760 / 1517)
  - Lookback I = 96 (fixed across horizons)
  - Horizons T in {96, 192, 336, 720}
  - Val tuning: state space = train, lookback from train+val tail
  - Test eval:  state space = train + val, lookback from val+test tail
  - StandardScaler fit on train only
  - MSE/MAE on standardized data
  - GPU PyTorch fp32 kernel
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


HORIZONS = [96, 192, 336, 720]
L = 96
DEVICE = 'cuda'
DTYPE = torch.float32


def build_configs():
    configs = []
    for sigma in [0.02, 0.05, 0.10, 0.25, 0.50]:
        for alpha in [1.0, 0.99]:
            configs.append(('raw', sigma, alpha))
    for sigma in [0.10, 0.25, 0.50, 1.00]:
        for alpha in [1.0, 0.99, 0.95]:
            configs.append(('diff', sigma, alpha))
    return configs


def make_primes_truths(series, L_match, T):
    s = np.asarray(series, dtype=np.float64)
    n = len(s); n_w = max(0, n - L_match - T + 1)
    if n_w == 0: return np.empty((0, L_match)), np.empty((0, T))
    starts = np.arange(n_w)
    p_idx = np.arange(L_match)[None, :] + starts[:, None]
    t_idx = np.arange(L_match, L_match + T)[None, :] + starts[:, None]
    return s[p_idx], s[t_idx]


def eval_one(state_series, eval_lookback, eval_target_series, L_match, T,
             kind, sigma_frac, alpha):
    if kind == 'diff':
        d_state = np.diff(state_series)
        sigma = max(float(np.std(d_state)) * sigma_frac, 1e-9)
        beta = max((sigma * np.sqrt(L_match)) ** 2, 1e-9)
        full = np.concatenate([eval_lookback[-(L_match+1):], eval_target_series])
        ext_primes, _ = make_primes_truths(full, L_match + 1, T)
        diffed_primes = np.diff(ext_primes, axis=1)
        anchors = ext_primes[:, -1]
        truths_idx = np.arange(L_match + 1, L_match + 1 + T)[None, :] + np.arange(diffed_primes.shape[0])[:, None]
        truths = full[truths_idx]
        forecast_d = forecast_many_torch(d_state, beta, alpha, 0.0,
                                          diffed_primes, T,
                                          device=DEVICE, dtype=DTYPE)
        if torch.is_tensor(forecast_d): forecast_d = forecast_d.cpu().numpy().astype(np.float64)
        forecasts = anchors[:, None] + np.cumsum(forecast_d, axis=1)
    else:
        sigma = max(float(np.std(state_series)) * sigma_frac, 1e-9)
        beta = max((sigma * np.sqrt(L_match)) ** 2, 1e-9)
        full = np.concatenate([eval_lookback[-L_match:], eval_target_series])
        primes, truths = make_primes_truths(full, L_match, T)
        forecasts = forecast_many_torch(state_series, beta, alpha, 0.0,
                                         primes, T,
                                         device=DEVICE, dtype=DTYPE)
        if torch.is_tensor(forecasts): forecasts = forecasts.cpu().numpy().astype(np.float64)
    diff = truths - forecasts
    return float((diff**2).mean()), float(np.abs(diff).mean())


def main():
    print(f"=== Exchange univariate full sweep: GDC val-tuned, L=I={L}, GPU fp32 ===\n")
    train, val, test, mu, sd = load_univariate('Exchange')
    print(f"train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Device: {torch.cuda.get_device_name(0)}, dtype: {DTYPE}\n")

    state_train_only = train
    state_train_val = np.concatenate([train, val])
    configs = build_configs()
    print(f"Sweeping {len(configs)} configs per horizon\n")
    rows = []
    t_total0 = time.time()
    for T in HORIZONS:
        print(f"--- T={T} ---", flush=True)
        t0 = time.time()
        val_results = []
        for kind, sigma, alpha in configs:
            v_mse, v_mae = eval_one(state_train_only, train, val,
                                     L, T, kind, sigma, alpha)
            val_results.append((v_mse, kind, sigma, alpha))
        val_results.sort(key=lambda x: x[0])
        best = val_results[0]
        t_mse, t_mae = eval_one(state_train_val, val, test,
                                 L, T, best[1], best[2], best[3])
        elapsed = time.time() - t0
        rows.append((T, best[1], best[2], best[3], best[0], t_mse, t_mae, elapsed))
        print(f"  Top 3 by val: ", end='')
        for v, k, s, a in val_results[:3]:
            print(f"{k}/sigma={s}/alpha={a}->{v:.4f}", end='  ')
        print()
        print(f"  PICK {best[1]} sigma={best[2]} alpha={best[3]}: "
              f"val MSE={best[0]:.4f}  test MSE={t_mse:.4f}  MAE={t_mae:.4f}  "
              f"({elapsed:.1f}s)\n", flush=True)
    print(f"Total: {time.time()-t_total0:.1f}s\n")

    # Autoformer Table 2 univariate Exchange baselines
    pub = {
        96:  dict(arima=(0.112,0.245), prophet=(0.828,0.762),
                  deepar=(0.417,0.515), nbeats=(0.156,0.299),
                  reformer=(1.327,0.944), logtrans=(0.279,0.441),
                  informer=(0.591,0.615), autoformer=(0.241,0.387)),
        192: dict(arima=(0.304,0.404), prophet=(0.909,0.974),
                  deepar=(0.813,0.735), nbeats=(0.669,0.665),
                  reformer=(1.258,0.929), logtrans=(1.950,1.048),
                  informer=(1.183,0.912), autoformer=(0.273,0.403)),
        336: dict(arima=(0.736,0.598), prophet=(1.304,0.988),
                  deepar=(1.331,0.962), nbeats=(0.611,0.605),
                  reformer=(2.179,1.296), logtrans=(2.438,1.262),
                  informer=(1.367,0.984), autoformer=(0.508,0.539)),
        720: dict(arima=(1.871,0.935), prophet=(3.238,1.566),
                  deepar=(1.894,1.181), nbeats=(1.111,0.860),
                  reformer=(2.285,1.243), logtrans=(2.010,1.247),
                  informer=(1.872,1.072), autoformer=(0.991,0.768)),
    }
    print(f"=== Exchange Univariate (Autoformer convention I=96, T in {HORIZONS}) ===")
    methods = ['arima', 'prophet', 'deepar', 'nbeats', 'reformer', 'logtrans',
               'informer', 'autoformer']
    header = f"{'T':>4s}  {'GDC':>14s}  " + "  ".join(f"{m:>13s}" for m in methods)
    print(header); print("-" * len(header))
    for T, kind, sigma, alpha, v_mse, t_mse, t_mae, _ in rows:
        gdc_str = f"{t_mse:.3f}/{t_mae:.3f}"
        cells = [f"{gdc_str:>14s}"]
        for m in methods:
            mse, mae = pub[T][m]
            cells.append(f"{mse:.3f}/{mae:.3f}")
        print(f"{T:>4d}  " + "  ".join(c.rjust(13) for c in cells))

    print(f"\n=== GDC vs best non-GDC per horizon ===")
    print(f"{'T':>4s}  {'GDC MSE':>9s}  {'best other':>30s}  {'GDC ratio':>10s}")
    for T, kind, sigma, alpha, v_mse, t_mse, t_mae, _ in rows:
        all_mses = [(pub[T][m][0], m) for m in methods]
        all_mses.sort()
        best_mse, best_method = all_mses[0]
        ratio = t_mse / best_mse
        rel = (1 - ratio) * 100
        print(f"{T:>4d}  {t_mse:>9.4f}  {best_method:>20s}={best_mse:.3f}  "
              f"{ratio:>5.2f}x ({rel:+.0f}%)")

    out = os.path.join(HERE, 'results', 'gdc_exchange_autoformer.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['T', 'kind', 'sigma', 'alpha', 'val_mse', 'test_mse', 'test_mae', 'time_s'])
        for r in rows: w.writerow(r)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
