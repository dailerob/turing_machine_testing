"""GDC val-tuned sweep on ECL/Traffic/ILI, Autoformer convention.

Univariate target = OT, splits 7:1:2 (Autoformer Table 1).
ECL/Traffic: I=96, horizons {96,192,336,720}.
ILI:         I=36, horizons {24,36,48,60} (Autoformer scripts use I=36 for ILI).
StandardScaler fit on train only. MSE/MAE on standardized data.
"""
from __future__ import annotations
import os, sys, time, csv, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SKOLR_BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(SKOLR_BENCH)
sys.path.insert(0, HERE); sys.path.insert(0, SKOLR_BENCH); sys.path.insert(0, ROOT)
from informer_loaders import load_univariate
from gdc_torch import forecast_many_torch


DEVICE = 'cuda'
DTYPE = torch.float32

DATASET_CONFIG = {
    'ECL_AF':     dict(L=96, horizons=[96, 192, 336, 720]),
    'Traffic_AF': dict(L=96, horizons=[96, 192, 336, 720]),
    'ILI_AF':     dict(L=36, horizons=[24, 36, 48, 60]),
}


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


def run_dataset(name):
    cfg = DATASET_CONFIG[name]
    L = cfg['L']
    HORIZONS = cfg['horizons']
    print(f"\n=== {name}: GDC val-tuned, L={L}, GPU fp32 ===")
    train, val, test, mu, sd = load_univariate(name)
    print(f"train={len(train)}, val={len(val)}, test={len(test)}")
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
        print(f"  PICK {best[1]} sigma={best[2]} alpha={best[3]}: "
              f"val MSE={best[0]:.4f}  test MSE={t_mse:.4f}  MAE={t_mae:.4f}  "
              f"({elapsed:.1f}s)", flush=True)
    print(f"Total {name}: {time.time()-t_total0:.1f}s")
    out = os.path.join(HERE, 'results', f'gdc_{name.lower()}_autoformer.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['T','kind','sigma','alpha','val_mse','test_mse','test_mae','time_s'])
        for r in rows: w.writerow(r)
    print(f"Wrote {out}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+',
                    default=['ILI_AF','ECL_AF','Traffic_AF'])
    args = ap.parse_args()
    print(f"Device: {torch.cuda.get_device_name(0)}, dtype: {DTYPE}")
    all_rows = {}
    for name in args.datasets:
        all_rows[name] = run_dataset(name)
    print("\n=== SUMMARY ===")
    for name, rows in all_rows.items():
        print(f"\n{name}:")
        for T, kind, sigma, alpha, v, tm, ta, _ in rows:
            print(f"  T={T:>4d}  {kind}/sigma={sigma}/alpha={alpha}  "
                  f"test MSE={tm:.4f}  MAE={ta:.4f}")


if __name__ == '__main__':
    main()
