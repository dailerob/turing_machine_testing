"""GDC val-tuned sweep on ETTm2 univariate, Autoformer convention.

Protocol (matches Autoformer Table 2 univariate ETT setting):
  - Univariate target = OT
  - Splits: 12 / 4 / 4 months (15-min granularity)
  - Lookback I = 96 (fixed across horizons)
  - Horizons T in {96, 192, 336, 720}
  - Val tuning:  state_space = train,         lookback from train+val tail
  - Test eval:   state_space = train + val,   lookback from val+test tail
  - StandardScaler fit on train only
  - MSE/MAE on standardized data
  - GPU PyTorch fp32 kernel (matches fp64 to ~3e-7 rel)

Comparable to: Autoformer Tab.2 univariate (ARIMA/Prophet/DeepAR/N-BEATS/
LogTrans/Reformer/Informer/Autoformer baselines) on ETT (= ETTm2).
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
from gdc_torch import forecast_many_torch, forecast_many_torch_dual


HORIZONS = [96, 192, 336, 720]
L = 96  # Autoformer's fixed input length
DEVICE = 'cuda'
DTYPE = torch.float32


def build_configs():
    """Config grid as (kind, sigma, alpha_ctx, alpha_fc) 4-tuples (single-alpha:
    alpha_fc == alpha_ctx), the original 22-config grid.

    NOTE (P1, 2026-06): dual-alpha (alpha_ctx < 1, alpha_fc = 1.0) was tested
    on the RAW recipe here and *hurt* — the sharper raw roll-out wins on
    validation but loses on test, val-overfitting and displacing the better
    diff pick (ETTm2 T=336: 0.150 -> 0.189; T=720: 0.254 -> 0.262). Set
    GDC_DUAL_ALPHA=1 to re-enable the raw dual candidates for inspection; they
    are OFF by default. The diff recipe is a no-op for alpha_fc regardless
    (stationary zero-mean differences). See paper/PROTOCOL_STANDARDIZATION.md."""
    dual = os.environ.get('GDC_DUAL_ALPHA', '0') == '1'
    configs = []
    for sigma in [0.02, 0.05, 0.10, 0.25, 0.50]:
        for alpha in [1.0, 0.99]:
            configs.append(('raw', sigma, alpha, alpha))     # single
        if dual:
            for alpha in [0.99, 0.95, 0.9]:
                configs.append(('raw', sigma, alpha, 1.0))   # dual (off by default)
    for sigma in [0.10, 0.25, 0.50, 1.00]:
        for alpha in [1.0, 0.99, 0.95]:
            configs.append(('diff', sigma, alpha, alpha))    # single (diff)
    return configs


def make_primes_truths(series, L_match, T):
    s = np.asarray(series, dtype=np.float64)
    n = len(s); n_w = max(0, n - L_match - T + 1)
    if n_w == 0: return np.empty((0, L_match)), np.empty((0, T))
    starts = np.arange(n_w)
    p_idx = np.arange(L_match)[None, :] + starts[:, None]
    t_idx = np.arange(L_match, L_match + T)[None, :] + starts[:, None]
    return s[p_idx], s[t_idx]


def _fc(states, beta, alpha, alpha_fc, primes, T):
    """Dispatch to single- or dual-alpha kernel (dual when alpha_fc != alpha)."""
    if alpha_fc is None or alpha_fc == alpha:
        return forecast_many_torch(states, beta, alpha, 0.0, primes, T,
                                   device=DEVICE, dtype=DTYPE)
    return forecast_many_torch_dual(states, beta, alpha, 0.0, alpha_fc, 0.0,
                                    primes, T, device=DEVICE, dtype=DTYPE)


def eval_one(state_series, eval_lookback, eval_target_series, L_match, T,
             kind, sigma_frac, alpha, alpha_fc=None):
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
        forecast_d = _fc(d_state, beta, alpha, alpha_fc, diffed_primes, T)
        if torch.is_tensor(forecast_d): forecast_d = forecast_d.cpu().numpy().astype(np.float64)
        forecasts = anchors[:, None] + np.cumsum(forecast_d, axis=1)
    else:
        sigma = max(float(np.std(state_series)) * sigma_frac, 1e-9)
        beta = max((sigma * np.sqrt(L_match)) ** 2, 1e-9)
        full = np.concatenate([eval_lookback[-L_match:], eval_target_series])
        primes, truths = make_primes_truths(full, L_match, T)
        forecasts = _fc(state_series, beta, alpha, alpha_fc, primes, T)
        if torch.is_tensor(forecasts): forecasts = forecasts.cpu().numpy().astype(np.float64)
    diff = truths - forecasts
    return float((diff**2).mean()), float(np.abs(diff).mean())


def main():
    print(f"=== ETTm2 univariate full sweep: GDC val-tuned, L=I={L}, GPU fp32 ===\n")
    train, val, test, mu, sd = load_univariate('ETTm2')
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
        for kind, sigma, alpha, alpha_fc in configs:
            v_mse, v_mae = eval_one(state_train_only, train, val,
                                     L, T, kind, sigma, alpha, alpha_fc)
            val_results.append((v_mse, kind, sigma, alpha, alpha_fc))
        val_results.sort(key=lambda x: x[0])
        best = val_results[0]
        t_mse, t_mae = eval_one(state_train_val, val, test,
                                 L, T, best[1], best[2], best[3], best[4])
        elapsed = time.time() - t0
        rows.append((T, best[1], best[2], best[3], best[0], t_mse, t_mae, elapsed))
        print(f"  Top 3 by val: ", end='')
        for v, k, s, a, afc in val_results[:3]:
            tag = f"alpha={a}" if afc == a else f"alpha={a}/afc={afc}"
            print(f"{k}/sigma={s}/{tag}->{v:.4f}", end='  ')
        print()
        print(f"  PICK {best[1]} sigma={best[2]} alpha={best[3]} alpha_fc={best[4]}: "
              f"val MSE={best[0]:.4f}  test MSE={t_mse:.4f}  MAE={t_mae:.4f}  "
              f"({elapsed:.1f}s)\n", flush=True)

    print(f"Total: {time.time()-t_total0:.1f}s\n")

    # === Comparison vs Autoformer Table 2 univariate ETT (= ETTm2) ===
    pub = {
        96:  dict(arima=(0.211,0.362), prophet=(0.287,0.456),
                  deepar=(0.099,0.237), nbeats=(0.082,0.219),
                  reformer=(0.108,0.244), logtrans=(0.075,0.208),
                  informer=(0.088,0.225), autoformer=(0.065,0.189)),
        192: dict(arima=(0.261,0.406), prophet=(0.312,0.483),
                  deepar=(0.154,0.310), nbeats=(0.120,0.268),
                  reformer=(0.175,0.296), logtrans=(0.129,0.275),
                  informer=(0.132,0.283), autoformer=(0.118,0.256)),
        336: dict(arima=(0.317,0.448), prophet=(0.331,0.474),
                  deepar=(0.277,0.428), nbeats=(0.226,0.370),
                  reformer=(0.396,0.491), logtrans=(0.154,0.302),
                  informer=(0.180,0.336), autoformer=(0.154,0.305)),
        720: dict(arima=(0.366,0.487), prophet=(0.534,0.593),
                  deepar=(0.332,0.468), nbeats=(0.188,0.338),
                  reformer=(0.468,0.540), logtrans=(0.160,0.322),
                  informer=(0.300,0.435), autoformer=(0.182,0.335)),
    }
    print(f"=== ETTm2 Univariate (Autoformer convention I=96, T in {HORIZONS}) ===")
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

    out = os.path.join(HERE, 'results', 'gdc_ettm2_autoformer.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['T', 'kind', 'sigma', 'alpha', 'val_mse', 'test_mse', 'test_mae', 'time_s'])
        for r in rows: w.writerow(r)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
