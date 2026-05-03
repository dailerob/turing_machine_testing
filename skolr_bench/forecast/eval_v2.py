"""V2: expanded recipes for ETT improvement experiments.

Adds 3 new recipe families on top of raw / diff:
  * 'detrend':   subtract local linear trend from the L-window, run
                  raw GDC on residuals, add forecast trend back
  * 'twostep':   two-step transition kernel (sequential mixing inside
                  the GDC kernel)
  * 'multiscale_avg': average forecasts from {raw L=L, raw L=L/2, diff L=L}
                  for the same window — cheap ensemble
Wider hyperparameter grid:
  * L in {seq_len/4, seq_len/2, seq_len, 2*seq_len}
  * σ% in {0.02, 0.05, 0.10, 0.25, 0.50}
  * α in {1.0, 0.99, 0.95, 0.9, 0.8}

Parallelism: scheduled at (channel, config) granularity so all 32 cores
are saturated. Picks are made after gather.
"""
from __future__ import annotations
import os, sys, csv, time, argparse
import numpy as np
import multiprocessing as mp
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SKOLR_BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(SKOLR_BENCH)
sys.path.insert(0, HERE); sys.path.insert(0, SKOLR_BENCH); sys.path.insert(0, ROOT)
from generative_dense_chain_timeseries import GenerativeDenseChainTimeSeries  # noqa: E402
from loaders import load, HORIZONS, DATASETS  # noqa: E402


def _make_gdc(states, beta, alpha, theta, transition_type='self_loop'):
    return GenerativeDenseChainTimeSeries(
        states, beta=beta, alpha=alpha, theta=theta,
        transition_type=transition_type,
        terminal_behavior='absorb',
        initial_dist='uniform')


def _gdc_run(history_1d, window_len, sigma_frac, alpha, theta, h,
             transition_type='self_loop'):
    """Run GDC-TS on a 1-D series, return h-step forecast."""
    sigma_per_step = float(np.std(history_1d)) * sigma_frac
    if sigma_per_step <= 0:
        return np.full(h, history_1d[-1])
    sigma_gdc = sigma_per_step * np.sqrt(window_len)
    beta = max(sigma_gdc ** 2, 1e-9)
    states = history_1d.reshape(-1, 1)
    gdc = _make_gdc(states, beta, alpha, theta, transition_type)
    prime = history_1d[-window_len:].reshape(-1, 1)
    _, sd = gdc.forecast_gdc_style(prime, n_steps=h)
    nt = (~gdc.terminal_mask).astype(float)
    sd_nt = sd * nt[None, :]
    sd_sum = sd_nt.sum(axis=1, keepdims=True)
    safe = np.where(sd_sum > 1e-12, sd_sum, 1.0)
    return ((sd_nt / safe) @ gdc.states)[:, 0]


def gdc_raw(history, window_len, sigma_frac, alpha, theta, h,
            transition_type='self_loop'):
    history = np.asarray(history, dtype=np.float64)
    if len(history) < window_len + 1:
        return np.full(h, history[-1])
    return _gdc_run(history, window_len, sigma_frac, alpha, theta, h,
                    transition_type)


def gdc_diff(history, window_len, sigma_frac, alpha, theta, h,
             transition_type='self_loop'):
    history = np.asarray(history, dtype=np.float64)
    if len(history) < window_len + 2:
        return np.full(h, history[-1])
    d = np.diff(history)
    if len(d) < window_len + 1:
        return np.full(h, history[-1])
    forecast_d = _gdc_run(d, window_len, sigma_frac, alpha, theta, h,
                          transition_type)
    return history[-1] + np.cumsum(forecast_d)


def gdc_detrend(history, window_len, sigma_frac, alpha, theta, h,
                transition_type='self_loop'):
    """Subtract linear trend from the L window, run raw GDC on residuals,
    then add the extrapolated trend back."""
    history = np.asarray(history, dtype=np.float64)
    if len(history) < window_len + 1:
        return np.full(h, history[-1])
    L = window_len
    win = history[-L:]
    x = np.arange(L, dtype=np.float64)
    # Linear regression on the window
    A = np.vstack([x, np.ones(L)]).T
    slope, intercept = np.linalg.lstsq(A, win, rcond=None)[0]
    trend = slope * x + intercept
    residuals = win - trend
    # We replace the last L of history with residuals (for matching consistency)
    # But for similarity matching we want the same detrending applied to
    # historical windows too — that's what the raw GDC kernel naturally
    # does on a residual series. So feed the full history minus trend.
    # Simpler approximation: just use residuals as the prime, and as the
    # state space the residuals of the entire history.
    full_x = np.arange(len(history), dtype=np.float64)
    # Subtract a global trend over the history (so historical windows are
    # also "detrended"). Use the same fit but extended.
    full_trend = slope * (full_x - (len(history) - L)) + intercept
    full_residuals = history - full_trend
    forecast_residuals = _gdc_run(full_residuals, window_len, sigma_frac,
                                   alpha, theta, h, transition_type)
    fut_x = np.arange(len(history), len(history) + h, dtype=np.float64)
    fut_trend = slope * (fut_x - (len(history) - L)) + intercept
    return forecast_residuals + fut_trend


def gdc_multiscale_avg(history, window_len, sigma_frac, alpha, theta, h,
                       transition_type='self_loop'):
    """Average of {raw at L, raw at L/2, diff at L} forecasts."""
    L = window_len
    raws = []
    raws.append(gdc_raw(history, L, sigma_frac, alpha, theta, h, transition_type))
    if L >= 4:
        raws.append(gdc_raw(history, L // 2, sigma_frac, alpha, theta, h,
                            transition_type))
    raws.append(gdc_diff(history, L, max(sigma_frac, 0.25), alpha, theta, h,
                         transition_type))
    return np.mean(np.stack(raws, axis=0), axis=0)


PREDICTORS = {
    'raw':            gdc_raw,
    'diff':           gdc_diff,
    'detrend':        gdc_detrend,
    'twostep_raw':    lambda h, **kw: gdc_raw(h, transition_type='self_loop_two_step', **kw),
    'twostep_diff':   lambda h, **kw: gdc_diff(h, transition_type='self_loop_two_step', **kw),
    'multiscale':     gdc_multiscale_avg,
}


def predict(kind, cfg, history, h):
    fn = PREDICTORS[kind]
    return fn(history, h=h, **cfg)


def build_configs(seq_len):
    configs = []
    L_options = [max(2, seq_len // 4), seq_len // 2, seq_len, min(2 * seq_len, 1024)]
    L_options = sorted(set(L_options))
    sigmas_raw = [0.02, 0.05, 0.10, 0.25]
    sigmas_diff = [0.25, 0.5, 1.0]
    alphas = [1.0, 0.99, 0.95, 0.9]
    # raw + twostep_raw
    for kind in ['raw', 'twostep_raw', 'detrend']:
        for L in L_options:
            for s in sigmas_raw:
                for a in alphas:
                    configs.append((kind, dict(window_len=L, sigma_frac=s,
                                                alpha=a, theta=0.0)))
    for kind in ['diff', 'twostep_diff']:
        for L in L_options:
            for s in sigmas_diff:
                for a in alphas:
                    configs.append((kind, dict(window_len=L, sigma_frac=s,
                                                alpha=a, theta=0.0)))
    # Multiscale uses fewer combos (it's an ensemble of 3 already)
    for L in L_options:
        for s in [0.05, 0.10, 0.25]:
            for a in [1.0, 0.95]:
                configs.append(('multiscale', dict(window_len=L, sigma_frac=s,
                                                    alpha=a, theta=0.0)))
    return configs


def eval_window_seq(series, kind, cfg, seq_len, pred_len, max_windows=None):
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    n_windows = max(0, n - seq_len - pred_len + 1)
    if n_windows == 0:
        return 0.0, 0.0, 0
    if max_windows is not None and max_windows < n_windows:
        idxs = np.linspace(0, n_windows - 1, max_windows, dtype=int)
    else:
        idxs = np.arange(n_windows)
    sse = 0.0; sae = 0.0; n_pts = 0
    for i in idxs:
        hist = series[i:i + seq_len]
        truth = series[i + seq_len: i + seq_len + pred_len]
        try:
            pred = predict(kind, cfg, hist, h=pred_len)
        except Exception:
            continue
        d = truth - pred
        sse += float(np.sum(d ** 2)); sae += float(np.sum(np.abs(d)))
        n_pts += pred_len
    return sse, sae, n_pts


def task_one(args):
    """One (channel, T, kind, cfg) val + test evaluation."""
    (dataset, T, ch_idx, val_series, test_series, kind, cfg, val_max) = args
    seq_len = 2 * T
    pred_len = T
    val_sse, val_sae, val_n = eval_window_seq(
        val_series, kind, cfg, seq_len, pred_len, max_windows=val_max)
    if val_n == 0:
        return (dataset, T, ch_idx, kind, cfg, float('inf'),
                0.0, 0.0, 0)
    val_mse = val_sse / val_n
    test_sse, test_sae, test_n = eval_window_seq(
        test_series, kind, cfg, seq_len, pred_len)
    return (dataset, T, ch_idx, kind, cfg, val_mse,
            test_sse, test_sae, test_n)


def run_dataset(dataset, val_max=200, n_workers=None):
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4))
    horizons = HORIZONS[dataset]
    max_seq = 2 * max(horizons)
    train, val, test, cols = load(dataset, seq_len=max_seq)
    n_ch = len(cols)
    print(f"  {dataset}: {n_ch} channels, train={train.shape}, val={val.shape}, "
          f"test={test.shape}, n_workers={n_workers}", flush=True)
    out = {}
    for T in horizons:
        seq_len = 2 * T
        configs = build_configs(seq_len)
        tasks = []
        for ch in range(n_ch):
            for kind, cfg in configs:
                tasks.append((dataset, T, ch, val[:, ch], test[:, ch],
                              kind, cfg, val_max))
        t0 = time.time()
        with mp.Pool(processes=n_workers) as pool:
            results = list(pool.imap_unordered(task_one, tasks, chunksize=4))
        # Pick best (kind, cfg) per channel by val MSE
        per_ch_best = {}  # ch -> (val_mse, kind, cfg, test_sse, test_sae, test_n)
        for r in results:
            ds, t_, ch, kind, cfg, val_mse, ts_sse, ts_sae, ts_n = r
            if ch not in per_ch_best or val_mse < per_ch_best[ch][0]:
                per_ch_best[ch] = (val_mse, kind, cfg, ts_sse, ts_sae, ts_n)
        # Aggregate
        total_sse = 0.0; total_sae = 0.0; total_n = 0
        picks = defaultdict(int)
        for ch in range(n_ch):
            v = per_ch_best.get(ch)
            if v is None: continue
            total_sse += v[3]; total_sae += v[4]; total_n += v[5]
            picks[(v[1], tuple(sorted(v[2].items())))] += 1
        mse = total_sse / total_n if total_n else float('nan')
        mae = total_sae / total_n if total_n else float('nan')
        out[T] = dict(mse=mse, mae=mae,
                      picks=picks,
                      per_ch=per_ch_best)
        print(f"    T={T}: MSE={mse:.4f} MAE={mae:.4f} "
              f"({len(configs)} cfgs/ch × {n_ch} ch in "
              f"{time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='+')
    ap.add_argument('--val-max', type=int, default=300)
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'results'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'forecast_v2_results.csv')
    summary_rows = []
    t_start = time.time()
    for ds in args.datasets:
        print(f"=== {ds} ===", flush=True)
        out = run_dataset(ds, val_max=args.val_max)
        for T, d in out.items():
            top_picks = sorted(d['picks'].items(), key=lambda x: -x[1])[:5]
            summary_rows.append((ds, T, d['mse'], d['mae'], top_picks))
    print(f"\nTotal: {time.time()-t_start:.0f}s", flush=True)

    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['dataset', 'T', 'test_mse', 'test_mae', 'top_picks'])
        for ds, T, mse, mae, picks in summary_rows:
            w.writerow([ds, T, mse, mae, str(picks)])
    print(f"Wrote {csv_path}")

    skolr_target = {
        'ETTh1':   {48: (0.333, 0.373), 96: (0.371, 0.398), 144: (0.405, 0.417), 192: (0.422, 0.432)},
        'ETTh2':   {48: (0.238, 0.306), 96: (0.299, 0.352), 144: (0.335, 0.377), 192: (0.365, 0.397)},
        'ETTm1':   {48: (0.280, 0.330), 96: (0.289, 0.340), 144: (0.319, 0.361), 192: (0.328, 0.373)},
        'ETTm2':   {48: (0.134, 0.228), 96: (0.171, 0.255), 144: (0.241, 0.304), 192: (0.241, 0.304)},
        'ECL':     {48: (0.137, 0.229), 96: (0.132, 0.225), 144: (0.143, 0.236), 192: (0.149, 0.244)},
        'Traffic': {48: (0.400, 0.258), 96: (0.368, 0.248), 144: (0.375, 0.255), 192: (0.377, 0.256)},
        'Weather': {48: (0.131, 0.170), 96: (0.154, 0.202), 144: (0.172, 0.220), 192: (0.193, 0.241)},
        'ILI':     {24: (1.556, 0.760), 36: (1.462, 0.728), 48: (1.537, 0.798), 60: (2.187, 0.995)},
    }
    print(f"\n=== v2 vs SKOLR Table 1 ===")
    print(f"{'dataset':>10s}  {'T':>4s}  {'GDC MSE':>9s}  {'GDC MAE':>9s}  "
          f"{'SKOLR MSE':>9s}  {'SKOLR MAE':>9s}  top picks (k:cfg : count)")
    for ds, T, mse, mae, picks in summary_rows:
        sk = skolr_target.get(ds, {}).get(T, (None, None))
        sk_mse = f"{sk[0]:.3f}" if sk[0] is not None else "—"
        sk_mae = f"{sk[1]:.3f}" if sk[1] is not None else "—"
        pick_summary = ", ".join(
            f"{k[0]}:L={dict(k[1])['window_len']}/s={dict(k[1])['sigma_frac']}/a={dict(k[1])['alpha']}={c}"
            for k, c in picks[:3]
        )
        print(f"{ds:>10s}  {T:>4d}  {mse:>9.4f}  {mae:>9.4f}  "
              f"{sk_mse:>9s}  {sk_mae:>9s}  {pick_summary}")


if __name__ == "__main__":
    main()
