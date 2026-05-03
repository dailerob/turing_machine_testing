"""Per-channel val-tuned GDC for SKOLR forecasting benchmarks.

Protocol (matches SKOLR Table 1):
  L = 2T lookback, T-step prediction, sliding window over test split.
  StandardScaler fit on train, applied to all splits.
  MSE/MAE on standardized data.
  Multivariate via channel-independence (per-channel univariate forecast,
  averaged across channels and windows).

For each (dataset, T):
  Per channel:
    Sweep small candidate set on subsampled val windows -> pick best by val MSE.
  Apply picks to ALL test windows; report mean MSE and MAE.
"""
from __future__ import annotations
import os, sys, csv, time, argparse
import numpy as np
import multiprocessing as mp
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))           # skolr_bench/forecast/
SKOLR_BENCH = os.path.dirname(HERE)                          # skolr_bench/
ROOT = os.path.dirname(SKOLR_BENCH)                          # repo root
sys.path.insert(0, HERE); sys.path.insert(0, SKOLR_BENCH); sys.path.insert(0, ROOT)
from generative_dense_chain_timeseries import GenerativeDenseChainTimeSeries  # noqa: E402
from loaders import load, HORIZONS, DATASETS  # noqa: E402


# Same gdc_raw / gdc_diff helpers as nlds_eval.py
def gdc_raw_forecast(history, window_len, sigma_frac, alpha, theta, h):
    history = np.asarray(history, dtype=np.float64)
    if len(history) < window_len + 1:
        return np.full(h, history[-1])
    sigma_per_step = float(np.std(history)) * sigma_frac
    if sigma_per_step <= 0:
        return np.full(h, history[-1])
    sigma_gdc = sigma_per_step * np.sqrt(window_len)
    beta = max(sigma_gdc ** 2, 1e-9)
    states = history.reshape(-1, 1)
    gdc = GenerativeDenseChainTimeSeries(
        states, beta=beta, alpha=alpha, theta=theta,
        transition_type='self_loop',
        terminal_behavior='absorb',
        initial_dist='uniform')
    prime = history[-window_len:].reshape(-1, 1)
    _, sd = gdc.forecast_gdc_style(prime, n_steps=h)
    nt = (~gdc.terminal_mask).astype(float)
    sd_nt = sd * nt[None, :]
    sd_sum = sd_nt.sum(axis=1, keepdims=True)
    safe = np.where(sd_sum > 1e-12, sd_sum, 1.0)
    return ((sd_nt / safe) @ gdc.states)[:, 0]


def gdc_diff_forecast(history, window_len, sigma_frac, alpha, theta, h):
    history = np.asarray(history, dtype=np.float64)
    if len(history) < window_len + 2:
        return np.full(h, history[-1])
    d = np.diff(history)
    if len(d) < window_len + 1:
        return np.full(h, history[-1])
    sigma_per_step = float(np.std(d)) * sigma_frac
    if sigma_per_step <= 0:
        return np.full(h, history[-1])
    sigma_gdc = sigma_per_step * np.sqrt(window_len)
    beta = max(sigma_gdc ** 2, 1e-9)
    states = d.reshape(-1, 1)
    gdc = GenerativeDenseChainTimeSeries(
        states, beta=beta, alpha=alpha, theta=theta,
        transition_type='self_loop',
        terminal_behavior='absorb',
        initial_dist='uniform')
    prime = d[-window_len:].reshape(-1, 1)
    _, sd = gdc.forecast_gdc_style(prime, n_steps=h)
    nt = (~gdc.terminal_mask).astype(float)
    sd_nt = sd * nt[None, :]
    sd_sum = sd_nt.sum(axis=1, keepdims=True)
    safe = np.where(sd_sum > 1e-12, sd_sum, 1.0)
    forecast_d = ((sd_nt / safe) @ gdc.states)[:, 0]
    return history[-1] + np.cumsum(forecast_d)


def predict(kind, cfg, history, h):
    if kind == 'raw':
        return gdc_raw_forecast(history, h=h, **cfg)
    return gdc_diff_forecast(history, h=h, **cfg)


def build_configs(seq_len):
    """Small candidate config grid."""
    configs = []
    # raw with α near 1 (good for cyclic/smooth)
    for L in [seq_len // 2, seq_len]:
        for s in [0.05, 0.10, 0.25]:
            for a in [1.0, 0.99, 0.95]:
                configs.append(('raw', dict(window_len=L, sigma_frac=s,
                                             alpha=a, theta=0.0)))
    # diff with various alphas (good for drift/noise)
    for L in [seq_len // 2, seq_len]:
        for s in [0.25, 0.5]:
            for a in [1.0, 0.95, 0.9]:
                configs.append(('diff', dict(window_len=L, sigma_frac=s,
                                              alpha=a, theta=0.0)))
    return configs


def eval_window_seq(series, kind, cfg, seq_len, pred_len,
                    max_windows=None):
    """Slide window over a 1-D series; return total SSE, SAE, n_pts."""
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
        pred = predict(kind, cfg, hist, h=pred_len)
        d = truth - pred
        sse += float(np.sum(d ** 2)); sae += float(np.sum(np.abs(d)))
        n_pts += pred_len
    return sse, sae, n_pts


def process_channel(args):
    """Sweep + test for one (dataset, T, channel)."""
    dataset, T, ch_idx, val_series, test_series, val_max = args
    seq_len = 2 * T
    pred_len = T
    configs = build_configs(seq_len)
    # Val sweep — subsample to keep cost down
    best = (float('inf'), None, None)
    for kind, cfg in configs:
        sse, sae, n_pts = eval_window_seq(val_series, kind, cfg,
                                          seq_len, pred_len,
                                          max_windows=val_max)
        if n_pts == 0: continue
        mse = sse / n_pts
        if mse < best[0]:
            best = (mse, kind, cfg)
    if best[1] is None:
        return dataset, T, ch_idx, float('nan'), float('nan'), 0, ('none', {})
    # Full test eval
    test_sse, test_sae, test_n = eval_window_seq(
        test_series, best[1], best[2], seq_len, pred_len)
    return (dataset, T, ch_idx, test_sse, test_sae, test_n,
            (best[1], best[2]))


def run_dataset(dataset, val_max=200, n_workers=None, channels_subset=None):
    """Returns dict T -> dict {mse, mae, picks}."""
    if n_workers is None:
        n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    horizons = HORIZONS[dataset]
    # Load with the largest seq_len so we have enough overlap for all T;
    # then per-T we just slice the prefix accordingly.
    max_seq = 2 * max(horizons)
    train, val, test, cols = load(dataset, seq_len=max_seq)
    n_ch = len(cols)
    if channels_subset is not None:
        ch_indices = list(channels_subset)
    else:
        ch_indices = list(range(n_ch))
    print(f"  {dataset}: {n_ch} channels (using {len(ch_indices)}), "
          f"train={train.shape}, val={val.shape}, test={test.shape}", flush=True)
    out = {}
    t0 = time.time()
    for T in horizons:
        seq_len = 2 * T
        # Build per-channel val/test prefix slices that exactly match
        # the SKOLR convention (border overlaps included by load() with
        # max_seq; trim to seq_len overlap for this T).
        # The val/test arrays returned by load() include max_seq-rows of
        # prefix; for this T we take all of val/test (the extra prefix
        # is harmless — it just means more starting positions for windows).
        tasks = []
        for ch in ch_indices:
            tasks.append((dataset, T, ch, val[:, ch], test[:, ch], val_max))
        with mp.Pool(processes=n_workers) as pool:
            results = list(pool.imap_unordered(process_channel, tasks))
        # Aggregate: pool SSE/SAE across channels and windows
        total_sse = 0.0; total_sae = 0.0; total_n = 0
        picks_count = defaultdict(int)
        per_channel_picks = []
        for r in results:
            ds, t_, ch, sse, sae, n_pts, pick = r
            total_sse += sse; total_sae += sae; total_n += n_pts
            picks_count[(pick[0], tuple(sorted(pick[1].items())))] += 1
            per_channel_picks.append((ch, pick))
        mse = total_sse / total_n if total_n else float('nan')
        mae = total_sae / total_n if total_n else float('nan')
        out[T] = dict(mse=mse, mae=mae, picks=picks_count,
                      per_channel_picks=per_channel_picks)
        elapsed = time.time() - t0
        print(f"    T={T}: MSE={mse:.4f}  MAE={mae:.4f}  "
              f"[{elapsed:.0f}s elapsed]", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='+',
                    help=f"one or more of {list(DATASETS.keys())}")
    ap.add_argument('--val-max', type=int, default=200,
                    help='subsample to at most this many val windows for picking')
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'results'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_rows = []
    t0 = time.time()
    for ds in args.datasets:
        print(f"=== {ds} ===", flush=True)
        out = run_dataset(ds, val_max=args.val_max)
        for T, d in out.items():
            all_rows.append((ds, T, d['mse'], d['mae']))

    print(f"\nDone in {time.time()-t0:.1f}s", flush=True)
    csv_path = os.path.join(args.out_dir, 'forecast_results.csv')
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['dataset', 'T', 'test_mse', 'test_mae'])
        for r in all_rows: w.writerow(r)
    print(f"Wrote {csv_path}")

    # SKOLR Table 1 reference numbers for comparison
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
    print(f"\n=== Comparison vs SKOLR Table 1 ===")
    print(f"{'dataset':>10s}  {'T':>4s}  {'GDC MSE':>9s}  {'GDC MAE':>9s}  "
          f"{'SKOLR MSE':>9s}  {'SKOLR MAE':>9s}")
    for ds, T, mse, mae in all_rows:
        sk = skolr_target.get(ds, {}).get(T, (None, None))
        sk_mse = f"{sk[0]:.3f}" if sk[0] is not None else "—"
        sk_mae = f"{sk[1]:.3f}" if sk[1] is not None else "—"
        print(f"{ds:>10s}  {T:>4d}  {mse:>9.4f}  {mae:>9.4f}  "
              f"{sk_mse:>9s}  {sk_mae:>9s}")


if __name__ == "__main__":
    main()
