"""V3: focused recipe, longer history.

Hypothesis: SKOLR's L=2T is short for hourly ETT data (T=96 = 4 days,
L=192 = 8 days). Longer context (L = 4T or 8T) might capture weekly
patterns the 2T window misses.

Small, focused config grid (no detrend/twostep/multiscale — they didn't
help in v2):
  raw  + diff
  L_lookback in {2T, 4T, 8T} (beyond SKOLR's 2T)
  L_match (the GDC window) in {T, 2T} of the L_lookback
  σ% in {0.05, 0.10, 0.25} for raw, {0.25, 0.5} for diff
  α in {1.0, 0.99, 0.95}

NOTE: we keep PREDICTION at L_match (the actual matching window length
used by GDC), but the historical context used by GDC's similarity
search comes from the entire L_lookback prefix (more candidate
historical windows to match against).
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
from loaders import load, HORIZONS  # noqa: E402


def _gdc_run(history_1d, window_len, sigma_frac, alpha, theta, h):
    sigma_per_step = float(np.std(history_1d)) * sigma_frac
    if sigma_per_step <= 0:
        return np.full(h, history_1d[-1])
    sigma_gdc = sigma_per_step * np.sqrt(window_len)
    beta = max(sigma_gdc ** 2, 1e-9)
    states = history_1d.reshape(-1, 1)
    gdc = GenerativeDenseChainTimeSeries(
        states, beta=beta, alpha=alpha, theta=theta,
        transition_type='self_loop',
        terminal_behavior='absorb',
        initial_dist='uniform')
    prime = history_1d[-window_len:].reshape(-1, 1)
    _, sd = gdc.forecast_gdc_style(prime, n_steps=h)
    nt = (~gdc.terminal_mask).astype(float)
    sd_nt = sd * nt[None, :]
    sd_sum = sd_nt.sum(axis=1, keepdims=True)
    safe = np.where(sd_sum > 1e-12, sd_sum, 1.0)
    return ((sd_nt / safe) @ gdc.states)[:, 0]


def gdc_raw(history, window_len, sigma_frac, alpha, theta, h):
    history = np.asarray(history, dtype=np.float64)
    if len(history) < window_len + 1:
        return np.full(h, history[-1])
    return _gdc_run(history, window_len, sigma_frac, alpha, theta, h)


def gdc_diff(history, window_len, sigma_frac, alpha, theta, h):
    history = np.asarray(history, dtype=np.float64)
    if len(history) < window_len + 2:
        return np.full(h, history[-1])
    d = np.diff(history)
    if len(d) < window_len + 1:
        return np.full(h, history[-1])
    forecast_d = _gdc_run(d, window_len, sigma_frac, alpha, theta, h)
    return history[-1] + np.cumsum(forecast_d)


PREDICT = {'raw': gdc_raw, 'diff': gdc_diff}


def predict(kind, cfg, history, h):
    return PREDICT[kind](history, h=h, **cfg)


def build_configs(T):
    """L_lookback ∈ {2T, 4T, 8T}, L_match ∈ {T, 2T}.
    Each cfg is (kind, dict(window_len=L_match, sigma_frac, alpha, theta=0)).
    L_lookback determines slicing of history before passing to GDC.
    """
    configs = []
    for L_look in [2*T, 4*T, 8*T]:
        for L_match in [T, 2*T]:
            if L_match > L_look:
                continue
            for s in [0.05, 0.10, 0.25]:
                for a in [1.0, 0.99, 0.95]:
                    configs.append(('raw', dict(window_len=L_match,
                                                sigma_frac=s, alpha=a,
                                                theta=0.0,
                                                _Llook=L_look)))
            for s in [0.25, 0.5]:
                for a in [1.0, 0.95, 0.9]:
                    configs.append(('diff', dict(window_len=L_match,
                                                  sigma_frac=s, alpha=a,
                                                  theta=0.0,
                                                  _Llook=L_look)))
    return configs


def eval_window_seq(series, kind, cfg, T, max_windows=None):
    """Slide windows over a 1-D series. Each window: take last L_look
    of history, predict T steps, compare to next T true points."""
    series = np.asarray(series, dtype=np.float64)
    L_look = cfg['_Llook']
    cfg_clean = {k: v for k, v in cfg.items() if not k.startswith('_')}
    n = len(series)
    n_windows = max(0, n - L_look - T + 1)
    if n_windows == 0:
        return 0.0, 0.0, 0
    if max_windows is not None and max_windows < n_windows:
        idxs = np.linspace(0, n_windows - 1, max_windows, dtype=int)
    else:
        idxs = np.arange(n_windows)
    sse = 0.0; sae = 0.0; n_pts = 0
    for i in idxs:
        hist = series[i:i + L_look]
        truth = series[i + L_look: i + L_look + T]
        try:
            pred = predict(kind, cfg_clean, hist, h=T)
        except Exception:
            continue
        d = truth - pred
        sse += float(np.sum(d ** 2)); sae += float(np.sum(np.abs(d)))
        n_pts += T
    return sse, sae, n_pts


def task_one(args):
    (dataset, T, ch_idx, val_series, test_series, kind, cfg, val_max) = args
    val_sse, val_sae, val_n = eval_window_seq(
        val_series, kind, cfg, T, max_windows=val_max)
    if val_n == 0:
        return (dataset, T, ch_idx, kind, cfg, float('inf'),
                0.0, 0.0, 0)
    val_mse = val_sse / val_n
    test_sse, test_sae, test_n = eval_window_seq(test_series, kind, cfg, T)
    return (dataset, T, ch_idx, kind, cfg, val_mse,
            test_sse, test_sae, test_n)


def run_dataset(dataset, val_max=300, n_workers=None):
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4))
    horizons = HORIZONS[dataset]
    # Load with the largest L_look needed (8 * max T)
    max_seq = 8 * max(horizons)
    train, val, test, cols = load(dataset, seq_len=max_seq)
    n_ch = len(cols)
    print(f"  {dataset}: {n_ch} ch, train={train.shape}, val={val.shape}, "
          f"test={test.shape}, n_workers={n_workers}", flush=True)
    out = {}
    for T in horizons:
        configs = build_configs(T)
        tasks = []
        for ch in range(n_ch):
            for kind, cfg in configs:
                tasks.append((dataset, T, ch, val[:, ch], test[:, ch],
                              kind, cfg, val_max))
        t0 = time.time()
        with mp.Pool(processes=n_workers) as pool:
            results = list(pool.imap_unordered(task_one, tasks, chunksize=8))
        per_ch_best = {}
        for r in results:
            ds, t_, ch, kind, cfg, val_mse, ts_sse, ts_sae, ts_n = r
            if ch not in per_ch_best or val_mse < per_ch_best[ch][0]:
                per_ch_best[ch] = (val_mse, kind, cfg, ts_sse, ts_sae, ts_n)
        total_sse = total_sae = 0.0; total_n = 0
        picks = defaultdict(int)
        for ch in range(n_ch):
            v = per_ch_best.get(ch)
            if v is None: continue
            total_sse += v[3]; total_sae += v[4]; total_n += v[5]
            picks[(v[1], v[2]['window_len'], v[2]['_Llook'],
                   v[2]['sigma_frac'], v[2]['alpha'])] += 1
        mse = total_sse / total_n if total_n else float('nan')
        mae = total_sae / total_n if total_n else float('nan')
        out[T] = dict(mse=mse, mae=mae, picks=picks)
        print(f"    T={T}: MSE={mse:.4f} MAE={mae:.4f} "
              f"({len(configs)} cfgs/ch × {n_ch} ch in "
              f"{time.time()-t0:.0f}s)", flush=True)
        for k, c in sorted(picks.items(), key=lambda x: -x[1])[:5]:
            print(f"      pick {c}× {k}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='+')
    ap.add_argument('--val-max', type=int, default=300)
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'results'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    t0 = time.time()
    for ds in args.datasets:
        print(f"=== {ds} ===", flush=True)
        out = run_dataset(ds, val_max=args.val_max)
        for T, d in out.items():
            rows.append((ds, T, d['mse'], d['mae']))
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)

    csv_path = os.path.join(args.out_dir, 'forecast_v3_results.csv')
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
            w.writerow(['dataset', 'T', 'test_mse', 'test_mae'])
        for r in rows: w.writerow(r)

    skolr = {
        'ETTh1': {48: (0.333, 0.373), 96: (0.371, 0.398), 144: (0.405, 0.417), 192: (0.422, 0.432)},
        'ETTh2': {48: (0.238, 0.306), 96: (0.299, 0.352), 144: (0.335, 0.377), 192: (0.365, 0.397)},
    }
    print(f"\n=== v3 vs SKOLR (longer L_lookback) ===")
    print(f"{'ds':>8s}  {'T':>4s}  {'GDC':>9s}  {'SKOLR':>9s}  {'gap':>6s}")
    for ds, T, mse, mae in rows:
        sk = skolr.get(ds, {}).get(T, (None, None))
        gap = (mse / sk[0] - 1) * 100 if sk[0] else float('nan')
        sk_str = f"{sk[0]:.3f}" if sk[0] else "—"
        print(f"{ds:>8s}  {T:>4d}  {mse:>9.4f}  {sk_str:>9s}  "
              f"{gap:>+5.1f}%")


if __name__ == "__main__":
    main()
