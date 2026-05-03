"""V6: full train+val state space + Numba-JIT'd batched forecast.

Architecture:
  - For each (dataset, T, channel, config):
    Build GDC ONCE with state space = train+val.
    Run forecast_many(numba) over ALL val primes and ALL test primes
    in one parallel batch each. ~32× speedup vs the per-prime numpy path.
  - No multiprocessing pool — Numba parallel uses all cores internally.
  - Picks best config per channel by val MSE.
"""
from __future__ import annotations
import os, sys, csv, time, argparse
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SKOLR_BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(SKOLR_BENCH)
sys.path.insert(0, HERE); sys.path.insert(0, SKOLR_BENCH); sys.path.insert(0, ROOT)
from generative_dense_chain_timeseries import GenerativeDenseChainTimeSeries  # noqa: E402
from loaders import load, HORIZONS  # noqa: E402
from gdc_numba import forecast_many  # noqa: E402


def make_state_series(train_col, val_col, max_seq):
    """Concat train + val (excluding the seq_len overlap from train)."""
    return np.concatenate([train_col, val_col[max_seq:]])


def primes_from_series(series, L_match, T):
    """Build (B, L_match) array of all sliding L_match-windows from a 1-D series
    such that there are T true points after each window.
    Returns (primes, truths_2d)  where truths_2d shape (B, T).
    """
    series = np.asarray(series, dtype=np.float64)
    n = len(series)
    B = max(0, n - L_match - T + 1)
    if B == 0:
        return np.empty((0, L_match)), np.empty((0, T))
    indices = np.arange(L_match)[None, :] + np.arange(B)[:, None]
    primes = series[indices]
    truth_idx = np.arange(L_match, L_match + T)[None, :] + np.arange(B)[:, None]
    truths = series[truth_idx]
    return primes, truths


def make_diff_series_and_primes(state_series, eval_series, L_match, T):
    """For 'diff' recipe: GDC's state space = diff(state_series).
    For each window in eval_series:
      prime_d = diff(eval[i:i+L_match+1])  (length L_match)
      anchor = eval[i+L_match]
      truth = eval[i+L_match+1 : i+L_match+1+T]
      pred = anchor + cumsum(forecast_d)
    """
    state_series = np.asarray(state_series, dtype=np.float64)
    eval_series = np.asarray(eval_series, dtype=np.float64)
    d_state = np.diff(state_series)
    n = len(eval_series)
    B = max(0, n - L_match - 1 - T + 1)
    if B == 0:
        return d_state, np.empty((0, L_match)), np.empty((0, T)), np.empty(0)
    # primes_d[b, t] = eval[b+t+1] - eval[b+t] for t in [0, L_match)
    diffs_eval = np.diff(eval_series)
    indices = np.arange(L_match)[None, :] + np.arange(B)[:, None]
    primes_d = diffs_eval[indices]  # (B, L_match)
    anchors = eval_series[L_match + np.arange(B)]
    truth_idx = np.arange(L_match + 1, L_match + 1 + T)[None, :] + np.arange(B)[:, None]
    truths = eval_series[truth_idx]
    return d_state, primes_d, truths, anchors


def build_gdc_1d(state_series_1d, window_len, sigma_frac, alpha, theta=0.0):
    sigma_per_step = float(np.std(state_series_1d)) * sigma_frac
    sigma_per_step = max(sigma_per_step, 1e-9)
    sigma_gdc = sigma_per_step * np.sqrt(window_len)
    beta = max(sigma_gdc ** 2, 1e-9)
    states = np.asarray(state_series_1d, dtype=np.float64).reshape(-1, 1)
    gdc = GenerativeDenseChainTimeSeries(
        states, beta=beta, alpha=alpha, theta=theta,
        transition_type='self_loop',
        terminal_behavior='absorb',
        initial_dist='uniform')
    return gdc


def eval_recipe_batched(state_series, eval_series, kind, L_match, T,
                        sigma_frac, alpha, max_windows=None):
    """Build GDC once, predict all windows in one Numba batch."""
    if kind == 'raw':
        gdc = build_gdc_1d(state_series, L_match, sigma_frac, alpha)
        primes, truths = primes_from_series(eval_series, L_match, T)
        if len(primes) == 0:
            return 0.0, 0.0, 0
        if max_windows and max_windows < len(primes):
            idxs = np.linspace(0, len(primes)-1, max_windows, dtype=int)
            primes = primes[idxs]; truths = truths[idxs]
        states_1d = gdc.states[:, 0]
        terminal_idx = int(np.where(gdc.terminal_mask)[0][-1])
        forecasts = forecast_many(states_1d, terminal_idx, gdc.beta,
                                   gdc.alpha, gdc.theta, primes, T)
        d = truths - forecasts
        sse = float(np.sum(d ** 2)); sae = float(np.sum(np.abs(d)))
        return sse, sae, primes.shape[0] * T
    elif kind == 'diff':
        d_state, primes_d, truths, anchors = make_diff_series_and_primes(
            state_series, eval_series, L_match, T)
        if len(primes_d) == 0:
            return 0.0, 0.0, 0
        gdc = build_gdc_1d(d_state, L_match, sigma_frac, alpha)
        if max_windows and max_windows < len(primes_d):
            idxs = np.linspace(0, len(primes_d)-1, max_windows, dtype=int)
            primes_d = primes_d[idxs]; truths = truths[idxs]; anchors = anchors[idxs]
        states_1d = gdc.states[:, 0]
        terminal_idx = int(np.where(gdc.terminal_mask)[0][-1])
        forecast_d = forecast_many(states_1d, terminal_idx, gdc.beta,
                                    gdc.alpha, gdc.theta, primes_d, T)
        # cumsum + anchor per row
        cum = np.cumsum(forecast_d, axis=1)
        forecasts = anchors[:, None] + cum
        d = truths - forecasts
        sse = float(np.sum(d ** 2)); sae = float(np.sum(np.abs(d)))
        return sse, sae, primes_d.shape[0] * T
    else:
        raise ValueError(kind)


def build_configs(T):
    """Focused grid based on v3 picks."""
    configs = []
    for kind in ['raw', 'diff']:
        for L in [T, 2*T, 4*T]:
            sigmas = [0.05, 0.10, 0.25] if kind == 'raw' else [0.25, 0.5]
            for s in sigmas:
                for a in [1.0, 0.99, 0.95]:
                    configs.append((kind, L, s, a))
    return configs


def run_dataset(dataset, val_max=300, verbose=True):
    horizons = HORIZONS[dataset]
    max_seq = 4 * max(horizons)
    train, val, test, cols = load(dataset, seq_len=max_seq)
    n_ch = len(cols)
    if verbose:
        print(f"  {dataset}: {n_ch} ch, train={train.shape}, val={val.shape}, "
              f"test={test.shape}", flush=True)
    out = {}
    for T in horizons:
        configs = build_configs(T)
        t0 = time.time()
        # For each channel, evaluate all configs, pick best by val MSE
        per_ch_test = {}
        for ch in range(n_ch):
            state_series = make_state_series(train[:, ch], val[:, ch], max_seq)
            best_val_mse = float('inf')
            best_cfg = None
            best_test = None
            for kind, L, s, a in configs:
                vsse, vsae, vn = eval_recipe_batched(
                    state_series, val[:, ch], kind, L, T, s, a,
                    max_windows=val_max)
                if vn == 0: continue
                vmse = vsse / vn
                if vmse < best_val_mse:
                    best_val_mse = vmse
                    best_cfg = (kind, L, s, a)
                    # Cache only val info; do test eval after picking
            # Test eval with the best config
            kind, L, s, a = best_cfg
            tsse, tsae, tn = eval_recipe_batched(
                state_series, test[:, ch], kind, L, T, s, a)
            per_ch_test[ch] = (best_cfg, tsse, tsae, tn)
        total_sse = total_sae = 0.0; total_n = 0
        picks = defaultdict(int)
        for ch, (cfg, tsse, tsae, tn) in per_ch_test.items():
            total_sse += tsse; total_sae += tsae; total_n += tn
            picks[cfg] += 1
        mse = total_sse / total_n if total_n else float('nan')
        mae = total_sae / total_n if total_n else float('nan')
        out[T] = dict(mse=mse, mae=mae, picks=dict(picks))
        if verbose:
            print(f"    T={T}: MSE={mse:.4f} MAE={mae:.4f} "
                  f"({len(configs)} cfgs/ch × {n_ch} ch in {time.time()-t0:.1f}s)",
                  flush=True)
            top = sorted(picks.items(), key=lambda x: -x[1])[:3]
            for cfg, c in top:
                print(f"      pick {c}x kind={cfg[0]} L={cfg[1]} s={cfg[2]} a={cfg[3]}",
                      flush=True)
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

    csv_path = os.path.join(args.out_dir, 'forecast_v6_results.csv')
    write_header = not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['dataset', 'T', 'test_mse', 'test_mae'])
        for r in rows: w.writerow(r)
    print(f"Wrote {csv_path}")

    skolr = {
        'ETTh1': {48: 0.333, 96: 0.371, 144: 0.405, 192: 0.422},
        'ETTh2': {48: 0.238, 96: 0.299, 144: 0.335, 192: 0.365},
        'ETTm1': {48: 0.280, 96: 0.289, 144: 0.319, 192: 0.328},
        'ETTm2': {48: 0.134, 96: 0.171, 144: 0.241, 192: 0.241},
        'ECL':   {48: 0.137, 96: 0.132, 144: 0.143, 192: 0.149},
        'Traffic':{48: 0.400, 96: 0.368, 144: 0.375, 192: 0.377},
        'Weather':{48: 0.131, 96: 0.154, 144: 0.172, 192: 0.193},
        'ILI':   {24: 1.556, 36: 1.462, 48: 1.537, 60: 2.187},
    }
    print(f"\n=== v6 (numba+full-state) vs SKOLR ===")
    print(f"{'ds':>10s}  {'T':>4s}  {'GDC MSE':>9s}  {'SKOLR':>9s}  {'gap':>6s}")
    for ds, T, mse, mae in rows:
        sk = skolr.get(ds, {}).get(T)
        sk_str = f"{sk:.3f}" if sk else "—"
        gap = (mse / sk - 1) * 100 if sk else float('nan')
        print(f"{ds:>10s}  {T:>4d}  {mse:>9.4f}  {sk_str:>9s}  {gap:>+5.1f}%")


if __name__ == "__main__":
    main()
