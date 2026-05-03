"""V5: state space = full (train + val), built ONCE per (channel, T, cfg).

Key change from v3: GDC is constructed once per (channel, T, recipe-config)
with state_space = concat(train, val). Then for each val and test window,
we just call forecast_gdc_style(prime). This is:

  - much faster (no per-window build cost)
  - much more historical context for similarity matching
    (~12k states for ETTh1 vs v3's max 1536)

Protocol:
  For each (channel, T):
    Sweep small config grid (L_match, sigma_frac, alpha, kind ∈ {raw, diff}):
      Build GDC once with full train+val as state space.
      Score on val windows (sliding, prime = val[i+L_look_val_offset : i+L_look_val_offset+L_match]).
      Pick best by val MSE.
    Apply to all test windows (prime = test[i:i+L_match], predict T ahead).

Note: val and test windows use the *same* prebuilt GDC for that config —
state space does NOT include the rolling test prefix.
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


def build_gdc(state_series_1d, window_len, sigma_frac, alpha, theta=0.0):
    """Build a GDC-TS once with the full historical 1-D series as states.
    Returns (gdc, beta) — gdc is reusable for many primes."""
    state_series_1d = np.asarray(state_series_1d, dtype=np.float64)
    sigma_per_step = float(np.std(state_series_1d)) * sigma_frac
    sigma_per_step = max(sigma_per_step, 1e-9)
    sigma_gdc = sigma_per_step * np.sqrt(window_len)
    beta = max(sigma_gdc ** 2, 1e-9)
    states = state_series_1d.reshape(-1, 1)
    gdc = GenerativeDenseChainTimeSeries(
        states, beta=beta, alpha=alpha, theta=theta,
        transition_type='self_loop',
        terminal_behavior='absorb',
        initial_dist='uniform')
    return gdc


def forecast_with(gdc, prime_1d, h):
    prime = np.asarray(prime_1d, dtype=np.float64).reshape(-1, 1)
    _, sd = gdc.forecast_gdc_style(prime, n_steps=h)
    nt = (~gdc.terminal_mask).astype(float)
    sd_nt = sd * nt[None, :]
    sd_sum = sd_nt.sum(axis=1, keepdims=True)
    safe = np.where(sd_sum > 1e-12, sd_sum, 1.0)
    return ((sd_nt / safe) @ gdc.states)[:, 0]


def eval_recipe(state_series, eval_series, window_len, sigma_frac, alpha,
                kind, T, max_windows=None):
    """For one config, build a GDC on `state_series` (full train+val) and
    score it by sliding T-prediction windows over `eval_series`.

    `eval_series` is a 1-D series we slide over (could be val or test).
    For each starting position i:
        prime = eval_series[i:i+window_len]
        truth = eval_series[i+window_len:i+window_len+T]
        prediction by GDC.

    For 'diff' recipe: build GDC on diff(state_series); prime = diff(eval[i:i+window_len+1]);
    forecast h diffs; cumsum onto eval[i+window_len].
    """
    if kind == 'raw':
        gdc = build_gdc(state_series, window_len, sigma_frac, alpha)
        eval_series = np.asarray(eval_series, dtype=np.float64)
        n = len(eval_series); n_w = max(0, n - window_len - T + 1)
        if n_w == 0: return 0.0, 0.0, 0
        idxs = np.linspace(0, n_w-1, max_windows, dtype=int) if max_windows and max_windows < n_w else np.arange(n_w)
        sse = sae = 0.0; n_pts = 0
        for i in idxs:
            prime = eval_series[i:i+window_len]
            truth = eval_series[i+window_len:i+window_len+T]
            try: pred = forecast_with(gdc, prime, h=T)
            except Exception: continue
            d = truth - pred
            sse += float(np.sum(d**2)); sae += float(np.sum(np.abs(d)))
            n_pts += T
        return sse, sae, n_pts
    elif kind == 'diff':
        d_state = np.diff(state_series)
        gdc = build_gdc(d_state, window_len, sigma_frac, alpha)
        eval_series = np.asarray(eval_series, dtype=np.float64)
        n = len(eval_series); n_w = max(0, n - window_len - T + 1)
        if n_w == 0: return 0.0, 0.0, 0
        idxs = np.linspace(0, n_w-1, max_windows, dtype=int) if max_windows and max_windows < n_w else np.arange(n_w)
        sse = sae = 0.0; n_pts = 0
        for i in idxs:
            window = eval_series[i:i+window_len+1]  # need +1 for the diff sequence
            if len(window) < window_len + 1:
                continue
            prime_d = np.diff(window)
            anchor = eval_series[i+window_len]
            truth = eval_series[i+window_len+1:i+window_len+1+T]
            if len(truth) < T:
                continue
            try: pred_d = forecast_with(gdc, prime_d, h=T)
            except Exception: continue
            pred = anchor + np.cumsum(pred_d)
            d = truth - pred
            sse += float(np.sum(d**2)); sae += float(np.sum(np.abs(d)))
            n_pts += T
        return sse, sae, n_pts
    else:
        raise ValueError(kind)


def task_one(args):
    (dataset, T, ch_idx, state_series, val_series, test_series,
     kind, window_len, sigma_frac, alpha, val_max) = args
    val_sse, val_sae, val_n = eval_recipe(
        state_series, val_series, window_len, sigma_frac, alpha, kind, T,
        max_windows=val_max)
    if val_n == 0:
        return (dataset, T, ch_idx, kind, window_len, sigma_frac, alpha,
                float('inf'), 0.0, 0.0, 0)
    val_mse = val_sse / val_n
    test_sse, test_sae, test_n = eval_recipe(
        state_series, test_series, window_len, sigma_frac, alpha, kind, T)
    return (dataset, T, ch_idx, kind, window_len, sigma_frac, alpha,
            val_mse, test_sse, test_sae, test_n)


def build_configs(T):
    """Small focused grid based on v3 picks: raw dominates, σ=0.25, α∈{1.0, 0.99}."""
    configs = []
    for kind in ['raw', 'diff']:
        for L in [T, 2*T, 4*T]:  # match-window length
            for s in [0.05, 0.10, 0.25] if kind == 'raw' else [0.25, 0.5]:
                for a in [1.0, 0.99, 0.95]:
                    configs.append((kind, L, s, a))
    return configs


def run_dataset(dataset, val_max=300, n_workers=None):
    if n_workers is None:
        n_workers = max(1, os.cpu_count() or 4)
    horizons = HORIZONS[dataset]
    # Load with the largest L_match needed (4 * max T)
    max_seq = 4 * max(horizons)
    train, val, test, cols = load(dataset, seq_len=max_seq)
    n_ch = len(cols)
    print(f"  {dataset}: {n_ch} ch, train={train.shape}, val={val.shape}, "
          f"test={test.shape}, workers={n_workers}", flush=True)
    out = {}
    for T in horizons:
        configs = build_configs(T)
        # state_series for each channel = full (train + val)
        # Note: load() returns val with `max_seq` rows of overlap from train,
        # so concat(train, val) double-counts that overlap region.
        # To avoid: we build state_series as just train followed by val[max_seq:].
        # Actually val starts at b1s[1] = num_train - max_seq, so val[max_seq:]
        # corresponds to data starting at num_train (the actual val region).
        tasks = []
        for ch in range(n_ch):
            state_series = np.concatenate([train[:, ch], val[max_seq:, ch]])
            for kind, L, s, a in configs:
                tasks.append((dataset, T, ch, state_series,
                              val[:, ch], test[:, ch], kind, L, s, a, val_max))
        t0 = time.time()
        with mp.Pool(processes=n_workers) as pool:
            results = list(pool.imap_unordered(task_one, tasks, chunksize=4))
        per_ch = {}
        for r in results:
            ds, t_, ch, kind, L, s, a, vm, tsse, tsae, tn = r
            if ch not in per_ch or vm < per_ch[ch][0]:
                per_ch[ch] = (vm, kind, L, s, a, tsse, tsae, tn)
        total_sse = total_sae = 0.0; total_n = 0
        picks = defaultdict(int)
        for ch in range(n_ch):
            v = per_ch.get(ch)
            if v is None: continue
            total_sse += v[5]; total_sae += v[6]; total_n += v[7]
            picks[(v[1], v[2], v[3], v[4])] += 1
        mse = total_sse / total_n if total_n else float('nan')
        mae = total_sae / total_n if total_n else float('nan')
        out[T] = dict(mse=mse, mae=mae, picks=dict(picks))
        print(f"    T={T}: MSE={mse:.4f} MAE={mae:.4f} "
              f"({len(configs)} cfgs/ch × {n_ch} ch in {time.time()-t0:.0f}s)",
              flush=True)
        for k, c in sorted(picks.items(), key=lambda x: -x[1])[:5]:
            print(f"      pick {c}× kind={k[0]} L={k[1]} σ={k[2]} α={k[3]}",
                  flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='+')
    ap.add_argument('--val-max', type=int, default=300)
    args = ap.parse_args()
    skolr = {
        'ETTh1': {48: 0.333, 96: 0.371, 144: 0.405, 192: 0.422},
        'ETTh2': {48: 0.238, 96: 0.299, 144: 0.335, 192: 0.365},
    }
    rows = []
    t0 = time.time()
    for ds in args.datasets:
        print(f"=== {ds} ===", flush=True)
        out = run_dataset(ds, val_max=args.val_max)
        for T, d in out.items():
            rows.append((ds, T, d['mse'], d['mae']))
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)
    print(f"\n=== v5 (state=full train+val, built once per cfg) vs SKOLR ===")
    print(f"{'ds':>8s}  {'T':>4s}  {'GDC MSE':>9s}  {'GDC MAE':>9s}  "
          f"{'SKOLR':>9s}  {'gap':>6s}")
    for ds, T, mse, mae in rows:
        sk = skolr.get(ds, {}).get(T)
        gap = (mse / sk - 1) * 100 if sk else float('nan')
        sk_str = f"{sk:.3f}" if sk else "—"
        print(f"{ds:>8s}  {T:>4d}  {mse:>9.4f}  {mae:>9.4f}  {sk_str:>9s}  "
              f"{gap:>+5.1f}%")


if __name__ == "__main__":
    main()
