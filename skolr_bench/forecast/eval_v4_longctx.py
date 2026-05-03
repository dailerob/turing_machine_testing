"""V4: Focused longer-context experiment.

Fix the dominant pick from v3 (raw, alpha=1.0, sigma=0.25), sweep ONLY
L_lookback over {2T, 4T, 8T, 16T, 32T} (capped at train_size/2).
Pick L_look per-channel by val MSE. ~5 configs per channel — fast.
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


def gdc_raw(history, window_len, sigma_frac, alpha, theta, h):
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


# Fixed: raw, alpha=1.0, sigma_frac=0.25, theta=0; sweep L_look only.
def task_one(args):
    dataset, T, ch_idx, val_series, test_series, L_look, val_max = args
    L_match = 2 * T   # L_match=2T was most common winner from v3
    cfg = dict(window_len=L_match, sigma_frac=0.25, alpha=1.0, theta=0.0)

    def eval_seq(series, max_w=None):
        s = np.asarray(series, dtype=np.float64)
        n = len(s)
        n_w = max(0, n - L_look - T + 1)
        if n_w == 0: return 0.0, 0.0, 0
        idxs = np.linspace(0, n_w - 1, max_w, dtype=int) if max_w and max_w < n_w else np.arange(n_w)
        sse = sae = 0.0; n_pts = 0
        for i in idxs:
            hist = s[i:i + L_look]
            truth = s[i + L_look:i + L_look + T]
            try: pred = gdc_raw(hist, h=T, **cfg)
            except Exception: continue
            d = truth - pred
            sse += float(np.sum(d**2)); sae += float(np.sum(np.abs(d)))
            n_pts += T
        return sse, sae, n_pts

    val_sse, val_sae, val_n = eval_seq(val_series, max_w=val_max)
    if val_n == 0: return (dataset, T, ch_idx, L_look, float('inf'), 0.0, 0.0, 0)
    test_sse, test_sae, test_n = eval_seq(test_series)
    return (dataset, T, ch_idx, L_look, val_sse / val_n, test_sse, test_sae, test_n)


def run_dataset(dataset, val_max=300, n_workers=None):
    if n_workers is None: n_workers = max(1, os.cpu_count() or 4)
    horizons = HORIZONS[dataset]
    ratios = [2, 4, 8, 16, 32]
    max_seq = max(r * t for r in ratios for t in horizons)
    train, val, test, cols = load(dataset, seq_len=max_seq)
    n_ch = len(cols)
    out = {}
    print(f"  {dataset}: {n_ch} ch, train={train.shape}, val={val.shape}, test={test.shape}, "
          f"workers={n_workers}", flush=True)
    for T in horizons:
        L_options = [r * T for r in ratios]
        # Skip configs whose L_look exceeds half of train (so val has enough overlap)
        L_options = [L for L in L_options if L <= train.shape[0] // 2]
        tasks = []
        for ch in range(n_ch):
            for L in L_options:
                tasks.append((dataset, T, ch, val[:, ch], test[:, ch], L, val_max))
        t0 = time.time()
        with mp.Pool(processes=n_workers) as pool:
            results = list(pool.imap_unordered(task_one, tasks, chunksize=4))
        per_ch = {}  # ch -> (val_mse, L_look, ts_sse, ts_sae, ts_n)
        for r in results:
            ds, t_, ch, L, vm, tsse, tsae, tn = r
            if ch not in per_ch or vm < per_ch[ch][0]:
                per_ch[ch] = (vm, L, tsse, tsae, tn)
        L_picks = defaultdict(int)
        total_sse = total_sae = 0.0; total_n = 0
        for ch in range(n_ch):
            v = per_ch.get(ch)
            if v is None: continue
            total_sse += v[2]; total_sae += v[3]; total_n += v[4]
            L_picks[v[1]] += 1
        mse = total_sse / total_n if total_n else float('nan')
        mae = total_sae / total_n if total_n else float('nan')
        out[T] = dict(mse=mse, mae=mae, picks=dict(L_picks))
        print(f"    T={T}: MSE={mse:.4f} MAE={mae:.4f}  "
              f"({len(L_options)} cfgs/ch × {n_ch} ch in {time.time()-t0:.0f}s)  "
              f"L_picks={dict(L_picks)}", flush=True)
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
            rows.append((ds, T, d['mse'], d['mae'], d['picks']))
    print(f"\nTotal: {time.time()-t0:.0f}s", flush=True)
    print(f"\n=== v4 (L_look only, fixed raw/α=1.0/σ=0.25) vs SKOLR ===")
    print(f"{'ds':>8s}  {'T':>4s}  {'MSE':>8s}  {'SKOLR':>8s}  L_pick distribution")
    for ds, T, mse, mae, picks in rows:
        sk = skolr.get(ds, {}).get(T)
        sk_str = f"{sk:.3f}" if sk else "—"
        print(f"{ds:>8s}  {T:>4d}  {mse:>8.4f}  {sk_str:>8s}  {picks}")


if __name__ == "__main__":
    main()
