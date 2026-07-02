"""Evaluate GDC-TS on the 4 NLDS using SKOLR's protocol.

Per (system, seed):
  1. Load 14k/2k/4k trajectory.
  2. Per-dim StandardScaler fit on train, applied to all.
  3. Sweep GDC configs (raw + diff variants) on a sliding window over val.
  4. Pick best config (by val MSE) per system/seed.
  5. Apply to sliding window over test.
  6. Report MSE, MAE on standardized scale (matches SKOLR Table 11).

Aggregated across seeds: mean ± std per system.
"""
from __future__ import annotations
import os, sys, csv, time, argparse
import numpy as np
import multiprocessing as mp
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))            # skolr_bench/nlds/
SKOLR_BENCH = os.path.dirname(HERE)                            # skolr_bench/
ROOT = os.path.dirname(SKOLR_BENCH)                            # repo root (has GDC module)
sys.path.insert(0, HERE); sys.path.insert(0, SKOLR_BENCH); sys.path.insert(0, ROOT)
from generative_dense_chain_timeseries import GenerativeDenseChainTimeSeries  # noqa: E402

DATA_DIR = os.path.join(HERE, 'nlds_data')
SYSTEMS = ['pendulum', 'duffing', 'lotka_volterra', 'lorenz63']
SEEDS = [0, 1, 2, 3, 4]
SEQ_LEN = 96   # L  (SKOLR run_longExp default)
PRED_LEN = 96  # T


def gdc_raw_forecast(history, window_len, sigma_frac, alpha, theta, h,
                     alpha_fc=None):
    """GDC-TS on raw 1-D history. Returns h-step forecast.

    alpha_fc (optional) sets a separate forecast-roll-out alpha (dual-alpha);
    None reuses alpha (single-alpha)."""
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
    _, sd = gdc.forecast_gdc_style(prime, n_steps=h, alpha_fc=alpha_fc)
    nt = (~gdc.terminal_mask).astype(float)
    sd_nt = sd * nt[None, :]
    sd_sum = sd_nt.sum(axis=1, keepdims=True)
    safe = np.where(sd_sum > 1e-12, sd_sum, 1.0)
    return ((sd_nt / safe) @ gdc.states)[:, 0]


def gdc_diff_forecast(history, window_len, sigma_frac, alpha, theta, h,
                      alpha_fc=None):
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
    _, sd = gdc.forecast_gdc_style(prime, n_steps=h, alpha_fc=alpha_fc)
    nt = (~gdc.terminal_mask).astype(float)
    sd_nt = sd * nt[None, :]
    sd_sum = sd_nt.sum(axis=1, keepdims=True)
    safe = np.where(sd_sum > 1e-12, sd_sum, 1.0)
    forecast_d = ((sd_nt / safe) @ gdc.states)[:, 0]
    return history[-1] + np.cumsum(forecast_d)


# Candidate configs: raw + diff, alpha sweep, sigma sweep.
# NOTE (P1, 2026-06): dual-alpha (alpha_ctx < 1, alpha_fc = 1.0) was tested and
# is NEUTRAL on NLDS — the val-pick selects equivalent configs and the results
# are identical to single-alpha (pendulum 0.0003, duffing 0.0005, LV 0.0000,
# lorenz 1.171). It is therefore OFF by default (set GDC_DUAL_ALPHA=1 to add the
# dual candidates). Diff is a no-op for alpha_fc regardless; dual-alpha actively
# helps only TM and dysts. See paper/PROTOCOL_STANDARDIZATION.md.
_DUAL = os.environ.get('GDC_DUAL_ALPHA', '0') == '1'
CONFIGS = []
for L in [48, 96]:
    for s in [0.05, 0.1, 0.25]:
        for a in [1.0, 0.99, 0.95, 0.9]:
            CONFIGS.append(('raw', dict(window_len=L, sigma_frac=s,
                                         alpha=a, theta=0.0, alpha_fc=a)))
        if _DUAL:
            for a in [0.99, 0.95, 0.9]:
                CONFIGS.append(('raw', dict(window_len=L, sigma_frac=s,
                                             alpha=a, theta=0.0, alpha_fc=1.0)))
for L in [48, 96]:
    for s in [0.25, 0.5, 1.0]:
        for a in [1.0, 0.99, 0.95, 0.9]:
            CONFIGS.append(('diff', dict(window_len=L, sigma_frac=s,
                                          alpha=a, theta=0.0, alpha_fc=a)))
        if _DUAL:
            for a in [0.99, 0.95, 0.9]:
                CONFIGS.append(('diff', dict(window_len=L, sigma_frac=s,
                                              alpha=a, theta=0.0, alpha_fc=1.0)))


def predict(kind, cfg, history, h):
    if kind == 'raw':
        return gdc_raw_forecast(history, h=h, **cfg)
    return gdc_diff_forecast(history, h=h, **cfg)


def eval_window_seq(data_split, kind, cfg, seq_len=SEQ_LEN, pred_len=PRED_LEN,
                    max_windows=None, stride=1):
    """Slide window over a single 1-D series; return per-window (mse, mae) means."""
    data_split = np.asarray(data_split, dtype=np.float64)
    n = len(data_split)
    n_windows = max(0, n - seq_len - pred_len + 1)
    if n_windows == 0:
        return float('nan'), float('nan'), 0
    if max_windows is not None:
        # Subsample windows uniformly
        idxs = np.linspace(0, n_windows - 1, min(max_windows, n_windows), dtype=int)
    else:
        idxs = np.arange(0, n_windows, stride)
    sse = 0.0; sae = 0.0; n_pts = 0
    for i in idxs:
        hist = data_split[i:i + seq_len]
        truth = data_split[i + seq_len: i + seq_len + pred_len]
        pred = predict(kind, cfg, hist, h=pred_len)
        d = truth - pred
        sse += float(np.sum(d ** 2)); sae += float(np.sum(np.abs(d)))
        n_pts += pred_len
    return sse / n_pts, sae / n_pts, len(idxs)


def process_system_seed(args):
    name, seed = args
    npz = np.load(os.path.join(DATA_DIR, f'{name}_seed{seed}.npz'),
                  allow_pickle=True)
    train = npz['train']; val = npz['val']; test = npz['test']
    n_dims = train.shape[1]
    # Per-dim standardize using train stats
    mu = train.mean(axis=0); sd = train.std(axis=0); sd = np.where(sd > 1e-9, sd, 1.0)
    train_s = (train - mu) / sd
    val_s = (val - mu) / sd
    test_s = (test - mu) / sd
    # Concatenate train+val for inference history (val window border per SKOLR)
    train_then_val = np.concatenate([train_s, val_s], axis=0)
    # For each dim, sweep configs on val (val border = train_s, eval over val windows
    # using the SKOLR border1 = num_train - seq_len).
    val_border1 = len(train_s) - SEQ_LEN
    val_data = np.concatenate([train_s[val_border1:], val_s], axis=0)
    test_border_data = np.concatenate(
        [train_then_val[len(train_then_val) - SEQ_LEN:], test_s], axis=0)
    # i.e. test data has its own (seq_len) head from end of train+val
    # to enable the first window in the test split.
    # Per-dim val sweep: pick argmin val MSE per (dim).
    per_dim_picks = []
    per_dim_test_mse = []; per_dim_test_mae = []
    for d in range(n_dims):
        # Val sweep with stride to keep cost tractable (~150 windows max per dim).
        val_scores = []
        for kind, cfg in CONFIGS:
            mse, mae, n_w = eval_window_seq(val_data[:, d], kind, cfg,
                                             max_windows=120)
            val_scores.append((mse, kind, cfg))
        val_scores.sort(key=lambda x: x[0])
        best_mse, best_kind, best_cfg = val_scores[0]
        per_dim_picks.append((best_kind, best_cfg))
        # Test eval over ALL windows for the picked config
        test_mse, test_mae, _ = eval_window_seq(test_border_data[:, d],
                                                 best_kind, best_cfg)
        per_dim_test_mse.append(test_mse)
        per_dim_test_mae.append(test_mae)
    avg_mse = float(np.mean(per_dim_test_mse))
    avg_mae = float(np.mean(per_dim_test_mae))
    return name, seed, avg_mse, avg_mae, per_dim_picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'nlds_results.csv'))
    args = ap.parse_args()
    tasks = [(name, seed) for name in SYSTEMS for seed in SEEDS]
    n_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
    print(f"NLDS eval: {len(tasks)} tasks × {len(CONFIGS)} configs, "
          f"{n_workers} workers", flush=True)
    rows = []
    t0 = time.time()
    with mp.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(process_system_seed, tasks):
            name, seed, mse, mae, picks = r
            print(f"  {name:>16s} seed={seed}: MSE={mse:.4f} MAE={mae:.4f}  "
                  f"picks={picks}", flush=True)
            rows.append((name, seed, mse, mae, picks))
    print(f"Done in {time.time()-t0:.1f}s", flush=True)

    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['system', 'seed', 'test_mse', 'test_mae', 'picks'])
        for r in rows: w.writerow(r)

    # Aggregate per system: mean ± std
    print(f"\n=== NLDS GDC results (mean ± std across 5 seeds) ===")
    print(f"{'system':>16s}  {'MSE':>16s}  {'MAE':>16s}  "
          f"  SKOLR target  KooPA target")
    targets = {  # SKOLR Table 11
        'pendulum':       ((0.0001, 0.0083), (0.0039, 0.0470)),
        'duffing':        ((0.0047, 0.0518), (0.0365, 0.1479)),
        'lotka_volterra': ((0.0018, 0.0354), (0.0178, 0.1050)),
        'lorenz63':       ((0.9740, 0.7941), (1.0937, 0.8325)),
    }
    by_sys = defaultdict(lambda: ([], []))
    for name, seed, mse, mae, _ in rows:
        by_sys[name][0].append(mse); by_sys[name][1].append(mae)
    for name in SYSTEMS:
        mses, maes = by_sys[name]
        m_mse = float(np.mean(mses)); s_mse = float(np.std(mses))
        m_mae = float(np.mean(maes)); s_mae = float(np.std(maes))
        sk = targets[name][0]; kp = targets[name][1]
        print(f"{name:>16s}  {m_mse:.4f}±{s_mse:.4f}  "
              f"{m_mae:.4f}±{s_mae:.4f}  "
              f"  {sk[0]}/{sk[1]}  {kp[0]}/{kp[1]}")


if __name__ == "__main__":
    main()
