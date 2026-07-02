"""Context parroting on the SKOLR NLDS benchmark, leakage-free.

Mirrors `skolr_bench/nlds/nlds_eval.py` (GDC-NLDS) protocol exactly:
  - Per (system, seed): 14k/2k/4k splits, per-dim StandardScaler on train.
  - Sliding L=96, T=96 windows over val (sub-sampled) for tuning, all
    test windows for evaluation.
  - Per-dim val pick → test eval → mean-across-dims per (system, seed).
  - Aggregate mean ± std across 5 seeds.

The crucial difference from the SKOLR/Informer parrot sweep is that
GDC-NLDS does *in-context* forecasting: each test window's GDC sees
ONLY the 96-point lookback as its state space. To match that, we use
**autoregressive in-context parroting**: at each forecast step, the
search pool is the current 96-point context (lookback augmented with
previous predictions), and we copy the next value of the best L'-prefix
match, then slide.

Variants swept (mirrors GDC's per-dim grid):
  raw  L' ∈ {12, 24, 48}, k ∈ {1, 5}
  diff L' ∈ {12, 24, 48}, k ∈ {1, 5}

(L' is the *sub-window* length used for matching within the 96-point
context; L=96 is the SKOLR lookback, fixed.)
"""
from __future__ import annotations
import os, sys, csv, time, argparse
import numpy as np
import multiprocessing as mp
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))            # skolr_bench/nlds/
SKOLR_BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(SKOLR_BENCH)
sys.path.insert(0, HERE); sys.path.insert(0, SKOLR_BENCH); sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(HERE, 'nlds_data')
SYSTEMS = ['pendulum', 'duffing', 'lotka_volterra', 'lorenz63']
SEEDS = [0, 1, 2, 3, 4]
SEQ_LEN = 96
PRED_LEN = 96


def _topk_continuations_1d(W, C, prime, k):
    """W: (n_w, L'), C: (n_w,), prime: (L',). Returns scalar mean of top-k C entries."""
    if W.shape[0] == 0:
        return None
    k = min(k, W.shape[0])
    diff = W - prime[None, :]
    d = (diff * diff).sum(axis=1)
    idx = np.argpartition(d, k - 1)[:k]
    return float(C[idx].mean())


def parrot_raw_step(context, Lp, k):
    """One autoregressive parrot step on raw values.

    Returns the predicted next value given the current 96-point context.
    Searches L'-windows in context with valid 1-step continuations.
    """
    n = len(context)
    if n < Lp + 2:
        return float(context[-1])
    n_w = n - Lp  # last index has 1-step continuation
    starts = np.arange(n_w)
    W = context[starts[:, None] + np.arange(Lp)[None, :]]
    C = context[starts + Lp]
    prime = context[-Lp:]
    pred = _topk_continuations_1d(W, C, prime, k)
    return float(context[-1]) if pred is None else pred


def parrot_diff_step(context, Lp, k):
    """One autoregressive parrot step in diff space, return predicted next raw value."""
    n = len(context)
    if n < Lp + 3:
        return float(context[-1])
    d = np.diff(context)
    nd = len(d)
    n_w = nd - Lp
    if n_w < 1:
        return float(context[-1])
    starts = np.arange(n_w)
    Wd = d[starts[:, None] + np.arange(Lp)[None, :]]
    Cd = d[starts + Lp]
    prime_d = d[-Lp:]
    pred_d = _topk_continuations_1d(Wd, Cd, prime_d, k)
    return float(context[-1]) if pred_d is None else float(context[-1]) + pred_d


def parrot_forecast(history_1d, Lp, T, mode, k):
    """Autoregressive in-context parroting over T steps.

    history_1d: (L,) 1-D lookback (typically L=96).
    Returns (T,) raw-space forecast.
    """
    history_1d = np.asarray(history_1d, dtype=np.float64)
    if mode == 'raw':
        step_fn = parrot_raw_step
    else:
        step_fn = parrot_diff_step
    preds = []
    ctx = history_1d.copy()
    for _ in range(T):
        nxt = step_fn(ctx, Lp, k)
        preds.append(nxt)
        ctx = np.concatenate([ctx, [nxt]])
    return np.asarray(preds, dtype=np.float64)


# Variant grid: (mode, Lp, k)
VARIANTS = []
for Lp in (12, 24, 48):
    for k in (1, 5):
        VARIANTS.append(('raw',  Lp, k))
        VARIANTS.append(('diff', Lp, k))


def predict(mode, Lp, k, history, T):
    return parrot_forecast(history, Lp, T, mode, k)


def eval_window_seq(data_split, mode, Lp, k, seq_len=SEQ_LEN, pred_len=PRED_LEN,
                    max_windows=None):
    """Slide over a single 1-D series; return mean (mse, mae)."""
    data_split = np.asarray(data_split, dtype=np.float64)
    n = len(data_split)
    n_w = max(0, n - seq_len - pred_len + 1)
    if n_w == 0:
        return float('nan'), float('nan'), 0
    if max_windows is not None:
        idxs = np.linspace(0, n_w - 1, min(max_windows, n_w), dtype=int)
    else:
        idxs = np.arange(n_w)
    sse = 0.0; sae = 0.0; n_pts = 0
    for i in idxs:
        hist = data_split[i:i + seq_len]
        truth = data_split[i + seq_len: i + seq_len + pred_len]
        pred = predict(mode, Lp, k, hist, pred_len)
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
    mu = train.mean(axis=0); sd = train.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    train_s = (train - mu) / sd
    val_s = (val - mu) / sd
    test_s = (test - mu) / sd
    train_then_val = np.concatenate([train_s, val_s], axis=0)
    val_border1 = len(train_s) - SEQ_LEN
    val_data = np.concatenate([train_s[val_border1:], val_s], axis=0)
    test_border_data = np.concatenate(
        [train_then_val[len(train_then_val) - SEQ_LEN:], test_s], axis=0)

    per_dim_picks = []
    per_dim_test_mse = []
    per_dim_test_mae = []
    for d in range(n_dims):
        # Val sweep with subsampling for speed
        val_scores = []
        for mode, Lp, k in VARIANTS:
            mse, mae, _ = eval_window_seq(val_data[:, d], mode, Lp, k,
                                           max_windows=80)
            val_scores.append((mse, mode, Lp, k))
        val_scores.sort(key=lambda x: x[0])
        best_mse, best_mode, best_Lp, best_k = val_scores[0]
        per_dim_picks.append((best_mode, best_Lp, best_k))
        # Test eval over ALL windows for the picked variant
        test_mse, test_mae, _ = eval_window_seq(test_border_data[:, d],
                                                 best_mode, best_Lp, best_k)
        per_dim_test_mse.append(test_mse)
        per_dim_test_mae.append(test_mae)
    avg_mse = float(np.mean(per_dim_test_mse))
    avg_mae = float(np.mean(per_dim_test_mae))
    return name, seed, avg_mse, avg_mae, per_dim_picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'parrot_results.csv'))
    args = ap.parse_args()
    tasks = [(name, seed) for name in SYSTEMS for seed in SEEDS]
    n_workers = max(1, min(20, (os.cpu_count() or 4) - 1))
    print(f"NLDS parrot eval: {len(tasks)} tasks × {len(VARIANTS)} variants, "
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
    print(f"\n=== NLDS Parrot results (mean ± std across 5 seeds) ===")
    print(f"{'system':>16s}  {'Parrot MSE':>16s}  {'Parrot MAE':>16s}  "
          f"  GDC target  SKOLR target")
    targets_skolr = {  # SKOLR Table 11
        'pendulum':       ((0.0001, 0.0083), (0.0039, 0.0470)),
        'duffing':        ((0.0047, 0.0518), (0.0365, 0.1479)),
        'lotka_volterra': ((0.0018, 0.0354), (0.0178, 0.1050)),
        'lorenz63':       ((0.9740, 0.7941), (1.0937, 0.8325)),
    }
    targets_gdc = {  # repo's NLDS_RESULTS.md
        'pendulum':       (0.0003, 0.0112),
        'duffing':        (0.0005, 0.0132),
        'lotka_volterra': (0.0000, 0.0011),
        'lorenz63':       (1.171,  0.847),
    }
    by_sys = defaultdict(lambda: ([], []))
    for name, seed, mse, mae, _ in rows:
        by_sys[name][0].append(mse); by_sys[name][1].append(mae)
    for name in SYSTEMS:
        mses, maes = by_sys[name]
        m_mse = float(np.mean(mses)); s_mse = float(np.std(mses))
        m_mae = float(np.mean(maes)); s_mae = float(np.std(maes))
        gdc = targets_gdc[name]
        sk = targets_skolr[name][0]
        print(f"{name:>16s}  {m_mse:.4f}±{s_mse:.4f}  "
              f"{m_mae:.4f}±{s_mae:.4f}  "
              f"  {gdc[0]:.4f}/{gdc[1]:.4f}  {sk[0]:.4f}/{sk[1]:.4f}")


if __name__ == "__main__":
    main()
