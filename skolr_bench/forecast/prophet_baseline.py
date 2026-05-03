"""Prophet baseline matching Informer's protocol.

Per Informer issue #302: refit per window with fixed input length.
Each window:
  - Convert lookback to (ds, y) DataFrame with synthetic hourly timestamps
  - Fit Prophet
  - Predict T steps ahead

Prophet is slow (~1-5s per fit). Parallelized across windows.
"""
from __future__ import annotations
import os, sys, time, csv, argparse
import numpy as np
import multiprocessing as mp
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
SKOLR_BENCH = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, SKOLR_BENCH)
from informer_loaders import load_univariate, DATASETS  # noqa: E402

warnings.filterwarnings('ignore')


def _suppress():
    warnings.filterwarnings('ignore')
    import logging
    for name in ['prophet', 'cmdstanpy', 'fbprophet']:
        try: logging.getLogger(name).setLevel(logging.CRITICAL)
        except: pass


def fit_predict_one(args):
    sid, lookback, T, hours_per_row = args
    _suppress()
    import pandas as pd
    from prophet import Prophet
    L = len(lookback)
    # Synthetic hourly timestamps for the lookback
    base = pd.Timestamp('2018-01-01')
    ts = pd.date_range(base, periods=L + T,
                        freq=f'{int(60*hours_per_row)}min')
    df = pd.DataFrame({'ds': ts[:L], 'y': lookback})
    try:
        m = Prophet(daily_seasonality=True, weekly_seasonality=True,
                    yearly_seasonality=False)
        m.fit(df)
        future = pd.DataFrame({'ds': ts[L:L+T]})
        f = m.predict(future)
        return sid, np.asarray(f['yhat'].values, dtype=np.float64)
    except Exception:
        # fallback random walk
        return sid, np.full(T, lookback[-1], dtype=np.float64)


def run(dataset, T, L, n_workers=None, stride=1, max_windows=None):
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4))
    train, val, test, mu, sd = load_univariate(dataset)
    info = DATASETS[dataset]
    hpr = info['hours_per_row']
    print(f"  {dataset}: train={len(train)}, val={len(val)}, test={len(test)}",
          flush=True)
    test_with_lookback = np.concatenate([val[-L:], test])
    n = len(test_with_lookback); n_w = max(0, n - L - T + 1)
    starts = np.arange(0, n_w, stride)
    if max_windows is not None and max_windows < len(starts):
        idxs = np.linspace(0, len(starts)-1, max_windows, dtype=int)
        starts = starts[idxs]
    print(f"  Prophet refit per window: L={L}, T={T}, n_windows={len(starts)}, "
          f"workers={n_workers}", flush=True)
    tasks = [(sid, test_with_lookback[i:i+L].copy(), T, hpr)
              for sid, i in enumerate(starts)]
    truths = np.empty((len(starts), T), dtype=np.float64)
    for sid, i in enumerate(starts):
        truths[sid] = test_with_lookback[i+L:i+L+T]
    forecasts = np.empty((len(starts), T), dtype=np.float64)
    t0 = time.time(); done = 0
    every = max(1, len(tasks) // 20)
    with mp.Pool(processes=n_workers) as pool:
        for sid, fcst in pool.imap_unordered(fit_predict_one, tasks,
                                              chunksize=2):
            forecasts[sid] = fcst
            done += 1
            if done % every == 0 or done == len(tasks):
                pct = 100 * done / len(tasks)
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(tasks) - done) / rate
                print(f"    {done}/{len(tasks)} ({pct:.0f}%) "
                      f"[{time.time()-t0:.0f}s, ~{eta:.0f}s left]", flush=True)
    elapsed = time.time() - t0
    diff = truths - forecasts
    mse = float((diff ** 2).mean())
    mae = float(np.abs(diff).mean())
    print(f"  Done in {elapsed:.1f}s.  MSE={mse:.4f}  MAE={mae:.4f}", flush=True)
    return mse, mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--T', type=int, default=24)
    ap.add_argument('--L', type=int, default=720)
    ap.add_argument('--max-windows', type=int, default=None,
                    help='subsample test windows (uniform spacing) for speed')
    args = ap.parse_args()
    print(f"=== Prophet baseline: {args.dataset} T={args.T} L={args.L} ===")
    mse, mae = run(args.dataset, args.T, args.L,
                    max_windows=args.max_windows)
    ref = {
        ('ETTh1', 24):  (0.115, 0.275),
        ('ETTh1', 48):  (0.168, 0.330),
        ('ETTh1', 168): (1.224, 0.763),
        ('ETTh1', 336): (1.549, 1.820),
        ('ETTh1', 720): (2.735, 3.253),
        ('ETTh2', 24):  (0.199, 0.381),
        ('ETTh2', 48):  (0.304, 0.462),
        ('ETTh2', 168): (2.145, 1.068),
        ('ETTh2', 336): (2.096, 2.543),
        ('ETTh2', 720): (3.355, 4.664),
    }
    r = ref.get((args.dataset, args.T))
    if r:
        print(f"  Informer Tab.1 Prophet published: MSE={r[0]:.3f}, MAE={r[1]:.3f}")
        print(f"  Our Prophet: MSE={mse:.3f}, MAE={mae:.3f}")
        print(f"  Ratio (ours/published): MSE={mse/r[0]:.2f}, MAE={mae/r[1]:.2f}")


if __name__ == "__main__":
    main()
