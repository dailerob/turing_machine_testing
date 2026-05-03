"""ARIMA baseline matching Informer's protocol.

From Informer issue #302 (cookieminions, Informer author):
  "we fix the input length of model, and refit the ARIMA model to get
   prediction with output length for every test step."

So: for each test sample (= each rolling window of stride 1):
  - input  = lookback of length L
  - refit  = pmdarima.auto_arima on those L points
  - output = forecast T steps in one shot

Parallelized across windows using multiprocessing.
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


def _suppress():
    """Silence pmdarima/statsmodels warnings inside workers."""
    warnings.filterwarnings('ignore')
    import logging
    logging.getLogger('pmdarima').setLevel(logging.CRITICAL)


def fit_predict_one(args):
    sid, lookback, T = args
    _suppress()
    import pmdarima as pm
    try:
        model = pm.auto_arima(
            lookback,
            suppress_warnings=True, seasonal=False,
            error_action='ignore', stepwise=True,
            max_p=5, max_q=5, max_d=2,
        )
        f = model.predict(n_periods=T)
        return sid, np.asarray(f, dtype=np.float64), model.order
    except Exception:
        # Fallback: random walk
        return sid, np.full(T, lookback[-1], dtype=np.float64), None


def run(dataset, T, L, n_workers=None, stride=1):
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4))
    train, val, test, mu, sd = load_univariate(dataset)
    print(f"  {dataset}: train={len(train)}, val={len(val)}, test={len(test)}", flush=True)
    test_with_lookback = np.concatenate([val[-L:], test])
    n = len(test_with_lookback)
    n_w = max(0, n - L - T + 1)
    starts = np.arange(0, n_w, stride)
    print(f"  ARIMA refit per window: L={L}, T={T}, n_windows={len(starts)}, "
          f"workers={n_workers}", flush=True)
    tasks = []
    for sid, i in enumerate(starts):
        lookback = test_with_lookback[i:i+L].copy()
        tasks.append((sid, lookback, T))
    truths = np.empty((len(starts), T), dtype=np.float64)
    for sid, i in enumerate(starts):
        truths[sid] = test_with_lookback[i+L:i+L+T]
    forecasts = np.empty((len(starts), T), dtype=np.float64)
    orders = []
    t0 = time.time(); done = 0
    every = max(1, len(tasks) // 20)
    with mp.Pool(processes=n_workers) as pool:
        for sid, fcst, order in pool.imap_unordered(fit_predict_one, tasks,
                                                      chunksize=8):
            forecasts[sid] = fcst
            orders.append(order)
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
    # Order distribution
    valid = [o for o in orders if o is not None]
    if valid:
        from collections import Counter
        order_counts = Counter(valid).most_common(5)
        print(f"  ARIMA order distribution (top 5): {order_counts}", flush=True)
    return mse, mae, forecasts, truths, orders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--T', type=int, default=24)
    ap.add_argument('--L', type=int, default=None,
                    help='lookback length (default: 2*T)')
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()
    if args.L is None:
        args.L = 2 * args.T
    print(f"=== ARIMA baseline: {args.dataset} T={args.T} L={args.L} ===")
    mse, mae, _, _, _ = run(args.dataset, args.T, args.L, stride=args.stride)
    # Reference numbers from Informer Table 1
    ref = {
        ('ETTh1', 24):  (0.108, 0.284),
        ('ETTh1', 48):  (0.175, 0.424),
        ('ETTh1', 168): (0.396, 0.504),
        ('ETTh1', 336): (0.468, 0.593),
        ('ETTh1', 720): (0.659, 0.766),
        ('ETTh2', 24):  (3.554, 0.445),
        ('ETTh2', 48):  (3.190, 0.474),
        ('ETTh2', 168): (2.800, 0.595),
        ('ETTh2', 336): (2.753, 0.738),
        ('ETTh2', 720): (2.878, 1.044),
    }
    r = ref.get((args.dataset, args.T))
    if r:
        print(f"  Informer Tab.1 ARIMA published: MSE={r[0]:.3f}, MAE={r[1]:.3f}")
        print(f"  Our ARIMA (stride={args.stride}):  MSE={mse:.3f}, MAE={mae:.3f}")
        print(f"  Ratio (ours/published): MSE={mse/r[0]:.2f}, MAE={mae/r[1]:.2f}")


if __name__ == "__main__":
    main()
