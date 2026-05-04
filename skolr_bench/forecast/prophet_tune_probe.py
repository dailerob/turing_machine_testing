"""Probe Prophet hyperparameters on Traffic T=96 to fix L=96 explosion.

Tests several configs on 50 windows and reports MSE/MAE on standardized data.
Goal: get Prophet at least as sane as ARIMA (MSE ~1.26 on Traffic T=96).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import multiprocessing as mp
import warnings; warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from informer_loaders import load_univariate, DATASETS


def fit_predict_one(args):
    sid, lookback, T, hpr, cfg = args
    warnings.filterwarnings('ignore')
    import logging
    for n in ['prophet','cmdstanpy','fbprophet']:
        try: logging.getLogger(n).setLevel(logging.CRITICAL)
        except: pass
    import pandas as pd
    from prophet import Prophet
    L = len(lookback)
    base = pd.Timestamp('2018-01-01')
    ts = pd.date_range(base, periods=L+T, freq=f'{int(60*hpr)}min')
    df = pd.DataFrame({'ds': ts[:L], 'y': lookback})
    try:
        m = Prophet(**cfg)
        m.fit(df)
        future = pd.DataFrame({'ds': ts[L:L+T]})
        f = m.predict(future)
        return sid, np.asarray(f['yhat'].values, dtype=np.float64)
    except Exception as e:
        return sid, np.full(T, lookback[-1], dtype=np.float64)


CONFIGS = {
    'baseline': dict(daily_seasonality=True, weekly_seasonality=True,
                     yearly_seasonality=False),
    'no_weekly': dict(daily_seasonality=True, weekly_seasonality=False,
                      yearly_seasonality=False),
    'no_weekly_rigid_trend': dict(daily_seasonality=True, weekly_seasonality=False,
                                   yearly_seasonality=False,
                                   changepoint_prior_scale=0.001,
                                   n_changepoints=5),
    'no_weekly_rigid_all': dict(daily_seasonality=True, weekly_seasonality=False,
                                 yearly_seasonality=False,
                                 changepoint_prior_scale=0.001,
                                 seasonality_prior_scale=1.0,
                                 n_changepoints=5),
    'flat_trend': dict(daily_seasonality=True, weekly_seasonality=False,
                       yearly_seasonality=False, growth='flat'),
    'flat_trend_rigid_seas': dict(daily_seasonality=True, weekly_seasonality=False,
                                   yearly_seasonality=False, growth='flat',
                                   seasonality_prior_scale=1.0),
    'no_seas_flat': dict(daily_seasonality=False, weekly_seasonality=False,
                          yearly_seasonality=False, growth='flat'),
}


def probe(name='Traffic_AF', T=96, L=96, n_windows=50):
    train, val, test, mu, sd = load_univariate(name)
    info = DATASETS[name]
    hpr = info['hours_per_row']
    full = np.concatenate([val[-L:], test])
    n_total = len(full) - L - T + 1
    starts = np.linspace(0, n_total - 1, n_windows, dtype=int)
    truths = np.stack([full[s+L:s+L+T] for s in starts])
    print(f"=== {name} T={T} L={L} ({len(starts)} windows, {info['hours_per_row']}h step) ===")
    n_workers = max(1, os.cpu_count() or 4)
    for cfg_name, cfg in CONFIGS.items():
        tasks = [(i, full[s:s+L].copy(), T, hpr, cfg) for i,s in enumerate(starts)]
        forecasts = np.empty((len(tasks), T))
        t0 = time.time()
        with mp.Pool(n_workers) as pool:
            for sid, f in pool.imap_unordered(fit_predict_one, tasks, chunksize=2):
                forecasts[sid] = f
        dt = time.time() - t0
        diff = truths - forecasts
        # Robust stats: also report median window-MSE (filters explosions)
        mse_per_w = (diff**2).mean(axis=1)
        mse_mean = float(mse_per_w.mean())
        mse_median = float(np.median(mse_per_w))
        mae_mean = float(np.abs(diff).mean())
        n_explode = int((mse_per_w > 100).sum())
        print(f"  {cfg_name:>26s}: mean_MSE={mse_mean:>10.3f}  median_MSE={mse_median:>7.3f}  "
              f"MAE={mae_mean:>7.3f}  n_explode={n_explode:>3d}/{len(tasks)}  ({dt:.0f}s)")


if __name__ == '__main__':
    probe('Traffic_AF', T=96, L=96, n_windows=50)
    probe('ECL_AF',     T=96, L=96, n_windows=50)
    probe('ILI_AF',     T=24, L=36, n_windows=50)
