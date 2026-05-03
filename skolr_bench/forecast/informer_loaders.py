"""Informer-style univariate loaders + horizons.

Splits per Informer paper Table 1:
  ETTh1, ETTh2: 12 / 4 / 4 months hourly  (8640 / 2880 / 2880 rows)
  ETTm1:        12 / 4 / 4 months 15-min   (34560 / 11520 / 11520 rows)
  ECL:          15 / 3 / 4 months hourly  (10920 / 2160 / 2880 rows)
  Weather:      28 / 10 / 10 months hourly (20160 / 7200 / 7200 rows)

Targets (univariate Informer Table 1):
  ETT*:    'OT'
  ECL:     'MT_320'
  Weather: 'wet bulb'

Horizons:
  ETTh, ECL, Weather: {24, 48, 168, 336, 720}     (and 960 for ECL)
  ETTm:               {24, 48, 96, 288, 672}

Lookback L:
  Informer used variable lookbacks per dataset; their default for ETTh
  was L=48 for short horizons, L=168 for long. We'll standardize at
  L=2T (the SKOLR convention) for now to keep the comparison simple,
  but expose the choice.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SKOLR_BENCH = os.path.dirname(HERE)
DATA_ROOT = os.path.join(SKOLR_BENCH, 'data_original')

# Per Informer paper, splits in months.
# ETT-hour: 30 days * 24 hr = 720 rows per month.
# ETT-15min: 30 days * 24 * 4 = 2880 rows per month.
# ECL hourly: 30 * 24 = 720 rows per month.
# Weather hourly (Informer used the "Local Climatological Data" weather): 30 * 24 = 720 per month.
DATASETS = {
    'ETTh1':   dict(rel='ETT-small/ETTh1.csv', target='OT',
                     train_mo=12, val_mo=4, test_mo=4, hours_per_row=1,
                     horizons=[24, 48, 168, 336, 720]),
    'ETTh2':   dict(rel='ETT-small/ETTh2.csv', target='OT',
                     train_mo=12, val_mo=4, test_mo=4, hours_per_row=1,
                     horizons=[24, 48, 168, 336, 720]),
    'ETTm1':   dict(rel='ETT-small/ETTm1.csv', target='OT',
                     train_mo=12, val_mo=4, test_mo=4, hours_per_row=0.25,
                     horizons=[24, 48, 96, 288, 672]),
    'ETTm2':   dict(rel='ETT-small/ETTm2.csv', target='OT',
                     train_mo=12, val_mo=4, test_mo=4, hours_per_row=0.25,
                     horizons=[24, 48, 96, 288, 672]),
    # ECL informer's loader uses last 4 months as test, prev 3 as val
    'ECL':     dict(rel='electricity/electricity.csv', target='MT_320',
                     train_mo=15, val_mo=3, test_mo=4, hours_per_row=1,
                     horizons=[48, 168, 336, 720, 960]),
    'Weather': dict(rel='weather/weather.csv', target='wet bulb',
                     train_mo=28, val_mo=10, test_mo=10, hours_per_row=1,
                     horizons=[48, 168, 336, 720]),
}


def load_univariate(name):
    """Load (train, val, test) 1-D arrays standardized by train stats.

    Returns (train, val, test, mu, sd) where each split is 1-D.
    Splits use Informer's month counts (no overlap; val/test do NOT include
    a lookback prefix from train -- the caller must supply lookback during
    eval by accessing earlier slices, e.g., concat(train, val) for first
    val windows).
    """
    info = DATASETS[name]
    df = pd.read_csv(os.path.join(DATA_ROOT, info['rel']))
    cols = list(df.columns)
    target = info['target']
    if target not in cols:
        # ECL: many columns, MT_320 is one of 321 customers
        # weather: target name might differ in our dataset version
        raise ValueError(
            f"target {target!r} not in columns of {name}. "
            f"First 10 cols: {cols[:10]}. Total cols: {len(cols)}.")
    arr = df[target].values.astype(np.float64)
    n_total = arr.shape[0]
    rows_per_mo = round(30 * 24 / info['hours_per_row'])
    n_train = info['train_mo'] * rows_per_mo
    n_val   = info['val_mo'] * rows_per_mo
    n_test  = info['test_mo'] * rows_per_mo
    end = n_train + n_val + n_test
    if end > n_total:
        # Fall back to ratio split (some datasets have less data than
        # informer's exact months).
        raise ValueError(
            f"{name}: requested {n_train+n_val+n_test} rows but only "
            f"{n_total} available")
    train = arr[:n_train]
    val   = arr[n_train: n_train + n_val]
    test  = arr[n_train + n_val: end]
    mu = train.mean(); sd = train.std()
    if sd < 1e-9: sd = 1.0
    train = (train - mu) / sd
    val   = (val   - mu) / sd
    test  = (test  - mu) / sd
    return train, val, test, mu, sd


def list_targets(name):
    """Print all column names for a dataset (for debugging)."""
    info = DATASETS[name]
    df = pd.read_csv(os.path.join(DATA_ROOT, info['rel']), nrows=1)
    return list(df.columns)


if __name__ == "__main__":
    # Sanity: load each dataset, report shapes.
    for name in DATASETS:
        try:
            train, val, test, mu, sd = load_univariate(name)
            print(f"{name:>10s}  train={train.shape[0]}, val={val.shape[0]}, "
                  f"test={test.shape[0]}, mu={mu:.3g}, sd={sd:.3g}")
        except Exception as e:
            cols = list_targets(name)
            short = cols[:5] + (['...'] if len(cols) > 5 else [])
            print(f"{name:>10s}  ERROR: {e}")
            print(f"           cols sample: {short}")
