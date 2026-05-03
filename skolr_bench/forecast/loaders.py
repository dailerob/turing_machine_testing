"""Torch-free data loaders matching the Koopa / SKOLR / TSLib convention.

Returns standardized train/val/test arrays with the same splits as
the official dataset classes in `data_provider/data_loader.py`.

Splits:
  ETT_hour (ETTh1, ETTh2): fixed (12 mo / 4 mo / 4 mo) = 8640/2880/2880
  ETT_minute (ETTm1, ETTm2): fixed at minute granularity
  Custom (Weather, Traffic, ECL, ILI, Exchange): 70/10/20 ratio
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SKOLR_BENCH = os.path.dirname(HERE)
DATA_ROOT = os.path.join(SKOLR_BENCH, 'data_original')

# Map dataset name -> (csv_relative_path, loader_kind)
DATASETS = {
    'ETTh1':       ('ETT-small/ETTh1.csv',           'ett_hour'),
    'ETTh2':       ('ETT-small/ETTh2.csv',           'ett_hour'),
    'ETTm1':       ('ETT-small/ETTm1.csv',           'ett_minute'),
    'ETTm2':       ('ETT-small/ETTm2.csv',           'ett_minute'),
    'ECL':         ('electricity/electricity.csv',   'custom'),
    'Traffic':     ('traffic/traffic.csv',           'custom'),
    'Weather':     ('weather/weather.csv',           'custom'),
    'ILI':         ('illness/national_illness.csv',  'custom'),
}

# SKOLR forecasting horizons
HORIZONS = {
    'ETTh1':   [48, 96, 144, 192],
    'ETTh2':   [48, 96, 144, 192],
    'ETTm1':   [48, 96, 144, 192],
    'ETTm2':   [48, 96, 144, 192],
    'ECL':     [48, 96, 144, 192],
    'Traffic': [48, 96, 144, 192],
    'Weather': [48, 96, 144, 192],
    'ILI':     [24, 36, 48, 60],
}


def _borders(kind, n_total, seq_len):
    """Per-split (border1, border2) following data_loader convention."""
    if kind == 'ett_hour':
        b1s = [0, 12*30*24 - seq_len,           12*30*24 + 4*30*24 - seq_len]
        b2s = [12*30*24, 12*30*24 + 4*30*24,    12*30*24 + 8*30*24]
    elif kind == 'ett_minute':
        b1s = [0, 12*30*24*4 - seq_len,         12*30*24*4 + 4*30*24*4 - seq_len]
        b2s = [12*30*24*4, 12*30*24*4 + 4*30*24*4, 12*30*24*4 + 8*30*24*4]
    elif kind == 'custom':
        n_train = int(n_total * 0.7)
        n_test = int(n_total * 0.2)
        n_val = n_total - n_train - n_test
        b1s = [0, n_train - seq_len, n_total - n_test - seq_len]
        b2s = [n_train, n_train + n_val, n_total]
    else:
        raise ValueError(kind)
    return b1s, b2s


def load(name, seq_len):
    """Returns (train, val, test) standardized arrays of shape (T, C).

    Standardization: per-channel StandardScaler fit on the train slice
    only (using border1=0..b2s[0]), applied to all three slices.

    seq_len is the lookback length used to determine val/test border
    overlap (they each include `seq_len` rows from the prior split so
    the first window in val/test can be formed).
    """
    rel, kind = DATASETS[name]
    df = pd.read_csv(os.path.join(DATA_ROOT, rel))
    cols = list(df.columns[1:])  # drop date column
    arr = df[cols].values.astype(np.float64)
    n_total = arr.shape[0]
    b1s, b2s = _borders(kind, n_total, seq_len)
    train_full = arr[b1s[0]:b2s[0]]
    mu = train_full.mean(axis=0)
    sd = train_full.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    arr_z = (arr - mu) / sd
    train = arr_z[b1s[0]:b2s[0]]
    val   = arr_z[b1s[1]:b2s[1]]
    test  = arr_z[b1s[2]:b2s[2]]
    return train, val, test, cols
