"""M4 Naive 2 implementation, matching the R/statistical-benchmark version
that produced the official Naive 2 numbers in the competition supplementary.

Reference: https://github.com/Mcompetitions/M4-methods/blob/master/ML_benchmarks.py
(but with ACF1 squared, per the R version — the Python-script's
non-squared ACF1 is a documented bug retained for reproducibility of the
ML benchmarks; it does not match the published statistical Naive2 scores).

M4 seasonality periods used for deseasonalize() AND MASE denominator:
  Hourly=24, Daily=1, Weekly=1, Monthly=12, Quarterly=4, Yearly=1.

Quick check: published Naive 2 sMAPE / MASE per frequency
  Yearly    16.342 / 3.974
  Quarterly 11.012 / 1.371
  Monthly   14.427 / 1.063
  Weekly     9.161 / 2.777
  Daily      3.045 / 3.278
  Hourly    18.383 / 2.395
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data_loader as dl


# M4 official seasonality period for MASE / Naive2 (NOT same as our data_loader's
# `seasonality` which had Daily=7).
M4_PERIOD = {
    "Yearly": 1,
    "Quarterly": 4,
    "Monthly": 12,
    "Weekly": 1,
    "Daily": 1,
    "Hourly": 24,
}


def acf(x, k):
    """Sample autocorrelation at lag k. Matches the M4 ACF function."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x); m = x.mean()
    s1 = np.sum((x[k:] - m) * (x[:-k] - m))
    s2 = np.sum((x - m) ** 2)
    if s2 <= 0:
        return 0.0
    return float(s1 / s2)


def seasonality_test(x, m):
    """Returns True if series x is seasonal at period m.
    Uses the R-version test (ALL ACFs squared in the sum)."""
    if m <= 1 or len(x) < 3 * m:
        return False
    n = len(x)
    s = sum(acf(x, i) ** 2 for i in range(1, m))
    limit = 1.645 * np.sqrt((1 + 2 * s) / n)
    return abs(acf(x, m)) > limit


def moving_averages(x, w):
    """M4 reference centered moving average (with the documented
    `len(x) % 2` bug retained for reproducibility)."""
    s = pd.Series(np.asarray(x, dtype=np.float64))
    if len(x) % 2 == 0:
        ma = s.rolling(window=w, center=True).mean()
        ma = ma.rolling(window=2, center=True).mean()
        return np.roll(ma.values, -1)
    return s.rolling(window=w, center=True).mean().values


def deseasonalize(x, m):
    """Returns seasonal indices (length m, scaled around 100) per M4 ref."""
    x = np.asarray(x, dtype=np.float64)
    if seasonality_test(x, m):
        ma = moving_averages(x, m)
        le = x * 100.0 / ma
        pad = m - (len(le) % m) if (len(le) % m) else 0
        if pad:
            le = np.hstack([le, np.full(pad, np.nan)])
        le = le.reshape(-1, m)
        si = np.nanmean(le, axis=0)
        norm = np.sum(si) / (m * 100.0)
        si = si / norm
    else:
        si = np.full(m, 100.0)
    return si


def naive2_forecast(train, h, m):
    """Naive 2: deseasonalize -> Naive (last value) -> re-seasonalize."""
    train = np.asarray(train, dtype=np.float64)
    n = len(train)
    si = deseasonalize(train, m)
    # deseasonalize using index modulo m (matches M4 ref code)
    des = train * 100.0 / si[np.arange(n) % m]
    last = des[-1]
    fcast_des = np.full(h, last)
    fut_si = si[(np.arange(n, n + h)) % m]
    return fcast_des * fut_si / 100.0


def smape(actual, forecast):
    """M4 sMAPE (returned as a percent: 0-200 range)."""
    a = np.asarray(actual, dtype=np.float64); f = np.asarray(forecast, dtype=np.float64)
    return float(np.mean(2.0 * np.abs(a - f) / (np.abs(a) + np.abs(f))) * 100.0)


def mase(insample, actual, forecast, m):
    """M4 MASE: mean|y - f| / mean|insample[m:] - insample[:-m|]."""
    ins = np.asarray(insample, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64); f = np.asarray(forecast, dtype=np.float64)
    masep = float(np.mean(np.abs(ins[m:] - ins[:-m])))
    return float(np.mean(np.abs(a - f)) / masep)


def evaluate_naive2(freq):
    """Return per-series sMAPE/MASE arrays + their means."""
    train = dl.load_train(freq); test = dl.load_test(freq)
    h = dl.horizon(freq); m = M4_PERIOD[freq]
    ids = sorted(train.keys(), key=lambda s: int(s[1:]))
    sm = np.empty(len(ids)); ma = np.empty(len(ids))
    for i, sid in enumerate(ids):
        tr = train[sid]; te = test[sid]
        f = naive2_forecast(tr, h, m)
        sm[i] = smape(te, f)
        ma[i] = mase(tr, te, f, m)
    return sm, ma


def main():
    expected = {
        "Yearly":    (16.342, 3.974),
        "Quarterly": (11.012, 1.371),
        "Monthly":   (14.427, 1.063),
        "Weekly":    ( 9.161, 2.777),
        "Daily":     ( 3.045, 3.278),
        "Hourly":    (18.383, 2.395),
    }
    print(f"{'freq':>10s}  {'series':>7s}  {'h':>3s}  {'m':>3s}  "
          f"{'sMAPE':>7s}  {'MASE':>7s}  {'pub sMAPE':>9s}  {'pub MASE':>9s}")
    for freq in ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]:
        sm, ma = evaluate_naive2(freq)
        es, em = expected[freq]
        print(f"{freq:>10s}  {len(sm):>7d}  {dl.horizon(freq):>3d}  "
              f"{M4_PERIOD[freq]:>3d}  {sm.mean():>6.3f}  {ma.mean():>6.3f}  "
              f"{es:>9.3f}  {em:>9.3f}")


if __name__ == "__main__":
    main()
