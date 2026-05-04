"""M4-spec Naive 1, Naive S, Naive 2 baselines on Autoformer-convention windows.

Strict M4 protocol (Makridakis et al. 2020 supplementary, Appendix E):
  - Naive 1:  F_{n+h} = Y_n (random walk)
  - Naive S:  F_{n+h} = Y_{n+h-m}  (last value at same seasonal phase)
  - Naive 2:  seasonality test on full history, if seasonal -> multiplicative
              decomposition (R's decompose()), forecast deseasonalized series
              with Naive 1, re-seasonalize. Else Naive 2 = Naive 1.

Seasonality test (90% confidence, Bartlett bound):
  |ACF_m| > 1.645 * sqrt( (1 + 2 * sum_{i=1..m-1} ACF_i^2) / n )
  Skip (treat non-seasonal) if m=1 or n < 3m.

M4 frequencies (Section 3.1): m=12 monthly, m=4 quarterly, m=24 hourly,
                              m=1 for yearly/weekly/daily.

Eval:
  - Sliding windows match GDC: primes of length L from (val_tail + test),
    truths of length T after the prime. Number of windows = len(test) - T + 1.
  - Series state for seasonality test / decomposition = train + val (raw scale).
  - Forecasts computed on raw scale, then standardized with train (mu, sd)
    to compare apples-to-apples with GDC/Autoformer MSE on standardized data.
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SKOLR_BENCH = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, SKOLR_BENCH)

from informer_loaders import DATASETS as INFORMER_DATASETS

# (dataset_name, m, L, horizons)
PROTOCOLS = [
    ('ETTm2',      1,  96, [96, 192, 336, 720]),   # 15-min, not in M4 -> m=1
    ('Exchange',   1,  96, [96, 192, 336, 720]),   # daily   -> m=1
    ('ECL_AF',     24, 96, [96, 192, 336, 720]),   # hourly  -> m=24
    ('Traffic_AF', 24, 96, [96, 192, 336, 720]),   # hourly  -> m=24
    ('ILI_AF',     1,  36, [24, 36, 48, 60]),      # weekly  -> m=1
]

# Autoformer-convention split (7:1:2). For ETT use the Informer-style 12/4/4
# month split (consistent with our other tables).
def load_raw_ot(name):
    info = INFORMER_DATASETS[name]
    path = os.path.join(SKOLR_BENCH, 'data_original', info['rel'])
    df = pd.read_csv(path)
    arr = df[info['target']].values.astype(np.float64)
    n_total = len(arr)
    if info.get('split') == 'ratio_7_1_2':
        n_train = int(n_total * 0.7)
        n_test  = int(n_total * 0.2)
        n_val   = n_total - n_train - n_test
    else:
        rows_per_mo = round(30 * 24 / info['hours_per_row'])
        n_train = info['train_mo'] * rows_per_mo
        n_val   = info['val_mo'] * rows_per_mo
        n_test  = info['test_mo'] * rows_per_mo
    train = arr[:n_train]
    val   = arr[n_train: n_train + n_val]
    test  = arr[n_train + n_val: n_train + n_val + n_test]
    return train, val, test


def m4_seasonality_test(series, m):
    """Returns True if series is seasonal at period m per M4 (90% CI)."""
    n = len(series)
    if m <= 1 or n < 3 * m:
        return False
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    var = float((x * x).mean())
    if var < 1e-12:
        return False
    acf = np.empty(m + 1)
    acf[0] = 1.0
    for k in range(1, m + 1):
        acf[k] = float((x[:n - k] * x[k:]).mean()) / var
    threshold = 1.645 * np.sqrt((1.0 + 2.0 * float((acf[1:m] ** 2).sum())) / n)
    return abs(acf[m]) > threshold


def classical_multiplicative_decompose(series, m):
    """R's decompose(., type='multiplicative') equivalent.

    Trend: centered MA of length m.  For even m, this is the standard
    2xm filter (average of two consecutive MA(m)).  For odd m, plain MA(m).
    Seasonal: detrended ratios averaged per phase, normalized to mean 1.
    Returns seasonal_indices array of length m, where seasonal_indices[k]
    is the multiplier for series positions with (i mod m) == k.
    """
    n = len(series)
    s = np.asarray(series, dtype=np.float64)
    if m % 2 == 0:
        kernel = np.concatenate(([1.0/(2*m)], np.full(m-1, 1.0/m), [1.0/(2*m)]))
        trend_len = m + 1
    else:
        kernel = np.full(m, 1.0/m)
        trend_len = m
    trend = np.convolve(s, kernel, mode='valid')   # length n - trend_len + 1
    offset = (trend_len - 1) // 2                   # leading positions w/o trend
    if (trend <= 0).any():
        # Multiplicative decomposition undefined.  Fallback: additive seasonal.
        return None
    detrended = s[offset: offset + len(trend)] / trend
    indices = np.zeros(m)
    counts = np.zeros(m, dtype=np.int64)
    for i, val in enumerate(detrended):
        phase = (offset + i) % m
        indices[phase] += val
        counts[phase] += 1
    indices = np.where(counts > 0, indices / np.maximum(counts, 1), 1.0)
    indices /= indices.mean()
    return indices


def make_windows(series, L, T):
    """Sliding windows: primes of len L, truths of len T, stride 1."""
    n = len(series)
    n_w = max(0, n - L - T + 1)
    if n_w == 0:
        return np.empty((0, L)), np.empty((0, T))
    starts = np.arange(n_w)
    p_idx = np.arange(L)[None, :] + starts[:, None]
    t_idx = np.arange(L, L + T)[None, :] + starts[:, None]
    return series[p_idx], series[t_idx]


def naive1(primes, T):
    return np.repeat(primes[:, -1:], T, axis=1)


def naive_s(primes, T, m):
    if m <= 1:
        return naive1(primes, T)
    last_m = primes[:, -m:]                       # (W, m)
    reps = T // m + 1
    tiled = np.tile(last_m, (1, reps))[:, :T]     # (W, T)
    return tiled


def naive2(primes, T, m, history, history_origin_phase):
    """Naive 2 per M4 spec.

    history: long historical series for seasonality test + decomposition.
    history_origin_phase: phase index (0..m-1) at history[0]
                          (we use 0 — series start is phase 0 by convention).
    For each prime, last_value is at phase = (history_origin_phase + n_hist + W_offset) % m;
    in our streaming setup we treat each prime's last index as phase
    (start_in_full + L - 1) mod m, but because primes are sliding by 1,
    last_phase for window w is just (last_phase_of_window_0 + w) mod m.
    """
    if not m4_seasonality_test(history, m):
        return naive1(primes, T)
    indices = classical_multiplicative_decompose(history, m)
    if indices is None:
        return naive1(primes, T)
    W, L = primes.shape
    # last_phase for each window: primes[w, -1] is at series position
    #   (history_origin + n_history + w + L - L) for window w?  Actually
    # we don't track absolute positions; simpler: use the prime itself
    # to identify the phase via correlation OR pass last_phase per window.
    # Simplest robust method: compute last_phase from total elapsed length.
    # CALLER PROVIDES last_phase via a side-channel (see eval_one).
    raise NotImplementedError("use naive2_eval below; needs phase info")


def naive2_eval(primes_full_start, primes, T, m, history, last_phases):
    """Vectorized Naive 2 with explicit per-window last-phase array."""
    if not m4_seasonality_test(history, m):
        return naive1(primes, T), False
    indices = classical_multiplicative_decompose(history, m)
    if indices is None:
        return naive1(primes, T), False
    W = primes.shape[0]
    last_vals = primes[:, -1]                                    # (W,)
    last_idx_seasonal = indices[last_phases]                     # (W,)
    deseason_last = last_vals / last_idx_seasonal                # (W,) — constant forecast
    # forecast phase for h=1..T at window w: (last_phase[w] + h) % m
    phase_grid = (last_phases[:, None] + 1 + np.arange(T)[None, :]) % m  # (W, T)
    seasonal_at_forecast = indices[phase_grid]                   # (W, T)
    return deseason_last[:, None] * seasonal_at_forecast, True


def metrics_standardized(forecasts_raw, truths_raw, mu, sd):
    f = (forecasts_raw - mu) / sd
    t = (truths_raw - mu) / sd
    diff = t - f
    return float((diff ** 2).mean()), float(np.abs(diff).mean())


def run_one(name, m, L, horizons):
    train, val, test = load_raw_ot(name)
    mu = train.mean()
    sd = train.std()
    if sd < 1e-9: sd = 1.0
    history = np.concatenate([train, val])    # decomposition + seasonality test base
    n_history = len(history)
    full_eval = np.concatenate([val[-L:], test])  # primes from val tail + test
    n_train_val = n_history
    rows = []
    for T in horizons:
        primes, truths = make_windows(full_eval, L, T)
        W = primes.shape[0]
        if W == 0:
            rows.append((T, 0, None, None, None, None, None, None, False))
            continue
        # Last index of primes[0] in absolute series coords = (n_train_val - L) + L - 1 = n_train_val - 1
        # primes[w] last index = n_train_val - 1 + w
        last_phases = (np.arange(W) + (n_train_val - 1)) % m
        # Naive 1
        f1 = naive1(primes, T)
        m1_mse, m1_mae = metrics_standardized(f1, truths, mu, sd)
        # Naive S
        fS = naive_s(primes, T, m)
        mS_mse, mS_mae = metrics_standardized(fS, truths, mu, sd)
        # Naive 2
        f2, was_seasonal = naive2_eval(0, primes, T, m, history, last_phases)
        m2_mse, m2_mae = metrics_standardized(f2, truths, mu, sd)
        rows.append((T, W, m1_mse, m1_mae, mS_mse, mS_mae,
                     m2_mse, m2_mae, was_seasonal))
    return rows


def main():
    print(f"{'dataset':>12s}  {'m':>3s}  {'L':>4s}  {'T':>4s}  {'W':>5s}  "
          f"{'N1 MSE':>8s} {'N1 MAE':>8s}  "
          f"{'NS MSE':>8s} {'NS MAE':>8s}  "
          f"{'N2 MSE':>8s} {'N2 MAE':>8s}  seasonal?")
    all_rows = []
    for name, m, L, horizons in PROTOCOLS:
        t0 = time.time()
        rows = run_one(name, m, L, horizons)
        for r in rows:
            T, W, m1_mse, m1_mae, mS_mse, mS_mae, m2_mse, m2_mae, was_seasonal = r
            print(f"{name:>12s}  {m:>3d}  {L:>4d}  {T:>4d}  {W:>5d}  "
                  f"{m1_mse:>8.4f} {m1_mae:>8.4f}  "
                  f"{mS_mse:>8.4f} {mS_mae:>8.4f}  "
                  f"{m2_mse:>8.4f} {m2_mae:>8.4f}  {was_seasonal}")
            all_rows.append((name, m, L, T, W, m1_mse, m1_mae,
                             mS_mse, mS_mae, m2_mse, m2_mae, was_seasonal))
        print(f"  ({time.time()-t0:.1f}s)")
    out = os.path.join(HERE, 'results', 'm4_naives.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'm', 'L', 'T', 'W',
                    'naive1_mse', 'naive1_mae',
                    'naiveS_mse', 'naiveS_mae',
                    'naive2_mse', 'naive2_mae',
                    'naive2_was_seasonal'])
        for r in all_rows:
            w.writerow(r)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
