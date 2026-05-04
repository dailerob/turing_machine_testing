"""ARIMA + Prophet on ILI/ECL/Traffic, Autoformer convention.

Refits per window per Informer issue #302 protocol.
Records MSE/MAE on standardized data (consistent with GDC sweep).
"""
from __future__ import annotations
import os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arima_baseline import run as arima_run
from prophet_baseline import run as prophet_run

CONFIG = [
    ('ILI_AF',     36, [24, 36, 48, 60]),
    ('Traffic_AF', 96, [96, 192, 336, 720]),
    ('ECL_AF',     96, [96, 192, 336, 720]),
]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'results', 'arima_prophet_autoformer.csv')


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    write_header = not os.path.exists(OUT)
    f = open(OUT, 'a', newline='')
    w = csv.writer(f)
    if write_header:
        w.writerow(['method', 'dataset', 'L', 'T', 'mse', 'mae', 'time_s'])
    f.flush()
    t_start = time.time()
    for name, L, horizons in CONFIG:
        for method, runner, kwargs in [
            ('arima', arima_run, {}),
            ('prophet', prophet_run, {}),
        ]:
            for T in horizons:
                print(f"\n=== {method.upper()} {name} L={L} T={T} ===", flush=True)
                t0 = time.time()
                if method == 'arima':
                    mse, mae, _, _, _ = runner(name, T, L)
                else:
                    mse, mae = runner(name, T, L)
                dt = time.time() - t0
                rows.append((method, name, L, T, mse, mae, dt))
                w.writerow([method, name, L, T, mse, mae, dt]); f.flush()
                print(f"  -> {method} {name} T={T}: MSE={mse:.4f} "
                      f"MAE={mae:.4f}  ({dt:.0f}s)", flush=True)
    f.close()
    print(f"\nTotal wall: {time.time()-t_start:.0f}s")
    print(f"Wrote {OUT}")
    print("\n=== SUMMARY ===")
    for m, n, L, T, mse, mae, dt in rows:
        print(f"  {m:>7s}  {n:>12s}  T={T:>4d}  MSE={mse:.4f}  MAE={mae:.4f}  ({dt:.0f}s)")


if __name__ == '__main__':
    main()
