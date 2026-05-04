"""Prophet-only re-run with flat-trend config on ILI/Traffic/ECL."""
from __future__ import annotations
import os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prophet_baseline import run as prophet_run

CONFIG = [
    ('ILI_AF',     36, [24, 36, 48, 60]),
    ('Traffic_AF', 96, [96, 192, 336, 720]),
    ('ECL_AF',     96, [96, 192, 336, 720]),
]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'results', 'prophet_flat_autoformer.csv')


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    f = open(OUT, 'w', newline='')
    w = csv.writer(f)
    w.writerow(['method','dataset','L','T','mse','mae','time_s'])
    f.flush()
    rows = []
    t_start = time.time()
    for name, L, horizons in CONFIG:
        for T in horizons:
            print(f"\n=== prophet_flat {name} L={L} T={T} ===", flush=True)
            t0 = time.time()
            mse, mae = prophet_run(name, T, L)
            dt = time.time() - t0
            rows.append((name, L, T, mse, mae, dt))
            w.writerow(['prophet_flat', name, L, T, mse, mae, dt]); f.flush()
            print(f"  -> {name} T={T}: MSE={mse:.4f} MAE={mae:.4f}  ({dt:.0f}s)", flush=True)
    f.close()
    print(f"\nTotal: {time.time()-t_start:.0f}s")
    print("\n=== SUMMARY ===")
    for n,L,T,mse,mae,dt in rows:
        print(f"  {n:>12s}  T={T:>4d}  MSE={mse:.4f}  MAE={mae:.4f}  ({dt:.0f}s)")


if __name__ == '__main__':
    main()
