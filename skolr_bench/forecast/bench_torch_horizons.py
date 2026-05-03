"""Benchmark torch GDC across all Informer horizons for ETTh1.

Estimates total runtime for a full leakage-free sweep
(state space = train+val, lookback L=720 to match Informer).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SKOLR_BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(SKOLR_BENCH)
sys.path.insert(0, HERE); sys.path.insert(0, SKOLR_BENCH); sys.path.insert(0, ROOT)
from informer_loaders import load_univariate
from gdc_torch import forecast_many_torch


def main():
    train, val, test, mu, sd = load_univariate('ETTh1')
    state = np.concatenate([train, val])  # ~11520 pts
    print(f"State space N = {len(state)} (train+val)")
    L = 720
    primes_template = np.lib.stride_tricks.sliding_window_view(
        np.concatenate([val[-L:], test]), L)
    print(f"Test windows B = {primes_template.shape[0]} (with sliding L={L})")

    # Warm-up
    _ = forecast_many_torch(state, beta=1.0, alpha=1.0, theta=0.0,
                             primes=primes_template[:1], T=24,
                             device='cuda', dtype=torch.float32)
    torch.cuda.synchronize()

    print(f"\n{'T':>4s}  {'B used':>8s}  {'fp64':>8s}  {'fp32':>8s}")
    total64 = 0.0; total32 = 0.0
    for T in [24, 48, 168, 336, 720]:
        # Number of usable windows = (test_len + L) - L - T + 1 = test_len - T + 1
        B = primes_template.shape[0] - T  # rough
        primes = primes_template[:B]
        # fp64
        torch.cuda.synchronize(); t0 = time.time()
        _ = forecast_many_torch(state, beta=1.0, alpha=1.0, theta=0.0,
                                 primes=primes, T=T, device='cuda',
                                 dtype=torch.float64)
        torch.cuda.synchronize(); t64 = time.time() - t0
        # fp32
        torch.cuda.synchronize(); t0 = time.time()
        _ = forecast_many_torch(state, beta=1.0, alpha=1.0, theta=0.0,
                                 primes=primes, T=T, device='cuda',
                                 dtype=torch.float32)
        torch.cuda.synchronize(); t32 = time.time() - t0
        print(f"{T:>4d}  {B:>8d}  {t64:>7.2f}s  {t32:>7.2f}s")
        total64 += t64; total32 += t32

    print(f"\nFor 1 config across all 5 horizons: fp64={total64:.1f}s  fp32={total32:.1f}s")
    print(f"For 22 val configs × 5 horizons:   fp64={total64*22:.0f}s  fp32={total32*22:.0f}s")
    print(f"  (i.e., full ETTh1 sweep: fp64 ~{total64*22/60:.1f} min, fp32 ~{total32*22/60:.1f} min)")
    print(f"\nFor 8 datasets (avg same scale): fp32 ~{total32*22*8/60:.0f} min")


if __name__ == "__main__":
    main()
