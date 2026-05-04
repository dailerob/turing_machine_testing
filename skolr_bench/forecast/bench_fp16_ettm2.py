"""Quick benchmark: GDC kernel on ETTm2 problem at fp64 / fp32 / fp16.

Goal: time per-config and report numerical accuracy of fp16 vs fp64.
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
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"  Tensor cores fp16/bf16/tf32 available")
    print()

    train, val, test, mu, sd = load_univariate('ETTm2')
    state = np.concatenate([train, val])  # ~46080 pts
    N = len(state)
    print(f"ETTm2 state space N = {N} (train+val)")

    L = 96  # Autoformer convention
    sigma_frac = 0.10; alpha = 1.0; theta = 0.0
    sigma = max(float(np.std(state)) * sigma_frac, 1e-9)
    beta = max((sigma * np.sqrt(L)) ** 2, 1e-9)

    print(f"\n{'T':>4s}  {'B':>6s}  {'fp64':>8s}  {'fp32':>8s}  {'fp16':>8s}  "
          f"{'fp32 vs64':>10s}  {'fp16 vs64':>10s}")
    for T in [96, 192, 336, 720]:
        full = np.concatenate([val[-L:], test])
        n = len(full); n_w = n - L - T + 1
        starts = np.arange(n_w)
        primes = full[np.arange(L)[None, :] + starts[:, None]]
        truths = full[np.arange(L, L+T)[None, :] + starts[:, None]]
        B = primes.shape[0]

        # Warm-up each dtype
        for dtype in [torch.float64, torch.float32, torch.float16]:
            try:
                _ = forecast_many_torch(state, beta, alpha, theta, primes[:1], T,
                                        device='cuda', dtype=dtype)
            except Exception as e:
                print(f"  warm-up failed for {dtype}: {e}")
        torch.cuda.synchronize()

        # Time fp64
        torch.cuda.synchronize(); t0 = time.time()
        ref64 = forecast_many_torch(state, beta, alpha, theta, primes, T,
                                     device='cuda', dtype=torch.float64).cpu().numpy()
        torch.cuda.synchronize(); t64 = time.time() - t0

        # Time fp32
        torch.cuda.synchronize(); t0 = time.time()
        out32 = forecast_many_torch(state, beta, alpha, theta, primes, T,
                                     device='cuda', dtype=torch.float32).cpu().numpy()
        torch.cuda.synchronize(); t32 = time.time() - t0

        # Time fp16
        torch.cuda.synchronize(); t0 = time.time()
        try:
            out16 = forecast_many_torch(state, beta, alpha, theta, primes, T,
                                         device='cuda', dtype=torch.float16).cpu().numpy().astype(np.float64)
            torch.cuda.synchronize(); t16 = time.time() - t0
            d16 = float(np.abs(out16 - ref64).max())
            r16 = d16 / max(np.abs(ref64).max(), 1e-9)
            t16_str = f"{t16:>7.2f}s"
            d16_str = f"{r16:.2e}"
        except Exception as e:
            t16_str = "FAIL"; d16_str = str(e)[:30]
            out16 = None

        d32 = float(np.abs(out32 - ref64).max())
        r32 = d32 / max(np.abs(ref64).max(), 1e-9)
        # Test MSE differences
        mse64 = float(((ref64 - truths) ** 2).mean())
        mse32 = float(((out32 - truths) ** 2).mean())
        mse16 = float(((out16 - truths) ** 2).mean()) if out16 is not None else float('nan')

        print(f"{T:>4d}  {B:>6d}  {t64:>7.2f}s  {t32:>7.2f}s  {t16_str:>8s}  "
              f"{r32:.2e}      {d16_str:>10s}")
        print(f"      test MSE: fp64={mse64:.4f}  fp32={mse32:.4f}  fp16={mse16:.4f}")
    print()
    print("Sweep estimate (22 configs val + 1 test ≈ 23x test cost per horizon):")


if __name__ == "__main__":
    main()
