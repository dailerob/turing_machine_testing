"""Validate the torch GDC kernel against the numpy GenerativeDenseChain.

Compare horizon_emission outputs across a small grid of HMM-style configs.
Pass if max-abs-diff < 1e-6 in float32 / 1e-12 in float64 across all test cells.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from generative_dense_chain import GenerativeDenseChain
from gdc_torch_discrete import horizon_emission_many


def build_test_data(N_seqs=10, T_train=20, nA=4, seed=0):
    rng = np.random.default_rng(seed)
    seqs = [rng.integers(0, nA, size=T_train) for _ in range(N_seqs)]
    return seqs


def numpy_predict(seqs, primes, horizons, nA, alpha, theta, beta,
                   transition_type, terminal_behavior, initial_dist):
    seq_arrays = [s.reshape(-1, 1).astype(np.int64) for s in seqs]
    gdc = GenerativeDenseChain(
        seq_arrays,
        alpha=alpha, theta=theta, gamma=0.0, beta=beta,
        transition_type=transition_type,
        initial_dist=initial_dist,
        terminal_behavior=terminal_behavior,
    )
    sym = gdc.states[:, 0].astype(np.int64)
    out = np.zeros((len(primes), len(horizons), nA))
    for i, p in enumerate(primes):
        obs = p.reshape(-1, 1).astype(np.int64)
        final_dist = gdc.forward_pass(obs, return_history=False)
        for j, h in enumerate(horizons):
            forecast = gdc.forecast(final_dist, n_steps=h)
            symdist = np.zeros(nA)
            np.add.at(symdist, sym, forecast)
            s = symdist.sum()
            if s > 0:
                symdist = symdist / s
            else:
                symdist = np.full(nA, 1.0 / nA)
            out[i, j, :] = symdist
    return out, gdc, sym


def torch_predict(gdc, sym, primes, horizons, nA, alpha, theta, beta,
                   transition_type, terminal_behavior, initial_dist,
                   device, dtype):
    primes_arr = np.stack([np.asarray(p, dtype=np.int64) for p in primes])
    out = horizon_emission_many(
        symbol_of_state=sym,
        terminal_mask=gdc.terminal_mask,
        start_mask=gdc.start_mask,
        primes=primes_arr,
        horizons=horizons,
        nA=nA,
        alpha=alpha, theta=theta, beta=beta,
        transition_type=transition_type,
        terminal_behavior=terminal_behavior,
        initial_dist=initial_dist,
        device=device, dtype=dtype,
    )
    return out.cpu().numpy()


def run_one(name, seqs, primes, horizons, nA, alpha, theta, beta,
             transition_type, terminal_behavior, initial_dist):
    np_out, gdc, sym = numpy_predict(seqs, primes, horizons, nA,
                                      alpha, theta, beta,
                                      transition_type, terminal_behavior,
                                      initial_dist)
    pt_out_64 = torch_predict(gdc, sym, primes, horizons, nA,
                               alpha, theta, beta,
                               transition_type, terminal_behavior,
                               initial_dist, 'cuda', torch.float64)
    pt_out_32 = torch_predict(gdc, sym, primes, horizons, nA,
                               alpha, theta, beta,
                               transition_type, terminal_behavior,
                               initial_dist, 'cuda', torch.float32)
    diff64 = np.abs(np_out - pt_out_64).max()
    diff32 = np.abs(np_out - pt_out_32).max()
    print(f"  {name:>50s}  max|np-torch_fp64|={diff64:.2e}  "
          f"max|np-torch_fp32|={diff32:.2e}")
    return diff64, diff32


def main():
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    seqs = build_test_data(N_seqs=10, T_train=20, nA=4, seed=0)
    rng = np.random.default_rng(1)
    primes = [rng.integers(0, 4, size=15) for _ in range(8)]
    horizons = [1, 5, 20]

    print("\n=== Validation: numpy vs torch GDC discrete ===")
    cases = []
    for tb in ['diffuse', 'absorb']:
        for init in ['uniform', 'sequence_starts']:
            for (a, t, b) in [(0.5, 0.05, 0.0), (0.5, 0.05, 0.2),
                              (0.8, 0.0, 0.1), (0.1, 0.001, 0.2)]:
                cases.append((tb, init, a, t, b))
    max64 = 0.0; max32 = 0.0
    for (tb, init, a, t, b) in cases:
        name = f"{tb}/{init}/a={a}/t={t}/b={b}"
        d64, d32 = run_one(name, seqs, primes, horizons, 4,
                            a, t, b, 'self_loop', tb, init)
        max64 = max(max64, d64); max32 = max(max32, d32)
    print(f"\nGlobal max diff: fp64={max64:.2e}  fp32={max32:.2e}")
    ok64 = max64 < 1e-10
    ok32 = max32 < 1e-5
    print(f"PASS fp64: {ok64}, PASS fp32: {ok32}")

    # Speed comparison
    print("\n=== Speed (single forward+forecast for B prefixes) ===")
    big_seqs = build_test_data(N_seqs=100, T_train=50, nA=8, seed=0)
    big_primes = [np.random.default_rng(i).integers(0, 8, size=20)
                   for i in range(100)]
    horizons_big = [1, 5, 20]
    t0 = time.time()
    np_out, gdc, sym = numpy_predict(big_seqs, big_primes, horizons_big, 8,
                                      0.5, 0.05, 0.1, 'self_loop',
                                      'diffuse', 'sequence_starts')
    t_np = time.time() - t0
    t0 = time.time()
    _ = torch_predict(gdc, sym, big_primes, horizons_big, 8,
                       0.5, 0.05, 0.1, 'self_loop',
                       'diffuse', 'sequence_starts', 'cuda', torch.float32)
    t_pt = time.time() - t0
    print(f"  numpy:        {t_np:.3f}s")
    print(f"  torch fp32:   {t_pt:.3f}s   (speedup: {t_np/max(t_pt, 1e-6):.1f}x)")


if __name__ == "__main__":
    main()
