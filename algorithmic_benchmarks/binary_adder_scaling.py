"""Sequential-training scaling experiment for binary_adder.

Take the best GDC configuration from the 4× sweep
(α=0.95, θ=0.05, β=0, transition=self_loop, terminal=diffuse,
initial_dist=sequence_starts) — the one that reaches 0/72217 on
binary_adder-noread — and ask:

  How fast does GDC's accuracy converge as we add training tapes,
  one at a time, on each OOD bit-length OOD bin?

Protocol:
  - Train pool: 200 tapes from num_range=(0, 32) (≤5-bit operands,
    same as the canonical training distribution). Same seed=42.
  - Test bins (10 problems each, fixed seeds):
      * 5-10 bits  : num_range=(32, 1024)        seed=123
      * 10-15 bits : num_range=(1024, 32768)     seed=124
      * 15-20 bits : num_range=(32768, 1048576)  seed=125
  - Variant: noread (the canonical headline-zero variant).
  - For K = 1, 2, 3, ..., 30, 35, 40, ..., 100, 110, ..., 200:
      * Build TorchTMGDC from first K tapes
      * Eval tuple errors on each test bin
      * Record (K, chain_len, bin, errors, perfect_tapes)

GPU-batched scoring via TorchTMGDC (the same scorer used in the main
benchmarks). Output: binary_adder_scaling.csv.
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

from binary_alphabet_adder import (                  # noqa: E402
    simulate_random_binary_alphabet_adders, BINARY_ALPHABET_ADDER)
from _tm_common import apply_noread_to_runs          # noqa: E402
from torch_tm_adapters import TorchTMGDC             # noqa: E402
from run_benchmarks import (                         # noqa: E402
    reduced_alphabet, encode_reduced_for_torch,
    torch_gdc_eval_tm_reduced)


# ---- Best binary_adder-noread GDC config from the 4× sweep ----
GDC_CFG = dict(
    alpha=0.95, theta=0.05, beta=0.0,
    transition_type='self_loop',
    initial_dist='sequence_starts',
    terminal_behavior='diffuse',
)

VARIANT = 'noread'   # set to 'original' for the unmasked variant
N_TRAIN_POOL = 200
N_TEST_PER_BIN = 10

# K schedule on small bin (cheap inference).
K_LIST = list(range(1, 11))

# NB: the binary-alphabet TM uses unary increment, so trace length
# scales with b * n_bits. With 1M+ step traces (10+ bits) the torch
# GDC's per-step kernel launches dominate at ~600µs per step, making
# a single K iteration ~10 min. We restrict to the 5-10 bit bin for
# a tractable scaling experiment; the 10-15 bit bin needs a faster
# scoring path (or a linear-time ripple-carry TM).
TEST_BINS = [
    ('5-10 bits', (32, 1024), 200_000),
]


def main():
    print(f"=== Binary adder sequential-training scaling experiment ===")
    print(f"GDC config: {GDC_CFG}")
    print(f"Variant: {VARIANT}")
    print(f"Train pool: {N_TRAIN_POOL} tapes, num_range=(0, 32) (≤5-bit)")
    print(f"Test bins: {[b[0] for b in TEST_BINS]}, {N_TEST_PER_BIN} problems each")
    print()

    # 1. Generate training pool and test sets
    t0 = time.time()
    train = simulate_random_binary_alphabet_adders(
        n_runs=N_TRAIN_POOL, num_range=(0, 32),
        max_steps=200_000, seed=42)
    print(f"Train pool simulated: {N_TRAIN_POOL} tapes "
          f"(halted={sum(train['halted_flags'])}/{N_TRAIN_POOL}, "
          f"correct={sum(train['correct'])}/{N_TRAIN_POOL}), "
          f"[{time.time()-t0:.1f}s]")

    test_sets = []
    for i, (name, num_range, max_steps) in enumerate(TEST_BINS):
        t0 = time.time()
        te = simulate_random_binary_alphabet_adders(
            n_runs=N_TEST_PER_BIN, num_range=num_range,
            max_steps=max_steps, seed=123 + i)
        halted = sum(te['halted_flags'])
        correct = sum(te['correct'])
        avg_len = np.mean([t.shape[0] for t in te['runs']])
        print(f"Test {name}: {N_TEST_PER_BIN} tapes "
              f"(halted={halted}/{N_TEST_PER_BIN}, correct={correct}/{N_TEST_PER_BIN}, "
              f"avg_trace_len={avg_len:.0f}) [{time.time()-t0:.1f}s]")
        test_sets.append((name, te))

    # 2. Apply noread masking with merged encodings (so train + all test
    # bins share the same NO_READ marker).
    if VARIANT == 'noread':
        merged_se = dict(train['symbol_encoding'])
        merged_st = dict(train['state_encoding'])
        for _, te in test_sets:
            for k in te['symbol_encoding']:
                if k not in merged_se: merged_se[k] = len(merged_se)
            for k in te['state_encoding']:
                if k not in merged_st: merged_st[k] = len(merged_st)
        train['runs'], _ = apply_noread_to_runs(
            train['runs'], BINARY_ALPHABET_ADDER, merged_st, merged_se)
        for _, te in test_sets:
            te['runs'], _ = apply_noread_to_runs(
                te['runs'], BINARY_ALPHABET_ADDER, merged_st, merged_se)
        print(f"Applied noread masking; alphabet entries={len(merged_se)}")

    # 3. Build alphabet from full training pool (so all 200 tapes share
    # the same tuple_to_id mapping; this lets us add tapes one-by-one
    # without re-encoding).
    tuple_to_id, id_to_tuple = reduced_alphabet(train['runs'])
    nA = len(id_to_tuple)
    train_red_full = [encode_reduced_for_torch(t, tuple_to_id)
                       for t in train['runs']]
    train_red_full = [s for s in train_red_full if len(s) > 0]
    print(f"Reduced alphabet size: {nA}")
    print(f"Train pool encoded: {len(train_red_full)} non-empty tapes, "
          f"total chain positions={sum(len(s) for s in train_red_full):,}\n")

    # 4. Sequential sweep
    rows = []
    bin_names = [b[0] for b in TEST_BINS]
    print(f"{'K':>4} {'chain_len':>10} " +
          " ".join(f"{n+' err':>11}" for n in bin_names) +
          " " + " ".join(f"{n+' perf':>11}" for n in bin_names) +
          f" {'time':>5}")
    print('-' * 80)
    for K in K_LIST:
        if K > len(train_red_full):
            continue
        train_subset = train_red_full[:K]
        chain_len = sum(len(s) for s in train_subset)
        t0 = time.time()
        gdc = TorchTMGDC(**GDC_CFG)
        gdc.fit(train_subset, alphabet_size=nA)
        bin_terr = []
        bin_perf = []
        bin_npred = []
        for name, te in test_sets:
            acc, total, terr, perf = torch_gdc_eval_tm_reduced(
                gdc, te['runs'], tuple_to_id, id_to_tuple)
            bin_terr.append(terr)
            bin_npred.append(int(total[0]))
            bin_perf.append(perf)
            rows.append(dict(K=K, chain_len=chain_len,
                              bin=name, tuple_errors=terr,
                              n_pred=int(total[0]),
                              perfect_tapes=perf,
                              n_test=N_TEST_PER_BIN))
        dt = time.time() - t0
        err_cols = " ".join(f"{te:>5}/{n:<5}"
                             for te, n in zip(bin_terr, bin_npred))
        perf_cols = " ".join(f"{p:>5}/{N_TEST_PER_BIN:<3}"
                              for p in bin_perf)
        print(f"{K:>4} {chain_len:>10,} {err_cols} {perf_cols} "
              f"{dt:>4.1f}s", flush=True)

    # 5. Save
    out_csv = os.path.join(HERE, f'binary_adder_scaling_{VARIANT}.csv')
    fields = ['K', 'chain_len', 'bin', 'tuple_errors', 'n_pred',
              'perfect_tapes', 'n_test']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out_csv}")


if __name__ == '__main__':
    main()
