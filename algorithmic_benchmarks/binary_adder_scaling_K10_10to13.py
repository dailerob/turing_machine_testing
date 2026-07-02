"""One-off: K=10 GDC on 10-13 bit binary_adder noread.

Same training pool as binary_adder_scaling.py (first 10 tapes from
seed=42, num_range=(0, 32) = ≤5-bit). Same GDC config (α=0.95,
θ=0.05, self_loop, diffuse). Test: 10 problems, operands in
num_range=(1024, 8192) (bit-length 11-13).
"""
from __future__ import annotations
import os, sys, time
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

GDC_CFG = dict(alpha=0.95, theta=0.05, beta=0.0,
                transition_type='self_loop',
                initial_dist='sequence_starts',
                terminal_behavior='diffuse')
K = 10
N_TEST = 10
TEST_NUM_RANGE = (1024, 8192)   # 11-13 bit operands
TEST_MAX_STEPS = 5_000_000

# Train: first K=10 tapes from the 200-tape pool with seed=42
train = simulate_random_binary_alphabet_adders(
    n_runs=200, num_range=(0, 32), max_steps=200_000, seed=42)
te = simulate_random_binary_alphabet_adders(
    n_runs=N_TEST, num_range=TEST_NUM_RANGE,
    max_steps=TEST_MAX_STEPS, seed=124)

print(f"Train: {K} tapes from pool (200 total), num_range=(0, 32) ≤5-bit")
print(f"Test: {N_TEST} tapes num_range={TEST_NUM_RANGE} (11-13 bit), "
      f"halted={sum(te['halted_flags'])}/{N_TEST}, "
      f"correct={sum(te['correct'])}/{N_TEST}, "
      f"avg_trace_len={np.mean([t.shape[0] for t in te['runs']]):.0f}")

# Apply noread merging
merged_se = dict(train['symbol_encoding'])
merged_st = dict(train['state_encoding'])
for src in (te['symbol_encoding'],):
    for k in src:
        if k not in merged_se: merged_se[k] = len(merged_se)
for src in (te['state_encoding'],):
    for k in src:
        if k not in merged_st: merged_st[k] = len(merged_st)
train['runs'], _ = apply_noread_to_runs(
    train['runs'], BINARY_ALPHABET_ADDER, merged_st, merged_se)
te['runs'], _ = apply_noread_to_runs(
    te['runs'], BINARY_ALPHABET_ADDER, merged_st, merged_se)

# Build alphabet from full training pool, take first K
tuple_to_id, id_to_tuple = reduced_alphabet(train['runs'])
nA = len(id_to_tuple)
train_red_full = [encode_reduced_for_torch(t, tuple_to_id) for t in train['runs']]
train_red_full = [s for s in train_red_full if len(s) > 0]
train_subset = train_red_full[:K]
chain_len = sum(len(s) for s in train_subset)
print(f"Reduced alphabet size: {nA}; chain_len at K={K}: {chain_len}")

t0 = time.time()
gdc = TorchTMGDC(**GDC_CFG)
gdc.fit(train_subset, alphabet_size=nA)
acc, total, terr, perf = torch_gdc_eval_tm_reduced(
    gdc, te['runs'], tuple_to_id, id_to_tuple)
dt = time.time() - t0
n_pred = int(total[0])
print(f"\nGDC K={K} on 11-13 bit binary_adder noread:")
print(f"  tuple errors: {terr}/{n_pred} ({100*terr/max(n_pred,1):.4f}%)")
print(f"  perfect tapes: {perf}/{N_TEST}")
print(f"  read/write/dir acc: {acc[0]:.6f} {acc[1]:.6f} {acc[2]:.6f}")
print(f"  inference time: {dt:.1f}s")
