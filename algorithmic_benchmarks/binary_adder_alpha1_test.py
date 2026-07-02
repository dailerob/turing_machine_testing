"""α=1 vs α<1 ablation: confirm the FSM-emulation theory.

Test setup: K=10 training tapes (≤5-bit additions), test on 10 problems
with 11-13 bit operands. Run GDC at several configurations and compare
tuple-error counts.

Theoretical predictions:
  - α=1.0, θ=0, terminal='absorb' : predicts exactly until test trace
    exceeds the longest training tape (~50 tuples), then degenerates.
    Expect HIGH error rate.
  - α=1.0, θ=0, terminal='diffuse': terminal mass redistributes
    uniformly. May provide a soft reset, but lacks per-step
    diffusion. Could still degrade.
  - α=0.99, θ=0, terminal='diffuse': 1% per-step diffusion provides
    enough background mass for filter step to re-concentrate.
    Expect LOW error rate.
  - α=0.95, θ=0.05, terminal='diffuse': baseline. Empirical 0/1M.
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


CONFIGS = [
    ('α=1.0, θ=0, absorb',  dict(alpha=1.0,  theta=0.0,  beta=0.0,
                                  terminal_behavior='absorb')),
    ('α=1.0, θ=0, diffuse', dict(alpha=1.0,  theta=0.0,  beta=0.0,
                                  terminal_behavior='diffuse')),
    ('α=0.99, θ=0, diffuse', dict(alpha=0.99, theta=0.0,  beta=0.0,
                                   terminal_behavior='diffuse')),
    ('α=0.95, θ=0.05, diffuse (baseline)',
     dict(alpha=0.95, theta=0.05, beta=0.0,
          terminal_behavior='diffuse')),
]
COMMON_CFG = dict(transition_type='self_loop',
                   initial_dist='sequence_starts')

K = 10
N_TEST = 10
TEST_NUM_RANGE = (1024, 8192)   # 11-13 bit
TEST_MAX_STEPS = 5_000_000

# Generate train pool & test set ONCE
train = simulate_random_binary_alphabet_adders(
    n_runs=200, num_range=(0, 32), max_steps=200_000, seed=42)
te = simulate_random_binary_alphabet_adders(
    n_runs=N_TEST, num_range=TEST_NUM_RANGE,
    max_steps=TEST_MAX_STEPS, seed=124)

print(f"Train: {K} tapes from pool, num_range=(0, 32) ≤5-bit")
train_lens = [t.shape[0] for t in train['runs'][:K]]
print(f"  Training tape trace lengths: min={min(train_lens)}, "
      f"max={max(train_lens)}, mean={np.mean(train_lens):.0f}")
print(f"Test: {N_TEST} tapes num_range={TEST_NUM_RANGE} (11-13 bit)")
print(f"  Test trace mean length: {np.mean([t.shape[0] for t in te['runs']]):.0f}")

# Apply noread
merged_se = dict(train['symbol_encoding'])
merged_st = dict(train['state_encoding'])
for k in te['symbol_encoding']:
    if k not in merged_se: merged_se[k] = len(merged_se)
for k in te['state_encoding']:
    if k not in merged_st: merged_st[k] = len(merged_st)
train['runs'], _ = apply_noread_to_runs(
    train['runs'], BINARY_ALPHABET_ADDER, merged_st, merged_se)
te['runs'], _ = apply_noread_to_runs(
    te['runs'], BINARY_ALPHABET_ADDER, merged_st, merged_se)

# Encode
tuple_to_id, id_to_tuple = reduced_alphabet(train['runs'])
nA = len(id_to_tuple)
train_red_full = [encode_reduced_for_torch(t, tuple_to_id) for t in train['runs']]
train_red_full = [s for s in train_red_full if len(s) > 0]
train_subset = train_red_full[:K]
chain_len = sum(len(s) for s in train_subset)
print(f"\nReduced alphabet size: {nA}; chain length at K={K}: {chain_len}")
print(f"  Training tape encoded lengths: {[len(s) for s in train_subset]}")
print()

# Run each config
print(f"{'config':<45} {'errors':>10} {'rate':>10} {'perfect':>9} {'time':>7}")
print('-' * 90)
for name, cfg_overrides in CONFIGS:
    cfg = {**COMMON_CFG, **cfg_overrides}
    t0 = time.time()
    try:
        gdc = TorchTMGDC(**cfg)
        gdc.fit(train_subset, alphabet_size=nA)
        acc, total, terr, perf = torch_gdc_eval_tm_reduced(
            gdc, te['runs'], tuple_to_id, id_to_tuple)
        n_pred = int(total[0])
        rate = 100 * terr / max(n_pred, 1)
        dt = time.time() - t0
        print(f"{name:<45} {terr:>5}/{n_pred:<8} {rate:>8.4f}% {perf:>3}/{N_TEST:<3} {dt:>5.1f}s",
              flush=True)
    except Exception as e:
        print(f"{name:<45} ERROR: {e}", flush=True)
