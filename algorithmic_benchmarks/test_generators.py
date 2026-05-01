"""Sanity tests for each algorithmic-task generator.

Each test runs ~200 random instances and confirms the TM (or
generator) produces the correct ground truth.  Run:

    python algorithmic_benchmarks/test_generators.py
"""

from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import parity_tm, increment_tm, reverse_tm, dyck1  # noqa: E402


def banner(s):
    print('\n' + '=' * 60)
    print(s)
    print('=' * 60, flush=True)


def show_trace(arr, n=10):
    print(f"  trace shape: {arr.shape}")
    print(f"  first {n} rows: (state, read, write, dir, next_state)")
    for row in arr[:n]:
        print(f"    {tuple(int(x) for x in row)}")


def test_parity():
    banner("Parity TM")
    res = parity_tm.simulate(n_runs=200, length_range=(3, 8),
                             max_steps=100, seed=1)
    n_halted = sum(res['halted_flags'])
    n_correct = sum(res['correct'])
    print(f"halted: {n_halted}/200, correct: {n_correct}/200")
    print(f"state_encoding: {res['state_encoding']}")
    print(f"symbol_encoding: {res['symbol_encoding']}")
    show_trace(res['runs'][0], n=10)
    print(f"  example input bits: {res['inputs'][0][0]}")
    print(f"  parity computed: {res['results'][0]}, "
          f"expected: {sum(res['inputs'][0][0]) % 2}")
    assert n_halted == 200, f"only {n_halted}/200 halted"
    assert n_correct == 200, f"only {n_correct}/200 correct"
    print("PASS")


def test_increment():
    banner("Increment TM")
    res = increment_tm.simulate(n_runs=200, length_range=(1, 6),
                                max_steps=200, seed=2)
    n_halted = sum(res['halted_flags'])
    n_correct = sum(res['correct'])
    print(f"halted: {n_halted}/200, correct: {n_correct}/200")
    print(f"state_encoding: {res['state_encoding']}")
    print(f"symbol_encoding: {res['symbol_encoding']}")
    show_trace(res['runs'][0], n=12)
    print(f"  example input: {res['inputs'][0][0]} -> "
          f"computed: {res['results'][0]} "
          f"(expected: {res['inputs'][0][0] + 1})")
    assert n_halted == 200, f"only {n_halted}/200 halted"
    assert n_correct == 200, f"only {n_correct}/200 correct"
    print("PASS")


def test_reverse():
    banner("Reverse TM")
    res = reverse_tm.simulate(n_runs=200, length_range=(3, 6),
                              max_steps=2000, seed=3)
    n_halted = sum(res['halted_flags'])
    n_correct = sum(res['correct'])
    print(f"halted: {n_halted}/200, correct: {n_correct}/200")
    print(f"state_encoding: {res['state_encoding']}")
    print(f"symbol_encoding: {res['symbol_encoding']}")
    show_trace(res['runs'][0], n=15)
    print(f"  example input: {res['inputs'][0][0]} -> "
          f"computed: {res['results'][0]} "
          f"(expected: {res['inputs'][0][0][::-1]})")
    assert n_halted == 200, f"only {n_halted}/200 halted"
    assert n_correct == 200, f"only {n_correct}/200 correct"
    print("PASS")


def test_dyck1():
    banner("Dyck-1 sampler")
    res = dyck1.simulate(n_runs=200, max_depth=4, length_min=4,
                         length_max=200, seed=4)
    n_correct = sum(res['correct_walks'])
    lens = [len(s) for s in res['sequences']]
    print(f"correct_walks: {n_correct}/200")
    print(f"length stats: min={min(lens)}, max={max(lens)}, "
          f"mean={np.mean(lens):.1f}")
    print(f"first sequence (tokens): {res['sequences'][0].tolist()}")
    sym = {0: '(', 1: ')', 2: 'END'}
    print(f"first sequence (chars):  "
          f"{' '.join(sym[t] for t in res['sequences'][0])}")
    assert n_correct == 200, f"only {n_correct}/200 balanced"
    print("PASS")


if __name__ == "__main__":
    test_parity()
    test_increment()
    test_reverse()
    test_dyck1()
    print("\nAll generator tests passed.")
