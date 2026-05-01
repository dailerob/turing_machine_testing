"""Increment Turing machine: A in binary -> A+1.  See TASKS.md Task 2."""

from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from _tm_common import simulate_task  # noqa: E402

START_STATE = 'FIND_LSB'
HALT_STATE = 'H'
BLANK_SYMBOL = '_'

PROGRAM = [
    ('FIND_LSB', '0', '0', 'R', 'FIND_LSB'),
    ('FIND_LSB', '1', '1', 'R', 'FIND_LSB'),
    ('FIND_LSB', '_', '_', 'L', 'INC'),
    ('INC', '0', '1', 'R', 'DONE'),
    ('INC', '1', '0', 'L', 'INC'),
    ('INC', '_', '1', 'R', 'DONE'),
    ('DONE', '0', '0', 'R', 'DONE'),
    ('DONE', '1', '1', 'R', 'DONE'),
    ('DONE', '_', '_', 'L', 'H'),
]


def sample_input(rng, length_range):
    """Sample a non-negative integer whose binary length is in
    [length_range[0], length_range[1]]."""
    lo, hi = length_range
    bits_n = int(rng.integers(lo, hi + 1))
    if bits_n <= 0:
        return (0,)
    n = int(rng.integers(0, 2 ** bits_n))
    return (n,)


def make_initial_tape(input_args):
    (n,) = input_args
    s = bin(n)[2:] if n > 0 else '0'
    return {i: c for i, c in enumerate(s)}


def start_position(input_args):
    return 0


def decode_result(final_tape):
    if not final_tape:
        return None
    positions = sorted(final_tape.keys())
    digits = []
    for p in positions:
        s = final_tape[p]
        if s in ('0', '1'):
            digits.append(s)
        else:
            if digits:
                break
    if not digits:
        return None
    try:
        return int(''.join(digits), 2)
    except ValueError:
        return None


def is_correct(input_args, result):
    if result is None:
        return False
    (n,) = input_args
    return result == n + 1


def simulate(n_runs, length_range, max_steps=10000, seed=42, noread=False):
    return simulate_task(sys.modules[__name__], n_runs, length_range,
                         max_steps, seed, noread=noread)
