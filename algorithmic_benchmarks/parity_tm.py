"""Parity Turing machine.  See `TASKS.md` Task 1 for the spec."""

from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from _tm_common import simulate_task  # noqa: E402

START_STATE = 'SCAN0'
HALT_STATE = 'H'
BLANK_SYMBOL = '_'

PROGRAM = [
    ('SCAN0', '0', '0', 'R', 'SCAN0'),
    ('SCAN0', '1', '1', 'R', 'SCAN1'),
    ('SCAN0', '_', '0', 'R', 'H'),
    ('SCAN1', '0', '0', 'R', 'SCAN1'),
    ('SCAN1', '1', '1', 'R', 'SCAN0'),
    ('SCAN1', '_', '1', 'R', 'H'),
]


def sample_input(rng, length_range):
    """Return (bits,) where bits is a tuple of 0/1."""
    n = int(rng.integers(length_range[0], length_range[1] + 1))
    bits = tuple(int(b) for b in rng.integers(0, 2, size=n))
    return (bits,)


def make_initial_tape(input_args):
    (bits,) = input_args
    return {i: str(b) for i, b in enumerate(bits)}


def start_position(input_args):
    return 0


def decode_result(final_tape):
    """Parity bit is at position N (the first blank past input)."""
    if not final_tape:
        return None
    n = max(p for p in final_tape if final_tape[p] in ('0', '1', '_'))
    # Find the rightmost char that is in {'0','1'} that comes after the
    # input chars.  Simplest: parity is whatever is at the maximum
    # position.
    max_pos = max(final_tape.keys())
    sym = final_tape.get(max_pos)
    if sym in ('0', '1'):
        return int(sym)
    return None


def is_correct(input_args, result):
    if result is None:
        return False
    (bits,) = input_args
    return result == (sum(bits) % 2)


def simulate(n_runs, length_range, max_steps=10000, seed=42, noread=False):
    return simulate_task(sys.modules[__name__], n_runs, length_range,
                         max_steps, seed, noread=noread)
