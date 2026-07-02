"""Shift-left-by-k Turing machine: shift a binary input left by k
positions (i.e., multiply by 2^k). MSB-first convention. k is fixed.

Algorithm: walk right past the input to the right boundary, then
write k zeros into successive blank cells.

Tape alphabet: {'0', '1', '_'} (ternary).
"""

from __future__ import annotations
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from _tm_common import simulate_task  # noqa: E402

START_STATE = 'WALK_RIGHT'
HALT_STATE = 'H'
BLANK_SYMBOL = '_'

# Fixed shift amount.
K = 2

# Build write states WRITE_1, WRITE_2, ..., WRITE_K
PROGRAM = [
    # Walk right through input to the right boundary blank.
    ('WALK_RIGHT', '0', '0', 'R', 'WALK_RIGHT'),
    ('WALK_RIGHT', '1', '1', 'R', 'WALK_RIGHT'),
]
# At the boundary, write K zeros sequentially.
for i in range(1, K + 1):
    next_state = f'WRITE_{i + 1}' if i < K else 'H'
    PROGRAM.append(
        (f'WRITE_{i}' if i > 1 else 'WALK_RIGHT', '_', '0', 'R', next_state))


def sample_input(rng, length_range):
    lo, hi = length_range
    n = int(rng.integers(lo, hi + 1))
    if n <= 0:
        return ((),)
    bits = tuple(int(b) for b in rng.integers(0, 2, size=n))
    return (bits,)


def make_initial_tape(input_args):
    (bits,) = input_args
    return {i: str(b) for i, b in enumerate(bits)}


def start_position(input_args):
    return 0


def decode_result(final_tape):
    if not final_tape:
        return None
    positions = sorted(final_tape.keys())
    digits = []
    for p in positions:
        s = final_tape.get(p, '_')
        if s in ('0', '1'):
            digits.append(s)
        else:
            if digits:
                break
    if not digits:
        return ()
    return tuple(int(d) for d in digits)


def is_correct(input_args, result):
    if result is None:
        return False
    (bits,) = input_args
    expected = tuple(bits) + tuple([0] * K)
    return tuple(result) == expected


def simulate(n_runs, length_range, max_steps=10000, seed=42, noread=False):
    return simulate_task(sys.modules[__name__], n_runs, length_range,
                         max_steps, seed, noread=noread)
