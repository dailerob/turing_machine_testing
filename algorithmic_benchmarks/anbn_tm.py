"""anbn: recognize the language a^n b^n (any non-negative n).

Input: a binary string (uses 0 for 'a', 1 for 'b').
Output: 1 if the input is exactly k zeros followed by k ones for some
k >= 0; else 0.

Algorithm: erase-as-you-go. Repeatedly strip the leftmost 0 and the
rightmost 1. If at any iteration the leftmost is not 0 or the rightmost
is not 1, fail. If both are simultaneously erased, accept.

Tape alphabet: {'0', '1', '_'} (ternary).
"""

from __future__ import annotations
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from _tm_common import simulate_task  # noqa: E402

START_STATE = 'LEFT_READ'
HALT_STATE = 'H'
BLANK_SYMBOL = '_'

PROGRAM = [
    # LEFT_READ: at the leftmost remaining char.
    ('LEFT_READ', '0', '_', 'R', 'GO_RIGHT'),     # erased a 0; now look for matching 1
    ('LEFT_READ', '1', '1', 'R', 'FAIL'),         # leftmost is 1 → fail
    ('LEFT_READ', '_', '1', 'R', 'H'),            # everything matched → accept

    # GO_RIGHT: walk to the right boundary.
    ('GO_RIGHT', '0', '0', 'R', 'GO_RIGHT'),
    ('GO_RIGHT', '1', '1', 'R', 'GO_RIGHT'),
    ('GO_RIGHT', '_', '_', 'L', 'CHECK_RIGHT'),

    # CHECK_RIGHT: at the rightmost remaining char.
    ('CHECK_RIGHT', '1', '_', 'L', 'WALK_LEFT'),  # match
    ('CHECK_RIGHT', '0', '0', 'R', 'FAIL'),       # rightmost is 0 → fail
    ('CHECK_RIGHT', '_', '0', 'R', 'H'),          # we erased a 0 but no 1 to match → fail

    # WALK_LEFT: walk back left to the next leftmost char.
    ('WALK_LEFT', '0', '0', 'L', 'WALK_LEFT'),
    ('WALK_LEFT', '1', '1', 'L', 'WALK_LEFT'),
    ('WALK_LEFT', '_', '_', 'R', 'LEFT_READ'),

    # FAIL: walk to the right boundary, write 0 to indicate failure.
    ('FAIL', '0', '0', 'R', 'FAIL'),
    ('FAIL', '1', '1', 'R', 'FAIL'),
    ('FAIL', '_', '0', 'R', 'H'),
]


def sample_input(rng, length_range):
    """Sample a binary string of varying length. Half the time we generate
    a valid a^n b^n pattern; otherwise random bits."""
    lo, hi = length_range
    n = int(rng.integers(lo, hi + 1))
    if n <= 0:
        return ((),)
    if rng.random() < 0.5:
        # Generate valid a^k b^k for some k <= n // 2
        k = int(rng.integers(0, n // 2 + 1))
        bits = tuple([0] * k + [1] * k)
    else:
        # Random bits — mostly invalid
        bits = tuple(int(b) for b in rng.integers(0, 2, size=n))
    return (bits,)


def make_initial_tape(input_args):
    (bits,) = input_args
    return {i: str(b) for i, b in enumerate(bits)}


def start_position(input_args):
    return 0


def decode_result(final_tape):
    """The rightmost non-blank symbol on the tape is the answer (1=accept,
    0=reject)."""
    if not final_tape:
        return None
    keys = sorted(final_tape.keys())
    for p in reversed(keys):
        s = final_tape.get(p, '_')
        if s in ('0', '1'):
            return int(s)
    return None


def is_correct(input_args, result):
    if result is None:
        return False
    (bits,) = input_args
    n = len(bits)
    if n % 2 != 0:
        expected = 0
    else:
        k = n // 2
        expected = 1 if all(b == 0 for b in bits[:k]) and all(
            b == 1 for b in bits[k:]) else 0
    return result == expected


def simulate(n_runs, length_range, max_steps=10000, seed=42, noread=False):
    return simulate_task(sys.modules[__name__], n_runs, length_range,
                         max_steps, seed, noread=noread)
