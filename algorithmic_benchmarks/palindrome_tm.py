"""palindrome_check: decide whether a binary input is a palindrome.

Input: binary string in {0,1}*.
Output: 1 if palindrome, 0 otherwise.

Algorithm: erase-as-you-go. Read+erase the leftmost char (remember
its value in state), walk right to the rightmost remaining char and
verify match. If match, erase and walk back to the new leftmost.
Repeat until everything is erased (palindrome) or a mismatch is found
(not palindrome).

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
    ('LEFT_READ', '0', '_', 'R', 'GO_RIGHT_0'),
    ('LEFT_READ', '1', '_', 'R', 'GO_RIGHT_1'),
    ('LEFT_READ', '_', '1', 'R', 'H'),  # empty / fully erased → palindrome

    # GO_RIGHT_X: walk right with memory of the erased leftmost char.
    ('GO_RIGHT_0', '0', '0', 'R', 'GO_RIGHT_0'),
    ('GO_RIGHT_0', '1', '1', 'R', 'GO_RIGHT_0'),
    ('GO_RIGHT_0', '_', '_', 'L', 'CHECK_RIGHT_0'),
    ('GO_RIGHT_1', '0', '0', 'R', 'GO_RIGHT_1'),
    ('GO_RIGHT_1', '1', '1', 'R', 'GO_RIGHT_1'),
    ('GO_RIGHT_1', '_', '_', 'L', 'CHECK_RIGHT_1'),

    # CHECK_RIGHT_X: at the rightmost remaining char; expect X.
    ('CHECK_RIGHT_0', '0', '_', 'L', 'WALK_LEFT'),  # match
    ('CHECK_RIGHT_0', '1', '1', 'R', 'FAIL'),       # mismatch
    ('CHECK_RIGHT_0', '_', '1', 'R', 'H'),          # single-char palindrome

    ('CHECK_RIGHT_1', '0', '0', 'R', 'FAIL'),       # mismatch
    ('CHECK_RIGHT_1', '1', '_', 'L', 'WALK_LEFT'),  # match
    ('CHECK_RIGHT_1', '_', '1', 'R', 'H'),          # single-char palindrome

    # WALK_LEFT: walk back to leftmost remaining char.
    ('WALK_LEFT', '0', '0', 'L', 'WALK_LEFT'),
    ('WALK_LEFT', '1', '1', 'L', 'WALK_LEFT'),
    ('WALK_LEFT', '_', '_', 'R', 'LEFT_READ'),

    # FAIL: walk to right boundary, write 0 to indicate failure.
    ('FAIL', '0', '0', 'R', 'FAIL'),
    ('FAIL', '1', '1', 'R', 'FAIL'),
    ('FAIL', '_', '0', 'R', 'H'),
]


def sample_input(rng, length_range):
    """Half-and-half mix of palindromes and random strings."""
    lo, hi = length_range
    n = int(rng.integers(lo, hi + 1))
    if n <= 0:
        return ((),)
    if rng.random() < 0.5:
        # Build a random palindrome
        half = tuple(int(b) for b in rng.integers(0, 2, size=(n + 1) // 2))
        if n % 2 == 0:
            bits = half + half[::-1]
        else:
            bits = half + half[:-1][::-1]
    else:
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
    expected = 1 if list(bits) == list(reversed(bits)) else 0
    return result == expected


def simulate(n_runs, length_range, max_steps=10000, seed=42, noread=False):
    return simulate_task(sys.modules[__name__], n_runs, length_range,
                         max_steps, seed, noread=noread)
