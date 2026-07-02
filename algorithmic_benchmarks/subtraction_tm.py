"""Subtraction Turing machine: A - B in binary, A >= B both >= 0.

Tape alphabet: {'0', '1', '_'} only — same ternary alphabet as
binary_alphabet_adder.

Tape layout (MSB-first for both numbers):
    A_MSB ... A_LSB _ B_MSB ... B_LSB _ _ _ ...

Algorithm: repeated decrement loop:
    while B != 0:
        B := B - 1   (decrement B in place)
        A := A - 1   (decrement A in place)
Final tape has A = original_A - original_B in A's slot; B is all zeros.

Decrement of MSB-first binary at its LSB end:
    at LSB, walk left.
    '1' -> '0' and done.
    '0' -> '1' and borrow leftward.

Zero-check on B: scan from MSB to trailing blank.
    if no '1' seen, B is 0 -> halt.

States:
    S_FIND_SEP        scan right through A to the separator blank
    S_CHECK_ZERO_0    scanning B, no '1' seen yet
    S_CHECK_ZERO_1    scanning B, at least one '1' seen
    S_DEC_B           at B's LSB, decrementing (handles borrow)
    S_GOTO_A          walking left through B back to the separator and
                      stepping into A's LSB
    S_DEC_A           at A's LSB, decrementing (handles borrow)
    S_RESET           walking left through A to its MSB, then back to
                      start position
    H                 halt
"""

from __future__ import annotations
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from _tm_common import simulate_task  # noqa: E402

START_STATE = 'S_FIND_SEP'
HALT_STATE = 'H'
BLANK_SYMBOL = '_'

PROGRAM = [
    # S_FIND_SEP: walk right through A to the separator blank.
    ('S_FIND_SEP', '0', '0', 'R', 'S_FIND_SEP'),
    ('S_FIND_SEP', '1', '1', 'R', 'S_FIND_SEP'),
    ('S_FIND_SEP', '_', '_', 'R', 'S_CHECK_ZERO_0'),

    # S_CHECK_ZERO_0: scanning B from MSB, no '1' seen yet.
    ('S_CHECK_ZERO_0', '0', '0', 'R', 'S_CHECK_ZERO_0'),
    ('S_CHECK_ZERO_0', '1', '1', 'R', 'S_CHECK_ZERO_1'),
    ('S_CHECK_ZERO_0', '_', '_', 'L', 'H'),     # B == 0, halt

    # S_CHECK_ZERO_1: scanning B, have seen a '1'; walk to the end.
    ('S_CHECK_ZERO_1', '0', '0', 'R', 'S_CHECK_ZERO_1'),
    ('S_CHECK_ZERO_1', '1', '1', 'R', 'S_CHECK_ZERO_1'),
    ('S_CHECK_ZERO_1', '_', '_', 'L', 'S_DEC_B'),

    # S_DEC_B: at B's LSB, decrement; borrow leftward through B.
    ('S_DEC_B', '1', '0', 'L', 'S_GOTO_A'),     # done, no further borrow
    ('S_DEC_B', '0', '1', 'L', 'S_DEC_B'),      # borrow leftward

    # S_GOTO_A: walk left through B to the separator, then step left
    # into A's LSB.
    ('S_GOTO_A', '0', '0', 'L', 'S_GOTO_A'),
    ('S_GOTO_A', '1', '1', 'L', 'S_GOTO_A'),
    ('S_GOTO_A', '_', '_', 'L', 'S_DEC_A'),

    # S_DEC_A: at A's LSB, decrement; borrow leftward.
    ('S_DEC_A', '1', '0', 'L', 'S_RESET'),       # done
    ('S_DEC_A', '0', '1', 'L', 'S_DEC_A'),       # borrow leftward

    # S_RESET: walk left through A to its left boundary, then step right
    # back to A's MSB to restart the loop.
    ('S_RESET', '0', '0', 'L', 'S_RESET'),
    ('S_RESET', '1', '1', 'L', 'S_RESET'),
    ('S_RESET', '_', '_', 'R', 'S_FIND_SEP'),
]


def sample_input(rng, length_range):
    """Sample two non-negative integers (a, b) with a >= b."""
    lo, hi = length_range
    bits_n = int(rng.integers(lo, hi + 1))
    if bits_n <= 0:
        return (0, 0)
    a = int(rng.integers(0, 2 ** bits_n))
    b = int(rng.integers(0, a + 1)) if a > 0 else 0
    return (a, b)


def make_initial_tape(input_args):
    a, b = input_args
    a_bits = bin(a)[2:] if a > 0 else '0'
    b_bits = bin(b)[2:] if b > 0 else '0'
    tape = {}
    pos = 0
    for c in a_bits:
        tape[pos] = c; pos += 1
    pos += 1  # separator (left as default blank)
    for c in b_bits:
        tape[pos] = c; pos += 1
    return tape


def start_position(input_args):
    return 0


def decode_result(final_tape):
    """Read A's bits from position 0 until first blank → integer."""
    if not final_tape:
        return None
    pos = 0
    bits = []
    while final_tape.get(pos, '_') in ('0', '1'):
        bits.append(final_tape[pos]); pos += 1
    if not bits:
        return None
    try:
        return int(''.join(bits), 2)
    except ValueError:
        return None


def is_correct(input_args, result):
    if result is None:
        return False
    a, b = input_args
    return result == a - b


def simulate(n_runs, length_range, max_steps=200000, seed=42, noread=False):
    return simulate_task(sys.modules[__name__], n_runs, length_range,
                         max_steps, seed, noread=noread)
