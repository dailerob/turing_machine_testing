"""bit_count_mod_3: count number of 1-bits in input, output (count mod 3)
as a 2-bit MSB-first binary number. Generalizes parity (mod 2) to a
deeper finite-state counter.

Output encoding (count mod 3 → 2-bit MSB-first):
    0 → "00"
    1 → "01"
    2 → "10"

Tape alphabet: {'0', '1', '_'} (ternary).
"""

from __future__ import annotations
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from _tm_common import simulate_task  # noqa: E402

START_STATE = 'SCAN_0'
HALT_STATE = 'H'
BLANK_SYMBOL = '_'

PROGRAM = [
    # Counting phase: state SCAN_X = "have seen X ones (mod 3)"
    ('SCAN_0', '0', '0', 'R', 'SCAN_0'),
    ('SCAN_0', '1', '1', 'R', 'SCAN_1'),
    ('SCAN_0', '_', '0', 'R', 'WRITE_0_LSB'),  # mod=0 → write "00"

    ('SCAN_1', '0', '0', 'R', 'SCAN_1'),
    ('SCAN_1', '1', '1', 'R', 'SCAN_2'),
    ('SCAN_1', '_', '0', 'R', 'WRITE_1_LSB'),  # mod=1 → write "01"

    ('SCAN_2', '0', '0', 'R', 'SCAN_2'),
    ('SCAN_2', '1', '1', 'R', 'SCAN_0'),
    ('SCAN_2', '_', '1', 'R', 'WRITE_2_LSB'),  # mod=2 → write "10"

    # Writing phase: write LSB (second bit) of the 2-bit output
    ('WRITE_0_LSB', '_', '0', 'R', 'H'),
    ('WRITE_1_LSB', '_', '1', 'R', 'H'),
    ('WRITE_2_LSB', '_', '0', 'R', 'H'),
]


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
    """Read the 2-bit output appended past the input."""
    if not final_tape:
        return None
    # Find the contiguous binary string starting at position 0.
    pos = 0
    while final_tape.get(pos, '_') in ('0', '1'):
        pos += 1
    # Now pos is at the first blank past the input. The output is at
    # positions (input_len, input_len + 1) = (pos, pos + 1) — but those
    # are also written! Actually the output overwrites the blank at
    # input_len, so the output occupies positions [input_len, input_len + 1].
    # In our PROGRAM we wrote first bit at the SCAN _ transition (which
    # was the boundary blank), then the second bit one position right.
    # So actually the input is at [0, pos), and output is at the LAST
    # two written positions on the tape.
    # Cleanest: find the rightmost two non-blank positions.
    keys = sorted(final_tape.keys())
    # Walk right-to-left collecting non-blank
    out = []
    for p in reversed(keys):
        s = final_tape.get(p, '_')
        if s in ('0', '1'):
            out.append(s)
        if len(out) == 2:
            break
    if len(out) < 2:
        return None
    # out is in reverse order: rightmost first. So out[1] is MSB, out[0] is LSB.
    msb, lsb = out[1], out[0]
    return int(msb) * 2 + int(lsb)


def is_correct(input_args, result):
    if result is None:
        return False
    (bits,) = input_args
    return result == (sum(bits) % 3)


def simulate(n_runs, length_range, max_steps=10000, seed=42, noread=False):
    return simulate_task(sys.modules[__name__], n_runs, length_range,
                         max_steps, seed, noread=noread)
