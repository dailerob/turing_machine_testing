"""Reverse Turing machine.  See TASKS.md Task 3."""

from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from _tm_common import simulate_task  # noqa: E402

START_STATE = 'SCAN_RIGHT'
HALT_STATE = 'H'
BLANK_SYMBOL = '_'

# Symbols: '#', '0', '1', 'A', 'B', '_'
PROGRAM = [
    # Walk right past everything to find end of input
    ('SCAN_RIGHT', '#', '#', 'R', 'SCAN_RIGHT'),
    ('SCAN_RIGHT', 'A', 'A', 'R', 'SCAN_RIGHT'),
    ('SCAN_RIGHT', 'B', 'B', 'R', 'SCAN_RIGHT'),
    ('SCAN_RIGHT', '0', '0', 'R', 'SCAN_RIGHT'),
    ('SCAN_RIGHT', '1', '1', 'R', 'SCAN_RIGHT'),
    ('SCAN_RIGHT', '_', '_', 'L', 'BACKUP'),

    # Walk left looking for rightmost unmarked input bit
    ('BACKUP', 'A', 'A', 'L', 'BACKUP'),
    ('BACKUP', 'B', 'B', 'L', 'BACKUP'),
    ('BACKUP', '0', 'A', 'R', 'CARRY_0_THROUGH_INPUT'),
    ('BACKUP', '1', 'B', 'R', 'CARRY_1_THROUGH_INPUT'),
    ('BACKUP', '#', '#', 'R', 'H'),  # all marked => done

    # Carry a '0' to the right past remaining input/marks to the gap
    ('CARRY_0_THROUGH_INPUT', '0', '0', 'R', 'CARRY_0_THROUGH_INPUT'),
    ('CARRY_0_THROUGH_INPUT', '1', '1', 'R', 'CARRY_0_THROUGH_INPUT'),
    ('CARRY_0_THROUGH_INPUT', 'A', 'A', 'R', 'CARRY_0_THROUGH_INPUT'),
    ('CARRY_0_THROUGH_INPUT', 'B', 'B', 'R', 'CARRY_0_THROUGH_INPUT'),
    ('CARRY_0_THROUGH_INPUT', '_', '_', 'R', 'CARRY_0_AFTER_GAP'),

    ('CARRY_1_THROUGH_INPUT', '0', '0', 'R', 'CARRY_1_THROUGH_INPUT'),
    ('CARRY_1_THROUGH_INPUT', '1', '1', 'R', 'CARRY_1_THROUGH_INPUT'),
    ('CARRY_1_THROUGH_INPUT', 'A', 'A', 'R', 'CARRY_1_THROUGH_INPUT'),
    ('CARRY_1_THROUGH_INPUT', 'B', 'B', 'R', 'CARRY_1_THROUGH_INPUT'),
    ('CARRY_1_THROUGH_INPUT', '_', '_', 'R', 'CARRY_1_AFTER_GAP'),

    # In output region, walk right to first blank, write
    ('CARRY_0_AFTER_GAP', '0', '0', 'R', 'CARRY_0_AFTER_GAP'),
    ('CARRY_0_AFTER_GAP', '1', '1', 'R', 'CARRY_0_AFTER_GAP'),
    ('CARRY_0_AFTER_GAP', '_', '0', 'L', 'GO_BACK_OUTPUT'),

    ('CARRY_1_AFTER_GAP', '0', '0', 'R', 'CARRY_1_AFTER_GAP'),
    ('CARRY_1_AFTER_GAP', '1', '1', 'R', 'CARRY_1_AFTER_GAP'),
    ('CARRY_1_AFTER_GAP', '_', '1', 'L', 'GO_BACK_OUTPUT'),

    # Walk back left through output to the gap
    ('GO_BACK_OUTPUT', '0', '0', 'L', 'GO_BACK_OUTPUT'),
    ('GO_BACK_OUTPUT', '1', '1', 'L', 'GO_BACK_OUTPUT'),
    ('GO_BACK_OUTPUT', '_', '_', 'L', 'GO_BACK_THROUGH_GAP'),

    # Walk back left through input to the '#' sentinel
    ('GO_BACK_THROUGH_GAP', '0', '0', 'L', 'GO_BACK_THROUGH_GAP'),
    ('GO_BACK_THROUGH_GAP', '1', '1', 'L', 'GO_BACK_THROUGH_GAP'),
    ('GO_BACK_THROUGH_GAP', 'A', 'A', 'L', 'GO_BACK_THROUGH_GAP'),
    ('GO_BACK_THROUGH_GAP', 'B', 'B', 'L', 'GO_BACK_THROUGH_GAP'),
    ('GO_BACK_THROUGH_GAP', '#', '#', 'R', 'BACKUP_AT_LEFT'),

    # From '#', walk right to end of input, then BACKUP
    ('BACKUP_AT_LEFT', '0', '0', 'R', 'BACKUP_AT_LEFT'),
    ('BACKUP_AT_LEFT', '1', '1', 'R', 'BACKUP_AT_LEFT'),
    ('BACKUP_AT_LEFT', 'A', 'A', 'R', 'BACKUP_AT_LEFT'),
    ('BACKUP_AT_LEFT', 'B', 'B', 'R', 'BACKUP_AT_LEFT'),
    ('BACKUP_AT_LEFT', '_', '_', 'L', 'BACKUP'),
]


def sample_input(rng, length_range):
    n = int(rng.integers(length_range[0], length_range[1] + 1))
    bits = tuple(int(b) for b in rng.integers(0, 2, size=n))
    return (bits,)


def make_initial_tape(input_args):
    (bits,) = input_args
    tape = {-1: '#'}
    for i, b in enumerate(bits):
        tape[i] = str(b)
    return tape


def start_position(input_args):
    return -1


def decode_result(final_tape):
    """Find the output region (after the gap) and decode."""
    if not final_tape:
        return None
    max_pos = max(final_tape.keys())
    # Walk right from position 0 looking for the first '_' (the gap)
    # then read until next '_'.
    p = 0
    while p <= max_pos and final_tape.get(p, '_') in ('A', 'B', '0', '1'):
        p += 1
    # p is at first '_' or past end
    p += 1  # skip the gap
    out = []
    while p <= max_pos:
        s = final_tape.get(p, '_')
        if s in ('0', '1'):
            out.append(s)
            p += 1
        else:
            break
    if not out:
        return None
    return tuple(int(c) for c in out)


def is_correct(input_args, result):
    if result is None:
        return False
    (bits,) = input_args
    return result == bits[::-1]


def simulate(n_runs, length_range, max_steps=20000, seed=42, noread=False):
    return simulate_task(sys.modules[__name__], n_runs, length_range,
                         max_steps, seed, noread=noread)
