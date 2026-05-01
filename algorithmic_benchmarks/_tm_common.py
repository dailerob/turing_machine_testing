"""Shared helpers for TM-trace dataset generation.

Mirrors the conventions of `binary_alphabet_adder.simulate_random_*`.
Each task module defines:

    PROGRAM        : list of (state, read, write, dir, next_state) tuples
    START_STATE    : initial state name
    HALT_STATE     : halt state name (always 'H' here)
    BLANK_SYMBOL   : the tape blank
    SYMBOLS        : set of all tape symbols used
    STATES         : set of all state names used
    make_initial_tape(input_args, rng) -> dict[int, str]
    decode_result(final_tape) -> any
    is_correct(input_args, result) -> bool
    sample_input(rng, length_range) -> input_args
"""

from __future__ import annotations

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from collections import defaultdict  # noqa: E402

from turing_machine import run_turing_machine, history_to_numpy  # noqa: E402

NO_READ_CHAR = '?'


def passthrough_keys(program):
    """Identify (state, read) transitions that are 'pass-through': the
    machine writes back what it read, moves in some direction, and the
    same (dir, next_state) action is shared by at least one OTHER read
    in the same state.  In those cases, the read is not actually used
    by the machine to choose its action — it is just walking past.

    Returns a set of (state, read) tuples in the original program's
    symbol space.
    """
    by_state = defaultdict(list)
    for s, r, w, d, ns in program:
        by_state[s].append((r, w, d, ns))
    pass_set = set()
    for s, transitions in by_state.items():
        groups = defaultdict(list)
        for r, w, d, ns in transitions:
            groups[(d, ns)].append((r, w))
        for (_d, _ns), rws in groups.items():
            if len(rws) < 2:
                continue
            for r, w in rws:
                if w == r:
                    pass_set.add((s, r))
    return pass_set


def passthrough_int_keys(pass_set, state_encoding, symbol_encoding):
    """Lift a name-space pass-through set into the integer state/symbol
    encodings used by `history_to_numpy`."""
    out = set()
    for s, r in pass_set:
        if s in state_encoding and r in symbol_encoding:
            out.add((int(state_encoding[s]), int(symbol_encoding[r])))
    return out


def apply_noread_to_runs(runs, program, state_encoding, symbol_encoding):
    """Take a list of encoded traces (numpy arrays) plus the original
    program & encodings, and return a new list of traces where the
    read+write columns are replaced with a NO_READ integer wherever
    the (state, read) key is pass-through under the program.

    Returns (new_runs, new_symbol_encoding).  The encoding gets a
    NO_READ entry appended at the next free integer.
    """
    pass_set_str = passthrough_keys(program)
    pass_set_int = passthrough_int_keys(pass_set_str, state_encoding,
                                        symbol_encoding)
    new_se = dict(symbol_encoding)
    if NO_READ_CHAR not in new_se:
        new_se[NO_READ_CHAR] = len(new_se)
    nr = new_se[NO_READ_CHAR]
    new_runs = []
    for arr in runs:
        out = arr.copy()
        for i in range(out.shape[0]):
            if int(out[i, 0]) == -1:
                continue
            key = (int(out[i, 0]), int(out[i, 1]))
            if key in pass_set_int:
                out[i, 1] = nr
                out[i, 2] = nr
        new_runs.append(out)
    return new_runs, new_se


def apply_no_read_to_history(history, pass_set, no_read_char=NO_READ_CHAR):
    """Replace the read AND write columns with `no_read_char` whenever
    the transition is pass-through (so the trace no longer reveals which
    of {0, 1, A, B, ...} the machine was scanning past).  Returns a new
    history list."""
    out = []
    for s, r, w, d, ns in history:
        if (s, r) in pass_set:
            out.append((s, no_read_char, no_read_char, d, ns))
        else:
            out.append((s, r, w, d, ns))
    return out


def build_encodings(program, halt_state, blank_symbol):
    states = set([halt_state])
    symbols = set([blank_symbol])
    for s, r, w, _, ns in program:
        states.add(s); states.add(ns)
        symbols.add(r)
        if w is not None:
            symbols.add(w)
    state_encoding = {s: i for i, s in enumerate(sorted(states, key=str))}
    symbol_encoding = {s: i for i, s in enumerate(sorted(symbols, key=str))}
    return state_encoding, symbol_encoding


def simulate_task(task_module, n_runs, length_range, max_steps,
                  seed, include_halt_row=True, noread=False):
    """Run a TM task on n_runs random inputs.

    If `noread=True`, transitions where the machine's action is
    independent of the read symbol have their read+write columns
    replaced with a NO_READ marker (see `passthrough_keys`).

    Returns dict with keys runs, halted_flags, inputs, results,
    correct, state_encoding, symbol_encoding (mirrors
    simulate_random_binary_alphabet_adders).
    """
    rng = np.random.default_rng(seed)
    state_encoding, symbol_encoding = build_encodings(
        task_module.PROGRAM, task_module.HALT_STATE, task_module.BLANK_SYMBOL)
    if noread:
        # Reserve an integer for NO_READ before any traces exist so
        # the encoding is stable.
        if NO_READ_CHAR not in symbol_encoding:
            symbol_encoding[NO_READ_CHAR] = len(symbol_encoding)
        pass_set = passthrough_keys(task_module.PROGRAM)
    else:
        pass_set = set()

    runs, halted, inputs, results, correct = [], [], [], [], []
    for _ in range(n_runs):
        input_args = task_module.sample_input(rng, length_range)
        initial_tape = task_module.make_initial_tape(input_args)
        start_pos = task_module.start_position(input_args)
        tape, _, _, history, did_halt = run_turing_machine(
            task_module.PROGRAM,
            halt_state=task_module.HALT_STATE,
            initial_state=task_module.START_STATE,
            max_steps=max_steps, verbose=False,
            initial_tape=initial_tape,
            blank_symbol=task_module.BLANK_SYMBOL,
            start_position=start_pos)
        result = task_module.decode_result(tape) if did_halt else None
        ok = task_module.is_correct(input_args, result) if did_halt else False
        if noread:
            history = apply_no_read_to_history(history, pass_set)
        # Update encoding to handle any unseen symbols/states
        for s, r, w, _, ns in history:
            if s not in state_encoding:
                state_encoding[s] = len(state_encoding)
            if ns not in state_encoding:
                state_encoding[ns] = len(state_encoding)
            if r not in symbol_encoding:
                symbol_encoding[r] = len(symbol_encoding)
            if w is not None and w not in symbol_encoding:
                symbol_encoding[w] = len(symbol_encoding)
        add_halt = include_halt_row and did_halt
        arr, _, _ = history_to_numpy(history, state_encoding,
                                     symbol_encoding,
                                     include_halt_row=add_halt)
        runs.append(arr); halted.append(did_halt); inputs.append(input_args)
        results.append(result); correct.append(ok)
    return {
        'runs': runs, 'halted_flags': halted, 'inputs': inputs,
        'results': results, 'correct': correct,
        'state_encoding': state_encoding,
        'symbol_encoding': symbol_encoding,
    }
