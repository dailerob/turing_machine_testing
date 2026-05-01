"""
Binary-alphabet Turing machine adder.

Tape alphabet: {'0', '1', '_'} only. No extra markers, no '+' symbol.

Tape layout (MSB-first for both numbers):
    A_MSB ... A_LSB _ B_MSB ... B_LSB _ _ _ ...

Algorithm (repeated until B = 0):
    while B != 0:
        B := B - 1   (decrement B in place)
        A := A + 1   (increment A in place, extending leftward if needed)

Decrement of a MSB-first binary number done at its LSB end:
    at LSB, walk left. '1' -> '0' and done. '0' -> '1' and borrow leftward.
Increment of a MSB-first binary number done at its LSB end:
    at LSB, walk left. '0' -> '1' and done. '1' -> '0' and carry leftward.
    Blank to the left of MSB means A has grown; write '1' there.

Zero-check on B is done by scanning B from MSB to trailing blank:
    if no '1' is seen, B is zero -> halt.

States:
    S_FIND_SEP      scan right through A to the separator blank
    S_CHECK_ZERO_0  scanning B, no '1' seen yet
    S_CHECK_ZERO_1  scanning B, at least one '1' seen
    S_DEC_LSB       at B's LSB, decrementing (handling borrow)
    S_GOTO_A        walking left through B back to A (after dec done)
    S_GOTO_A_INA    walking left through A to its LSB (sep-1 is reached via GOTO_A)
    S_INC           incrementing A at LSB, propagating carry leftward
    H               halt
"""

import numpy as np
from turing_machine import run_turing_machine, string_to_tape, tape_to_string, history_to_numpy


BLANK = '_'


BINARY_ALPHABET_ADDER = [
    # ---- FIND_SEP: scan right through A to the separator ----
    ('FIND_SEP', '0', '0', 'R', 'FIND_SEP'),
    ('FIND_SEP', '1', '1', 'R', 'FIND_SEP'),
    ('FIND_SEP', '_', '_', 'R', 'CZ0'),

    # ---- CZ0: scan B for any '1'; if hit blank first, B is zero -> halt ----
    ('CZ0', '0', '0', 'R', 'CZ0'),
    ('CZ0', '1', '1', 'R', 'CZ1'),
    ('CZ0', '_', '_', 'L', 'H'),

    # ---- CZ1: seen a '1'; keep walking right to end of B ----
    ('CZ1', '0', '0', 'R', 'CZ1'),
    ('CZ1', '1', '1', 'R', 'CZ1'),
    ('CZ1', '_', '_', 'L', 'DEC'),   # step back to B's LSB

    # ---- DEC: at B's LSB; perform B := B - 1 with borrow propagating left ----
    ('DEC', '1', '0', 'L', 'GOTO_A'),   # 1 - 1 = 0, done
    ('DEC', '0', '1', 'L', 'DEC'),      # 0 - 1 = 1, borrow continues left
    # If DEC sees '_' it would mean all of B was 0, but CZ1 guaranteed a '1'
    # exists, so this case is unreachable.

    # ---- GOTO_A: walk left through remaining B to the separator, then into A ----
    ('GOTO_A', '0', '0', 'L', 'GOTO_A'),
    ('GOTO_A', '1', '1', 'L', 'GOTO_A'),
    ('GOTO_A', '_', '_', 'L', 'INC'),   # crossed sep -> now at A's LSB

    # ---- INC: A := A + 1, carry propagates leftward; may extend A ----
    ('INC', '0', '1', 'R', 'FIND_SEP'), # done; restart loop
    ('INC', '1', '0', 'L', 'INC'),      # carry continues
    ('INC', '_', '1', 'R', 'FIND_SEP'), # A grew leftward; new MSB = 1
]


def encode_tape(a: int, b: int, start: int = 0):
    """Encode two non-negative ints as tape dict: '<A>_<B>' starting at `start`."""
    s = f"{bin(a)[2:]}_{bin(b)[2:]}"
    return string_to_tape(s, start_position=start), s


def decode_tape(tape, blank=BLANK):
    """Read A off the tape: first contiguous 0/1 run, skipping leading blanks."""
    if not tape:
        return 0
    positions = sorted(tape.keys())
    bits = []
    seen_digit = False
    for p in positions:
        c = tape[p]
        if c == blank:
            if seen_digit:
                break
            else:
                continue
        seen_digit = True
        bits.append(str(c))
    return int(''.join(bits), 2) if bits else 0


def run_adder(a: int, b: int, verbose=False, max_steps=10_000_000):
    tape, s = encode_tape(a, b, start=0)
    final_tape, steps, _, history, halted = run_turing_machine(
        BINARY_ALPHABET_ADDER,
        halt_state='H',
        initial_state='FIND_SEP',
        max_steps=max_steps,
        verbose=verbose,
        initial_tape=tape,
        blank_symbol=BLANK,
        start_position=0,
    )
    return decode_tape(final_tape), steps, halted, final_tape, history, s


def simulate_random_binary_alphabet_adders(n_runs, num_range=(0, 255), max_steps=200_000,
                                           include_halt_row=True, seed=None, verbose=False):
    """
    Simulate the binary-alphabet adder on random addition problems.

    Mirrors turing_machine.simulate_random_adders but uses BINARY_ALPHABET_ADDER
    and the tape alphabet {'0', '1', '_'}.

    Returns a dict with: 'runs', 'halted_flags', 'inputs', 'results', 'correct',
    'state_encoding', 'symbol_encoding'.
    """
    if seed is not None:
        np.random.seed(seed)

    program = BINARY_ALPHABET_ADDER
    halt_state = 'H'
    start_state = 'FIND_SEP'

    # Build stable encodings from program (so state/symbol ids are fixed across runs).
    all_states, all_symbols = set(), set()
    for curr, read, write, _, nxt in program:
        all_states.update([curr, nxt])
        all_symbols.update([read, write])
    all_states.add(halt_state)
    all_symbols.add(BLANK)

    state_encoding = {s: i for i, s in enumerate(sorted(all_states, key=str))}
    symbol_encoding = {s: i for i, s in enumerate(sorted(all_symbols, key=str))}

    runs, halted_flags, inputs, results, correct = [], [], [], [], []
    lo, hi = num_range

    for idx in range(n_runs):
        a = int(np.random.randint(lo, hi + 1))
        b = int(np.random.randint(lo, hi + 1))
        inputs.append((a, b))

        initial_tape, _ = encode_tape(a, b, start=0)
        final_tape, steps, _, history, halted = run_turing_machine(
            program,
            halt_state=halt_state,
            initial_state=start_state,
            max_steps=max_steps,
            verbose=False,
            initial_tape=initial_tape,
            blank_symbol=BLANK,
            start_position=0,
        )
        halted_flags.append(halted)

        result_decimal = decode_tape(final_tape) if halted else None
        is_correct = halted and result_decimal == a + b
        results.append(result_decimal)
        correct.append(is_correct)

        if verbose:
            status = "OK" if is_correct else ("FAIL" if halted else "TIMEOUT")
            print(f"Run {idx+1}/{n_runs}: {a}+{b}={a+b} got {result_decimal} "
                  f"steps={steps} {status}")

        arr, _, _ = history_to_numpy(
            history, state_encoding, symbol_encoding,
            include_halt_row=(include_halt_row and halted),
        )
        runs.append(arr)

    return {
        'runs': runs,
        'halted_flags': halted_flags,
        'inputs': inputs,
        'results': results,
        'correct': correct,
        'state_encoding': state_encoding,
        'symbol_encoding': symbol_encoding,
    }


def validate(cases):
    passed = 0
    failed = []
    for a, b in cases:
        got, steps, halted, tape, hist, s_in = run_adder(a, b)
        expected = a + b
        if halted and got == expected:
            passed += 1
        else:
            failed.append((a, b, expected, got, halted, steps, s_in,
                           tape_to_string(tape, blank_symbol=BLANK)))
    print(f"Passed {passed}/{len(cases)}")
    for a, b, exp, got, halted, steps, s_in, out in failed[:10]:
        print(f"  FAIL {a}+{b}: expected {exp}, got {got}, "
              f"halted={halted}, steps={steps}")
        print(f"    input  : '{s_in}'")
        print(f"    output : '{out}'")
    return passed, failed


if __name__ == "__main__":
    import random
    random.seed(0)

    edge = [
        (0, 0), (0, 1), (1, 0), (1, 1),
        (2, 3), (3, 2), (7, 1), (1, 7),
        (5, 3), (3, 5), (15, 1), (1, 15),
        (255, 1), (127, 128), (100, 200),
        (0, 127), (127, 0),
        (31, 31), (63, 64),
    ]
    rand = [(random.randint(0, 255), random.randint(0, 255)) for _ in range(40)]
    cases = edge + rand

    print("Trace for 5 + 3:")
    got, steps, halted, tape, hist, s_in = run_adder(5, 3)
    print(f"  input  : '{s_in}'")
    print(f"  output : '{tape_to_string(tape, blank_symbol=BLANK)}'")
    print(f"  decoded: {got} (expected {5 + 3}), steps={steps}, halted={halted}\n")

    validate(cases)
