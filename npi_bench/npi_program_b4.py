"""NPI-style addition program + trace generator — BASE 4.

Same structure as `npi_program.py` (base 10) and `npi_program_b2.py` (base 2).
Addition table = 4×4×2 = 32 cells; obs space = 5^4 = 625 tuples (digits
0..3 + BLANK).
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np

BASE = 4
BLANK = 10

AT_HALT    = 0
AT_RETURN  = 1
AT_CALL    = 2
AT_MOVE    = 3
AT_WRITE   = 4
AT_INIT    = 5
AT_INIT_A  = 6
AT_INIT_B  = 7
N_ACTION_TYPES = 8

SUB_ADD    = 0
SUB_ADD1   = 1
SUB_CARRY  = 2
SUB_LSHIFT = 3

MOVE_p1_L = 0
MOVE_p2_L = 1
MOVE_p3_L = 2
MOVE_p4_L = 3
MOVE_p3_R = 4

WRITE_p3_1 = 0
def WRITE_p4(v: int) -> int:
    assert 0 <= v < BASE
    return 1 + v   # WRITE_p4_0..WRITE_p4_(BASE-1)

INIT_BEGIN   = 0
INIT_A_END   = 1
INIT_B_END   = 2
INIT_END     = 3


def to_base_str(n: int, base: int = BASE) -> str:
    """MSB-first base-`base` digit string."""
    if n == 0:
        return '0'
    digits = []
    while n:
        digits.append(str(n % base))
        n //= base
    return ''.join(reversed(digits))


def _width_in_base(n: int, base: int = BASE) -> int:
    return len(to_base_str(n, base)) if n > 0 else 1


class _NPIRunnerB4:
    def __init__(self, a: int, b: int):
        assert a >= 0 and b >= 0
        self.a_val, self.b_val = a, b
        a_w = _width_in_base(a); b_w = _width_in_base(b)
        max_in = max(a_w, b_w)
        self.n_cols = max_in + 2
        self.row1 = self._encode_lsb_first(a, self.n_cols)
        self.row2 = self._encode_lsb_first(b, self.n_cols)
        self.row3 = [0] * self.n_cols
        self.row4 = [None] * self.n_cols
        self.p = [None, 0, 0, 0, 0]
        self.a_msd = a_w - 1 if a > 0 else 0
        self.b_msd = b_w - 1 if b > 0 else 0
        self.trace: List[Tuple[int, int, int, int, int, int]] = []

    @staticmethod
    def _encode_lsb_first(n: int, width: int) -> List[int]:
        r = []
        for _ in range(width):
            r.append(n % BASE); n //= BASE
        return r

    def _obs_value(self, p_idx: int) -> int:
        col = self.p[p_idx]
        if not (0 <= col < self.n_cols):
            return BLANK
        row = (None, self.row1, self.row2, self.row3, self.row4)[p_idx]
        v = row[col]
        return BLANK if v is None else v

    def _current_obs(self):
        return (self._obs_value(1), self._obs_value(2),
                self._obs_value(3), self._obs_value(4))

    def _emit_init(self, at, arg):
        self.trace.append((BLANK, BLANK, BLANK, BLANK, at, arg))

    def _emit_action(self, at, arg):
        op1, op2, op3, op4 = self._current_obs()
        self.trace.append((op1, op2, op3, op4, at, arg))

    def act_move(self, p_idx, direction):
        if direction == 'L':
            arg = (MOVE_p1_L, MOVE_p2_L, MOVE_p3_L, MOVE_p4_L)[p_idx - 1]
            self._emit_action(AT_MOVE, arg)
            self.p[p_idx] += 1
        else:
            assert p_idx == 3
            self._emit_action(AT_MOVE, MOVE_p3_R)
            self.p[p_idx] -= 1

    def act_write_p3(self, v):
        assert v == 1
        self._emit_action(AT_WRITE, WRITE_p3_1)
        self.row3[self.p[3]] = v

    def act_write_p4(self, v):
        self._emit_action(AT_WRITE, WRITE_p4(v))
        self.row4[self.p[4]] = v

    def sub_carry(self):
        self._emit_action(AT_CALL, SUB_CARRY)
        self.act_move(3, 'L')
        self.act_write_p3(1)
        self.act_move(3, 'R')
        self._emit_action(AT_RETURN, 0)

    def sub_lshift(self):
        self._emit_action(AT_CALL, SUB_LSHIFT)
        self.act_move(1, 'L')
        self.act_move(2, 'L')
        self.act_move(3, 'L')
        self.act_move(4, 'L')
        self._emit_action(AT_RETURN, 0)

    def sub_add1(self):
        self._emit_action(AT_CALL, SUB_ADD1)
        a = self._obs_value(1); b = self._obs_value(2); c = self._obs_value(3)
        a = 0 if a == BLANK else a
        b = 0 if b == BLANK else b
        c = 0 if c == BLANK else c
        s = a + b + c
        self.act_write_p4(s % BASE)
        if s >= BASE:
            self.sub_carry()
        self._emit_action(AT_RETURN, 0)

    def sub_add(self):
        self._emit_action(AT_CALL, SUB_ADD)
        while True:
            past_a = self.p[1] > self.a_msd
            past_b = self.p[2] > self.b_msd
            c_here = self._obs_value(3)
            c_here = 0 if c_here == BLANK else c_here
            if past_a and past_b and c_here == 0:
                break
            self.sub_add1()
            self.sub_lshift()
        self._emit_action(AT_RETURN, 0)


def generate_trace(a: int, b: int) -> np.ndarray:
    r = _NPIRunnerB4(a, b)
    r._emit_init(AT_INIT, INIT_BEGIN)
    for d in to_base_str(a):
        r._emit_init(AT_INIT_A, int(d))
    r._emit_init(AT_INIT, INIT_A_END)
    for d in to_base_str(b):
        r._emit_init(AT_INIT_B, int(d))
    r._emit_init(AT_INIT, INIT_B_END)
    r._emit_init(AT_INIT, INIT_END)
    r.sub_add()
    r._emit_action(AT_HALT, 0)
    return np.array(r.trace, dtype=np.int64)


def decode_output_from_trace(trace: np.ndarray) -> int:
    p4_col = 0
    cells = {}
    for row in trace:
        at, arg = int(row[4]), int(row[5])
        if at == AT_MOVE:
            if arg == MOVE_p4_L: p4_col += 1
        elif at == AT_WRITE:
            if 1 <= arg <= BASE:
                cells[p4_col] = arg - 1
    if not cells:
        return 0
    max_col = max(cells.keys())
    digits = [str(cells.get(i, 0)) for i in range(max_col, -1, -1)]
    s = ''.join(digits).lstrip('0')
    return int(s, BASE) if s else 0


if __name__ == "__main__":
    cases = [(0, 0), (1, 0), (3, 0), (1, 3), (3, 3), (7, 7),
             (15, 1), (10, 6), (63, 1), (1, 63), (42, 21)]
    all_ok = True
    for a, b in cases:
        tr = generate_trace(a, b)
        out = decode_output_from_trace(tr)
        ok = (out == a + b)
        all_ok = all_ok and ok
        print(f"  {a:>5} + {b:>5} = {out:>5}  (expected {a+b:>5})  "
              f"{'OK' if ok else 'FAIL':>4}  trace.shape={tr.shape}")
    print(f"\n{'All passed.' if all_ok else 'FAILURES.'}")
