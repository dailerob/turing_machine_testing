"""NPI-style addition program + 6-column row trace generator.

Reproduces the 4-subroutine decomposition from Reed & de Freitas (ICLR 2016,
"Neural Programmer-Interpreters"):

    ADD()      — top-level loop over columns
    ADD1()     — read column, write sum digit, optionally CARRY
    CARRY()    — propagate +1 to the next-left column
    LSHIFT()   — move all 4 pointers one column to the left

Environment is a 4-row 2D tape:
    Row 1 — operand A digits (right-aligned, LSB at column 0)
    Row 2 — operand B digits (right-aligned, LSB at column 0)
    Row 3 — carry (initially 0)
    Row 4 — output (initially blank)
Each row has its own movable pointer (p1..p4); all start at column 0.

Trace format: a sequence of 6-integer rows, mirroring how the existing TM
benchmarks pack (state, read, write, dir, next_state) into 5-int rows.
Each row has columns:

    [obs_p1, obs_p2, obs_p3, obs_p4, action_type, arg]

obs_pi   — value at pointer i's current column (0..9, BLANK=10)
            BLANK during init rows (the environment hasn't been "stepped" yet)
action_type — one of {HALT, RETURN, CALL, MOVE, WRITE, INIT, INIT_A, INIT_B}
arg      — argument for the action; meaning depends on action_type
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Column / value enumerations
# ---------------------------------------------------------------------------
BLANK = 10  # used in any obs column whose pointer is past the operand digits

# action_type enum
AT_HALT    = 0
AT_RETURN  = 1
AT_CALL    = 2
AT_MOVE    = 3
AT_WRITE   = 4
AT_INIT    = 5   # init structural marker
AT_INIT_A  = 6   # init operand-A digit emission
AT_INIT_B  = 7   # init operand-B digit emission
N_ACTION_TYPES = 8

# CALL arg encoding (subprogram_id)
SUB_ADD    = 0
SUB_ADD1   = 1
SUB_CARRY  = 2
SUB_LSHIFT = 3

# MOVE arg encoding (pointer_id, direction)
MOVE_p1_L = 0
MOVE_p2_L = 1
MOVE_p3_L = 2
MOVE_p4_L = 3
MOVE_p3_R = 4

# WRITE arg encoding (pointer + value)
WRITE_p3_1 = 0
def WRITE_p4(v: int) -> int:
    assert 0 <= v <= 9
    return 1 + v   # so WRITE_p4_0=1, ..., WRITE_p4_9=10

# INIT marker arg
INIT_BEGIN   = 0
INIT_A_END   = 1
INIT_B_END   = 2
INIT_END     = 3

# Number of distinct values per column (used for sanity / vocab summary).
COL_VALUE_DOMAINS = {
    'obs_p1': list(range(0, 10)) + [BLANK],
    'obs_p2': list(range(0, 10)) + [BLANK],
    'obs_p3': [0, 1, BLANK],
    'obs_p4': list(range(0, 10)) + [BLANK],
    'action_type': list(range(N_ACTION_TYPES)),
    'arg': list(range(11)),  # max arg value across all action_types is 10 (WRITE_p4_9)
}


# ---------------------------------------------------------------------------
# Trace generator
# ---------------------------------------------------------------------------
class _NPIRunner:
    """Walks the NPI addition program and records the 6-column trace.

    Each emit_action() call appends one row to self.trace, capturing the
    CURRENT environment observation (4 pointer values) alongside the action
    being taken. After the action mutates the environment, the next emit
    will see the updated obs.
    """

    def __init__(self, a: int, b: int):
        assert a >= 0 and b >= 0
        self.a_val, self.b_val = a, b
        max_in = max(len(str(a)), len(str(b)))
        self.n_cols = max_in + 2   # +1 for possible leading carry, +1 slack
        self.row1 = self._encode_lsb_first(a, self.n_cols)
        self.row2 = self._encode_lsb_first(b, self.n_cols)
        self.row3 = [0] * self.n_cols          # carry row
        self.row4 = [None] * self.n_cols       # output row (None = blank)
        self.p = [None, 0, 0, 0, 0]            # 1-indexed; cols start at LSB
        self.a_msd = len(str(a)) - 1 if a > 0 else 0
        self.b_msd = len(str(b)) - 1 if b > 0 else 0
        self.trace: List[Tuple[int, int, int, int, int, int]] = []

    @staticmethod
    def _encode_lsb_first(n: int, width: int) -> List[int]:
        r = []
        for _ in range(width):
            r.append(n % 10); n //= 10
        return r

    def _obs_value(self, p_idx: int) -> int:
        """Return the value at pointer p_idx's current column, or BLANK."""
        col = self.p[p_idx]
        if not (0 <= col < self.n_cols):
            return BLANK
        row = (None, self.row1, self.row2, self.row3, self.row4)[p_idx]
        v = row[col]
        return BLANK if v is None else v

    def _current_obs(self) -> Tuple[int, int, int, int]:
        return (self._obs_value(1), self._obs_value(2),
                self._obs_value(3), self._obs_value(4))

    def _emit_init(self, action_type: int, arg: int):
        """Emit a row in the init block: obs is fully BLANK (no env yet)."""
        self.trace.append((BLANK, BLANK, BLANK, BLANK, action_type, arg))

    def _emit_action(self, action_type: int, arg: int):
        """Emit an action row using the CURRENT observation. The action
        will be applied separately (so the next emit sees updated obs)."""
        op1, op2, op3, op4 = self._current_obs()
        self.trace.append((op1, op2, op3, op4, action_type, arg))

    # ---- ACT primitives (emit + apply) ----------------------------------
    def act_move(self, p_idx: int, direction: str):
        if direction == 'L':
            arg = (MOVE_p1_L, MOVE_p2_L, MOVE_p3_L, MOVE_p4_L)[p_idx - 1]
            self._emit_action(AT_MOVE, arg)
            self.p[p_idx] += 1
        else:
            assert p_idx == 3, "only p3 ever moves right (inside CARRY)"
            self._emit_action(AT_MOVE, MOVE_p3_R)
            self.p[p_idx] -= 1

    def act_write_p3(self, v: int):
        assert v == 1, "row 3 only ever receives WRITE 1 (from CARRY)"
        self._emit_action(AT_WRITE, WRITE_p3_1)
        self.row3[self.p[3]] = v

    def act_write_p4(self, v: int):
        self._emit_action(AT_WRITE, WRITE_p4(v))
        self.row4[self.p[4]] = v

    # ---- Subroutines (emit CALL, body, RETURN) ---------------------------
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
        # Convert BLANK to 0 for arithmetic
        a = 0 if a == BLANK else a
        b = 0 if b == BLANK else b
        c = 0 if c == BLANK else c
        s = a + b + c
        self.act_write_p4(s % 10)
        if s >= 10:
            self.sub_carry()
        self._emit_action(AT_RETURN, 0)

    def sub_add(self):
        self._emit_action(AT_CALL, SUB_ADD)
        while True:
            # Termination: both operand pointers are past their MSB AND
            # the carry at the current column is zero.
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
    """Generate the full NPI-style 6-column trace for a + b.

    Returns
    -------
    np.ndarray of shape (n_rows, 6), dtype int64. Columns:
        [obs_p1, obs_p2, obs_p3, obs_p4, action_type, arg]
    """
    r = _NPIRunner(a, b)
    # ---- Init block ----
    r._emit_init(AT_INIT, INIT_BEGIN)
    for d in str(a):                       # operand A, MSB-first
        r._emit_init(AT_INIT_A, int(d))
    r._emit_init(AT_INIT, INIT_A_END)
    for d in str(b):                       # operand B, MSB-first
        r._emit_init(AT_INIT_B, int(d))
    r._emit_init(AT_INIT, INIT_B_END)
    r._emit_init(AT_INIT, INIT_END)
    # ---- Run ADD ----
    r.sub_add()
    # ---- HALT ----
    r._emit_action(AT_HALT, 0)
    return np.array(r.trace, dtype=np.int64)


def decode_output_from_trace(trace: np.ndarray) -> int:
    """Reconstruct the output integer from a 6-column trace.

    Walks the trace, simulating p4's column position and recording each
    WRITE_p4_v. Decodes the resulting row 4 digits MSB-first.
    """
    p4_col = 0
    cells = {}
    for row in trace:
        at, arg = int(row[4]), int(row[5])
        if at == AT_MOVE:
            if arg == MOVE_p4_L:
                p4_col += 1
        elif at == AT_WRITE:
            if 1 <= arg <= 10:     # WRITE_p4_v with v = arg - 1
                cells[p4_col] = arg - 1
    if not cells:
        return 0
    max_col = max(cells.keys())
    digits = ''.join(str(cells.get(i, 0)) for i in range(max_col, -1, -1)).lstrip('0')
    return int(digits) if digits else 0


def vocab_summary() -> str:
    out = []
    out.append("Column value domains:")
    for col, vals in COL_VALUE_DOMAINS.items():
        out.append(f"  {col:<12}: {len(vals)} distinct values  {vals}")
    # Enumerate the (action_type, arg) tuples that are actually valid
    pairs = []
    pairs.append(('HALT',     [(AT_HALT,    0)]))
    pairs.append(('RETURN',   [(AT_RETURN,  0)]))
    pairs.append(('CALL',     [(AT_CALL,    a) for a in range(4)]))
    pairs.append(('MOVE',     [(AT_MOVE,    a) for a in range(5)]))
    pairs.append(('WRITE',    [(AT_WRITE,   a) for a in range(11)]))
    pairs.append(('INIT',     [(AT_INIT,    a) for a in range(4)]))
    pairs.append(('INIT_A',   [(AT_INIT_A,  d) for d in range(10)]))
    pairs.append(('INIT_B',   [(AT_INIT_B,  d) for d in range(10)]))
    n_valid = sum(len(p) for _, p in pairs)
    out.append(f"\nTotal valid (action_type, arg) tuples: {n_valid}")
    for name, ps in pairs:
        out.append(f"  {name:<8}: {len(ps):>2d} variants")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(vocab_summary())
    print()
    test_cases = [(0, 0), (5, 0), (0, 7), (12, 7), (8, 5), (99, 1),
                  (123, 456), (9, 1), (999, 1), (54321, 12345),
                  (1000000, 1), (1, 999999)]
    all_ok = True
    for a, b in test_cases:
        tr = generate_trace(a, b)
        out = decode_output_from_trace(tr)
        ok = (out == a + b)
        all_ok = all_ok and ok
        print(f"{a:>8} + {b:>8} = {out:>9}  (expected {a+b:>9})  "
              f"{'OK' if ok else 'FAIL':>4}  trace_shape={tr.shape}")
    print(f"\n{'All passed.' if all_ok else 'FAILURES.'}")
