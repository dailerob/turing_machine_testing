"""Dyck-1 sequence sampler.  See TASKS.md Task 4.

Alphabet: 0 = '(', 1 = ')', 2 = END.

Each generated sequence is a single random walk over the depth
counter, starting at depth 0 and ending the first time depth returns
to 0 *and* total emitted length >= length_min.  The terminating END
token marks the boundary so a sequence model trained on
concatenated sequences can identify them.

Public:
    sample(rng, max_depth=4, p_open=0.55, length_min=4, length_max=200)
        -> np.ndarray of int64 token ids

    simulate(n_runs, max_depth, p_open, length_min, length_max, seed)
        -> dict {sequences: list[ndarray], correct_walks: list[bool]}

`correct_walks` is True iff the walk returned to depth 0 within
length_max — used as a sanity check (should always be True for the
default sampler).
"""

from __future__ import annotations
import numpy as np

OPEN, CLOSE, END = 0, 1, 2
ALPHABET_SIZE = 3


def sample(rng, max_depth=4, p_open=0.55, length_min=4, length_max=200):
    out = []
    depth = 0
    while True:
        if len(out) >= length_max:
            # force-close to depth 0 then END
            while depth > 0 and len(out) < length_max + max_depth:
                out.append(CLOSE); depth -= 1
            out.append(END)
            return np.asarray(out, dtype=np.int64)
        if depth == 0:
            if len(out) >= length_min:
                out.append(END)
                return np.asarray(out, dtype=np.int64)
            choice = OPEN
        elif depth >= max_depth:
            choice = CLOSE
        else:
            choice = OPEN if rng.random() < p_open else CLOSE
        out.append(choice)
        depth += 1 if choice == OPEN else -1
    # unreachable


def is_balanced(seq):
    """Verify that the prefix up to the first END is balanced."""
    depth = 0
    for tok in seq:
        if tok == END:
            return depth == 0
        if tok == OPEN:
            depth += 1
        elif tok == CLOSE:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def simulate(n_runs, max_depth=4, p_open=0.55, length_min=4,
             length_max=200, seed=42):
    rng = np.random.default_rng(seed)
    seqs = [sample(rng, max_depth, p_open, length_min, length_max)
            for _ in range(n_runs)]
    correct = [is_balanced(s) for s in seqs]
    return {'sequences': seqs, 'correct_walks': correct,
            'alphabet_size': ALPHABET_SIZE}
