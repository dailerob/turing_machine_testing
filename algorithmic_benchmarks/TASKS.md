# Algorithmic-task benchmarks for GDC vs CHMM

This folder is a small but extensible benchmark suite for sequence
models on **algorithmic / Turing-machine** tasks. It complements the
`hmm_comparison/` folder (random-HMM forecasting and hidden-state
alignment) and the existing binary-alphabet adder experiment
(`BINARY_ALPHABET_ADDER_EXPERIMENT.md`).

The point of this folder: we have GDC at 99.87% and CHMM at 99.99%
on the binary-alphabet adder. That is one task; the paper plan calls
for a broader sweep across qualitatively different algorithmic
structures so we can characterise the failure modes of each model.

## Common protocol

Three of the four tasks here are **Turing-machine traces**: the
machine runs deterministically on a tape, and each step yields a row
`(state, read, write, direction, next_state)`. The trace is the data.
This matches the existing binary-adder protocol exactly.

The fourth task (Dyck-1) is a pure **sequence-prediction** task — the
model sees a single symbol stream and must predict the next symbol.

For TM-trace tasks, two evaluation views are supported:

* **Full 5-column**: token at step *t* is the full tuple
  `(state, read, write, dir, next_state)`. Hidden TM state is
  observable. This is the easy setting.
* **Reduced 3-column**: token is `(read, write, dir)`. The hidden
  TM state must be inferred from history. This is where the
  meaningful comparison lives.

Both GDC and CHMM are evaluated 1-step-ahead, **conditional on the
observed next read symbol** (mirroring the binary-adder protocol).

### Train / test discipline

For each TM task:

* **Train** on tapes with input lengths drawn from a small range
  (e.g. inputs ≤ 6 bits).
* **Test** on tapes with input lengths drawn from a larger range
  (e.g. inputs ≤ 12 bits) — *out-of-distribution length*, which is
  the structurally interesting axis.

For Dyck-1: train on max-depth ≤ 6, test on max-depth ≤ 12.

This is deliberately the same recipe as the binary adder (train
B ≤ 32, test B ≤ 1000) so cross-task comparisons are clean.

---

## Task 1 — Parity TM (`parity_tm.py`)

**What it computes.** Parity of the input bit string (XOR of all
bits), written at the position immediately after the input.

**Tape format.**

```
position:   0 1 2 ... N-1   N
input:      b_0 b_1 ... b_{N-1}  '_'   '_'   '_'  ...
output:     b_0 b_1 ... b_{N-1}  p     '_'   '_'  ...      (p = parity bit)
```

**States.**

* `SCAN0` — even parity so far (initial state)
* `SCAN1` — odd parity so far
* `H` — halt

**Transition table.**

```
(SCAN0, '0') -> '0', R, SCAN0
(SCAN0, '1') -> '1', R, SCAN1
(SCAN0, '_') -> '0', R, H        # parity 0, written; halt
(SCAN1, '0') -> '0', R, SCAN1
(SCAN1, '1') -> '1', R, SCAN0
(SCAN1, '_') -> '1', R, H        # parity 1, written; halt
```

**Why it's interesting.** Cleanest possible TM with non-trivial
state — the model must track a 1-bit hidden state across an
arbitrary span of observations. Length extrapolation is the natural
stress: train short, test long.

**Tape lengths.** Train `N ∈ [3, 8]`. Test `N ∈ [16, 32]`.

---

## Task 2 — Increment TM (`increment_tm.py`)

**What it computes.** Adds 1 to a binary number. Input is a binary
number written MSB-first at positions 0..N-1; the head starts at
position 0.

**Tape format.**

```
input:   b_{N-1} b_{N-2} ... b_0  '_'  '_' ...     (MSB at position 0)
output:  bit-pattern of (input + 1), possibly extending leftward by 1 cell
```

**States.**

* `FIND_LSB` — walk right to LSB (initial state)
* `INC` — walk left applying +1 with carry
* `DONE` — walk right back to start (reset head)
* `H` — halt

**Transition table.**

```
# Walk right to find LSB
(FIND_LSB, '0') -> '0', R, FIND_LSB
(FIND_LSB, '1') -> '1', R, FIND_LSB
(FIND_LSB, '_') -> '_', L, INC

# Increment: 0 -> 1 stops, 1 -> 0 carries
(INC, '0') -> '1', R, DONE
(INC, '1') -> '0', L, INC
(INC, '_') -> '1', R, DONE        # carry past MSB, A grew

# Walk right back to LSB (just for symmetry — could halt directly)
(DONE, '0') -> '0', R, DONE
(DONE, '1') -> '1', R, DONE
(DONE, '_') -> '_', L, H
```

**Why it's interesting.** Carry propagation is a multi-step
deterministic dependency whose length depends on the input
(number of trailing 1s). Tests whether the model can track an
"in-carry" hidden state.

**Tape lengths.** Train inputs in `[0, 31]` (≤ 5 bits). Test inputs
in `[0, 4095]` (≤ 12 bits).

---

## Task 3 — Reverse TM (`reverse_tm.py`)

**What it computes.** Given a binary input of length N, writes the
reversed string to the right of the input.

**Tape format.**

```
positions: -1 0 1 ... N-1    N         N+1 ... 2N
initial:   #  b_0 b_1 ... b_{N-1}  '_'  '_'  ...  '_'
final:     #  A_0 A_1 ... A_{N-1}  '_'  b_{N-1} b_{N-2} ... b_0  '_'
```

`A` ∈ {`A`, `B`} are markers: `A = '0' done`, `B = '1' done`. The
character `#` at position −1 is a permanent left-edge sentinel.
A single blank `'_'` at position N separates input from output.

**States.** (10 states)

* `SCAN_RIGHT` — walk right to end of input (initial)
* `BACKUP` — walk left looking for rightmost unmarked input bit
* `CARRY_0_THROUGH_INPUT`, `CARRY_1_THROUGH_INPUT` — walk through
  remaining input + marks to the gap
* `CARRY_0_AFTER_GAP`, `CARRY_1_AFTER_GAP` — walk through output to
  the next blank
* `GO_BACK_OUTPUT` — walk left through output back to gap
* `GO_BACK_THROUGH_GAP` — walk left through input back to `#`
* `BACKUP_AT_LEFT` — walk right from `#` to end of input
* `H` — halt

**Transition table** (in code; see `reverse_tm.py`).

**Why it's interesting.** Multi-pass O(N²) structure with several
distinct phases. The same emission `(read, write, dir)` is produced
by several different states (e.g. moving R through input is the
same emission whether we're in `SCAN_RIGHT` or
`CARRY_0_THROUGH_INPUT`). This is the kind of state ambiguity CHMM's
clones are designed to disambiguate.

**Tape lengths.** Train N ∈ [3, 6]. Test N ∈ [10, 16].

---

## Task 4 — Dyck-1 sequence prediction (`dyck1.py`)

**Not a TM.** Sequence-only prediction task.

**What it computes.** Sample valid Dyck-1 strings (balanced
parentheses over `( )`) and predict the next symbol given the prefix.

**Sampling.** Random-walk over the depth counter:

* depth = 0: emit `(`
* depth ≥ max_depth: emit `)`
* otherwise: emit `(` with prob `p_open` (default 0.55), else `)`
* terminate when depth = 0 *and* total length ≥ length_min, after
  that always emit a special `END` token to allow forecasting to
  recognise sequence boundaries.

**Alphabet.** `{ '(' = 0, ')' = 1, END = 2 }`.

**Why it's interesting.** Counter automaton — to predict whether `)`
is legal you must track depth ≥ 1 across arbitrarily long spans.
Theoretically a non-regular language; tests whether GDC's prefix
form and CHMM's clones can approximate counter dynamics.

**Train / test.** Train: 1000 sequences with `max_depth=4`. Test: 200
sequences with `max_depth=8` (length / depth OOD).

**Eval metric.** Per-symbol next-token accuracy on the `(` / `)`
positions (skipping the `END` token positions where the choice is
trivial).

---

## Files in this folder

* `TASKS.md` — this file.
* `parity_tm.py` — Parity TM definition + dataset generator.
* `increment_tm.py` — Increment TM definition + dataset generator.
* `reverse_tm.py` — Reverse TM definition + dataset generator.
* `dyck1.py` — Dyck-1 sampler + dataset generator.
* `test_generators.py` — sanity tests; verifies each task's
  ground-truth correctness on a battery of inputs.
* `run_benchmarks.py` — GDC + CHMM benchmarking harness across
  all four tasks.
* `BENCHMARK_RESULTS.md` — written after the harness has run.

## What we expect

Building on the existing GDC-vs-CHMM finding on the binary adder
(CHMM 0.014% error, GDC 0.13% error, 9× compression in CHMM's favour):

| task | GDC expectation | CHMM expectation | structural reason |
|---|---|---|---|
| Parity | near-perfect in-distribution; OOD-length degrades sharply | near-perfect both; very small K suffices (K=2) | parity has only 2 hidden states; cloning matches exactly |
| Increment | near-perfect in-dist; OOD-length controlled by carry-chain length | near-perfect both | very few hidden states (4) |
| Reverse | strong in-dist; OOD-length stresses prefix memory | strong; CHMM probably wins by a margin | many same-emission different-state cases — CHMM's strength |
| Dyck-1 | strong on shallow depth; OOD depth degrades for both | comparable but CHMM may be slightly worse on deep counters | counter dynamics — neither has true counter state |

Headline question: *Does CHMM consistently match-or-beat GDC on
algorithmic TM-trace tasks, as it did on the binary adder?* Or is
the binary-adder result a special case?
