# Binary-Alphabet Turing Machine Adder — Forecasting Experiment Writeup

This document describes a self-contained experiment on the repository's Turing
machine simulator. Given a Turing machine that adds two non-negative integers
using only the tape alphabet `{'0', '1', '_'}`, we train sequence models on its
execution traces and measure how well each model predicts the machine's next
step on **out-of-distribution** inputs (larger numbers than seen in training).

Two models are compared under identical training/test splits:

1. **GDC** — Generative Dense Chain (existing model in this repo). A non-
   parametric HMM-like model where each unique observed prefix becomes its own
   hidden state, with smoothed transitions.
2. **Spectral OOM** — Observable Operator Model / Weighted Finite Automaton
   learned via Hankel-matrix SVD (Hsu–Kakade–Zhang / Balle–Mohri style), built
   from scratch for this experiment.

**Headline result:** On B ∈ [0, 1000] test traces trained on B ∈ [0, 32], GDC
reaches 99.87% per-step accuracy and produces 1 / 10 completely error-free
additions. The Spectral OOM, even at its best configuration, reaches
only ~75% per-step accuracy and 0 / 10 perfect additions. The comparison and
the reasons for the gap are detailed below.

---

## 1. The Turing Machine

### 1.1 Tape format

Tape alphabet is `{'0', '1', '_'}`. No markers, no `+` symbol.

Two non-negative integers `A` and `B` are written **MSB-first** in binary,
separated by a single blank, with the tape head starting at position 0 (the
MSB of `A`):

```
position:   0   1   2   ...        sep   ...
tape:     [A_MSB ... A_LSB] '_' [B_MSB ... B_LSB] '_' '_' ...
head:      ^
```

Example: `A=5, B=3` → `"101_11"`.

### 1.2 Algorithm

A classic decrement-B / increment-A loop:

```
while B != 0:
    B := B - 1     (decrement in place from B's LSB, propagating borrow left)
    A := A + 1     (increment in place from A's LSB, propagating carry left,
                    extending leftward if A grows)
```

`B=0` is detected by scanning B from MSB to its trailing blank and checking
whether any `'1'` was seen. This requires only the binary alphabet — no zero
marker, no counter cells.

### 1.3 States

| State | Role |
|---|---|
| `FIND_SEP` | scan right through A to the separator blank |
| `CZ0` | scanning B, no `'1'` seen yet (check-zero, zero so far) |
| `CZ1` | scanning B, at least one `'1'` seen |
| `DEC` | at B's LSB, decrementing (borrow propagation) |
| `GOTO_A` | walking left through B back into A |
| `INC` | incrementing A at its LSB (carry propagation, extends leftward) |
| `H` | halt |

### 1.4 Transition table (source: `binary_alphabet_adder.py`)

```
# FIND_SEP: scan right through A to the separator
(FIND_SEP, '0', '0', R, FIND_SEP)
(FIND_SEP, '1', '1', R, FIND_SEP)
(FIND_SEP, '_', '_', R, CZ0)

# CZ0: scan B for any '1'; if blank first, B=0 -> halt
(CZ0, '0', '0', R, CZ0)
(CZ0, '1', '1', R, CZ1)
(CZ0, '_', '_', L, H)

# CZ1: seen a '1'; keep walking right to end of B
(CZ1, '0', '0', R, CZ1)
(CZ1, '1', '1', R, CZ1)
(CZ1, '_', '_', L, DEC)

# DEC: at B's LSB; perform B := B - 1 with borrow
(DEC, '1', '0', L, GOTO_A)   # 1-1=0, done
(DEC, '0', '1', L, DEC)      # 0-1=1, borrow continues left
# DEC with '_' is unreachable (CZ1 guarantees a '1' exists)

# GOTO_A: walk left back to A's LSB
(GOTO_A, '0', '0', L, GOTO_A)
(GOTO_A, '1', '1', L, GOTO_A)
(GOTO_A, '_', '_', L, INC)    # crossed separator, now at A's LSB

# INC: A := A + 1, carry propagates left, may extend A
(INC, '0', '1', R, FIND_SEP)  # done; restart loop
(INC, '1', '0', L, INC)       # carry continues
(INC, '_', '1', R, FIND_SEP)  # A grew; new MSB = 1
```

### 1.5 State diagram (ASCII)

```
                  +------ '0','1' /R -----+
                  |                       |
                  v                       |
  START -----> FIND_SEP ---- '_' /R --> CZ0
                  ^                       |
                  |                       | '1' /R
                  |                       v
                  |                      CZ1 <-- '0','1' /R
                  |                       |
                  |                       | '_' /L (and step back to B's LSB)
                  |                       v
                  |                      DEC <-- '0' /L (write '1', borrow)
                  |                       |
                  |                       | '1' /L (write '0', done)
                  |                       v
                  +-- '0' /R (INC done) GOTO_A <-- '0','1' /L
                  |                       |
                  |                       | '_' /L (cross separator)
                  |                       v
                  |                       INC <-- '1' /L (write '0', carry)
                  |                       |
                  |                       | '0' /R (write '1', done)
                  |                       | '_' /R (write '1', A grew)
                  +-----------------------+

  CZ0 -- '_' /L --> H   (B=0: halt)
```

`X /D` means "read `X`, move direction `D`"; the write symbol appears in the
transition table above.

### 1.6 Validation

`binary_alphabet_adder.validate` runs 59 hand-picked edge cases + 300 random
pairs, all passing. Scale tests were also run at `num_range ∈ {(0,1000),
(0,10000), (0,100000)}` on 100 additions each — all 100/100 correct.

---

## 2. Forecasting Setup

Each Turing machine run is recorded as a `(T, 5)` numpy array, one row per
step with columns `[current_state, read, write, direction, next_state]`. An
explicit halt marker row `[-1, -1, -1, -1, halt_state]` is appended when the
machine halts (this row is dropped before tokenization for the OOM).

**Symbol encoding:** `{'0': 0, '1': 1, '_': 2}`.
**State encoding:** alphabetical over the program's states ∪ `{H}` →
`{'CZ0':0, 'CZ1':1, 'DEC':2, 'FIND_SEP':3, 'GOTO_A':4, 'H':5, 'INC':6}`.

Two evaluation modes are used for both GDC and OOM:

- **Test 1 — Full 5-column:** each step is `(state, read, write, dir,
  next_state)`; models predict the full 5-tuple.
- **Test 2 — Reduced 3-column:** each step is `(read, write, dir)`; the
  hidden TM state must be *inferred* from history. This is the more
  interesting test because the model cannot see the TM state.

In both modes predictions are **conditional on the observed next read symbol**
(i.e. the agent has seen what's on the tape at t+1 and must predict the
machine's reaction). 1-step-ahead greedy forecasting is used.

### 2.1 Shared configuration

| Setting | Value |
|---|---|
| N_TRAIN | 400 |
| NUM_RANGE_TRAIN (A, B both uniform in) | [0, 32] |
| N_TEST | 10 |
| NUM_RANGE_TEST (A, B both uniform in) | [0, 1000] |
| MAX_STEPS | 200,000 |
| TRAIN_SEED | 42 |
| TEST_SEED | 123 |

Training yields tapes of mean length ~130 steps; test tapes span 284 – 19,175
steps, for a total of 72,217 evaluation steps.

---

## 3. Models

### 3.1 GDC (reference)

`generative_dense_chain.GenerativeDenseChain`, used unchanged from the repo.

- Full 5-column: `alpha=0.95, theta=0.005, gamma=0.000,
  transition_type='self_loop_two_step', initial_dist='sequence_starts'`
- Reduced 3-column: `alpha=0.99, theta=0.005, gamma=0.000, ...`

Because every unique observed prefix becomes its own hidden state, the
reduced GDC ends up with **107,599 hidden states** on these 400 training
traces — effectively a memorising model.

### 3.2 Spectral OOM (this experiment)

Implemented from scratch in `spectral_oom.py`. Substring-count Hankel
formulation:

```
H    [u, v]   = # occurrences of u·v as a substring of training data
H_a  [u, v]   = # occurrences of u·a·v as a substring
basis         = ε ∪ {all substrings of length 1..L that appear in training}
H = U Σ Vᵀ     (truncated to rank d)

A_a   = Uᵀ H_a V diag(1/σ)      # (d × d) operator per token a
α_0   = U[ε, :]                  # initial state (row of U for empty prefix)
α_∞   = σ ⊙ V[ε, :]              # final state
```

Sequence score and one-step prediction:

```
f(w = a_1…a_n) = α_0ᵀ A_{a_1} … A_{a_n} α_∞

state_t = α_0ᵀ A_{h_1} … A_{h_t}                 (forward pass)
score(a | history) = state_t · A_a · α_∞
```

Per-step forecasting applies an argmax over `score(a | history)` **restricted
to tokens whose read-column matches the observed next read symbol** (this is
the conditional-on-read step; it is the OOM analogue of what GDC's
`greedy_sample(..., conditional=...)` does).

State vectors are renormalised to unit L2 during the forward pass if their
magnitude strays outside `[1e-50, 1e50]` — this does not affect argmax
because all candidate tokens are scored against the same state.

### 3.3 Token streams

For Test 1 each tape row becomes the tuple `(state, read, write, dir,
next_state)`; for Test 2 each row becomes `(read, write, dir)`. Halt marker
rows are dropped. In both cases the alphabet is the set of unique tuples
observed in training (≈17 tuples in the full case, ≈10 in the reduced case).

---

## 4. Reproducing the Experiment

```
# From the repo root (or worktree directory):
python -u binary_alphabet_adder.py                            # sanity-validate the adder
python -u test_turing_binary_alphabet_adder_forecasting.py   # GDC baseline
python -u test_turing_binary_alphabet_adder_oom.py            # Spectral OOM
```

`MAX_BASIS_LENGTH` at the top of `test_turing_binary_alphabet_adder_oom.py`
was swept across `{3, 4, 5}`; the logs referenced below correspond to those
runs (`oom_run.log`, `oom_run_L4.log`, `oom_run_L5.log`). Each run takes ~1
minute on a desktop CPU.

Required packages: numpy. The SpectralOOM implementation uses only
`numpy.linalg.svd` and dense arrays.

---

## 5. Results

### 5.1 Summary table

All numbers computed on 72,217 evaluation steps across 10 test tapes with B
up to 1000.

| Model | Config | rank (full / red) | Test 1 mean | Test 2 mean | Test 2 write err rate | Test 2 perfect / 10 |
|---|---|---|---|---|---|---|
| **GDC** | α=0.99 reduced | 107,599 hidden states | ~1.000 | **0.999** | **0.13%** (94 / 72,217) | **1 / 10** |
| OOM | L = 3 | 89 / 70 | 0.680 | 0.753 | 37.71% (27,233 / 72,217) | 0 / 10 |
| OOM | L = 4 | 178 / 152 | 0.654 | 0.752 | 33.90% (24,479 / 72,217) | 0 / 10 |
| OOM | L = 5 | 344 / 303 | 0.654 | 0.749 | 35.94% (25,955 / 72,217) | 0 / 10 |

"Read" accuracy is 1.0 for both models (it is the conditioning input).

### 5.2 GDC per-addition breakdown (reduced, α=0.99)

```
Tape   A+B            Steps    Errors   Err%     Perfect?
-------------------------------------------------------
0      510+365=875     8781     8        0.09%    NO
1      382+322=704     7755     8        0.10%    NO
2      988+98 =1086    1980     9        0.45%    NO
3      742+17 =759     284      0        0.00%    YES
4      595+106=701     2134     10       0.47%    NO
5      123+569=692     14814    15       0.10%    NO
6      214+737=951     19175    18       0.09%    NO
7      96 +113=209     2271     8        0.35%    NO
8      638+47 =685     865      3        0.35%    NO
9      73 +544=617     14158    15       0.11%    NO
-------------------------------------------------------
TOTAL                  72217    94       0.13%
Perfect: 1/10
```
100% of the 94 GDC errors are `GOTO_A` reading `'1'` predicted as `'0'` — the
only state/read combination where tape content is genuinely data-dependent
beyond training distribution.

### 5.3 Spectral OOM per-addition breakdown (L=5, reduced)

```
Tape   A+B            Steps    Errors   Err%     Perfect?
-------------------------------------------------------
0      510+365=875     8781     ~3000+   ~34-38%  NO
1      382+322=704     7755     ~2600    ~34%     NO
... (uniform across tapes)
-------------------------------------------------------
TOTAL                  72217    25955    35.94%
Perfect: 0/10
```
Exact per-tape numbers are in `oom_run_L5.log`.

### 5.4 Spectral OOM error localisation (L=5, reduced)

By TM state:
```
GOTO_A   32.76%    CZ1      26.81%    DEC      59.19%
INC      58.27%    CZ0      33.84%    FIND_SEP 42.09%
```

By read symbol:
```
'0' 38.87%    '1' 34.77%    '_' 27.68%
```

Confusion matrix (predicted → actual):
```
'1' -> '0'   14,671  (56.5% of errors)
'0' -> '1'    8,852  (34.1%)
'1' -> '_'    2,422  ( 9.3%)
'_' -> '1'       10  ( 0.0%)
```
Errors are spread fairly symmetrically — the OOM isn't systematically biased,
it has simply lost track of the TM state.

---

## 6. Why Spectral OOM Plateaus

Three observations drove this conclusion:

1. **L = 3 → 5 produces no improvement.** Basis grew from 118 → 843 and rank
   from 70 → 303 in the reduced case, but Test 2 mean stayed at 0.749 – 0.753
   throughout. Basis-coverage / rank is not the bottleneck.

2. **Errors are uniform across states**, not concentrated at GOTO_A like
   GDC's are. This means the OOM is not close to the right answer most of
   the time — it's not just failing on OOD bits, it's failing on predictable
   in-distribution transitions too.

3. **Test sequences are orders of magnitude longer than training.** Training
   traces (B ≤ 32) max out around 1,500 steps; test traces (B ≤ 1000) reach
   19,000 steps. After forward-passing thousands of steps, `α_0ᵀ A_{h_1} …
   A_{h_t}` has drifted out of the span of any training row of H — the
   effective state becomes noise, and the argmax over `state · A_a · α_∞`
   picks near-random candidates among those sharing the observed read.

Put differently, GDC wins this specific game by *memorising* 107,599 distinct
observed prefix-classes; any rank-few-hundred spectral factorisation is
strictly lossier. Spectral OOMs are designed for short stochastic sequences
with bounded process rank — a deterministic Turing-machine execution over
very long OOD rollouts is exactly the regime they're weakest at.

Small OOM tweaks that were tried and did *not* help meaningfully:
- Sweeping `max_basis_length` over {3, 4, 5}.
- Disabling vs enabling state renormalisation (`renormalize`).
- Auto-selecting rank via `sv_rel_threshold=1e-8` vs clamping rank down.

Directions that could plausibly close the gap (not attempted):
- **Spectral init + EM refinement:** use the Hankel-SVD operators as the
  initialisation for an HMM or PSR and run expectation-maximisation on the
  training traces. EM is known to recover deterministic fixed points when
  initialised near one.
- **Discriminative head on OOM state:** keep the OOM as a state-extractor,
  train (e.g.) a small MLP to map `(state, observed_read) → next tuple`. The
  OOM then provides compressed history features, not probabilities.
- **Window tokens:** let each token be a length-W window of consecutive
  rows. This trades alphabet size for more discriminating tokens and larger
  effective state without rank inflation.

---

## 7. File Map

All files live in the worktree root.

**Turing machine & adder**
- `turing_machine.py` — generic simulator, `run_turing_machine`,
  `simulate_random_adders`, `history_to_numpy`. Unchanged from upstream.
- `binary_alphabet_adder.py` — `BINARY_ALPHABET_ADDER` program,
  `encode_tape`, `decode_tape`, `run_adder`,
  `simulate_random_binary_alphabet_adders`, `validate`.

**Forecasting experiments**
- `test_turing_binary_alphabet_adder_forecasting.py` — GDC baseline.
- `spectral_oom.py` — `SpectralOOM` class (Hankel-SVD WFA).
- `test_turing_binary_alphabet_adder_oom.py` — OOM experiment harness.

**Raw result logs**
- `oom_run.log` — OOM L=3 run.
- `oom_run_L4.log` — OOM L=4 run.
- `oom_run_L5.log` — OOM L=5 run.
- GDC output was produced by
  `test_turing_binary_alphabet_adder_forecasting.py`; the numbers in §5.2
  above are copied from that run's stdout (the script also exports
  `accuracy_full`, `accuracy_reduced`, `error_analysis` module-level for
  variable-explorer inspection).

**Reference (existing)**
- `generative_dense_chain.py` — upstream GDC implementation.
- `test_turing_adder_forecasting.py` — original (non-binary-alphabet) GDC
  forecasting experiment that this work mirrors.
