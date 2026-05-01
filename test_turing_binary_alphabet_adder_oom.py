"""
Test Spectral OOM forecasting on the Binary-Alphabet Adder execution traces.

Mirrors test_turing_binary_alphabet_adder_forecasting.py (the GDC version) but
replaces GenerativeDenseChain with SpectralOOM (Hankel-matrix spectral WFA).

Each tape row [state, read, write, direction, next_state] is treated as a
single discrete token (its tuple). The OOM learns operators over the alphabet
of observed tuples. Forecasting "conditional on read" is implemented by
restricting the predicted-next-token argmax to tokens whose read-column
matches the ground-truth next read.

Two tests:
    Test 1 — full 5-column tokens, conditional on read
    Test 2 — reduced 3-column tokens (read, write, direction), conditional on read

Config matches the GDC test exactly:
    N_TRAIN=400, NUM_RANGE_TRAIN=(0,32), N_TEST=10, NUM_RANGE_TEST=(0,1000),
    seeds 42/123.
"""

import sys
import numpy as np
from collections import defaultdict
from binary_alphabet_adder import simulate_random_binary_alphabet_adders
from spectral_oom import SpectralOOM


def log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------
def tape_to_tokens(tape, cols):
    """Convert a (T, k) numpy tape into a list of tuples.

    Halt rows (first col == -1) are dropped; the halted status is captured
    separately if needed.
    """
    toks = []
    for row in tape:
        if row[0] == -1:
            # Halt marker row — skip; terminal state is implicit.
            continue
        toks.append(tuple(int(x) for x in row[list(cols)]))
    return toks


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_oom(oom, test_tapes, cols, read_col_in_tuple, test_label):
    """Evaluate OOM forecasting, conditional on observed read value.

    cols: which columns of the tape row define the token.
    read_col_in_tuple: index within the tuple of the read symbol used for
        conditioning.
    Returns (accuracy_per_position, total_per_position).
    """
    k = len(cols)
    correct_per_position = np.zeros(k)
    total_per_position = np.zeros(k)

    # Pre-index tokens by their read value for fast conditional argmax.
    tokens = oom.alphabet  # list of tuples
    read_vals = np.array([tok[read_col_in_tuple] for tok in tokens])
    token_arr = np.array(tokens, dtype=np.int64)  # (nA, k)

    unknown_skipped = 0
    for tape_idx, tape in enumerate(test_tapes):
        toks = tape_to_tokens(tape, cols)
        if len(toks) < 2:
            continue
        log(f"  [{test_label}] evaluating tape {tape_idx + 1}/{len(test_tapes)} "
            f"(tokens={len(toks)})...")

        # Any unknown tokens in this test tape get skipped (state unchanged)
        # by forward_pass; we'll still evaluate on transitions where both
        # ends are known.
        _, hist = oom.forward_pass(toks, return_history=True)

        for t in range(len(toks) - 1):
            state = hist[t]
            actual_next = toks[t + 1]
            actual_read = actual_next[read_col_in_tuple]

            scores = oom.predict_next_scores(state)  # (nA,)
            # Mask to tokens with matching read
            mask = (read_vals == actual_read)
            if not mask.any():
                unknown_skipped += 1
                continue
            masked_scores = np.where(mask, scores, -np.inf)
            best = int(np.argmax(masked_scores))
            predicted = token_arr[best]

            for p in range(k):
                total_per_position[p] += 1
                if predicted[p] == actual_next[p]:
                    correct_per_position[p] += 1

    acc = np.zeros(k)
    for p in range(k):
        if total_per_position[p] > 0:
            acc[p] = correct_per_position[p] / total_per_position[p]
    if unknown_skipped:
        log(f"  [{test_label}] skipped {unknown_skipped} steps with no matching-read token")
    return acc, total_per_position


def analyze_write_errors_oom(oom, test_tapes, cols, read_col_in_tuple,
                              write_col_in_tuple, state_col_in_tuple,
                              state_encoding, symbol_encoding):
    """Write-error breakdown for the reduced 3-column OOM.

    state_col_in_tuple: set to None if the token tuple doesn't include TM
    state directly — we then read TM state from column 0 of the raw tape row.
    """
    state_decoding = {v: k for k, v in state_encoding.items()}
    symbol_decoding = {v: k for k, v in symbol_encoding.items()}

    errors_by_state = defaultdict(int)
    total_by_state = defaultdict(int)
    confusion_matrix = defaultdict(int)
    errors_by_read_symbol = defaultdict(int)
    total_by_read_symbol = defaultdict(int)
    errors_by_state_and_read = defaultdict(int)
    total_by_state_and_read = defaultdict(int)
    error_records = []

    tokens = oom.alphabet
    read_vals = np.array([tok[read_col_in_tuple] for tok in tokens])
    token_arr = np.array(tokens, dtype=np.int64)

    for tape_idx, tape in enumerate(test_tapes):
        toks = tape_to_tokens(tape, cols)
        if len(toks) < 2:
            continue

        _, hist = oom.forward_pass(toks, return_history=True)

        # Map tape-row index (including halt) to token-index (halt skipped).
        # Build mapping from non-halt tape rows to token positions.
        nonhalt_rows = [i for i, row in enumerate(tape) if row[0] != -1]

        for t in range(len(toks) - 1):
            state = hist[t]
            actual_next = toks[t + 1]
            actual_read = actual_next[read_col_in_tuple]

            scores = oom.predict_next_scores(state)
            mask = (read_vals == actual_read)
            if not mask.any():
                continue
            masked_scores = np.where(mask, scores, -np.inf)
            best = int(np.argmax(masked_scores))
            predicted = token_arr[best]

            # Pull TM state from raw tape (column 0) for the predicted step
            raw_row_idx = nonhalt_rows[t + 1]
            tm_state_encoded = int(tape[raw_row_idx, 0])
            tm_state = state_decoding.get(tm_state_encoded, f"State_{tm_state_encoded}")

            read_symbol = symbol_decoding.get(int(actual_read), f"Sym_{actual_read}")

            total_by_state[tm_state] += 1
            total_by_read_symbol[read_symbol] += 1
            total_by_state_and_read[(tm_state, read_symbol)] += 1

            if predicted[write_col_in_tuple] != actual_next[write_col_in_tuple]:
                pw = int(predicted[write_col_in_tuple])
                aw = int(actual_next[write_col_in_tuple])
                pred_sym = symbol_decoding.get(pw, f"Sym_{pw}")
                actual_sym = symbol_decoding.get(aw, f"Sym_{aw}")

                errors_by_state[tm_state] += 1
                confusion_matrix[(pred_sym, actual_sym)] += 1
                errors_by_read_symbol[read_symbol] += 1
                errors_by_state_and_read[(tm_state, read_symbol)] += 1
                error_records.append({
                    'tape_idx': tape_idx,
                    'step': t + 1,
                    'tm_state': tm_state,
                    'read_symbol': read_symbol,
                    'predicted_write': pred_sym,
                    'actual_write': actual_sym,
                })

    return {
        'errors_by_state': dict(errors_by_state),
        'total_by_state': dict(total_by_state),
        'confusion_matrix': dict(confusion_matrix),
        'errors_by_read_symbol': dict(errors_by_read_symbol),
        'total_by_read_symbol': dict(total_by_read_symbol),
        'errors_by_state_and_read': dict(errors_by_state_and_read),
        'total_by_state_and_read': dict(total_by_state_and_read),
        'error_records': error_records,
        'state_decoding': state_decoding,
        'symbol_decoding': symbol_decoding,
    }


def print_error_analysis(analysis, header):
    log("\n" + "=" * 70)
    log(header)
    log("=" * 70)

    errors_by_state = analysis['errors_by_state']
    total_by_state = analysis['total_by_state']
    confusion_matrix = analysis['confusion_matrix']
    errors_by_read_symbol = analysis['errors_by_read_symbol']
    total_by_read_symbol = analysis['total_by_read_symbol']
    errors_by_state_and_read = analysis['errors_by_state_and_read']
    total_by_state_and_read = analysis['total_by_state_and_read']
    error_records = analysis['error_records']

    total_errors = sum(errors_by_state.values())
    total_predictions = sum(total_by_state.values())
    log(f"\nTotal write errors: {total_errors} / {total_predictions} "
        f"({100*total_errors/max(total_predictions,1):.2f}% error rate)")

    log("\n--- ERRORS BY TM STATE ---")
    log(f"{'State':<15} {'Errors':<10} {'Total':<10} {'Error Rate':<12}")
    log("-" * 47)
    for state in sorted(total_by_state.keys(),
                        key=lambda x: errors_by_state.get(x, 0), reverse=True):
        errors = errors_by_state.get(state, 0)
        total = total_by_state.get(state, 0)
        rate = 100 * errors / total if total > 0 else 0
        log(f"{state:<15} {errors:<10} {total:<10} {rate:.2f}%")

    log("\n--- ERRORS BY READ SYMBOL ---")
    log(f"{'Read':<15} {'Errors':<10} {'Total':<10} {'Error Rate':<12}")
    log("-" * 47)
    for symbol in sorted(total_by_read_symbol.keys(), key=str):
        errors = errors_by_read_symbol.get(symbol, 0)
        total = total_by_read_symbol.get(symbol, 0)
        rate = 100 * errors / total if total > 0 else 0
        log(f"{repr(symbol):<15} {errors:<10} {total:<10} {rate:.2f}%")

    log("\n--- SYMBOL CONFUSION MATRIX (Predicted vs Actual) ---")
    log(f"{'Predicted -> Actual':<25} {'Count':<10} {'% of Errors':<12}")
    log("-" * 47)
    for (pred, actual) in sorted(confusion_matrix.keys(),
                                  key=lambda x: confusion_matrix[x], reverse=True):
        count = confusion_matrix[(pred, actual)]
        pct = 100 * count / total_errors if total_errors > 0 else 0
        log(f"{repr(pred)} -> {repr(actual):<15} {count:<10} {pct:.1f}%")

    log("\n--- TOP ERROR COMBINATIONS (State + Read Symbol) ---")
    log(f"{'State':<15} {'Read':<10} {'Errors':<10} {'Total':<10} {'Error Rate':<12}")
    log("-" * 57)
    for (state, read) in sorted(errors_by_state_and_read.keys(),
                                 key=lambda x: errors_by_state_and_read[x], reverse=True)[:15]:
        errors = errors_by_state_and_read[(state, read)]
        total = total_by_state_and_read[(state, read)]
        rate = 100 * errors / total if total > 0 else 0
        log(f"{state:<15} {repr(read):<10} {errors:<10} {total:<10} {rate:.2f}%")

    if error_records:
        log("\n--- SAMPLE ERROR RECORDS (first 20) ---")
        log(f"{'Tape':<6} {'Step':<8} {'TM State':<12} {'Read':<8} {'Pred':<10} {'Actual':<10}")
        log("-" * 54)
        for rec in error_records[:20]:
            log(f"{rec['tape_idx']:<6} {rec['step']:<8} {rec['tm_state']:<12} "
                f"{repr(rec['read_symbol']):<8} {repr(rec['predicted_write']):<10} "
                f"{repr(rec['actual_write']):<10}")


def per_addition_accuracy(oom, test_tapes, test_inputs, cols,
                           read_col_in_tuple, write_col_in_tuple):
    """Count how many test additions produce zero write errors end-to-end."""
    tokens = oom.alphabet
    read_vals = np.array([tok[read_col_in_tuple] for tok in tokens])
    token_arr = np.array(tokens, dtype=np.int64)

    log("\n--- PER-ADDITION WRITE-ERROR ACCURACY ---")
    log(f"{'Tape':<6} {'A+B':<16} {'Steps':<8} {'Errors':<8} {'Err%':<8} {'Perfect?':<10}")
    log("-" * 56)
    perfect = 0
    total_steps = 0
    total_errs = 0
    for tape_idx, tape in enumerate(test_tapes):
        toks = tape_to_tokens(tape, cols)
        if len(toks) < 2:
            continue
        _, hist = oom.forward_pass(toks, return_history=True)
        errs = 0
        steps = 0
        for t in range(len(toks) - 1):
            state = hist[t]
            actual_next = toks[t + 1]
            actual_read = actual_next[read_col_in_tuple]
            scores = oom.predict_next_scores(state)
            mask = (read_vals == actual_read)
            if not mask.any():
                continue
            masked_scores = np.where(mask, scores, -np.inf)
            best = int(np.argmax(masked_scores))
            predicted = token_arr[best]
            steps += 1
            if predicted[write_col_in_tuple] != actual_next[write_col_in_tuple]:
                errs += 1
        a, b = test_inputs[tape_idx]
        flag = "YES" if errs == 0 else "NO"
        if errs == 0:
            perfect += 1
        total_steps += steps
        total_errs += errs
        log(f"{tape_idx:<6} {f'{a}+{b}={a+b}':<16} {steps:<8} {errs:<8} "
            f"{100*errs/max(steps,1):<8.2f} {flag:<10}")
    log("-" * 56)
    log(f"TOTAL                 {total_steps:<8} {total_errs:<8} "
        f"{100*total_errs/max(total_steps,1):.2f}%")
    log(f"\nPerfect additions (zero write errors): {perfect}/{len(test_tapes)}")
    return perfect


# =============================================================================
# Configuration — matched to GDC test for apples-to-apples comparison
# =============================================================================
N_TRAIN = 400
N_TEST = 10
NUM_RANGE_TRAIN = (0, 32)
NUM_RANGE_TEST = (0, 1000)
MAX_STEPS = 200_000
TRAIN_SEED = 42
TEST_SEED = 123

# OOM hyperparameters
MAX_BASIS_LENGTH = 5   # substrings up to length 5 in prefix/suffix basis
RANK = None            # auto-pick from SVD threshold
SV_REL_THRESHOLD = 1e-8

# =============================================================================
# Data generation
# =============================================================================
log("=" * 70)
log("TEST SPECTRAL OOM FORECASTING ON BINARY-ALPHABET ADDER TRACES")
log("=" * 70)

log(f"\n[1/4] Generating {N_TRAIN} training tapes "
    f"(num_range={NUM_RANGE_TRAIN}, seed={TRAIN_SEED})...")
train_results = simulate_random_binary_alphabet_adders(
    n_runs=N_TRAIN, num_range=NUM_RANGE_TRAIN,
    max_steps=MAX_STEPS, seed=TRAIN_SEED,
)
train_tapes     = train_results['runs']
train_halted    = train_results['halted_flags']
train_inputs    = train_results['inputs']
train_correct   = train_results['correct']
state_encoding  = train_results['state_encoding']
symbol_encoding = train_results['symbol_encoding']
log(f"  Done. Halted: {sum(train_halted)}/{N_TRAIN}, "
    f"Correct: {sum(train_correct)}/{N_TRAIN}")

log(f"\n[2/4] Generating {N_TEST} test tapes "
    f"(num_range={NUM_RANGE_TEST}, seed={TEST_SEED})...")
test_results = simulate_random_binary_alphabet_adders(
    n_runs=N_TEST, num_range=NUM_RANGE_TEST,
    max_steps=MAX_STEPS, seed=TEST_SEED,
)
test_tapes   = test_results['runs']
test_halted  = test_results['halted_flags']
test_inputs  = test_results['inputs']
log(f"  Done. Halted: {sum(test_halted)}/{N_TEST}")

train_steps = [t.shape[0] for t in train_tapes]
test_steps  = [t.shape[0] for t in test_tapes]
log(f"\nState encoding  ({len(state_encoding)} states):  {state_encoding}")
log(f"Symbol encoding ({len(symbol_encoding)} symbols): {symbol_encoding}")
log(f"Train tape lengths: min={min(train_steps)}, max={max(train_steps)}, "
    f"mean={np.mean(train_steps):.1f}")
log(f"Test  tape lengths: min={min(test_steps)},  max={max(test_steps)},  "
    f"mean={np.mean(test_steps):.1f}")

# =============================================================================
# TEST 1 — full 5-column tokens
# =============================================================================
log("\n" + "=" * 70)
log("TEST 1: FULL 5-COLUMN SPECTRAL OOM (conditional on read)")
log("=" * 70)

COLS_FULL = (0, 1, 2, 3, 4)
train_toks_full = [tape_to_tokens(t, COLS_FULL) for t in train_tapes]
n_full_toks = sum(len(x) for x in train_toks_full)
uniq_full = set()
for s in train_toks_full:
    uniq_full.update(s)
log(f"\nTraining token stream: {n_full_toks} tokens, "
    f"{len(uniq_full)} unique 5-tuples")

log(f"\n[Test 1] Fitting SpectralOOM (L={MAX_BASIS_LENGTH}, rank=auto)...")
oom_full = SpectralOOM(
    max_basis_length=MAX_BASIS_LENGTH, rank=RANK,
    sv_rel_threshold=SV_REL_THRESHOLD, renormalize=True, verbose=True,
)
oom_full.fit(train_toks_full)

log("\n[Test 1] Evaluating on test tapes...")
acc_full, tot_full = evaluate_oom(
    oom_full, test_tapes, COLS_FULL,
    read_col_in_tuple=1, test_label="Test 1",
)
pos_names_full = ['current_state', 'read', 'write', 'direction', 'next_state']
log("\n[Test 1] Per-position accuracy:")
for p, name in enumerate(pos_names_full):
    log(f"  {name}: {acc_full[p]:.4f} ({int(tot_full[p])} preds)")
log(f"  Overall mean: {acc_full.mean():.4f}")

# =============================================================================
# TEST 2 — reduced 3-column tokens (read, write, direction)
# =============================================================================
log("\n" + "=" * 70)
log("TEST 2: REDUCED 3-COLUMN SPECTRAL OOM (read/write/dir, conditional on read)")
log("=" * 70)

COLS_REDUCED = (1, 2, 3)
train_toks_red = [tape_to_tokens(t, COLS_REDUCED) for t in train_tapes]
n_red_toks = sum(len(x) for x in train_toks_red)
uniq_red = set()
for s in train_toks_red:
    uniq_red.update(s)
log(f"\nTraining token stream: {n_red_toks} tokens, "
    f"{len(uniq_red)} unique 3-tuples")

log(f"\n[Test 2] Fitting SpectralOOM (L={MAX_BASIS_LENGTH}, rank=auto)...")
oom_red = SpectralOOM(
    max_basis_length=MAX_BASIS_LENGTH, rank=RANK,
    sv_rel_threshold=SV_REL_THRESHOLD, renormalize=True, verbose=True,
)
oom_red.fit(train_toks_red)

log("\n[Test 2] Evaluating on test tapes...")
acc_red, tot_red = evaluate_oom(
    oom_red, test_tapes, COLS_REDUCED,
    read_col_in_tuple=0, test_label="Test 2",
)
pos_names_red = ['read', 'write', 'direction']
log("\n[Test 2] Per-position accuracy:")
for p, name in enumerate(pos_names_red):
    log(f"  {name}: {acc_red[p]:.4f} ({int(tot_red[p])} preds)")
log(f"  Overall mean: {acc_red.mean():.4f}")

# =============================================================================
# Error analysis & per-addition report (on reduced model — mirrors GDC)
# =============================================================================
log("\n[Analysis] Write-error breakdown on reduced OOM...")
analysis = analyze_write_errors_oom(
    oom_red, test_tapes, COLS_REDUCED,
    read_col_in_tuple=0, write_col_in_tuple=1, state_col_in_tuple=None,
    state_encoding=state_encoding, symbol_encoding=symbol_encoding,
)
print_error_analysis(analysis, "WRITE ERROR ANALYSIS — REDUCED 3-COLUMN OOM")

log("\n[Analysis] Per-addition accuracy on reduced OOM...")
perfect_red = per_addition_accuracy(
    oom_red, test_tapes, test_inputs, COLS_REDUCED,
    read_col_in_tuple=0, write_col_in_tuple=1,
)

# =============================================================================
# Summary
# =============================================================================
log("\n" + "=" * 70)
log("FINAL SUMMARY - SPECTRAL OOM vs (GDC reference below)")
log("=" * 70)
log(f"\nConfig: L={MAX_BASIS_LENGTH}, rank_used(full)={oom_full._rank_used}, "
    f"rank_used(reduced)={oom_red._rank_used}")
log(f"\nTest 1 (full 5-col):")
for p, name in enumerate(pos_names_full):
    log(f"  {name}: {acc_full[p]:.4f}")
log(f"  mean: {acc_full.mean():.4f}")
log(f"\nTest 2 (reduced 3-col):")
for p, name in enumerate(pos_names_red):
    log(f"  {name}: {acc_red[p]:.4f}")
log(f"  mean: {acc_red.mean():.4f}")
log(f"\nPer-addition perfect count (reduced): {perfect_red}/{len(test_tapes)}")
log("\nGDC reference (same data): Test2 mean~0.999, perfect=1/10, 94/72217 errors.")
log("\nDone.")
