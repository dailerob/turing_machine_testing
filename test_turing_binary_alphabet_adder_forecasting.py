"""
Test GDC forecasting on the Binary-Alphabet Adder execution traces.

Mirrors test_turing_adder_forecasting.py but uses the {'0','1','_'} adder
defined in binary_alphabet_adder.py.

Two tests are run using a single fully-trained model (no sweep):
- Test 1: Full 5-column GDC, forecasting with conditional on read (column 1)
- Test 2: Reduced 3-column GDC (read, write, direction only), conditional on read

Variables are kept at module level for inspection in a variable explorer.
All progress is flushed immediately so it can be monitored live.
"""

import sys
import numpy as np
from binary_alphabet_adder import simulate_random_binary_alphabet_adders
from generative_dense_chain import GenerativeDenseChain


def log(msg=""):
    """Print and immediately flush so live monitoring works."""
    print(msg, flush=True)


def evaluate_gdc_forecasting(gdc, test_tapes, test_halted, symbol_encoding):
    """Full 5-column GDC eval; conditional on read value (column 1). 1 step ahead."""
    correct_per_position = np.zeros(5)
    total_per_position = np.zeros(5)

    for tape_idx, (tape, halted) in enumerate(zip(test_tapes, test_halted)):
        if len(tape) < 2:
            continue
        log(f"  [Test 1] evaluating test tape {tape_idx + 1}/{len(test_tapes)} "
            f"(length={len(tape)})...")

        _, state_history = gdc.forward_pass(tape, return_history=True)

        for t in range(len(tape) - 1):
            if tape[t, 0] == -1 or tape[t + 1, 0] == -1:
                continue

            state_dist = state_history[t]
            forecast_dist = gdc.forecast(state_dist, n_steps=1)

            actual_next = tape[t + 1]
            conditional = np.array([np.nan, actual_next[1], np.nan, np.nan, np.nan])
            prediction = gdc.greedy_sample(forecast_dist, conditional=conditional)

            for pos in range(5):
                if not np.isnan(prediction[pos]):
                    total_per_position[pos] += 1
                    if prediction[pos] == actual_next[pos]:
                        correct_per_position[pos] += 1

    accuracy_per_position = np.zeros(5)
    for pos in range(5):
        if total_per_position[pos] > 0:
            accuracy_per_position[pos] = correct_per_position[pos] / total_per_position[pos]
    return accuracy_per_position, total_per_position


def evaluate_gdc_forecasting_reduced(gdc, test_tapes, test_halted):
    """Reduced 3-column GDC eval (read, write, direction); conditional on read."""
    correct_per_position = np.zeros(3)
    total_per_position = np.zeros(3)

    for tape_idx, (tape, halted) in enumerate(zip(test_tapes, test_halted)):
        if len(tape) < 2:
            continue
        log(f"  [Test 2] evaluating test tape {tape_idx + 1}/{len(test_tapes)} "
            f"(length={len(tape)})...")

        tape_reduced = tape[:, 1:4]
        _, state_history = gdc.forward_pass(tape_reduced, return_history=True)

        for t in range(len(tape) - 1):
            if tape[t, 0] == -1 or tape[t + 1, 0] == -1:
                continue

            state_dist = state_history[t]
            forecast_dist = gdc.forecast(state_dist, n_steps=1)

            actual_next = tape_reduced[t + 1]
            conditional = np.array([actual_next[0], np.nan, np.nan])
            prediction = gdc.greedy_sample(forecast_dist, conditional=conditional)

            for pos in range(3):
                if not np.isnan(prediction[pos]):
                    total_per_position[pos] += 1
                    if prediction[pos] == actual_next[pos]:
                        correct_per_position[pos] += 1

    accuracy_per_position = np.zeros(3)
    for pos in range(3):
        if total_per_position[pos] > 0:
            accuracy_per_position[pos] = correct_per_position[pos] / total_per_position[pos]
    return accuracy_per_position, total_per_position


def analyze_write_errors_reduced(gdc, test_tapes, test_halted, state_encoding, symbol_encoding):
    """Write-error breakdown for the reduced 3-column model."""
    from collections import defaultdict

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

    for tape_idx, (tape, halted) in enumerate(zip(test_tapes, test_halted)):
        if len(tape) < 2:
            continue

        tape_reduced = tape[:, 1:4]
        _, state_history = gdc.forward_pass(tape_reduced, return_history=True)

        for t in range(len(tape) - 1):
            if tape[t, 0] == -1 or tape[t + 1, 0] == -1:
                continue

            state_dist = state_history[t]
            forecast_dist = gdc.forecast(state_dist, n_steps=1)

            actual_next = tape_reduced[t + 1]
            conditional = np.array([actual_next[0], np.nan, np.nan])
            prediction = gdc.greedy_sample(forecast_dist, conditional=conditional)

            tm_state_encoded = int(tape[t + 1, 0])
            tm_state = state_decoding.get(tm_state_encoded, f"State_{tm_state_encoded}")

            read_encoded = int(actual_next[0])
            read_symbol = symbol_decoding.get(read_encoded, f"Sym_{read_encoded}")

            if not np.isnan(prediction[1]):
                total_by_state[tm_state] += 1
                total_by_read_symbol[read_symbol] += 1
                total_by_state_and_read[(tm_state, read_symbol)] += 1

            if not np.isnan(prediction[1]) and prediction[1] != actual_next[1]:
                predicted_write = int(prediction[1])
                actual_write = int(actual_next[1])
                predicted_symbol = symbol_decoding.get(predicted_write, f"Sym_{predicted_write}")
                actual_symbol = symbol_decoding.get(actual_write, f"Sym_{actual_write}")

                errors_by_state[tm_state] += 1
                confusion_matrix[(predicted_symbol, actual_symbol)] += 1
                errors_by_read_symbol[read_symbol] += 1
                errors_by_state_and_read[(tm_state, read_symbol)] += 1

                error_records.append({
                    'tape_idx': tape_idx,
                    'step': t + 1,
                    'tm_state': tm_state,
                    'read_symbol': read_symbol,
                    'predicted_write': predicted_symbol,
                    'actual_write': actual_symbol,
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


def print_error_analysis(analysis):
    log("\n" + "=" * 70)
    log("WRITE ERROR ANALYSIS FOR TEST 2 (REDUCED 3-COLUMN GDC)")
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
        f"({100*total_errors/max(total_predictions, 1):.2f}% error rate)")

    log("\n--- ERRORS BY TURING MACHINE STATE ---")
    log(f"{'State':<15} {'Errors':<10} {'Total':<10} {'Error Rate':<12}")
    log("-" * 47)
    for state in sorted(errors_by_state.keys(),
                        key=lambda x: errors_by_state.get(x, 0), reverse=True):
        errors = errors_by_state.get(state, 0)
        total = total_by_state.get(state, 0)
        rate = 100 * errors / total if total > 0 else 0
        log(f"{state:<15} {errors:<10} {total:<10} {rate:.2f}%")

    zero_error_states = [s for s in total_by_state.keys() if s not in errors_by_state]
    if zero_error_states:
        log("\nStates with ZERO errors:")
        for state in sorted(zero_error_states):
            log(f"  {state}: 0 / {total_by_state[state]}")

    log("\n--- ERRORS BY READ SYMBOL ---")
    log(f"{'Read Symbol':<15} {'Errors':<10} {'Total':<10} {'Error Rate':<12}")
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
    sorted_combos = sorted(errors_by_state_and_read.keys(),
                           key=lambda x: errors_by_state_and_read[x], reverse=True)
    for state, read in sorted_combos[:15]:
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


def run_tests(train_tapes, train_halted, test_tapes, test_halted,
              state_encoding, symbol_encoding):

    position_names_full = ['current_state', 'read', 'write', 'direction', 'next_state']
    position_names_reduced = ['read', 'write', 'direction']

    # -------------------------------------------------------------------------
    # TEST 1: Full 5-column GDC — single model trained on all training tapes
    # -------------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("TEST 1: FULL 5-COLUMN GDC (conditional on read)")
    log("=" * 70)

    log(f"\nBuilding GDC from all {len(train_tapes)} training tapes...")
    gdc_full = GenerativeDenseChain(
        train_tapes,
        alpha=0.95, theta=0.005, gamma=0.000,
        transition_type='self_loop_two_step',
        initial_dist='sequence_starts',
    )
    log(f"GDC built: {gdc_full.n_states} hidden states")

    log("\nEvaluating on test tapes...")
    accuracy_full, total_full = evaluate_gdc_forecasting(
        gdc_full, test_tapes, test_halted, symbol_encoding
    )

    log("\nTest 1 results:")
    for pos, name in enumerate(position_names_full):
        log(f"  {name}: {accuracy_full[pos]:.3f} ({int(total_full[pos])} predictions)")
    log(f"  Overall (mean): {accuracy_full.mean():.3f}")

    # -------------------------------------------------------------------------
    # TEST 2: Reduced 3-column GDC — single model trained on all training tapes
    # -------------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("TEST 2: REDUCED 3-COLUMN GDC (read, write, direction only)")
    log("=" * 70)

    training_reduced = [tape[:, 1:4] for tape in train_tapes]
    log(f"\nBuilding reduced GDC from all {len(training_reduced)} training tapes...")
    gdc_reduced = GenerativeDenseChain(
        training_reduced,
        alpha=0.99, theta=0.005, gamma=0.000,
        transition_type='self_loop_two_step',
        initial_dist='sequence_starts',
    )
    log(f"GDC built: {gdc_reduced.n_states} hidden states")

    log("\nEvaluating on test tapes...")
    accuracy_reduced, total_reduced = evaluate_gdc_forecasting_reduced(
        gdc_reduced, test_tapes, test_halted
    )

    log("\nTest 2 results:")
    for pos, name in enumerate(position_names_reduced):
        log(f"  {name}: {accuracy_reduced[pos]:.3f} ({int(total_reduced[pos])} predictions)")
    log(f"  Overall (mean): {accuracy_reduced.mean():.3f}")

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)

    log("\n--- TEST 1: Full 5-column GDC ---")
    log(f"  n_train: {len(train_tapes)}, n_states: {gdc_full.n_states}")
    for pos, name in enumerate(position_names_full):
        log(f"  {name}: {accuracy_full[pos]:.3f}")
    log(f"  Overall (mean): {accuracy_full.mean():.3f}")

    log("\n--- TEST 2: Reduced 3-column GDC ---")
    log(f"  n_train: {len(train_tapes)}, n_states: {gdc_reduced.n_states}")
    for pos, name in enumerate(position_names_reduced):
        log(f"  {name}: {accuracy_reduced[pos]:.3f}")
    log(f"  Overall (mean): {accuracy_reduced.mean():.3f}")

    log("\nRunning write-error analysis on Test 2 model...")
    error_analysis = analyze_write_errors_reduced(
        gdc_reduced, test_tapes, test_halted,
        state_encoding, symbol_encoding,
    )
    print_error_analysis(error_analysis)

    return accuracy_full, accuracy_reduced, error_analysis


# =============================================================================
# Configuration
# =============================================================================
N_TRAIN = 400
N_TEST = 10
NUM_RANGE_TRAIN = (0, 32)
NUM_RANGE_TEST = (0, 1000)
MAX_STEPS = 200_000
TRAIN_SEED = 42
TEST_SEED = 123

# =============================================================================
# Generate training and test sets
# =============================================================================
log("=" * 70)
log("TEST GDC FORECASTING ON BINARY-ALPHABET ADDER EXECUTION TRACES")
log("=" * 70)

log(f"\n[1/4] Generating {N_TRAIN} training tapes "
    f"(num_range={NUM_RANGE_TRAIN}, seed={TRAIN_SEED})...")
train_results = simulate_random_binary_alphabet_adders(
    n_runs=N_TRAIN, num_range=NUM_RANGE_TRAIN,
    max_steps=MAX_STEPS, seed=TRAIN_SEED,
)
train_tapes   = train_results['runs']
train_halted  = train_results['halted_flags']
train_inputs  = train_results['inputs']
train_correct = train_results['correct']
state_encoding  = train_results['state_encoding']
symbol_encoding = train_results['symbol_encoding']
log(f"  Done. Halted: {sum(train_halted)}/{N_TRAIN}, Correct: {sum(train_correct)}/{N_TRAIN}")

log(f"\n[2/4] Generating {N_TEST} test tapes "
    f"(num_range={NUM_RANGE_TEST}, seed={TEST_SEED})...")
test_results = simulate_random_binary_alphabet_adders(
    n_runs=N_TEST, num_range=NUM_RANGE_TEST,
    max_steps=MAX_STEPS, seed=TEST_SEED,
)
test_tapes   = test_results['runs']
test_halted  = test_results['halted_flags']
test_inputs  = test_results['inputs']
test_correct = test_results['correct']
log(f"  Done. Halted: {sum(test_halted)}/{N_TEST}, Correct: {sum(test_correct)}/{N_TEST}")

train_steps = [t.shape[0] for t in train_tapes]
test_steps  = [t.shape[0] for t in test_tapes]
log(f"\nState encoding  ({len(state_encoding)} states):  {state_encoding}")
log(f"Symbol encoding ({len(symbol_encoding)} symbols): {symbol_encoding}")
log(f"Train tape lengths: min={min(train_steps)}, max={max(train_steps)}, "
    f"mean={np.mean(train_steps):.1f}")
log(f"Test  tape lengths: min={min(test_steps)},  max={max(test_steps)},  "
    f"mean={np.mean(test_steps):.1f}")

accuracy_full = accuracy_reduced = error_analysis = None

if __name__ == "__main__":
    log("\n[3/4] Running Test 1 + Test 2...")
    accuracy_full, accuracy_reduced, error_analysis = run_tests(
        train_tapes, train_halted, test_tapes, test_halted,
        state_encoding, symbol_encoding,
    )
    log("\n[4/4] Done.")
