"""
Test GDC forecasting on Turing Machine Binary Adder execution traces.

This script:
1. Simulates the Binary Adder on random addition problems for training and test sets
2. Creates GDC models with increasing amounts of training data
3. Evaluates 1-step-ahead forecasting accuracy on test additions
4. Reports accuracy per position (current_state, read, write, direction, next_state)

Two tests are run:
- Test 1: Full 5-column GDC, forecasting with conditional on read (column 1)
- Test 2: Reduced 3-column GDC (read, write, direction only), forecasting with conditional on read

Variables are kept at module level for inspection in variable explorer.
"""

import numpy as np
from turing_machine import simulate_random_adders
from generative_dense_chain import GenerativeDenseChain


def evaluate_gdc_forecasting(gdc, test_tapes, test_halted, symbol_encoding):
    """
    Evaluate GDC forecasting accuracy on test tapes (full 5-column model).
    
    Uses greedy_sample with conditional on read value (column 1).
    Forecasts are 1 step ahead.
    
    Parameters
    ----------
    gdc : GenerativeDenseChain
        The trained GDC model.
    test_tapes : list of np.ndarray
        List of test tape execution traces.
    test_halted : list of bool
        Whether each test tape halted.
    symbol_encoding : dict
        Mapping from symbols to integers.
        
    Returns
    -------
    accuracy_per_position : np.ndarray
        Accuracy for each of the 5 positions.
    total_per_position : np.ndarray
        Total predictions made for each position.
    """
    # 5 positions: current_state, read, write, direction, next_state
    correct_per_position = np.zeros(5)
    total_per_position = np.zeros(5)
    
    for tape, halted in zip(test_tapes, test_halted):
        if len(tape) < 2:
            continue
        
        # Single forward pass to get state distribution at every timestep (O(T) instead of O(T²))
        _, state_history = gdc.forward_pass(tape, return_history=True)
            
        # For each step (except the last), predict the next step
        for t in range(len(tape) - 1):
            # Skip if current or next row is a halt row (contains -1 values)
            if tape[t, 0] == -1 or tape[t + 1, 0] == -1:
                continue
            
            # Use precomputed state distribution at step t
            state_dist = state_history[t]
            
            # Forecast 1 step ahead
            forecast_dist = gdc.forecast(state_dist, n_steps=1)
            
            # Greedy sample with conditional on read value (column 1)
            # The conditional specifies that we know the read value
            actual_next = tape[t + 1]
            conditional = np.array([np.nan, actual_next[1], np.nan, np.nan, np.nan])
            
            prediction = gdc.greedy_sample(forecast_dist, conditional=conditional)
            
            # Compare prediction to actual for each position
            for pos in range(5):
                if not np.isnan(prediction[pos]):
                    total_per_position[pos] += 1
                    if prediction[pos] == actual_next[pos]:
                        correct_per_position[pos] += 1
    
    # Compute accuracy per position
    accuracy_per_position = np.zeros(5)
    for pos in range(5):
        if total_per_position[pos] > 0:
            accuracy_per_position[pos] = correct_per_position[pos] / total_per_position[pos]
    
    return accuracy_per_position, total_per_position


def evaluate_gdc_forecasting_reduced(gdc, test_tapes, test_halted):
    """
    Evaluate GDC forecasting accuracy on test tapes (reduced 3-column model).
    
    The GDC is built on read, write, direction only (columns 1, 2, 3).
    Uses greedy_sample with conditional on read value (column 0 in reduced model).
    Forecasts are 1 step ahead.
    
    Parameters
    ----------
    gdc : GenerativeDenseChain
        The trained GDC model (3 columns: read, write, direction).
    test_tapes : list of np.ndarray
        List of test tape execution traces (full 5-column format).
    test_halted : list of bool
        Whether each test tape halted.
        
    Returns
    -------
    accuracy_per_position : np.ndarray
        Accuracy for each of the 3 positions (read, write, direction).
    total_per_position : np.ndarray
        Total predictions made for each position.
    """
    # 3 positions: read, write, direction
    correct_per_position = np.zeros(3)
    total_per_position = np.zeros(3)
    
    for tape, halted in zip(test_tapes, test_halted):
        if len(tape) < 2:
            continue
        
        # Extract reduced columns (read, write, direction = columns 1, 2, 3)
        tape_reduced = tape[:, 1:4]
        
        # Single forward pass to get state distribution at every timestep (O(T) instead of O(T²))
        _, state_history = gdc.forward_pass(tape_reduced, return_history=True)
            
        # For each step (except the last), predict the next step
        for t in range(len(tape) - 1):
            # Skip if current or next row is a halt row (contains -1 values)
            if tape[t, 0] == -1 or tape[t + 1, 0] == -1:
                continue
            
            # Use precomputed state distribution at step t
            state_dist = state_history[t]
            
            # Forecast 1 step ahead
            forecast_dist = gdc.forecast(state_dist, n_steps=1)
            
            # Greedy sample with conditional on read value (column 0 in reduced model)
            # The conditional specifies that we know the read value
            actual_next = tape_reduced[t + 1]
            conditional = np.array([actual_next[0], np.nan, np.nan])
            
            prediction = gdc.greedy_sample(forecast_dist, conditional=conditional)
            
            # Compare prediction to actual for each position
            for pos in range(3):
                if not np.isnan(prediction[pos]):
                    total_per_position[pos] += 1
                    if prediction[pos] == actual_next[pos]:
                        correct_per_position[pos] += 1
    
    # Compute accuracy per position
    accuracy_per_position = np.zeros(3)
    for pos in range(3):
        if total_per_position[pos] > 0:
            accuracy_per_position[pos] = correct_per_position[pos] / total_per_position[pos]
    
    return accuracy_per_position, total_per_position


def analyze_write_errors_reduced(gdc, test_tapes, test_halted, state_encoding, symbol_encoding):
    """
    Analyze write errors in the reduced 3-column model.
    
    Tracks:
    - Which Turing machine states have the most write errors
    - What symbol confusions occur (predicted vs actual)
    - What read symbol was present when errors occurred
    
    Parameters
    ----------
    gdc : GenerativeDenseChain
        The trained GDC model (3 columns: read, write, direction).
    test_tapes : list of np.ndarray
        List of test tape execution traces (full 5-column format).
    test_halted : list of bool
        Whether each test tape halted.
    state_encoding : dict
        Mapping from state names to integers.
    symbol_encoding : dict
        Mapping from symbols to integers.
        
    Returns
    -------
    error_analysis : dict
        Dictionary containing error analysis data.
    """
    from collections import defaultdict
    
    # Reverse mappings for readable output
    state_decoding = {v: k for k, v in state_encoding.items()}
    symbol_decoding = {v: k for k, v in symbol_encoding.items()}
    
    # Track errors by various dimensions
    errors_by_state = defaultdict(int)       # TM state -> count
    total_by_state = defaultdict(int)        # TM state -> total predictions
    confusion_matrix = defaultdict(int)      # (predicted, actual) -> count
    errors_by_read_symbol = defaultdict(int) # read symbol -> count
    total_by_read_symbol = defaultdict(int)  # read symbol -> total
    errors_by_state_and_read = defaultdict(int)  # (state, read) -> count
    total_by_state_and_read = defaultdict(int)   # (state, read) -> total
    
    # Store detailed error records
    error_records = []
    
    for tape_idx, (tape, halted) in enumerate(zip(test_tapes, test_halted)):
        if len(tape) < 2:
            continue
        
        # Extract reduced columns (read, write, direction = columns 1, 2, 3)
        tape_reduced = tape[:, 1:4]
        
        # Single forward pass to get state distribution at every timestep
        _, state_history = gdc.forward_pass(tape_reduced, return_history=True)
            
        # For each step (except the last), predict the next step
        for t in range(len(tape) - 1):
            # Skip if current or next row is a halt row (contains -1 values)
            if tape[t, 0] == -1 or tape[t + 1, 0] == -1:
                continue
            
            # Use precomputed state distribution at step t
            state_dist = state_history[t]
            
            # Forecast 1 step ahead
            forecast_dist = gdc.forecast(state_dist, n_steps=1)
            
            # Greedy sample with conditional on read value (column 0 in reduced model)
            actual_next = tape_reduced[t + 1]
            conditional = np.array([actual_next[0], np.nan, np.nan])
            
            prediction = gdc.greedy_sample(forecast_dist, conditional=conditional)
            
            # Get the Turing machine state from the full tape (next step's state = column 0)
            tm_state_encoded = tape[t + 1, 0]
            tm_state = state_decoding.get(tm_state_encoded, f"State_{tm_state_encoded}")
            
            # Get read symbol
            read_encoded = int(actual_next[0])
            read_symbol = symbol_decoding.get(read_encoded, f"Sym_{read_encoded}")
            
            # Track totals
            if not np.isnan(prediction[1]):  # position 1 is write in reduced model
                total_by_state[tm_state] += 1
                total_by_read_symbol[read_symbol] += 1
                total_by_state_and_read[(tm_state, read_symbol)] += 1
            
            # Check for write error (position 1 in reduced model)
            if not np.isnan(prediction[1]) and prediction[1] != actual_next[1]:
                predicted_write = int(prediction[1])
                actual_write = int(actual_next[1])
                
                predicted_symbol = symbol_decoding.get(predicted_write, f"Sym_{predicted_write}")
                actual_symbol = symbol_decoding.get(actual_write, f"Sym_{actual_write}")
                
                # Record error
                errors_by_state[tm_state] += 1
                confusion_matrix[(predicted_symbol, actual_symbol)] += 1
                errors_by_read_symbol[read_symbol] += 1
                errors_by_state_and_read[(tm_state, read_symbol)] += 1
                
                # Store detailed record
                error_records.append({
                    'tape_idx': tape_idx,
                    'step': t + 1,
                    'tm_state': tm_state,
                    'read_symbol': read_symbol,
                    'predicted_write': predicted_symbol,
                    'actual_write': actual_symbol
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
        'symbol_decoding': symbol_decoding
    }


def print_error_analysis(analysis):
    """Print a formatted summary of write error analysis."""
    
    print("\n" + "=" * 70)
    print("WRITE ERROR ANALYSIS FOR TEST 2 (REDUCED 3-COLUMN GDC)")
    print("=" * 70)
    
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
    
    print(f"\nTotal write errors: {total_errors} / {total_predictions} "
          f"({100*total_errors/total_predictions:.2f}% error rate)")
    
    # Errors by TM state
    print("\n--- ERRORS BY TURING MACHINE STATE ---")
    print(f"{'State':<15} {'Errors':<10} {'Total':<10} {'Error Rate':<12}")
    print("-" * 47)
    
    # Sort by error count (descending)
    for state in sorted(errors_by_state.keys(), key=lambda x: errors_by_state.get(x, 0), reverse=True):
        errors = errors_by_state.get(state, 0)
        total = total_by_state.get(state, 0)
        rate = 100 * errors / total if total > 0 else 0
        print(f"{state:<15} {errors:<10} {total:<10} {rate:.2f}%")
    
    # States with zero errors
    zero_error_states = [s for s in total_by_state.keys() if s not in errors_by_state]
    if zero_error_states:
        print("\nStates with ZERO errors:")
        for state in sorted(zero_error_states):
            print(f"  {state}: 0 / {total_by_state[state]}")
    
    # Errors by read symbol
    print("\n--- ERRORS BY READ SYMBOL ---")
    print(f"{'Read Symbol':<15} {'Errors':<10} {'Total':<10} {'Error Rate':<12}")
    print("-" * 47)
    
    for symbol in sorted(total_by_read_symbol.keys(), key=str):
        errors = errors_by_read_symbol.get(symbol, 0)
        total = total_by_read_symbol.get(symbol, 0)
        rate = 100 * errors / total if total > 0 else 0
        print(f"{repr(symbol):<15} {errors:<10} {total:<10} {rate:.2f}%")
    
    # Symbol confusion matrix
    print("\n--- SYMBOL CONFUSION MATRIX (Predicted vs Actual) ---")
    print(f"{'Predicted -> Actual':<25} {'Count':<10} {'% of Errors':<12}")
    print("-" * 47)
    
    for (pred, actual) in sorted(confusion_matrix.keys(), key=lambda x: confusion_matrix[x], reverse=True):
        count = confusion_matrix[(pred, actual)]
        pct = 100 * count / total_errors if total_errors > 0 else 0
        print(f"{repr(pred)} -> {repr(actual):<15} {count:<10} {pct:.1f}%")
    
    # Errors by state AND read symbol (most problematic combinations)
    print("\n--- TOP ERROR COMBINATIONS (State + Read Symbol) ---")
    print(f"{'State':<15} {'Read':<10} {'Errors':<10} {'Total':<10} {'Error Rate':<12}")
    print("-" * 57)
    
    # Sort by error count
    sorted_combos = sorted(errors_by_state_and_read.keys(), 
                           key=lambda x: errors_by_state_and_read[x], reverse=True)
    for state, read in sorted_combos[:15]:  # Top 15
        errors = errors_by_state_and_read[(state, read)]
        total = total_by_state_and_read[(state, read)]
        rate = 100 * errors / total if total > 0 else 0
        print(f"{state:<15} {repr(read):<10} {errors:<10} {total:<10} {rate:.2f}%")
    
    # Sample error records
    if error_records:
        print("\n--- SAMPLE ERROR RECORDS (first 20) ---")
        print(f"{'Tape':<6} {'Step':<8} {'TM State':<12} {'Read':<8} {'Pred':<10} {'Actual':<10}")
        print("-" * 54)
        for rec in error_records[:20]:
            print(f"{rec['tape_idx']:<6} {rec['step']:<8} {rec['tm_state']:<12} "
                  f"{repr(rec['read_symbol']):<8} {repr(rec['predicted_write']):<10} "
                  f"{repr(rec['actual_write']):<10}")


def run_tests(train_tapes, train_halted, test_tapes, test_halted, state_encoding, symbol_encoding):
    """
    Run both GDC forecasting tests.
    
    Returns results_full, results_reduced, and error_analysis dictionaries.
    """
    n_train_total = len(train_tapes)
    
    # =========================================================================
    # TEST 1: Full 5-column GDC
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: FULL 5-COLUMN GDC (conditional on read)")
    print("=" * 70)
    
    position_names_full = ['current_state', 'read', 'write', 'direction', 'next_state']
    
    # Store results
    results_full = []
    
    for n_train in range(10, 40, 10):
        # Create GDC with first n_train tapes
        training_subset = train_tapes[:n_train]
        
        # Create GDC model with sequence_starts initial distribution and self-loop two-step transition
        gdc = GenerativeDenseChain(
            training_subset, 
            alpha=0.95, 
            theta=0.005, 
            gamma=0.000,
            transition_type='self_loop_two_step',
            initial_dist='sequence_starts'
        )
        
        # Evaluate on test tapes
        accuracy_per_pos, total_per_pos = evaluate_gdc_forecasting(
            gdc, test_tapes, test_halted, symbol_encoding
        )
        
        results_full.append({
            'n_train': n_train,
            'n_states': gdc.n_states,
            'accuracy': accuracy_per_pos.copy(),
            'total': total_per_pos.copy()
        })
        
        # Print progress every 10 tapes and at key points
        overall_acc = accuracy_per_pos.mean()
        print(f"\nAfter {n_train} training tape(s):")
        print(f"  GDC n_states: {gdc.n_states}")
        for pos, name in enumerate(position_names_full):
            print(f"  {name}: {accuracy_per_pos[pos]:.3f} "
                  f"({int(total_per_pos[pos])} predictions)")
        print(f"  Overall (mean): {overall_acc:.3f}")
    
    # =========================================================================
    # TEST 2: Reduced 3-column GDC (read, write, direction only)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: REDUCED 3-COLUMN GDC (read, write, direction only)")
    print("=" * 70)
    
    position_names_reduced = ['read', 'write', 'direction']
    
    # Store results
    results_reduced = []
    gdc_reduced_final = None  # Keep reference to final model for error analysis
    
    for n_train in range(10, n_train_total, 10):
        # Create GDC with first n_train tapes, using only columns 1, 2, 3
        training_subset = [tape[:, 1:4] for tape in train_tapes[:n_train]]
        
        # Create GDC model with sequence_starts initial distribution and self-loop two-step transition
        gdc = GenerativeDenseChain(
            training_subset, 
            alpha=0.99, 
            theta=0.005, 
            gamma=0.000,
            transition_type='self_loop_two_step',
            initial_dist='sequence_starts'
        )
        
        # Evaluate on test tapes
        accuracy_per_pos, total_per_pos = evaluate_gdc_forecasting_reduced(
            gdc, test_tapes, test_halted
        )
        
        results_reduced.append({
            'n_train': n_train,
            'n_states': gdc.n_states,
            'accuracy': accuracy_per_pos.copy(),
            'total': total_per_pos.copy()
        })
        
        # Print progress every 10 tapes and at key points
        overall_acc = accuracy_per_pos.mean()
        print(f"\nAfter {n_train} training tape(s):")
        print(f"  GDC n_states: {gdc.n_states}")
        for pos, name in enumerate(position_names_reduced):
            print(f"  {name}: {accuracy_per_pos[pos]:.3f} "
                  f"({int(total_per_pos[pos])} predictions)")
        print(f"  Overall (mean): {overall_acc:.3f}")
        
        gdc_reduced_final = gdc  # Save final model
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    # Test 1 summary
    print("\n--- TEST 1: Full 5-column GDC ---")
    print("\nAccuracy progression (overall mean accuracy):")
    print(f"{'n_train':<10} {'n_states':<10} {'Accuracy':<10}")
    print("-" * 30)
    for r in results_full:
        if r['n_train'] <= 5 or r['n_train'] % 10 == 0 or r['n_train'] == n_train_total:
            print(f"{r['n_train']:<10} {r['n_states']:<10} {r['accuracy'].mean():.3f}")
    
    print("\nFinal model accuracy per position:")
    final_full = results_full[-1]
    for pos, name in enumerate(position_names_full):
        print(f"  {name}: {final_full['accuracy'][pos]:.3f}")
    print(f"  Overall (mean): {final_full['accuracy'].mean():.3f}")
    
    # Test 2 summary
    print("\n--- TEST 2: Reduced 3-column GDC ---")
    print("\nAccuracy progression (overall mean accuracy):")
    print(f"{'n_train':<10} {'n_states':<10} {'Accuracy':<10}")
    print("-" * 30)
    for r in results_reduced:
        if r['n_train'] <= 5 or r['n_train'] % 10 == 0 or r['n_train'] == n_train_total:
            print(f"{r['n_train']:<10} {r['n_states']:<10} {r['accuracy'].mean():.3f}")
    
    print("\nFinal model accuracy per position:")
    final_reduced = results_reduced[-1]
    for pos, name in enumerate(position_names_reduced):
        print(f"  {name}: {final_reduced['accuracy'][pos]:.3f}")
    print(f"  Overall (mean): {final_reduced['accuracy'].mean():.3f}")
    
    # Perform error analysis on the final reduced model
    error_analysis = None
    if gdc_reduced_final is not None:
        error_analysis = analyze_write_errors_reduced(
            gdc_reduced_final, test_tapes, test_halted, 
            state_encoding, symbol_encoding
        )
        print_error_analysis(error_analysis)
    
    return results_full, results_reduced, error_analysis


# =============================================================================
# Configuration - adjust these parameters as needed
# =============================================================================
N_TRAIN = 400              # Number of training addition problems
N_TEST = 10               # Number of test addition problems
NUM_RANGE_TRAIN = (0, 32) # Range of random numbers for training (5-bit)
NUM_RANGE_TEST = (0, 32)  # Range of random numbers for testing (5-bit)
MAX_STEPS = 5000          # Max steps before timeout (adder needs more steps than busy beaver)
TRAIN_SEED = 42
TEST_SEED = 123

# =============================================================================
# Generate training and test sets separately
# =============================================================================
print("=" * 70)
print("TEST GDC FORECASTING ON BINARY ADDER EXECUTION TRACES")
print("=" * 70)

# Generate training tapes
print(f"\nGenerating {N_TRAIN} training additions (num_range={NUM_RANGE_TRAIN}, max_steps={MAX_STEPS}, seed={TRAIN_SEED})...")
train_results = simulate_random_adders(
    n_runs=N_TRAIN,
    num_range=NUM_RANGE_TRAIN,
    max_steps=MAX_STEPS,
    seed=TRAIN_SEED
)
train_tapes = train_results['runs']
train_halted = train_results['halted_flags']
train_inputs = train_results['inputs']
train_correct = train_results['correct']
state_encoding = train_results['state_encoding']
symbol_encoding = train_results['symbol_encoding']

print(f"  Training runs that halted: {sum(train_halted)}/{N_TRAIN}")
print(f"  Training runs correct: {sum(train_correct)}/{N_TRAIN}")

# Generate test tapes (separate call with different seed and potentially different num_range)
print(f"\nGenerating {N_TEST} test additions (num_range={NUM_RANGE_TEST}, max_steps={MAX_STEPS}, seed={TEST_SEED})...")
test_results = simulate_random_adders(
    n_runs=N_TEST,
    num_range=NUM_RANGE_TEST,
    max_steps=MAX_STEPS,
    seed=TEST_SEED
)
test_tapes = test_results['runs']
test_halted = test_results['halted_flags']
test_inputs = test_results['inputs']
test_correct = test_results['correct']

print(f"  Test runs that halted: {sum(test_halted)}/{N_TEST}")
print(f"  Test runs correct: {sum(test_correct)}/{N_TEST}")

# Print encoding information
print(f"\nState encoding ({len(state_encoding)} states): {state_encoding}")
print(f"Symbol encoding ({len(symbol_encoding)} symbols): {symbol_encoding}")
print(f"Training tapes: {N_TRAIN} (num_range={NUM_RANGE_TRAIN})")
print(f"Test tapes: {N_TEST} (num_range={NUM_RANGE_TEST})")

# Print sample statistics
train_steps = [tape.shape[0] for tape in train_tapes]
test_steps = [tape.shape[0] for tape in test_tapes]
print(f"\nTraining tape lengths: min={min(train_steps)}, max={max(train_steps)}, mean={np.mean(train_steps):.1f}")
print(f"Test tape lengths: min={min(test_steps)}, max={max(test_steps)}, mean={np.mean(test_steps):.1f}")

# Store results at module level for variable explorer access
results_full = None
results_reduced = None
error_analysis = None

if __name__ == "__main__":
    results_full, results_reduced, error_analysis = run_tests(
        train_tapes, train_halted, test_tapes, test_halted, state_encoding, symbol_encoding
    )
