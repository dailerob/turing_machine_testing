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
            
        # For each step (except the last), predict the next step
        for t in range(len(tape) - 1):
            # Skip if current or next row is a halt row (contains -1 values)
            if tape[t, 0] == -1 or tape[t + 1, 0] == -1:
                continue
            
            # Forward pass on observations up to and including step t
            observations = tape[:t + 1]
            state_dist = gdc.forward_pass(observations)
            
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
            
        # For each step (except the last), predict the next step
        for t in range(len(tape) - 1):
            # Skip if current or next row is a halt row (contains -1 values)
            if tape[t, 0] == -1 or tape[t + 1, 0] == -1:
                continue
            
            # Forward pass on observations up to and including step t
            observations = tape_reduced[:t + 1]
            state_dist = gdc.forward_pass(observations)
            
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


def run_tests(train_tapes, train_halted, test_tapes, test_halted, symbol_encoding):
    """
    Run both GDC forecasting tests.
    
    Returns results_full and results_reduced dictionaries.
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
    
    for n_train in range(1, n_train_total + 1):
        # Create GDC with first n_train tapes
        training_subset = train_tapes[:n_train]
        
        # Create GDC model
        gdc = GenerativeDenseChain(training_subset, alpha=0.5)
        
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
        if n_train <= 5 or n_train % 10 == 0 or n_train == n_train_total:
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
    
    for n_train in range(1, n_train_total + 1):
        # Create GDC with first n_train tapes, using only columns 1, 2, 3
        training_subset = [tape[:, 1:4] for tape in train_tapes[:n_train]]
        
        # Create GDC model
        gdc = GenerativeDenseChain(training_subset, alpha=0.5)
        
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
        if n_train <= 5 or n_train % 10 == 0 or n_train == n_train_total:
            overall_acc = accuracy_per_pos.mean()
            print(f"\nAfter {n_train} training tape(s):")
            print(f"  GDC n_states: {gdc.n_states}")
            for pos, name in enumerate(position_names_reduced):
                print(f"  {name}: {accuracy_per_pos[pos]:.3f} "
                      f"({int(total_per_pos[pos])} predictions)")
            print(f"  Overall (mean): {overall_acc:.3f}")
    
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
    
    return results_full, results_reduced


# =============================================================================
# Configuration - adjust these parameters as needed
# =============================================================================
N_TRAIN = 160              # Number of training addition problems
N_TEST = 10               # Number of test addition problems
NUM_RANGE_TRAIN = (0, 16) # Range of random numbers for training (5-bit)
NUM_RANGE_TEST = (0, 31)  # Range of random numbers for testing (5-bit)
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

if __name__ == "__main__":
    results_full, results_reduced = run_tests(
        train_tapes, train_halted, test_tapes, test_halted, symbol_encoding
    )
