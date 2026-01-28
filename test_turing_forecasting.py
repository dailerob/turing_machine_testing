"""
Test GDC forecasting on Turing Machine execution traces.

This script:
1. Simulates BUSY_BEAVER_4 on separate training and test tapes
2. Creates GDC models with increasing amounts of training data
3. Evaluates 1-step-ahead forecasting accuracy on test tapes
4. Reports accuracy per position (current_state, read, write, direction, next_state)

Two tests are run:
- Test 1: Full 5-column GDC, forecasting with conditional on read (column 1)
- Test 2: Reduced 3-column GDC (read, write, direction only), forecasting with conditional on read

Variables are kept at module level for inspection in variable explorer.
"""

import numpy as np
from turing_machine import simulate_random_tapes, BUSY_BEAVER_4
from generative_dense_chain import GenerativeDenseChain


def evaluate_gdc_forecasting(gdc, test_tapes, test_halted):
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


def run_tests(train_tapes, train_halted, test_tapes, test_halted):
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
        gdc = GenerativeDenseChain(training_subset, alpha=0.9)
        
        # Evaluate on test tapes
        accuracy_per_pos, total_per_pos = evaluate_gdc_forecasting(
            gdc, test_tapes, test_halted
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
        gdc = GenerativeDenseChain(training_subset, alpha=0.9)
        
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
N_TRAIN = 20
N_TEST = 20
TAPE_LENGTH = 10
TRAIN_MAX_STEPS = 20
TEST_MAX_STEPS = 100
TRAIN_SEED = 42
TEST_SEED = 123

# =============================================================================
# Generate training and test sets separately
# =============================================================================
print("=" * 70)
print("TEST GDC FORECASTING ON TURING MACHINE EXECUTION TRACES")
print("=" * 70)

# Generate training tapes
print(f"\nGenerating {N_TRAIN} training tapes (tape_length={TAPE_LENGTH}, max_steps={TRAIN_MAX_STEPS}, seed={TRAIN_SEED})...")
train_tapes, train_halted, state_encoding = simulate_random_tapes(
    BUSY_BEAVER_4,
    n_runs=N_TRAIN,
    tape_length=TAPE_LENGTH,
    max_steps=TRAIN_MAX_STEPS,
    seed=TRAIN_SEED
)
print(f"  Training runs that halted: {sum(train_halted)}/{N_TRAIN}")

# Generate test tapes (separate call, can use different max_steps)
print(f"\nGenerating {N_TEST} test tapes (tape_length={TAPE_LENGTH}, max_steps={TEST_MAX_STEPS}, seed={TEST_SEED})...")
test_tapes, test_halted, _ = simulate_random_tapes(
    BUSY_BEAVER_4,
    n_runs=N_TEST,
    tape_length=TAPE_LENGTH,
    max_steps=TEST_MAX_STEPS,
    seed=TEST_SEED
)
print(f"  Test runs that halted: {sum(test_halted)}/{N_TEST}")

print(f"\nState encoding: {state_encoding}")
print(f"Training tapes: {N_TRAIN}")
print(f"Test tapes: {N_TEST}")

# Store results at module level for variable explorer access
results_full = None
results_reduced = None

if __name__ == "__main__":
    results_full, results_reduced = run_tests(
        train_tapes, train_halted, test_tapes, test_halted
    )
