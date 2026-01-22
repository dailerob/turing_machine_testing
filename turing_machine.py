"""
Turing Machine Simulator

Turing machines are represented as lists of 5-tuples:
    (current_state, read, write, direction, next_state)

Where:
    - current_state: the state the machine must be in for this rule to apply
    - read: the symbol read from the tape (0 or 1)
    - write: the symbol to write to the tape
    - direction: 'L' (left) or 'R' (right)
    - next_state: the state to transition to
"""

from collections import defaultdict
import numpy as np


def run_turing_machine(program, halt_state='H', initial_state='A', max_steps=None, verbose=True):
    """
    Run a Turing machine simulation.
    
    Args:
        program: List of 5-tuples (current_state, read, write, direction, next_state)
        halt_state: The state that causes the machine to halt
        initial_state: The starting state
        max_steps: Maximum steps before forced stop (None for unlimited)
        verbose: If True, print each step
    
    Returns:
        Tuple of (tape_dict, steps_taken, ones_count, history)
        where history is a list of 5-tuples executed at each step
    """
    # Build transition table from program
    transitions = {}
    for current_state, read, write, direction, next_state in program:
        transitions[(current_state, read)] = (write, direction, next_state)
    
    # Initialize tape (defaultdict returns 0 for unvisited cells)
    tape = defaultdict(int)
    head = 0
    state = initial_state
    steps = 0
    history = []  # Track which 5-tuple was executed at each step
    
    if verbose:
        print(f"Starting Turing Machine simulation")
        print(f"Initial state: {initial_state}, Halt state: {halt_state}")
        print(f"Program has {len(program)} transition rules")
        print("-" * 60)
    
    while state != halt_state:
        if max_steps is not None and steps >= max_steps:
            if verbose:
                print(f"\nReached maximum steps ({max_steps}), stopping.")
            break
        
        read_symbol = tape[head]
        
        if (state, read_symbol) not in transitions:
            if verbose:
                print(f"\nNo transition for state={state}, read={read_symbol}. Halting.")
            break
        
        write_symbol, direction, next_state = transitions[(state, read_symbol)]
        
        # Record the 5-tuple that was executed
        history.append((state, read_symbol, write_symbol, direction, next_state))
        
        if verbose:
            print(f"Step {steps + 1}: State={state}, Read={read_symbol} -> "
                  f"Write={write_symbol}, Move={direction}, Next={next_state}")
        
        # Execute transition
        tape[head] = write_symbol
        head += 1 if direction == 'R' else -1
        state = next_state
        steps += 1
    
    if state == halt_state and verbose:
        print(f"\nMachine halted after {steps} steps.")
    
    # Count ones on tape
    ones_count = sum(1 for v in tape.values() if v == 1)
    
    return dict(tape), steps, ones_count, history


def history_to_numpy(history, state_encoding=None):
    """
    Convert execution history to a numpy array of shape (n_steps, 5).
    
    Args:
        history: List of 5-tuples from run_turing_machine
        state_encoding: Optional dict mapping state names to integers.
                        If None, states are auto-encoded alphabetically.
    
    Returns:
        numpy array of shape (n_steps, 5) where columns are:
            [current_state, read, write, direction, next_state]
        Also returns the state_encoding dict used.
    
    Encoding:
        - States are encoded as integers (e.g., A=0, B=1, C=2, ...)
        - Read/Write symbols (0, 1) remain as integers
        - Direction: L=0, R=1
    """
    if not history:
        return np.array([]).reshape(0, 5), {}
    
    # Build state encoding if not provided
    if state_encoding is None:
        all_states = set()
        for curr, _, _, _, nxt in history:
            all_states.add(curr)
            all_states.add(nxt)
        state_encoding = {state: i for i, state in enumerate(sorted(all_states))}
    
    # Direction encoding
    dir_encoding = {'L': 0, 'R': 1}
    
    # Convert history to numpy array
    n_steps = len(history)
    arr = np.zeros((n_steps, 5), dtype=np.int32)
    
    for i, (curr_state, read, write, direction, next_state) in enumerate(history):
        arr[i, 0] = state_encoding[curr_state]
        arr[i, 1] = read
        arr[i, 2] = write
        arr[i, 3] = dir_encoding[direction]
        arr[i, 4] = state_encoding[next_state]
    
    return arr, state_encoding


def save_history_to_file(history, filepath, state_encoding=None):
    """
    Save execution history to a .npy file.
    
    Args:
        history: List of 5-tuples from run_turing_machine
        filepath: Path to save the numpy array (should end in .npy)
        state_encoding: Optional dict mapping state names to integers
    
    Returns:
        The state_encoding dict used
    """
    arr, encoding = history_to_numpy(history, state_encoding)
    np.save(filepath, arr)
    return encoding


def visualize_tape(tape, head=None, width=40):
    """
    Print a visual representation of the tape.
    
    Args:
        tape: Dictionary mapping positions to symbols
        head: Optional head position to highlight
        width: Number of cells to show around center
    """
    if not tape:
        print("Empty tape")
        return
    
    min_pos = min(tape.keys())
    max_pos = max(tape.keys())
    
    print("\nFinal tape state:")
    print("Position:", end=" ")
    for pos in range(min_pos, max_pos + 1):
        print(f"{pos:^3}", end="")
    print()
    
    print("   Value:", end=" ")
    for pos in range(min_pos, max_pos + 1):
        symbol = tape.get(pos, 0)
        if head is not None and pos == head:
            print(f"[{symbol}]", end="")
        else:
            print(f" {symbol} ", end="")
    print()


# 4-State Busy Beaver
# This machine writes 13 ones on the tape before halting
# It runs for 107 steps
BUSY_BEAVER_4 = [
    # (current_state, read, write, direction, next_state)
    ('A', 0, 1, 'R', 'B'),
    ('A', 1, 1, 'L', 'B'),
    ('B', 0, 1, 'L', 'A'),
    ('B', 1, 0, 'L', 'C'),
    ('C', 0, 1, 'R', 'H'),  # Halt when in state C reading 0
    ('C', 1, 1, 'L', 'D'),
    ('D', 0, 1, 'R', 'D'),
    ('D', 1, 0, 'R', 'A'),
]


if __name__ == "__main__":
    print("=" * 60)
    print("TURING MACHINE SIMULATOR")
    print("=" * 60)
    print("\nRunning 4-State Busy Beaver")
    print("Expected: 13 ones written, 107 steps to halt\n")
    
    tape, steps, ones, history = run_turing_machine(BUSY_BEAVER_4, verbose=True)
    
    visualize_tape(tape)
    
    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Total steps: {steps}")
    print(f"  Ones on tape: {ones}")
    print(f"{'=' * 60}")
    
    # Demonstrate history to numpy conversion
    print("\nConverting history to numpy array...")
    history_array, state_encoding = history_to_numpy(history)
    
    print(f"Array shape: {history_array.shape}")
    print(f"State encoding: {state_encoding}")
    print(f"Direction encoding: L=0, R=1")
    print(f"\nFirst 5 steps (as numpy array):")
    print("  [curr_state, read, write, direction, next_state]")
    print(history_array[:5])
    print(f"\nLast 5 steps:")
    print(history_array[-5:])
    
    # Save to file
    save_history_to_file(history, "busy_beaver_history.npy")
    print(f"\nHistory saved to 'busy_beaver_history.npy'")