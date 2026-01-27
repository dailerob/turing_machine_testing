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


def run_turing_machine(program, halt_state='H', initial_state='A', max_steps=None, verbose=True, initial_tape=None):
    """
    Run a Turing machine simulation.
    
    Args:
        program: List of 5-tuples (current_state, read, write, direction, next_state)
        halt_state: The state that causes the machine to halt
        initial_state: The starting state
        max_steps: Maximum steps before forced stop (None for unlimited)
        verbose: If True, print each step
        initial_tape: Optional dict mapping positions to symbols (0 or 1).
                      If None, tape starts as all zeros.
    
    Returns:
        Tuple of (tape_dict, steps_taken, ones_count, history, halted)
        where history is a list of 5-tuples executed at each step
        and halted is True if the machine reached the halt state
    """
    # Build transition table from program
    transitions = {}
    for current_state, read, write, direction, next_state in program:
        transitions[(current_state, read)] = (write, direction, next_state)
    
    # Initialize tape (defaultdict returns 0 for unvisited cells)
    tape = defaultdict(int)
    if initial_tape is not None:
        tape.update(initial_tape)
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
    
    halted = (state == halt_state)
    
    if halted and verbose:
        print(f"\nMachine halted after {steps} steps.")
    
    # Count ones on tape
    ones_count = sum(1 for v in tape.values() if v == 1)
    
    return dict(tape), steps, ones_count, history, halted


def history_to_numpy(history, state_encoding=None, include_halt_row=True):
    """
    Convert execution history to a numpy array of shape (n_steps, 5).
    
    Args:
        history: List of 5-tuples from run_turing_machine
        state_encoding: Optional dict mapping state names to integers.
                        If None, states are auto-encoded alphabetically.
        include_halt_row: If True (default), adds a final row representing the
                          halt state with [-1, -1, -1, -1, halt_state].
    
    Returns:
        numpy array of shape (n_steps, 5) where columns are:
            [current_state, read, write, direction, next_state]
        Also returns the state_encoding dict used.
    
    Encoding:
        - States are encoded as integers (e.g., A=0, B=1, C=2, ...)
        - Read/Write symbols (0, 1) remain as integers
        - Direction: L=0, R=1
        - Halt row uses -1 for current_state, read, write, direction
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
    
    # Determine array size
    n_steps = len(history)
    total_rows = n_steps + 1 if include_halt_row else n_steps
    arr = np.zeros((total_rows, 5), dtype=np.int8)
    
    for i, (curr_state, read, write, direction, next_state) in enumerate(history):
        arr[i, 0] = state_encoding[curr_state]
        arr[i, 1] = read
        arr[i, 2] = write
        arr[i, 3] = dir_encoding[direction]
        arr[i, 4] = state_encoding[next_state]
    
    # Add halt row: [-1, -1, -1, -1, halt_state]
    if include_halt_row and history:
        halt_state = history[-1][4]  # next_state of last transition
        arr[n_steps, 0] = -1  # No current state action
        arr[n_steps, 1] = -1  # No read
        arr[n_steps, 2] = -1  # No write
        arr[n_steps, 3] = -1  # No direction
        arr[n_steps, 4] = state_encoding[halt_state]  # Final state
    
    return arr, state_encoding


def simulate_random_tapes(program, n_runs, tape_length=10, halt_state='H', initial_state='A',
                          max_steps=100, include_halt_row=True, seed=None):
    """
    Simulate a Turing machine on multiple random initial tapes.
    
    Args:
        program: List of 5-tuples (current_state, read, write, direction, next_state)
        n_runs: Number of simulations to run
        tape_length: Length of the random tape (centered at position 0)
        halt_state: The state that causes the machine to halt
        initial_state: The starting state
        max_steps: Maximum steps before forced stop (default 100)
        include_halt_row: If True (default), adds a final halt state row
                          only when the machine actually halts
        seed: Optional random seed for reproducibility
    
    Returns:
        Tuple of (runs, halted_flags, state_encoding) where:
            - runs: List of numpy arrays, one per simulation
            - halted_flags: List of booleans indicating if each run halted
            - state_encoding: Dict mapping state names to integers
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Build state encoding from program
    all_states = set()
    for curr, _, _, _, nxt in program:
        all_states.add(curr)
        all_states.add(nxt)
    all_states.add(halt_state)
    state_encoding = {state: i for i, state in enumerate(sorted(all_states))}
    
    runs = []
    halted_flags = []
    
    for _ in range(n_runs):
        # Generate random tape centered at position 0
        start_pos = -(tape_length // 2)
        random_bits = np.random.randint(0, 2, size=tape_length)
        initial_tape = {start_pos + i: int(bit) for i, bit in enumerate(random_bits)}
        
        # Run simulation
        _, _, _, history, halted = run_turing_machine(
            program,
            halt_state=halt_state,
            initial_state=initial_state,
            max_steps=max_steps,
            verbose=False,
            initial_tape=initial_tape
        )
        
        # Convert to numpy - only add halt row if machine actually halted
        add_halt_row = include_halt_row and halted
        arr, _ = history_to_numpy(history, state_encoding, add_halt_row)
        runs.append(arr)
        halted_flags.append(halted)
    
    return runs, halted_flags, state_encoding


def save_history_to_file(history, filepath, state_encoding=None, include_halt_row=True):
    """
    Save execution history to a .npy file.
    
    Args:
        history: List of 5-tuples from run_turing_machine
        filepath: Path to save the numpy array (should end in .npy)
        state_encoding: Optional dict mapping state names to integers
        include_halt_row: If True (default), adds a final halt state row
    
    Returns:
        The state_encoding dict used
    """
    arr, encoding = history_to_numpy(history, state_encoding, include_halt_row)
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
    
    tape, steps, ones, history, halted = run_turing_machine(BUSY_BEAVER_4, verbose=True)
    
    visualize_tape(tape)
    
    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Total steps: {steps}")
    print(f"  Ones on tape: {ones}")
    print(f"  Halted: {halted}")
    print(f"{'=' * 60}")
    
    # Demonstrate history to numpy conversion
    print("\nConverting history to numpy array...")
    history_array, state_encoding = history_to_numpy(history, include_halt_row=halted)
    
    print(f"Array shape: {history_array.shape}")
    print(f"State encoding: {state_encoding}")
    print(f"Direction encoding: L=0, R=1")
    print(f"\nFirst 5 steps (as numpy array):")
    print("  [curr_state, read, write, direction, next_state]")
    print(history_array[:5])
    print(f"\nLast 5 steps:")
    print(history_array[-5:])
    
    # Demonstrate simulate_random_tapes
    print(f"\n{'=' * 60}")
    print("RANDOM TAPE SIMULATION")
    print("=" * 60)
    print("\nRunning 10 simulations with random initial tapes...")
    print("(tape_length=10, max_steps=100)\n")
    
    runs, halted_flags, encoding = simulate_random_tapes(
        BUSY_BEAVER_4,
        n_runs=10,
        tape_length=10,
        max_steps=100,
        seed=20
    )
    
    print(f"{'Run':<5} {'Steps':<8} {'Halted?':<10} {'Shape'}")
    print("-" * 35)
    
    for i, (run, halted) in enumerate(zip(runs, halted_flags)):
        n_steps = run.shape[0] - 1 if halted else run.shape[0]  # Subtract halt row only if halted
        halted_str = "Yes" if halted else "No (max)"
        print(f"{i+1:<5} {n_steps:<8} {halted_str:<10} {run.shape}")
    
    print(f"\nTotal runs: {len(runs)}")
    print(f"Runs that halted: {sum(halted_flags)}")
    
    # Show example runs
    halted_idx = next(i for i, h in enumerate(halted_flags) if h)
    not_halted_idx = next((i for i, h in enumerate(halted_flags) if not h), None)
    
    print(f"\nExample - Run {halted_idx+1} (halted, last 3 rows):")
    print(runs[halted_idx][-3:])
    
    if not_halted_idx is not None:
        print(f"\nExample - Run {not_halted_idx+1} (did NOT halt, last 3 rows):")
        print(runs[not_halted_idx][-3:])