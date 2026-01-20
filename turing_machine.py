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
        Tuple of (tape_dict, steps_taken, ones_count)
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
    
    return dict(tape), steps, ones_count


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
    
    tape, steps, ones = run_turing_machine(BUSY_BEAVER_4, verbose=True)
    
    visualize_tape(tape)
    
    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Total steps: {steps}")
    print(f"  Ones on tape: {ones}")
    print(f"{'=' * 60}")
