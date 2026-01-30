"""
Turing Machine Simulator

Turing machines are represented as lists of 5-tuples:
    (current_state, read, write, direction, next_state)

Where:
    - current_state: the state the machine must be in for this rule to apply
    - read: the symbol read from the tape (can be any hashable symbol)
    - write: the symbol to write to the tape
    - direction: 'L' (left) or 'R' (right)
    - next_state: the state to transition to

Extended Features:
    - Support for arbitrary alphabet (not just binary)
    - YAML format parsing for online Turing machine definitions
    - Custom blank/default symbol
    - Custom starting head position
    - String-to-tape conversion for input initialization
"""

from collections import defaultdict
import numpy as np
import yaml
import re


def run_turing_machine(program, halt_state='H', initial_state='A', max_steps=None, 
                       verbose=True, initial_tape=None, blank_symbol=0, start_position=0):
    """
    Run a Turing machine simulation.
    
    Args:
        program: List of 5-tuples (current_state, read, write, direction, next_state)
        halt_state: The state that causes the machine to halt
        initial_state: The starting state
        max_steps: Maximum steps before forced stop (None for unlimited)
        verbose: If True, print each step
        initial_tape: Optional dict mapping positions to symbols.
                      If None, tape starts as all blank_symbol.
        blank_symbol: The default symbol for empty tape cells (default: 0)
        start_position: The initial head position (default: 0)
    
    Returns:
        Tuple of (tape_dict, steps_taken, ones_count, history, halted)
        where history is a list of 5-tuples executed at each step
        and halted is True if the machine reached the halt state
    """
    # Build transition table from program
    transitions = {}
    for current_state, read, write, direction, next_state in program:
        transitions[(current_state, read)] = (write, direction, next_state)
    
    # Initialize tape with custom blank symbol
    tape = defaultdict(lambda: blank_symbol)
    if initial_tape is not None:
        tape.update(initial_tape)
    head = start_position
    state = initial_state
    steps = 0
    history = []  # Track which 5-tuple was executed at each step
    
    if verbose:
        print(f"Starting Turing Machine simulation")
        print(f"Initial state: {initial_state}, Halt state: {halt_state}")
        print(f"Blank symbol: {repr(blank_symbol)}, Start position: {start_position}")
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
                print(f"\nNo transition for state={state}, read={repr(read_symbol)}. Halting.")
            break
        
        write_symbol, direction, next_state = transitions[(state, read_symbol)]
        
        # Record the 5-tuple that was executed (write_symbol may be None for no-write)
        history.append((state, read_symbol, write_symbol, direction, next_state))
        
        if verbose:
            write_display = repr(write_symbol) if write_symbol is not None else "(no write)"
            print(f"Step {steps + 1}: State={state}, Read={repr(read_symbol)} -> "
                  f"Write={write_display}, Move={direction}, Next={next_state}")
        
        # Execute transition (if write_symbol is None, keep the current value)
        if write_symbol is not None:
            tape[head] = write_symbol
        head += 1 if direction == 'R' else -1
        state = next_state
        steps += 1
    
    halted = (state == halt_state)
    
    if halted and verbose:
        print(f"\nMachine halted after {steps} steps.")
    
    # Count ones on tape (works for both int 1 and string '1')
    ones_count = sum(1 for v in tape.values() if v == 1 or v == '1')
    
    return dict(tape), steps, ones_count, history, halted


def string_to_tape(input_string, start_position=0):
    """
    Convert an input string to a tape dictionary.
    
    Args:
        input_string: String to place on tape (e.g., '1101011+11001')
        start_position: Position of the first character (default: 0)
    
    Returns:
        Dict mapping positions to single-character symbols
    
    Example:
        string_to_tape('101', start_position=0) -> {0: '1', 1: '0', 2: '1'}
        string_to_tape('a+b', start_position=-1) -> {-1: 'a', 0: '+', 1: 'b'}
    """
    return {start_position + i: char for i, char in enumerate(input_string)}


def tape_to_string(tape, blank_symbol=' '):
    """
    Convert a tape dictionary back to a readable string.
    
    Args:
        tape: Dict mapping positions to symbols
        blank_symbol: Symbol to use for gaps (default: ' ')
    
    Returns:
        String representation of the tape from min to max position
    """
    if not tape:
        return ""
    
    min_pos = min(tape.keys())
    max_pos = max(tape.keys())
    
    result = []
    for pos in range(min_pos, max_pos + 1):
        symbol = tape.get(pos, blank_symbol)
        result.append(str(symbol))
    
    return ''.join(result)


def _preprocess_yaml_keys(yaml_string):
    """
    Preprocess YAML string to convert list-style keys [a,b,c] to quoted strings.
    
    YAML doesn't support lists as dictionary keys, but many online Turing machine
    definitions use this syntax for symbol groups. We convert them to quoted strings
    that can be parsed later.
    
    Example: '[0,1,+]: R' becomes '"[0,1,+]": R'
    """
    # Pattern to match unquoted list keys at the start of a line (with indentation)
    # Matches: '    [0,1,+]: ' and converts to '    "[0,1,+]": '
    pattern = r'^(\s*)(\[[^\]]+\])(\s*:)'
    
    lines = yaml_string.split('\n')
    processed_lines = []
    
    for line in lines:
        match = re.match(pattern, line)
        if match:
            indent, key, colon = match.groups()
            rest = line[match.end():]
            processed_lines.append(f'{indent}"{key}"{colon}{rest}')
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)


def parse_yaml_machine(yaml_string):
    """
    Parse a YAML-format Turing machine definition into 5-tuples.
    
    Supports the common online format with features like:
        - Symbol groups: [0,1,+] matches any of those symbols
        - Implicit transitions: 'R' means move right, don't change symbol, same state
        - Shorthand syntax: {write: x, L: next_state} or {L: next_state}
    
    Args:
        yaml_string: YAML string defining the Turing machine
    
    Returns:
        Dict with keys:
            - 'program': List of 5-tuples (state, read, write, direction, next_state)
            - 'input': Initial tape string (or None)
            - 'blank': Blank symbol (default ' ')
            - 'start_state': Initial state name
            - 'halt_states': Set of halt state names (states with no transitions)
    
    Example YAML format:
        input: '1101011+11001'
        blank: ' '
        start state: right
        table:
          right:
            [0,1,+]: R
            ' ': {L: read}
          read:
            0: {write: c, L: have0}
            1: {write: c, L: have1}
    """
    # Preprocess to handle list-style keys
    yaml_string = _preprocess_yaml_keys(yaml_string)
    
    # Parse the YAML
    data = yaml.safe_load(yaml_string)
    
    input_string = data.get('input', None)
    blank_symbol = data.get('blank', ' ')
    start_state = data.get('start state', data.get('start_state', None))
    table = data.get('table', {})
    
    program = []
    all_states = set(table.keys())
    states_with_transitions = set()
    
    for state_name, transitions in table.items():
        if transitions is None:
            # This is a halt state (no transitions defined)
            continue
            
        parsed_transitions = _parse_state_transitions(state_name, transitions)
        program.extend(parsed_transitions)
        
        if parsed_transitions:
            states_with_transitions.add(state_name)
    
    # Halt states are those defined but with no outgoing transitions
    halt_states = all_states - states_with_transitions
    
    # If no start state specified, try to find one
    if start_state is None and table:
        start_state = list(table.keys())[0]
    
    return {
        'program': program,
        'input': input_string,
        'blank': blank_symbol,
        'start_state': start_state,
        'halt_states': halt_states
    }


def _parse_symbol_key(key):
    """
    Parse a symbol key which may be a single symbol or a list of symbols.
    
    Examples:
        '0' -> ['0']
        ' ' -> [' ']
        [0, 1, '+'] -> ['0', '1', '+']
        '[0,1,+]' -> ['0', '1', '+']  (string representation of list)
    """
    if isinstance(key, list):
        return [str(s) for s in key]
    
    # Check if it's a string that looks like a list: "[0,1,+]"
    key_str = str(key)
    if key_str.startswith('[') and key_str.endswith(']'):
        # Parse the bracketed list
        inner = key_str[1:-1]
        # Split by comma, handling potential spaces
        symbols = [s.strip().strip("'\"") for s in inner.split(',')]
        return symbols
    
    # Single symbol
    return [str(key)]


def _parse_transition_value(state_name, read_symbol, value):
    """
    Parse a transition value into (write, direction, next_state).
    
    Handles various formats:
        'R' -> (None, 'R', state_name)  # Move right, no write, same state
        'L' -> (None, 'L', state_name)  # Move left, no write, same state
        {L: next_state} -> (None, 'L', next_state)
        {R: next_state} -> (None, 'R', next_state)
        {write: x, L: next_state} -> (x, 'L', next_state)
        {write: x, R: next_state} -> (x, 'R', next_state)
        {write: x, L} -> (x, 'L', state_name)  # write and move, same state
    
    Note: When no explicit write is specified, write_symbol is None.
          This indicates the tape cell should remain unchanged.
    """
    # Simple direction only: 'R' or 'L'
    if value in ('R', 'L'):
        return (None, value, state_name)
    
    if isinstance(value, dict):
        write_symbol = value.get('write', None)
        write_symbol = str(write_symbol) if write_symbol is not None else None
        
        # Find direction and next_state
        direction = None
        next_state = state_name  # Default to same state
        
        if 'L' in value:
            direction = 'L'
            if value['L'] is not None:
                next_state = value['L']
        elif 'R' in value:
            direction = 'R'
            if value['R'] is not None:
                next_state = value['R']
        
        if direction is None:
            raise ValueError(f"No direction (L/R) found in transition: {value}")
        
        return (write_symbol, direction, next_state)
    
    raise ValueError(f"Cannot parse transition value: {value}")


def _parse_state_transitions(state_name, transitions):
    """
    Parse all transitions for a given state.
    
    Args:
        state_name: Name of the current state
        transitions: Dict mapping read symbols to transition specs
    
    Returns:
        List of 5-tuples (state_name, read, write, direction, next_state)
    """
    result = []
    
    if not isinstance(transitions, dict):
        return result
    
    for key, value in transitions.items():
        # Parse the symbol(s) this rule applies to
        symbols = _parse_symbol_key(key)
        
        for read_symbol in symbols:
            # Parse the transition for this symbol
            write, direction, next_state = _parse_transition_value(
                state_name, read_symbol, value
            )
            result.append((state_name, read_symbol, write, direction, next_state))
    
    return result


def run_yaml_machine(yaml_string, max_steps=None, verbose=True, start_position=0):
    """
    Parse and run a Turing machine from YAML format.
    
    Convenience function that combines parse_yaml_machine and run_turing_machine.
    
    Args:
        yaml_string: YAML string defining the machine
        max_steps: Maximum steps before stopping (None for unlimited)
        verbose: If True, print each step
        start_position: Override starting head position (default: 0)
    
    Returns:
        Same as run_turing_machine: (tape_dict, steps, ones_count, history, halted)
        Plus 'parsed' key with the parsed machine data
    """
    parsed = parse_yaml_machine(yaml_string)
    
    # Convert input string to tape if provided
    initial_tape = None
    if parsed['input'] is not None:
        initial_tape = string_to_tape(str(parsed['input']), start_position=0)
    
    # Determine halt state (use first one if multiple, or a synthetic one)
    halt_states = parsed['halt_states']
    halt_state = list(halt_states)[0] if halt_states else '__HALT__'
    
    result = run_turing_machine(
        program=parsed['program'],
        halt_state=halt_state,
        initial_state=parsed['start_state'],
        max_steps=max_steps,
        verbose=verbose,
        initial_tape=initial_tape,
        blank_symbol=parsed['blank'],
        start_position=start_position
    )
    
    return result + (parsed,)


def history_to_numpy(history, state_encoding=None, symbol_encoding=None, include_halt_row=True):
    """
    Convert execution history to a numpy array of shape (n_steps, 5).
    
    Args:
        history: List of 5-tuples from run_turing_machine
        state_encoding: Optional dict mapping state names to integers.
                        If None, states are auto-encoded alphabetically.
        symbol_encoding: Optional dict mapping tape symbols to integers.
                         If None, symbols are auto-encoded (sorted).
                         For binary machines, {0: 0, 1: 1} is used automatically.
        include_halt_row: If True (default), adds a final row representing the
                          halt state with [-1, -1, -1, -1, halt_state].
    
    Returns:
        Tuple of (array, state_encoding, symbol_encoding) where:
            - array: numpy array of shape (n_steps, 5) with columns
                     [current_state, read, write, direction, next_state]
            - state_encoding: dict mapping state names to integers
            - symbol_encoding: dict mapping tape symbols to integers
    
    Encoding:
        - States are encoded as integers (e.g., A=0, B=1, C=2, ...)
        - Symbols are encoded as integers (auto-detected from history)
        - Direction: L=0, R=1
        - Write column: -1 indicates no write (tape unchanged)
        - Halt row uses -1 for current_state, read, write, direction
    """
    if not history:
        return np.array([]).reshape(0, 5), {}, {}
    
    # Build state encoding if not provided
    if state_encoding is None:
        all_states = set()
        for curr, _, _, _, nxt in history:
            all_states.add(curr)
            all_states.add(nxt)
        state_encoding = {state: i for i, state in enumerate(sorted(all_states, key=str))}
    
    # Build symbol encoding if not provided
    if symbol_encoding is None:
        all_symbols = set()
        for _, read, write, _, _ in history:
            all_symbols.add(read)
            if write is not None:  # None means no write, don't add to encoding
                all_symbols.add(write)
        # Sort symbols for consistent encoding
        symbol_encoding = {sym: i for i, sym in enumerate(sorted(all_symbols, key=str))}
    
    # Direction encoding
    dir_encoding = {'L': 0, 'R': 1}
    
    # Determine array size and dtype (use int16 to handle larger alphabets)
    n_steps = len(history)
    total_rows = n_steps + 1 if include_halt_row else n_steps
    arr = np.zeros((total_rows, 5), dtype=np.int16)
    
    for i, (curr_state, read, write, direction, next_state) in enumerate(history):
        arr[i, 0] = state_encoding[curr_state]
        arr[i, 1] = symbol_encoding[read]
        arr[i, 2] = symbol_encoding[write] if write is not None else -1  # -1 for no write
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
    
    return arr, state_encoding, symbol_encoding


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
        arr, _, _ = history_to_numpy(history, state_encoding, include_halt_row=add_halt_row)
        runs.append(arr)
        halted_flags.append(halted)
    
    return runs, halted_flags, state_encoding


def simulate_random_adders(n_runs, num_range=(0, 255), max_steps=10000, 
                           include_halt_row=True, seed=None, verbose=False):
    """
    Simulate the binary adder machine on random addition problems.
    
    Args:
        n_runs: Number of addition simulations to run
        num_range: Tuple (min, max) for random number range (inclusive)
                   Default (0, 255) for 8-bit numbers
        max_steps: Maximum steps before forced stop (default 10000)
        include_halt_row: If True (default), adds a final halt state row
                          only when the machine actually halts
        seed: Optional random seed for reproducibility
        verbose: If True, print progress for each run
    
    Returns:
        Dict with keys:
            - 'runs': List of numpy arrays (execution histories)
            - 'halted_flags': List of booleans indicating if each run halted
            - 'inputs': List of tuples (a, b) - the decimal input numbers
            - 'results': List of computed results (decimal, or None if didn't halt)
            - 'correct': List of booleans indicating if result matches a+b
            - 'state_encoding': Dict mapping state names to integers
            - 'symbol_encoding': Dict mapping symbols to integers
    
    Example:
        results = simulate_random_adders(100, num_range=(0, 1000), seed=42)
        print(f"Accuracy: {sum(results['correct'])/len(results['correct'])*100:.1f}%")
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Parse the binary adder machine
    parsed = parse_yaml_machine(BINARY_ADDER_YAML)
    program = parsed['program']
    blank_symbol = parsed['blank']
    start_state = parsed['start_state']
    halt_state = list(parsed['halt_states'])[0] if parsed['halt_states'] else '__HALT__'
    
    # Build encodings from the program
    all_states = set()
    all_symbols = set()
    for curr, read, write, _, nxt in program:
        all_states.add(curr)
        all_states.add(nxt)
        all_symbols.add(read)
        all_symbols.add(write)
    all_states.add(halt_state)
    all_symbols.add(blank_symbol)
    
    state_encoding = {state: i for i, state in enumerate(sorted(all_states, key=str))}
    symbol_encoding = {sym: i for i, sym in enumerate(sorted(all_symbols, key=str))}
    
    runs = []
    halted_flags = []
    inputs = []
    results = []
    correct = []
    
    min_num, max_num = num_range
    
    for run_idx in range(n_runs):
        # Generate random numbers
        a = np.random.randint(min_num, max_num + 1)
        b = np.random.randint(min_num, max_num + 1)
        inputs.append((a, b))
        
        # Convert to binary strings and create input tape
        a_binary = bin(a)[2:]  # Remove '0b' prefix
        b_binary = bin(b)[2:]
        input_string = f"{a_binary}+{b_binary}"
        initial_tape = string_to_tape(input_string, start_position=0)
        
        if verbose:
            print(f"Run {run_idx + 1}/{n_runs}: {a} + {b} = {a + b} "
                  f"({a_binary} + {b_binary})")
        
        # Run the adder
        tape, steps, _, history, halted = run_turing_machine(
            program,
            halt_state=halt_state,
            initial_state=start_state,
            max_steps=max_steps,
            verbose=False,
            initial_tape=initial_tape,
            blank_symbol=blank_symbol,
            start_position=0
        )
        
        halted_flags.append(halted)
        
        # Parse the result from the tape
        result_decimal = None
        is_correct = False
        
        if halted:
            result_string = tape_to_string(tape, blank_symbol=blank_symbol).strip()
            # The result format is "sum second_number"
            parts = result_string.split()
            if parts:
                try:
                    result_decimal = int(parts[0], 2)
                    is_correct = (result_decimal == a + b)
                except ValueError:
                    pass
        
        results.append(result_decimal)
        correct.append(is_correct)
        
        if verbose:
            status = "OK" if is_correct else ("FAIL" if halted else "TIMEOUT")
            print(f"  -> Result: {result_decimal}, Expected: {a + b}, Status: {status}")
        
        # Convert history to numpy
        add_halt_row = include_halt_row and halted
        
        # Update symbol encoding with any new symbols from this run
        for _, read, write, _, _ in history:
            if read not in symbol_encoding:
                symbol_encoding[read] = len(symbol_encoding)
            if write is not None and write not in symbol_encoding:
                symbol_encoding[write] = len(symbol_encoding)
        
        arr, _, _ = history_to_numpy(
            history, state_encoding, symbol_encoding, include_halt_row=add_halt_row
        )
        runs.append(arr)
    
    return {
        'runs': runs,
        'halted_flags': halted_flags,
        'inputs': inputs,
        'results': results,
        'correct': correct,
        'state_encoding': state_encoding,
        'symbol_encoding': symbol_encoding
    }


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
    arr, encoding, _ = history_to_numpy(history, state_encoding, include_halt_row=include_halt_row)
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


# Binary Adder Machine
# Adds two binary numbers: given input "a+b", produces "c b" where c = a+b
# Example: '11+1' => '100 1' (3+1=4)
BINARY_ADDER_YAML = """
input: '0+0'
blank: ' '
start state: right
table:
  # Start at the second number's rightmost digit.
  right:
    [0,1,+]: R
    ' ': {L: read}

  # Add each digit from right to left:
  # read the current digit of the second number,
  read:
    0: {write: c, L: have0}
    1: {write: c, L: have1}
    +: {write: ' ', L: rewrite}
  # and add it to the next place of the first number,
  # marking the place (using O or I) as already added.
  have0:
    [0,1]: L
    +: {L: add0}
  have1:
    [0,1]: L
    +: {L: add1}
  add0:
    [0,' ']: {write: O, R: back0}
    1: {write: I, R: back0}
    [O,I]: L
  add1:
    [0,' ']: {write: I, R: back1}
    1: {write: O, L: carry}
    [O,I]: L
  carry:
    [0,' ']: {write: 1, R: back1}
    1: {write: 0, L}
  # Then, restore the current digit, and repeat with the next digit.
  back0:
    [0,1,O,I,+]: R
    c: {write: 0, L: read}
  back1:
    [0,1,O,I,+]: R
    c: {write: 1, L: read}

  # Finish: rewrite place markers back to 0s and 1s.
  rewrite:
    O: {write: 0, L}
    I: {write: 1, L}
    [0,1]: L
    ' ': {R: done}
  done:
"""


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
    history_array, state_encoding, symbol_encoding = history_to_numpy(history, include_halt_row=halted)
    
    print(f"Array shape: {history_array.shape}")
    print(f"State encoding: {state_encoding}")
    print(f"Symbol encoding: {symbol_encoding}")
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
    
    # Demonstrate YAML machine parsing and extended alphabet
    print(f"\n{'=' * 60}")
    print("YAML MACHINE DEMO: BINARY ADDER")
    print("=" * 60)
    
    # Use a specific input for demo (override the module-level default)
    demo_yaml = BINARY_ADDER_YAML.replace("input: '0+0'", "input: '1101011+11001'")
    
    print("\nParsing YAML machine definition...")
    parsed = parse_yaml_machine(demo_yaml)
    print(f"Input: {parsed['input']}")
    print(f"Blank symbol: {repr(parsed['blank'])}")
    print(f"Start state: {parsed['start_state']}")
    print(f"Halt states: {parsed['halt_states']}")
    print(f"Number of transition rules: {len(parsed['program'])}")
    
    print("\nFirst 10 transition rules:")
    for i, rule in enumerate(parsed['program'][:10]):
        print(f"  {rule}")
    
    print("\nRunning binary adder: 1101011 + 11001 = ?")
    print("(1101011 binary = 107 decimal, 11001 binary = 25 decimal)")
    print("Expected result: 10000100 binary = 132 decimal")
    print("-" * 60)
    
    tape, steps, ones, history, halted, _ = run_yaml_machine(
        demo_yaml, 
        max_steps=1000, 
        verbose=False
    )
    
    print(f"\nMachine halted: {halted}")
    print(f"Steps taken: {steps}")
    result_string = tape_to_string(tape, blank_symbol=' ')
    print(f"Final tape: '{result_string}'")
    print(f"Final tape (stripped): '{result_string.strip()}'")
    
    # Parse the result
    parts = result_string.strip().split()
    if len(parts) >= 1:
        result_binary = parts[0]
        try:
            result_decimal = int(result_binary, 2)
            print(f"Result: {result_binary} binary = {result_decimal} decimal")
        except ValueError:
            print(f"Result: {result_binary}")
    
    # Demonstrate simulate_random_adders
    print(f"\n{'=' * 60}")
    print("RANDOM ADDER SIMULATION")
    print("=" * 60)
    print("\nRunning 20 random additions with numbers in range [0, 255]...")
    
    adder_results = simulate_random_adders(
        n_runs=20,
        num_range=(0, 255),
        max_steps=10000,
        seed=42,
        verbose=False
    )
    
    print(f"\n{'Run':<5} {'A':<6} {'B':<6} {'A+B':<8} {'Result':<8} {'Steps':<8} {'Status'}")
    print("-" * 55)
    
    for i in range(len(adder_results['runs'])):
        a, b = adder_results['inputs'][i]
        expected = a + b
        result = adder_results['results'][i]
        halted = adder_results['halted_flags'][i]
        correct = adder_results['correct'][i]
        steps = adder_results['runs'][i].shape[0]
        
        if correct:
            status = "OK"
        elif halted:
            status = "WRONG"
        else:
            status = "TIMEOUT"
        
        result_str = str(result) if result is not None else "N/A"
        print(f"{i+1:<5} {a:<6} {b:<6} {expected:<8} {result_str:<8} {steps:<8} {status}")
    
    # Summary
    n_correct = sum(adder_results['correct'])
    n_halted = sum(adder_results['halted_flags'])
    n_total = len(adder_results['runs'])
    
    print(f"\n{'=' * 55}")
    print(f"Summary:")
    print(f"  Total runs: {n_total}")
    print(f"  Halted: {n_halted}/{n_total}")
    print(f"  Correct: {n_correct}/{n_total} ({n_correct/n_total*100:.1f}%)")
    print(f"  State encoding: {len(adder_results['state_encoding'])} states")
    print(f"  Symbol encoding: {len(adder_results['symbol_encoding'])} symbols")