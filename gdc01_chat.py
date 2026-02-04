"""
GDC-01 Interactive Chat - A character-level language model using Generative Dense Chains.

This script loads training data from gdc01_training_data.txt and creates an interactive
chat where GDC-01 responds using greedy sampling with state updates based on emission
likelihood.
"""

import numpy as np
from typing import List, Tuple, Optional
from generative_dense_chain import GenerativeDenseChain

# Delimiter used in training data file
CONVERSATION_DELIMITER = "\n===CONVERSATION===\n"

# Special tokens for detecting turn boundaries
USER_PREFIX = "User:"
GDC01_PREFIX = "GDC-01:"


class TextEncoder:
    """
    Handles conversion between text (strings) and integer arrays.
    Uses character-level encoding with a dynamic vocabulary.
    
    Unknown characters (not seen during fit) are encoded with a special code (-1)
    that won't match any state, causing the model to fall back to uniform distribution.
    """
    
    # Special code for unknown characters (won't match any valid state)
    UNKNOWN_CODE = -1
    
    def __init__(self):
        self.char_to_int: dict = {}
        self.int_to_char: dict = {}
        self.vocab_size: int = 0
    
    def fit(self, texts: List[str]) -> 'TextEncoder':
        """
        Build vocabulary from a list of texts.
        
        Parameters
        ----------
        texts : List[str]
            List of text strings to build vocabulary from.
            
        Returns
        -------
        self
        """
        # Collect all unique characters
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        
        # Sort for deterministic ordering
        sorted_chars = sorted(all_chars)
        
        # Build mappings
        self.char_to_int = {char: i for i, char in enumerate(sorted_chars)}
        self.int_to_char = {i: char for i, char in enumerate(sorted_chars)}
        self.vocab_size = len(sorted_chars)
        
        return self
    
    def encode(self, text: str) -> np.ndarray:
        """
        Convert text to integer array.
        
        Unknown characters are encoded as UNKNOWN_CODE (-1), which won't match
        any state in the GDC model, causing it to fall back to uniform distribution.
        
        Parameters
        ----------
        text : str
            Text string to encode.
            
        Returns
        -------
        np.ndarray
            Array of shape (len(text), 1) with integer character codes.
        """
        encoded = np.array(
            [[self.char_to_int.get(char, self.UNKNOWN_CODE)] for char in text],
            dtype=np.int32
        )
        return encoded
    
    def decode(self, encoded: np.ndarray) -> str:
        """
        Convert integer array back to text.
        
        Parameters
        ----------
        encoded : np.ndarray
            Array of shape (n, 1) with integer character codes.
            
        Returns
        -------
        str
            Decoded text string.
        """
        # Handle both (n, 1) and (n,) shapes
        if encoded.ndim == 2:
            chars = [self.int_to_char[int(code[0])] for code in encoded]
        else:
            chars = [self.int_to_char[int(code)] for code in encoded]
        return ''.join(chars)
    
    def encode_char(self, char: str) -> int:
        """Encode a single character to its integer code. Returns UNKNOWN_CODE for unknown chars."""
        return self.char_to_int.get(char, self.UNKNOWN_CODE)
    
    def decode_int(self, code: int) -> str:
        """Decode a single integer code to its character."""
        return self.int_to_char.get(code, '')


def load_conversations(filename: str = "gdc01_training_data.txt") -> List[str]:
    """
    Load conversations from a text file into a list.
    
    Parameters
    ----------
    filename : str
        Input filename (relative to script directory or absolute path).
    
    Returns
    -------
    List[str]
        List where each element is a string containing one conversation.
    """
    import os
    
    # Try relative path first, then absolute
    if not os.path.isabs(filename):
        filepath = os.path.join(os.path.dirname(__file__), filename)
    else:
        filepath = filename
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by delimiter
    parts = content.split(CONVERSATION_DELIMITER)
    
    # Filter out header (parts that start with #) and empty strings
    conversations = []
    for part in parts:
        part = part.strip()
        if part and not part.startswith('#'):
            conversations.append(part)
    
    print(f"Loaded {len(conversations)} conversations from {filepath}")
    return conversations


def build_gdc_model(
    conversations: List[str],
    encoder: TextEncoder,
    alpha: float = 0.9,
    theta: float = 0.05,
    beta: float = 0.0,
    transition_type: str = 'self_loop',
    initial_dist: str = 'sequence_starts'
) -> GenerativeDenseChain:
    """
    Build a GDC model from conversations.
    
    Parameters
    ----------
    conversations : List[str]
        List of conversation strings.
    encoder : TextEncoder
        Fitted text encoder.
    alpha : float
        Transition probability to next sequential state.
    theta : float
        Self-loop probability.
    beta : float
        Emission noise probability (probability of emitting a random symbol).
    transition_type : str
        Type of transition structure.
    initial_dist : str
        Initial distribution type.
    
    Returns
    -------
    GenerativeDenseChain
        The constructed GDC model.
    """
    # Encode each conversation as a separate sequence
    sequences = []
    for conv in conversations:
        encoded = encoder.encode(conv)
        sequences.append(encoded)
    
    # Create GDC with list of sequences (tracks terminal/start states)
    gdc = GenerativeDenseChain(
        sequences=sequences,
        alpha=alpha,
        theta=theta,
        beta=beta,
        transition_type=transition_type,
        initial_dist=initial_dist
    )
    
    print(f"Built GDC model with {gdc.n_states} states from {len(sequences)} conversations")
    return gdc


def greedy_generate_with_state_update(
    gdc: GenerativeDenseChain,
    encoder: TextEncoder,
    state_dist: np.ndarray,
    max_chars: int = 500,
    stop_on_user: bool = True
) -> Tuple[str, np.ndarray, bool]:
    """
    Generate text using greedy sampling with state updates on each emission.
    
    Each generated character is treated as an observation, updating the state
    distribution based on emission likelihood.
    
    Parameters
    ----------
    gdc : GenerativeDenseChain
        The GDC model.
    encoder : TextEncoder
        Text encoder for decoding.
    state_dist : np.ndarray
        Current state distribution.
    max_chars : int
        Maximum characters to generate.
    stop_on_user : bool
        If True, stop generation when "User:" is detected.
    
    Returns
    -------
    generated_text : str
        The generated text.
    final_state_dist : np.ndarray
        State distribution after generation.
    stopped_on_user : bool
        True if generation stopped because "User:" was detected.
    """
    generated_chars = []
    current_dist = state_dist.copy()
    stopped_on_user = False
    
    for _ in range(max_chars):
        # Apply transition to get prediction distribution
        pred_dist = gdc._transition(current_dist)
        
        # Greedy sample: get the most likely next character
        sample = gdc.greedy_sample(pred_dist)
        char_code = int(sample[0])
        char = encoder.decode_int(char_code)
        
        if not char:
            # Unknown character, stop generation
            break
        
        generated_chars.append(char)
        
        # Update state distribution based on emission (the generated character is now an observation)
        # This is the key: treat the sample as an observation and update state
        observation = np.array([[char_code]])
        emission_likelihood = gdc._emission_likelihood(observation[0])
        
        # Apply emission to prediction distribution
        current_dist = pred_dist * emission_likelihood
        total = current_dist.sum()
        
        if total > 0:
            current_dist = current_dist / total
        else:
            # No matching state - this shouldn't happen with greedy sampling
            # Fall back to uniform
            current_dist = np.ones(gdc.n_states) / gdc.n_states
        
        # Check if we've generated "User:" (indicating turn boundary)
        if stop_on_user:
            generated_text = ''.join(generated_chars)
            if generated_text.endswith(USER_PREFIX):
                stopped_on_user = True
                # Remove "User:" from output since user will type their message
                generated_chars = generated_chars[:-len(USER_PREFIX)]
                break
    
    return ''.join(generated_chars), current_dist, stopped_on_user


def condition_on_text(
    gdc: GenerativeDenseChain,
    encoder: TextEncoder,
    text: str,
    initial_dist: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Condition the model on observed text (forward pass).
    
    Parameters
    ----------
    gdc : GenerativeDenseChain
        The GDC model.
    encoder : TextEncoder
        Text encoder.
    text : str
        Text to condition on.
    initial_dist : np.ndarray, optional
        Initial state distribution. If None, uses model's default.
    
    Returns
    -------
    np.ndarray
        State distribution after conditioning on the text.
    """
    if not text:
        if initial_dist is not None:
            return initial_dist
        return gdc._get_initial_distribution()
    
    # Encode text to observations
    observations = encoder.encode(text)
    
    # Forward pass to get conditioned state distribution
    if initial_dist is not None:
        # Manual forward pass starting from given distribution
        dist = initial_dist.copy()
        for t, obs in enumerate(observations):
            if t > 0:
                dist = gdc._transition(dist)
            
            # Apply emission likelihood
            emission = gdc._emission_likelihood(obs)
            dist = dist * emission
            total = dist.sum()
            
            if total > 0:
                dist = dist / total
            else:
                # No matching state, fall back to uniform
                dist = np.ones(gdc.n_states) / gdc.n_states
        
        return dist
    else:
        return gdc.forward_pass(observations)


def interactive_chat(
    gdc: GenerativeDenseChain,
    encoder: TextEncoder,
    max_response_chars: int = 1000
):
    """
    Run an interactive chat session with GDC-01.
    
    The model generates responses character by character using greedy sampling.
    When it predicts "User:", it stops and waits for user input.
    
    Parameters
    ----------
    gdc : GenerativeDenseChain
        The GDC model.
    encoder : TextEncoder
        Text encoder.
    max_response_chars : int
        Maximum characters per response.
    """
    print("\n" + "=" * 60)
    print("GDC-01 Interactive Chat")
    print("=" * 60)
    print("Type your message and press Enter. Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")
    
    # Start with initial distribution favoring conversation starts
    state_dist = gdc._get_initial_distribution()
    
    # Track conversation context
    conversation_so_far = ""
    
    while True:
        # Get user input
        try:
            user_input = input("User: ").strip()
        except EOFError:
            print("\nGoodbye!")
            break
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Build the context: previous conversation + current user message + GDC-01 prefix
        user_turn = f"User: {user_input}\n"
        gdc01_start = "GDC-01: "
        
        # Add to conversation history
        conversation_so_far += user_turn + gdc01_start
        
        # Condition model on the full context
        state_dist = condition_on_text(gdc, encoder, conversation_so_far)
        
        # Generate response
        print("GDC-01: ", end="", flush=True)
        
        response, state_dist, stopped_on_user = greedy_generate_with_state_update(
            gdc=gdc,
            encoder=encoder,
            state_dist=state_dist,
            max_chars=max_response_chars,
            stop_on_user=True
        )
        
        # Print response (character by character for effect, or all at once)
        print(response)
        
        # Update conversation history
        conversation_so_far += response
        if not conversation_so_far.endswith('\n'):
            conversation_so_far += '\n'
        
        # If generation stopped on "User:", that's expected - loop continues


def main():
    """Main function to load data and start interactive chat."""
    print("=" * 60)
    print("Loading GDC-01 Model")
    print("=" * 60)
    
    # Load conversations
    conversations = load_conversations("gdc01_training_data.txt")
    
    if not conversations:
        print("No conversations loaded. Please run generate_gdc01_training_data.py first.")
        return
    
    # Build encoder from all conversation text
    encoder = TextEncoder()
    encoder.fit(conversations)
    print(f"Vocabulary size: {encoder.vocab_size} characters")
    
    # Build GDC model
    gdc = build_gdc_model(
        conversations=conversations,
        encoder=encoder,
        alpha=0.9,      # High sequential transition probability
        theta=0.05,     # Small self-loop for repetition
        transition_type='self_loop',
        initial_dist='sequence_starts'  # Start at conversation beginnings
    )
    
    # Start interactive chat
    interactive_chat(gdc, encoder)


if __name__ == "__main__":
    main()
