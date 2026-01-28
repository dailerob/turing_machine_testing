"""
Generative Dense Chain (GDC) - A training-free Hidden Markov Model variant.

The model assumes:
- Emission: P(obs|state) = 1 if obs == state exactly, 0 otherwise
- Transition: Sequential bias (alpha) to next state, uniform otherwise
- Initial distribution: Uniform
"""

import numpy as np
from typing import Union, List, Tuple, Optional


class GenerativeDenseChain:
    """
    A training-free generative model where hidden states are defined by
    an input sequence, with assumed transition and emission distributions.
    """
    
    def __init__(self, sequences: Union[np.ndarray, List[np.ndarray]], alpha: float = 0.9):
        """
        Initialize a Generative Dense Chain.
        
        Parameters
        ----------
        sequences : np.ndarray or List[np.ndarray]
            Either a single n*k array or a list of arrays. Each row becomes a hidden state.
            If a list is provided, arrays are concatenated and terminal states (last state
            of each array) diffuse uniformly rather than with sequential bias.
        alpha : float
            Default transition probability to next sequential state. The remaining
            probability (1-alpha) is distributed uniformly among other states.
        """
        if isinstance(sequences, np.ndarray):
            self.states = sequences.copy()
            self.terminal_mask = np.zeros(len(sequences), dtype=bool)
            self.terminal_mask[-1] = True
        else:
            # List of arrays - concatenate and track terminal states
            self.states = np.vstack(sequences)
            self.terminal_mask = np.zeros(len(self.states), dtype=bool)
            cumsum = 0
            for seq in sequences:
                cumsum += len(seq)
                self.terminal_mask[cumsum - 1] = True
        
        self.n_states = len(self.states)
        self.k = self.states.shape[1]
        self.alpha = alpha
    
    def _transition(self, dist: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        """
        Apply transition to a state distribution.
        
        The transition matrix has:
        - P(next state | current non-terminal state) = alpha
        - P(other states | current non-terminal state) = (1-alpha)/(n-1) each
        - P(any state | current terminal state) = 1/n (uniform diffusion)
        
        Parameters
        ----------
        dist : np.ndarray
            Current state distribution (length n_states).
        alpha : float, optional
            Transition probability to next state. Uses default if not specified.
            
        Returns
        -------
        np.ndarray
            New state distribution after transition.
        """
        if alpha is None:
            alpha = self.alpha
        
        n = self.n_states
        
        # Handle edge case of single state
        if n == 1:
            return dist.copy()
        
        beta = (1 - alpha) / (n - 1)
        
        # Separate terminal and non-terminal contributions
        non_terminal_dist = dist * (~self.terminal_mask).astype(float)
        terminal_prob = (dist * self.terminal_mask.astype(float)).sum()
        non_terminal_sum = non_terminal_dist.sum()
        
        # Roll for sequential transition from non-terminal states
        rolled = np.roll(non_terminal_dist, 1)
        
        # Combine contributions:
        # - rolled * (alpha - beta): extra probability from sequential transition
        # - beta * non_terminal_sum: base uniform contribution from non-terminal states
        # - terminal_prob / n: uniform contribution from terminal states
        new_dist = rolled * (alpha - beta) + beta * non_terminal_sum + terminal_prob / n
        
        return new_dist
    
    def _emission_likelihood(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute emission likelihood for each state given an observation.
        
        Parameters
        ----------
        observation : np.ndarray
            Single observation vector (length k).
            
        Returns
        -------
        np.ndarray
            Likelihood for each state (1 if exact match, 0 otherwise).
        """
        return (self.states == observation).all(axis=1).astype(float)
    
    def forward_pass(
        self,
        observations: np.ndarray,
        alpha: Optional[float] = None,
        return_history: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform forward pass (filtering) given observations.
        
        Parameters
        ----------
        observations : np.ndarray
            T*k array of observations, where T is the number of timesteps.
        alpha : float, optional
            Transition probability. Uses default if not specified.
        return_history : bool, default False
            If True, also return the state distribution at each timestep.
            
        Returns
        -------
        final_dist : np.ndarray
            State distribution after the last observation (length n_states).
        history : np.ndarray, optional
            T*n_states array of state distributions at each timestep.
            Only returned if return_history=True.
        """
        if alpha is None:
            alpha = self.alpha
        
        # Start with uniform distribution
        dist = np.ones(self.n_states) / self.n_states
        
        if return_history:
            history = []
        
        for t, obs in enumerate(observations):
            # Apply transition (skip for first observation)
            if t > 0:
                dist = self._transition(dist, alpha)
            
            # Apply emission likelihood
            likelihood = self._emission_likelihood(obs)
            dist = dist * likelihood
            
            # Normalize, or return to uniform if no states match
            total = dist.sum()
            if total > 0:
                dist = dist / total
            else:
                # No state matches observation - return to uniform
                dist = np.ones(self.n_states) / self.n_states
            
            if return_history:
                history.append(dist.copy())
        
        if return_history:
            return dist, np.array(history)
        return dist
    
    def forecast(
        self,
        state_dist: np.ndarray,
        n_steps: int,
        alpha: Optional[float] = None,
        return_history: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Project state distribution forward using the transition matrix.
        
        Parameters
        ----------
        state_dist : np.ndarray
            Current state distribution (length n_states).
        n_steps : int
            Number of steps to project forward.
        alpha : float, optional
            Transition probability. Uses default if not specified.
        return_history : bool, default False
            If True, return distribution at each projected step.
            
        Returns
        -------
        final_dist : np.ndarray
            State distribution after n_steps transitions.
        history : np.ndarray, optional
            n_steps*n_states array of state distributions at each step.
            Only returned if return_history=True.
        """
        if alpha is None:
            alpha = self.alpha
        
        dist = state_dist.copy()
        
        if return_history:
            history = []
        
        for _ in range(n_steps):
            dist = self._transition(dist, alpha)
            if return_history:
                history.append(dist.copy())
        
        if return_history:
            return dist, np.array(history)
        return dist
    
    def greedy_sample(
        self,
        state_dist: np.ndarray,
        mask: Optional[np.ndarray] = None,
        conditional: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Return the most likely state vector, optionally grouping by masked elements.
        
        When a mask is provided, states are grouped by their values at the unmasked
        (mask=1) positions. The group with highest total probability is selected,
        and masked-out (mask=0) positions are set to np.nan.
        
        When a conditional is provided, states are first filtered to only those
        that match the non-NaN values in the conditional array. If no states match,
        the conditional is ignored.
        
        Parameters
        ----------
        state_dist : np.ndarray
            State distribution (length n_states).
        mask : np.ndarray, optional
            k-element binary mask. 1 = consider this element for grouping and output,
            0 = ignore for grouping, output as np.nan.
            If None, returns the full most likely state vector.
        conditional : np.ndarray, optional
            k-element array with np.nan for "don't care" positions and specific
            values for positions that must match. States are filtered to only
            those matching the non-NaN values. If no states match, ignored.
            
        Returns
        -------
        np.ndarray
            Most likely state vector (length k). Masked-out elements are np.nan.
        """
        # Apply conditional filtering if provided
        states = self.states
        dist = state_dist
        
        if conditional is not None:
            conditional = np.asarray(conditional, dtype=float)
            # Find which positions have non-NaN values (constraints)
            constraint_mask = ~np.isnan(conditional)
            
            if constraint_mask.any():
                # Find states that match the conditional values at constrained positions
                matches = (states[:, constraint_mask] == conditional[constraint_mask]).all(axis=1)
                
                # Only apply filtering if at least one state matches
                if matches.any():
                    matching_indices = np.where(matches)[0]
                    states = states[matching_indices]
                    dist = state_dist[matching_indices]
        
        if mask is None:
            # No grouping, return most likely state
            best_idx = np.argmax(dist)
            return states[best_idx].astype(float).copy()
        
        mask = np.asarray(mask, dtype=bool)
        
        # Group states by their values at unmasked (mask=1) positions
        groups = {}
        for idx, state in enumerate(states):
            key = tuple(state[mask])
            if key not in groups:
                groups[key] = 0.0
            groups[key] += dist[idx]
        
        # Find group with highest total probability
        best_key = max(groups.keys(), key=lambda k: groups[k])
        
        # Build result with NaN in masked-out positions
        result = np.full(self.k, np.nan)
        result[mask] = np.array(best_key)
        
        return result
    
    def random_sample(
        self,
        state_dist: np.ndarray,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Randomly sample a state vector, optionally grouping by masked elements.
        
        When a mask is provided, states are grouped by their values at the unmasked
        (mask=1) positions. A group is sampled proportional to its total probability,
        and masked-out (mask=0) positions are set to np.nan.
        
        Parameters
        ----------
        state_dist : np.ndarray
            State distribution (length n_states).
        mask : np.ndarray, optional
            k-element binary mask. 1 = consider this element for grouping and output,
            0 = ignore for grouping, output as np.nan.
            If None, samples a full state vector.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
            
        Returns
        -------
        np.ndarray
            Sampled state vector (length k). Masked-out elements are np.nan.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if mask is None:
            # No grouping, sample directly from distribution
            # Handle unnormalized or zero distributions
            total = state_dist.sum()
            if total > 0:
                probs = state_dist / total
            else:
                probs = np.ones(self.n_states) / self.n_states
            
            idx = rng.choice(self.n_states, p=probs)
            return self.states[idx].astype(float).copy()
        
        mask = np.asarray(mask, dtype=bool)
        
        # Group states by their values at unmasked (mask=1) positions
        groups = {}
        for idx, state in enumerate(self.states):
            key = tuple(state[mask])
            if key not in groups:
                groups[key] = 0.0
            groups[key] += state_dist[idx]
        
        # Sample group proportional to total probability
        keys = list(groups.keys())
        probs = np.array([groups[k] for k in keys])
        
        # Normalize probabilities
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones(len(keys)) / len(keys)
        
        chosen_idx = rng.choice(len(keys), p=probs)
        chosen_key = keys[chosen_idx]
        
        # Build result with NaN in masked-out positions
        result = np.full(self.k, np.nan)
        result[mask] = np.array(chosen_key)
        
        return result


if __name__ == "__main__":
    # Simple demonstration
    print("=== Generative Dense Chain Demo ===\n")
    
    # Create a simple sequence
    sequence = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    
    gdc = GenerativeDenseChain(sequence, alpha=0.8)
    print(f"States:\n{gdc.states}\n")
    print(f"Terminal mask: {gdc.terminal_mask}\n")
    
    # Forward pass with observations
    observations = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
    ])
    
    final_dist, history = gdc.forward_pass(observations, return_history=True)
    print(f"Observations:\n{observations}\n")
    print(f"Final distribution: {final_dist}")
    print(f"History:\n{history}\n")
    
    # Forecast
    forecast_dist = gdc.forecast(final_dist, n_steps=2)
    print(f"Forecast (2 steps): {forecast_dist}\n")
    
    # Greedy sample
    greedy = gdc.greedy_sample(final_dist)
    print(f"Greedy sample (no mask): {greedy}")
    
    greedy_masked = gdc.greedy_sample(final_dist, mask=[1, 0])
    print(f"Greedy sample (mask=[1,0]): {greedy_masked}\n")
    
    # Random sample
    rng = np.random.default_rng(42)
    random = gdc.random_sample(final_dist, rng=rng)
    print(f"Random sample (no mask): {random}")
    
    random_masked = gdc.random_sample(final_dist, mask=[1, 0], rng=rng)
    print(f"Random sample (mask=[1,0]): {random_masked}\n")
    
    # Demo with multiple sequences
    print("=== Multiple Sequences Demo ===\n")
    
    seq1 = np.array([[0, 0], [0, 1]])
    seq2 = np.array([[1, 0], [1, 1], [2, 0]])
    
    gdc_multi = GenerativeDenseChain([seq1, seq2], alpha=0.8)
    print(f"States:\n{gdc_multi.states}\n")
    print(f"Terminal mask: {gdc_multi.terminal_mask}")
    print("(Terminal states diffuse uniformly instead of with sequential bias)")
