"""
Generative Dense Chain (GDC) - A training-free Hidden Markov Model variant.

The model assumes:
- Emission: P(obs|state) configurable via beta parameter:
  - beta=0 (default): Deterministic, P(obs|state) = 1 if obs == state, 0 otherwise
  - beta>0: Noisy, P(obs|state) = (1-beta) if obs==state else 0, plus beta/V uniform
    where V is the vocabulary size (number of unique states)
- Transition: Configurable (see transition_type parameter)
- Initial distribution: Uniform (over all states or sequence starts)

Transition Types (all no-wrap: state 0 gets no inflow from the chain; wrap mass goes to state 0):
-----------------
1. 'sequential' (default):
   - P(next state | non-terminal) = alpha, (1-alpha)/(n-1) diffuse; no wrap.
   - P(any state | terminal) = 1/n (uniform).

2. 'self_loop':
   - P(same state | any state) = theta (self-loop)
   - P(next state | non-terminal) = alpha, rest diffuse; no wrap.
   - P(other states | terminal) = (1-theta)/(n-1) each.

3. 'self_loop_two_step':
   - Self-loop (theta), next (alpha), two ahead (gamma), rest diffuse; no wrap.
   - At pre-terminal/terminal, transitions diffuse as in (1) and (2).
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Literal

# Type alias for transition types
TransitionType = Literal['sequential', 'self_loop', 'self_loop_two_step']


class GenerativeDenseChain:
    """
    A training-free generative model where hidden states are defined by
    an input sequence, with assumed transition and emission distributions.
    """
    
    def __init__(
        self,
        sequences: Union[np.ndarray, List[np.ndarray]],
        alpha: float = 0.8,
        theta: float = 0.1,
        gamma: float = 0.0,
        beta: float = 0.0,
        transition_type: TransitionType = 'sequential',
        initial_dist: str = 'uniform',
        partial_match: bool = False
    ):
        """
        Initialize a Generative Dense Chain.
        
        Parameters
        ----------
        sequences : np.ndarray or List[np.ndarray]
            Either a single n*k array or a list of arrays. Each row becomes a hidden state.
            If a list is provided, arrays are concatenated and terminal states (last state
            of each array) diffuse uniformly rather than with sequential bias.
        alpha : float, default 0.8
            Transition probability to next sequential state.
        theta : float, default 0.1
            Self-loop probability (used when transition_type='self_loop' or 'self_loop_two_step').
            For 'self_loop': alpha + theta <= 1. For 'self_loop_two_step': alpha + theta + gamma <= 1.
        gamma : float, default 0.0
            Transition probability to state two steps ahead (only for transition_type='self_loop_two_step').
            At sequence end (pre-terminal and terminal), transitions diffuse as in original.
        beta : float, default 0.0
            Emission noise probability.
            - When partial_match=False:
              - beta=0: Deterministic emission (P(obs|state)=1 if match, 0 otherwise)
              - beta>0: P(obs|state) = (1-beta) + beta/V if match, beta/V otherwise
              where V is the vocabulary size.
            - When partial_match=True:
              - P(obs|state) = (1-beta) * (m/k) + beta/n
              where m is the number of matching elements, k is vector length, n is number of states.
              beta acts as a floor probability for complete mismatches.
        transition_type : str, default 'sequential'
            Type of transition structure:
            - 'sequential': alpha to next, (1-alpha) diffuses
            - 'self_loop': theta to self, alpha to next, rest diffuses
            - 'self_loop_two_step': theta to self, alpha to next, gamma to two ahead, rest diffuses
        initial_dist : str, default 'uniform'
            Initial distribution type:
            - 'uniform': Uniform over all states.
            - 'sequence_starts': Uniform over the first state of each sequence.
        partial_match : bool, default False
            If True, emission likelihood is based on the proportion of matching elements
            rather than requiring exact matches. Enables soft matching where states with
            more matching elements get higher likelihood.
        """
        if initial_dist not in ('uniform', 'sequence_starts'):
            raise ValueError(f"initial_dist must be 'uniform' or 'sequence_starts', got '{initial_dist}'")
        
        if not 0 <= beta <= 1:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        
        if transition_type not in ('sequential', 'self_loop', 'self_loop_two_step'):
            raise ValueError(f"transition_type must be 'sequential', 'self_loop', or 'self_loop_two_step', got '{transition_type}'")
        
        if transition_type == 'self_loop' and alpha + theta > 1:
            raise ValueError(f"alpha + theta must be <= 1, got {alpha} + {theta} = {alpha + theta}")
        
        if transition_type == 'self_loop_two_step' and alpha + theta + gamma > 1:
            raise ValueError(f"alpha + theta + gamma must be <= 1, got {alpha} + {theta} + {gamma} = {alpha + theta + gamma}")
        
        if isinstance(sequences, np.ndarray):
            self.states = sequences.copy()
            self.terminal_mask = np.zeros(len(sequences), dtype=bool)
            self.terminal_mask[-1] = True
            self.start_mask = np.zeros(len(sequences), dtype=bool)
            self.start_mask[0] = True
        else:
            # List of arrays - concatenate and track terminal/start states
            self.states = np.vstack(sequences)
            self.terminal_mask = np.zeros(len(self.states), dtype=bool)
            self.start_mask = np.zeros(len(self.states), dtype=bool)
            cumsum = 0
            for seq in sequences:
                self.start_mask[cumsum] = True
                cumsum += len(seq)
                self.terminal_mask[cumsum - 1] = True
        
        self.n_states = len(self.states)
        self.k = self.states.shape[1]
        self.alpha = alpha
        self.theta = theta
        self.gamma = gamma
        self.beta = beta
        self.transition_type = transition_type
        self.initial_dist = initial_dist
        self.partial_match = partial_match
        
        # Build state index for O(1) emission lookups (exact match)
        self._state_to_indices: dict[tuple, list[int]] = {}
        for i, row in enumerate(self.states):
            key = tuple(row)
            if key not in self._state_to_indices:
                self._state_to_indices[key] = []
            self._state_to_indices[key].append(i)
        
        # Vocabulary size for noisy emission (beta > 0)
        self.vocab_size = len(self._state_to_indices)
        
        # Build position-value indices for O(k) partial match lookups
        # _position_value_to_indices[pos][value] = list of state indices with that value at pos
        if partial_match:
            self._position_value_to_indices: List[dict] = []
            for pos in range(self.k):
                pos_dict: dict = {}
                for i, row in enumerate(self.states):
                    val = row[pos]
                    if val not in pos_dict:
                        pos_dict[val] = []
                    pos_dict[val].append(i)
                self._position_value_to_indices.append(pos_dict)
    
    def _transition(
        self,
        dist: np.ndarray,
        alpha: Optional[float] = None,
        theta: Optional[float] = None,
        gamma: Optional[float] = None,
        transition_type: Optional[TransitionType] = None
    ) -> np.ndarray:
        """
        Apply transition to a state distribution.
        
        Dispatches to the appropriate transition implementation based on
        transition_type. Override parameters take precedence over instance defaults.
        
        Parameters
        ----------
        dist : np.ndarray
            Current state distribution (length n_states).
        alpha : float, optional
            Transition probability to next state. Uses instance default if not specified.
        theta : float, optional
            Self-loop probability. Uses instance default if not specified.
        gamma : float, optional
            Transition probability to state two steps ahead. Uses instance default if not specified.
        transition_type : str, optional
            Transition structure type. Uses instance default if not specified.
            
        Returns
        -------
        np.ndarray
            New state distribution after transition.
        """
        # Use instance defaults for unspecified parameters
        if alpha is None:
            alpha = self.alpha
        if theta is None:
            theta = self.theta
        if gamma is None:
            gamma = self.gamma
        if transition_type is None:
            transition_type = self.transition_type
        
        n = self.n_states
        
        # Handle edge case of single state (applies to all transition types)
        if n == 1:
            return dist.copy()
        
        # Dispatch to appropriate transition implementation
        if transition_type == 'sequential':
            return self._transition_sequential(dist, alpha)
        elif transition_type == 'self_loop':
            return self._transition_self_loop(dist, alpha, theta)
        elif transition_type == 'self_loop_two_step':
            return self._transition_self_loop_two_step(dist, alpha, theta, gamma)
        else:
            raise ValueError(f"Unknown transition type: {transition_type}")
    
    def _transition_sequential(self, dist: np.ndarray, alpha: float) -> np.ndarray:
        """
        Sequential transition (no-wrap): alpha to next state, (1-alpha) diffuses uniformly.
        State 0 gets no inflow from the chain; mass that would wrap is added to state 0.
        """
        n = self.n_states
        diffuse_rate = (1 - alpha) / (n - 1)

        non_terminal_dist = dist * (~self.terminal_mask).astype(float)
        terminal_prob = (dist * self.terminal_mask.astype(float)).sum()
        non_terminal_sum = non_terminal_dist.sum()
        shifted = np.zeros(n)
        shifted[1:n] = non_terminal_dist[: n - 1]
        last_nt_idx = np.where(~self.terminal_mask)[0][-1]
        wrap_to_zero = np.zeros(n)
        #wrap_to_zero[0] = (alpha - diffuse_rate) * non_terminal_dist[last_nt_idx]

        new_dist = (alpha - diffuse_rate) * shifted + diffuse_rate * non_terminal_sum + terminal_prob / n + wrap_to_zero
        return new_dist
    
    def _transition_self_loop(self, dist: np.ndarray, alpha: float, theta: float) -> np.ndarray:
        """
        Self-loop transition (no-wrap): theta to self, alpha to next, rest diffuse.
        State 0 gets no inflow from the chain; wrap mass is added to state 0.
        """
        n = self.n_states

        if n == 2:
            return theta * dist + (1 - theta) * np.roll(dist, 1)

        gamma_diff = 1 - alpha - theta
        diffuse_nt = gamma_diff / (n - 2)
        diffuse_t = (1 - theta) / (n - 1)

        non_terminal_dist = dist * (~self.terminal_mask).astype(float)
        terminal_dist = dist * self.terminal_mask.astype(float)
        non_terminal_sum = non_terminal_dist.sum()
        terminal_sum = terminal_dist.sum()
        shifted = np.zeros(n)
        shifted[1:n] = non_terminal_dist[: n - 1]
        last_nt_idx = np.where(~self.terminal_mask)[0][-1]

        self_loop = theta * dist
        sequential = alpha * shifted
        wrap_to_zero = np.zeros(n)
        #wrap_to_zero[0] = alpha * non_terminal_dist[last_nt_idx]
        nt_diffusion = diffuse_nt * non_terminal_sum - diffuse_nt * non_terminal_dist - diffuse_nt * shifted
        nt_diffusion[0] -= diffuse_nt * non_terminal_dist[last_nt_idx]
        t_diffusion = diffuse_t * terminal_sum - diffuse_t * terminal_dist

        return self_loop + sequential + nt_diffusion + t_diffusion + wrap_to_zero
    
    def _transition_self_loop_two_step(self, dist: np.ndarray, alpha: float, theta: float, gamma: float) -> np.ndarray:
        """
        Self-loop + two-step (no-wrap): theta to self, alpha to next, gamma to two ahead, rest diffuse.
        At pre-terminal and terminal states, transition diffuses as in original types.
        """
        n = self.n_states

        if n == 2:
            return theta * dist + (1 - theta) * np.roll(dist, 1)

        terminal_dist = dist * self.terminal_mask.astype(float)
        terminal_sum = terminal_dist.sum()
        diffuse_t = (1 - theta) / (n - 1)
        pre_terminal_mask = np.roll(self.terminal_mask, -1)
        pre_terminal_dist = dist * (~self.terminal_mask).astype(float) * pre_terminal_mask.astype(float)
        pre_terminal_sum = pre_terminal_dist.sum()
        has_two_mask = (~self.terminal_mask) & (~pre_terminal_mask)
        has_two_dist = dist * has_two_mask.astype(float)
        has_two_sum = has_two_dist.sum()

        self_loop = theta * dist
        t_diffusion = diffuse_t * terminal_sum - diffuse_t * terminal_dist

        if pre_terminal_sum > 0 and n >= 3:
            diffuse_pre = (1 - theta - alpha) / (n - 2)
            shifted_pre = np.zeros(n)
            shifted_pre[1:n] = pre_terminal_dist[: n - 1]
            pre_contrib = alpha * shifted_pre + diffuse_pre * (pre_terminal_sum - pre_terminal_dist - shifted_pre)
        else:
            pre_contrib = np.zeros(n)

        if has_two_sum > 0:
            shifted_1_ht = np.zeros(n)
            shifted_1_ht[1:n] = has_two_dist[: n - 1]
            shifted_2_ht = np.zeros(n)
            shifted_2_ht[2:n] = has_two_dist[: n - 2]
            if n == 3:
                diffuse_ht = (1 - theta - alpha - gamma) / n
                has_two_contrib = (theta - diffuse_ht) * has_two_dist + (alpha - diffuse_ht) * shifted_1_ht + (gamma - diffuse_ht) * shifted_2_ht + diffuse_ht * has_two_sum
            else:
                diffuse_ht = (1 - theta - alpha - gamma) / (n - 3)
                has_two_contrib = (theta - diffuse_ht) * has_two_dist + (alpha - diffuse_ht) * shifted_1_ht + (gamma - diffuse_ht) * shifted_2_ht + diffuse_ht * has_two_sum
        else:
            has_two_contrib = np.zeros(n)

        return self_loop + t_diffusion + pre_contrib + has_two_contrib
    
    def _emission_likelihood(
        self,
        observation: np.ndarray,
        beta: Optional[float] = None,
        partial_match: Optional[bool] = None
    ) -> np.ndarray:
        """
        Compute emission likelihood for each state given an observation.
        
        Parameters
        ----------
        observation : np.ndarray
            Single observation vector (length k).
        beta : float, optional
            Emission noise probability. Uses instance default if not specified.
        partial_match : bool, optional
            If True, use partial matching. Uses instance default if not specified.
            
        Returns
        -------
        np.ndarray
            Likelihood for each state (length n_states).
        """
        if beta is None:
            beta = self.beta
        if partial_match is None:
            partial_match = self.partial_match
        
        n = self.n_states
        k = self.k
        
        if partial_match:
            if not self.partial_match:
                raise ValueError(
                    "Cannot use partial_match=True unless the model was "
                    "initialized with partial_match=True (position indices not built)"
                )
            # Count matches per state
            match_counts = np.zeros(n)
            for pos in range(k):
                matching_at_pos = self._position_value_to_indices[pos].get(observation[pos], [])
                match_counts[matching_at_pos] += 1
            
            # P(obs|state) = (1-beta) * (m/k) + beta/n
            match_proportions = match_counts / k
            return (1 - beta) * match_proportions + beta / n
        else:
            # Exact matching
            if beta == 0:
                return (self.states == observation).all(axis=1).astype(float)
            else:
                # Noisy emission: (1-beta) + beta/V if match, beta/V otherwise
                V = self.vocab_size
                exact_match = (self.states == observation).all(axis=1).astype(float)
                return (1 - beta) * exact_match + beta / V
    
    def _get_initial_distribution(self, initial_dist: Optional[str] = None) -> np.ndarray:
        """
        Get the initial state distribution.
        
        Parameters
        ----------
        initial_dist : str, optional
            Initial distribution type. Uses instance default if not specified.
            - 'uniform': Uniform over all states.
            - 'sequence_starts': Uniform over the first state of each sequence.
            
        Returns
        -------
        np.ndarray
            Initial state distribution (length n_states).
        """
        if initial_dist is None:
            initial_dist = self.initial_dist
        
        if initial_dist == 'uniform':
            return np.ones(self.n_states) / self.n_states
        elif initial_dist == 'sequence_starts':
            dist = self.start_mask.astype(float)
            return dist / dist.sum()
        else:
            raise ValueError(f"initial_dist must be 'uniform' or 'sequence_starts', got '{initial_dist}'")
    
    def forward_pass(
        self,
        observations: np.ndarray,
        alpha: Optional[float] = None,
        theta: Optional[float] = None,
        gamma: Optional[float] = None,
        beta: Optional[float] = None,
        transition_type: Optional[TransitionType] = None,
        initial_dist: Optional[str] = None,
        partial_match: Optional[bool] = None,
        return_history: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform forward pass (filtering) given observations.
        
        Parameters
        ----------
        observations : np.ndarray
            T*k array of observations, where T is the number of timesteps.
        alpha : float, optional
            Transition probability to next state. Uses instance default if not specified.
        theta : float, optional
            Self-loop probability. Uses instance default if not specified.
        gamma : float, optional
            Transition probability to state two steps ahead. Uses instance default if not specified.
        beta : float, optional
            Emission noise probability. Uses instance default if not specified.
            - When partial_match=False:
              - beta=0: Deterministic emission (P(obs|state)=1 if match, 0 otherwise)
              - beta>0: P(obs|state) = (1-beta) + beta/V if match, beta/V otherwise
            - When partial_match=True:
              - P(obs|state) = (1-beta) * (m/k) + beta/n
              where m is matching elements, k is vector length, n is number of states.
        transition_type : str, optional
            Transition structure type. Uses instance default if not specified.
        initial_dist : str, optional
            Initial distribution type. Uses instance default if not specified.
            - 'uniform': Uniform over all states.
            - 'sequence_starts': Uniform over the first state of each sequence.
        partial_match : bool, optional
            If True, use partial matching for emissions. Uses instance default if not specified.
            Note: partial_match=True requires the model to be initialized with partial_match=True
            to have the position indices built.
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
        n = self.n_states
        k = self.k
        
        # Use instance defaults for unspecified parameters
        if beta is None:
            beta = self.beta
        if partial_match is None:
            partial_match = self.partial_match
        
        # Validate partial_match usage
        if partial_match and not self.partial_match:
            raise ValueError(
                "Cannot use partial_match=True at runtime unless the model was "
                "initialized with partial_match=True (position indices not built)"
            )
        
        # Get initial distribution
        dist = self._get_initial_distribution(initial_dist)
        
        # Pre-allocate history array if needed
        if return_history:
            history = np.empty((len(observations), n))
        
        # Precompute emission constants
        if not partial_match and beta > 0:
            V = self.vocab_size
            beta_over_V = beta / V
        if partial_match:
            beta_over_n = beta / n
        
        for t, obs in enumerate(observations):
            # Apply transition (skip for first observation)
            if t > 0:
                dist = self._transition(dist, alpha, theta, gamma, transition_type)
            
            if partial_match:
                # Partial matching: P(obs|state) = (1-beta) * (m/k) + beta/n
                # where m is number of matching elements
                
                # Count matches per state using position indices: O(k + total_matches)
                match_counts = np.zeros(n)
                for pos in range(k):
                    matching_at_pos = self._position_value_to_indices[pos].get(obs[pos], [])
                    match_counts[matching_at_pos] += 1
                
                # Compute emission likelihood: (1-beta) * (m/k) + beta/n
                match_proportions = match_counts / k
                emission_likelihood = (1 - beta) * match_proportions + beta_over_n
                
                # Apply emission and normalize
                unnorm_dist = dist * emission_likelihood
                total = unnorm_dist.sum()
                
                if total > 0:
                    dist = unnorm_dist / total
                else:
                    # All states have zero probability - return to uniform
                    dist = np.ones(n) / n
            else:
                # Exact matching (original behavior)
                matching_indices = self._state_to_indices.get(tuple(obs), [])
                
                if beta == 0:
                    # Deterministic emission: O(1) lookup, O(m) update for m matches
                    if matching_indices:
                        # Extract probabilities only at matching indices
                        matched_probs = dist[matching_indices]
                        total = matched_probs.sum()
                        
                        if total > 0:
                            # Zero out and set only matching states
                            new_dist = np.zeros(n)
                            new_dist[matching_indices] = matched_probs / total
                            dist = new_dist
                        else:
                            # Matching states have zero probability - return to uniform
                            dist = np.ones(n) / n
                    else:
                        # No state matches observation - return to uniform
                        dist = np.ones(n) / n
                else:
                    # Noisy emission: P(obs|state) = (1-beta) + beta/V if match, beta/V otherwise
                    # Uses O(1) lookup for matches, O(n) vectorized update
                    if matching_indices:
                        matched_probs = dist[matching_indices]
                        sum_match = matched_probs.sum()
                    else:
                        sum_match = 0.0
                    
                    total_dist = dist.sum()
                    
                    # Unnormalized probability mass:
                    # sum_match * ((1-beta) + beta/V) + (total - sum_match) * (beta/V)
                    # = sum_match * (1-beta) + total * (beta/V)
                    unnorm_total = sum_match * (1 - beta) + total_dist * beta_over_V
                    
                    if unnorm_total > 0:
                        # Scale all states by beta/V / unnorm_total
                        new_dist = dist * (beta_over_V / unnorm_total)
                        # Add bonus (1-beta) / unnorm_total to matching states
                        if matching_indices:
                            new_dist[matching_indices] += matched_probs * ((1 - beta) / unnorm_total)
                        dist = new_dist
                    else:
                        # All states have zero unnormalized probability - return to uniform
                        dist = np.ones(n) / n
            
            if return_history:
                history[t] = dist
        
        if return_history:
            return dist, history
        return dist
    
    def forecast(
        self,
        state_dist: np.ndarray,
        n_steps: int,
        alpha: Optional[float] = None,
        theta: Optional[float] = None,
        gamma: Optional[float] = None,
        transition_type: Optional[TransitionType] = None,
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
            Transition probability to next state. Uses instance default if not specified.
        theta : float, optional
            Self-loop probability. Uses instance default if not specified.
        gamma : float, optional
            Transition probability to state two steps ahead. Uses instance default if not specified.
        transition_type : str, optional
            Transition structure type. Uses instance default if not specified.
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
        dist = state_dist.copy()
        
        if return_history:
            history = []
        
        for _ in range(n_steps):
            dist = self._transition(dist, alpha, theta, gamma, transition_type)
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
    
    def mean_sample(
        self,
        state_dist: np.ndarray,
        conditional: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Return the expected value (probability-weighted average) of state vectors.
        
        Computes E[state] = sum(state_i * P(state_i)) / sum(P(state_i))
        
        When a conditional is provided, states are first filtered to only those
        that match the non-NaN values in the conditional array. If no states match,
        the conditional is ignored.
        
        Parameters
        ----------
        state_dist : np.ndarray
            State distribution (length n_states).
        conditional : np.ndarray, optional
            k-element array with np.nan for "don't care" positions and specific
            values for positions that must match. States are filtered to only
            those matching the non-NaN values. If no states match, ignored.
            
        Returns
        -------
        np.ndarray
            Expected value vector (length k). This is a weighted average of all
            (filtered) state vectors, so values may not correspond to any actual state.
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
        
        # Normalize distribution
        total = dist.sum()
        if total > 0:
            weights = dist / total
        else:
            # Uniform weights if distribution sums to zero
            weights = np.ones(len(dist)) / len(dist)
        
        # Compute expected value: weighted average of states
        expected_value = (states.T @ weights).astype(float)
        
        return expected_value
    
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
    print(f"Terminal mask: {gdc.terminal_mask}")
    print(f"Transition type: {gdc.transition_type}\n")
    
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
    
    # Mean sample (expected value)
    mean = gdc.mean_sample(final_dist)
    print(f"Mean sample (no conditional): {mean}")
    print(f"  (final_dist={final_dist}, so expected = state[2] = [1,0])\n")
    
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
    print(f"Start mask: {gdc_multi.start_mask}")
    print("(Terminal states diffuse uniformly instead of with sequential bias)\n")
    
    # Demo with sequence_starts initial distribution
    print("=== Sequence Starts Initial Distribution Demo ===\n")
    
    gdc_starts = GenerativeDenseChain([seq1, seq2], alpha=0.8, initial_dist='sequence_starts')
    print(f"Initial distribution (uniform): {gdc_multi._get_initial_distribution()}")
    print(f"Initial distribution (sequence_starts): {gdc_starts._get_initial_distribution()}")
    
    # Forward pass comparison
    obs = np.array([[0, 0]])
    dist_uniform, _ = gdc_multi.forward_pass(obs, return_history=True)
    dist_starts, _ = gdc_starts.forward_pass(obs, return_history=True)
    print(f"\nAfter observing [0, 0]:")
    print(f"  With uniform init: {dist_uniform}")
    print(f"  With sequence_starts init: {dist_starts}")
    
    # Demo with self-loop transition type
    print("\n=== Self-Loop Transition Demo ===\n")
    
    gdc_self_loop = GenerativeDenseChain(
        sequence, 
        alpha=0.8, 
        theta=0.1, 
        transition_type='self_loop'
    )
    print(f"Transition type: {gdc_self_loop.transition_type}")
    print(f"Alpha (sequential): {gdc_self_loop.alpha}")
    print(f"Theta (self-loop): {gdc_self_loop.theta}")
    print(f"Remaining (diffusion): {1 - gdc_self_loop.alpha - gdc_self_loop.theta}\n")
    
    # Compare transition behavior
    print("Comparing transitions from state 0 (dist=[1,0,0,0]):")
    initial = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Sequential transition
    seq_trans = gdc._transition(initial)
    print(f"  Sequential (alpha=0.8): {seq_trans}")
    
    # Self-loop transition  
    sl_trans = gdc_self_loop._transition(initial)
    print(f"  Self-loop (alpha=0.8, theta=0.1): {sl_trans}")
    
    # Verify probabilities
    print(f"\n  Sequential sum: {seq_trans.sum():.6f}")
    print(f"  Self-loop sum: {sl_trans.sum():.6f}")
    
    # Forecast comparison
    print("\nForecasting 3 steps from uniform distribution:")
    uniform = np.ones(4) / 4
    
    seq_forecast, seq_hist = gdc.forecast(uniform, n_steps=3, return_history=True)
    sl_forecast, sl_hist = gdc_self_loop.forecast(uniform, n_steps=3, return_history=True)
    
    print(f"  Sequential final: {seq_forecast}")
    print(f"  Self-loop final: {sl_forecast}")
    
    # Runtime override of transition type
    print("\n=== Runtime Override Demo ===\n")
    print("Using sequential model but overriding to self_loop at runtime:")
    override_trans = gdc._transition(initial, transition_type='self_loop', theta=0.15)
    print(f"  Override result: {override_trans}")

    # Third option: self_loop_two_step (self-loop + two-step gamma)
    print("\n=== Self-Loop Two-Step (Gamma) Demo ===\n")
    gdc_sl2 = GenerativeDenseChain(
        sequence, alpha=0.5, theta=0.1, gamma=0.2, transition_type='self_loop_two_step'
    )
    print(f"Transition type: {gdc_sl2.transition_type}")
    print(f"Alpha (next): {gdc_sl2.alpha}, Theta (self): {gdc_sl2.theta}, Gamma (two ahead): {gdc_sl2.gamma}")
    sl2_trans = gdc_sl2._transition(initial)
    print(f"  From state 0: {sl2_trans}")
    print(f"  Sum: {sl2_trans.sum():.6f}")
    
    # Demo with emission noise (beta)
    print("\n=== Emission Noise (Beta) Demo ===\n")
    
    # Create model with noisy emissions
    gdc_noisy = GenerativeDenseChain(sequence, alpha=0.8, beta=0.2)
    print(f"Beta (emission noise): {gdc_noisy.beta}")
    print(f"Vocabulary size: {gdc_noisy.vocab_size}")
    print(f"P(obs|match) = (1-beta) + beta/V = {1 - gdc_noisy.beta + gdc_noisy.beta / gdc_noisy.vocab_size:.4f}")
    print(f"P(obs|no match) = beta/V = {gdc_noisy.beta / gdc_noisy.vocab_size:.4f}\n")
    
    # Compare forward pass with and without beta
    print("Comparing forward pass with observation [0, 0]:")
    obs_single = np.array([[0, 0]])
    
    # Deterministic (beta=0)
    dist_det = gdc.forward_pass(obs_single)
    print(f"  Deterministic (beta=0): {dist_det}")
    
    # Noisy (beta=0.2)
    dist_noisy = gdc_noisy.forward_pass(obs_single)
    print(f"  Noisy (beta=0.2): {dist_noisy}")
    
    # Runtime override
    dist_override = gdc.forward_pass(obs_single, beta=0.2)
    print(f"  Runtime override (beta=0.2): {dist_override}")
    
    # Multiple observations - noisy emission retains some probability on non-matching states
    print("\nAfter observing [[0, 0], [0, 1], [1, 0]]:")
    obs_multi = np.array([[0, 0], [0, 1], [1, 0]])
    
    dist_det_multi = gdc.forward_pass(obs_multi)
    dist_noisy_multi = gdc_noisy.forward_pass(obs_multi)
    
    print(f"  Deterministic: {dist_det_multi}")
    print(f"  Noisy (beta=0.2): {dist_noisy_multi}")
    print(f"  Note: Noisy emission retains probability on non-matching states")
    
    # Demo with partial matching
    print("\n=== Partial Matching Demo ===\n")
    
    # Create model with partial matching enabled
    gdc_partial = GenerativeDenseChain(sequence, alpha=0.8, beta=0.1, partial_match=True)
    print(f"Partial match: {gdc_partial.partial_match}")
    print(f"Beta (floor for no match): {gdc_partial.beta}")
    print(f"States:\n{gdc_partial.states}\n")
    
    # Show position-value indices
    print("Position-value indices (maps value at each position to state indices):")
    for pos in range(gdc_partial.k):
        print(f"  Position {pos}: {gdc_partial._position_value_to_indices[pos]}")
    
    # Test with an observation that partially matches multiple states
    print("\nEmission likelihoods for observation [0, 1]:")
    obs_partial = np.array([0, 1])
    print(f"  States:  {[list(s) for s in gdc_partial.states]}")
    print(f"  Matches: [0,0]->1/2, [0,1]->2/2, [1,0]->0/2, [1,1]->1/2")
    
    likelihood = gdc_partial._emission_likelihood(obs_partial)
    print(f"  Likelihoods: {likelihood}")
    print(f"  Formula: P(obs|state) = (1-beta)*(m/k) + beta/n")
    print(f"           = 0.9*(m/2) + 0.1/4")
    
    # Compare forward pass behavior
    print("\nForward pass comparison after observing [[0, 1]]:")
    obs_test = np.array([[0, 1]])
    
    # Exact matching (should concentrate on state [0,1] only)
    dist_exact = gdc.forward_pass(obs_test)
    print(f"  Exact match (beta=0):    {dist_exact}")
    
    # Partial matching (should give some probability to [0,0] and [1,1] too)
    dist_partial = gdc_partial.forward_pass(obs_test)
    print(f"  Partial match (beta=0.1): {dist_partial}")
    
    # Observation with no exact match
    print("\nForward pass with observation [[0, 2]] (no exact match in states):")
    obs_nomatch = np.array([[0, 2]])
    
    dist_exact_nomatch = gdc.forward_pass(obs_nomatch)
    print(f"  Exact match: {dist_exact_nomatch} (falls back to uniform)")
    
    dist_partial_nomatch = gdc_partial.forward_pass(obs_nomatch)
    print(f"  Partial match: {dist_partial_nomatch}")
    print(f"  (States [0,0] and [0,1] have 1/2 match, [1,0] and [1,1] have 0/2 match)")
    
    # Demo with mean_sample
    print("\n=== Mean Sample (Expected Value) Demo ===\n")
    
    # Create a distribution where multiple states have probability
    print("States:")
    print(f"  {[list(s) for s in gdc.states]}")
    
    # Uniform distribution
    uniform_dist = np.array([0.25, 0.25, 0.25, 0.25])
    print(f"\nUniform distribution: {uniform_dist}")
    mean_uniform = gdc.mean_sample(uniform_dist)
    print(f"  Mean sample: {mean_uniform}")
    print(f"  (Expected: 0.25*[0,0] + 0.25*[0,1] + 0.25*[1,0] + 0.25*[1,1] = [0.5, 0.5])")
    
    # Weighted distribution
    weighted_dist = np.array([0.4, 0.3, 0.2, 0.1])
    print(f"\nWeighted distribution: {weighted_dist}")
    mean_weighted = gdc.mean_sample(weighted_dist)
    print(f"  Mean sample: {mean_weighted}")
    print(f"  (Expected: 0.4*[0,0] + 0.3*[0,1] + 0.2*[1,0] + 0.1*[1,1] = [0.3, 0.4])")
    
    # With conditional filtering
    print("\nMean sample with conditional filtering:")
    print(f"  States: {[list(s) for s in gdc.states]}")
    print(f"  Distribution: {uniform_dist}")
    
    # Conditional: first element must be 0
    cond_first_0 = np.array([0, np.nan])
    mean_cond = gdc.mean_sample(uniform_dist, conditional=cond_first_0)
    print(f"\n  Conditional [0, nan] (first element must be 0):")
    print(f"    Matching states: [0,0], [0,1]")
    print(f"    Mean sample: {mean_cond}")
    print(f"    (Expected: 0.5*[0,0] + 0.5*[0,1] = [0, 0.5])")
    
    # Conditional: second element must be 1
    cond_second_1 = np.array([np.nan, 1])
    mean_cond2 = gdc.mean_sample(uniform_dist, conditional=cond_second_1)
    print(f"\n  Conditional [nan, 1] (second element must be 1):")
    print(f"    Matching states: [0,1], [1,1]")
    print(f"    Mean sample: {mean_cond2}")
    print(f"    (Expected: 0.5*[0,1] + 0.5*[1,1] = [0.5, 1])")
    
    # Conditional with no matches (should be ignored)
    cond_no_match = np.array([5, 5])
    mean_no_match = gdc.mean_sample(uniform_dist, conditional=cond_no_match)
    print(f"\n  Conditional [5, 5] (no matches, ignored):")
    print(f"    Mean sample: {mean_no_match}")
    print(f"    (Falls back to full distribution: [0.5, 0.5])")
