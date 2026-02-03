"""
Generative Dense Chain for Time Series (GDC-TS).

Same transition structure as GenerativeDenseChain, but emissions are Gaussian:
- Emission: P(obs|state) = N(obs; state, beta*I)  (diagonal covariance, variance beta per dimension)
- Transition: Same as GDC (sequential, self_loop, self_loop_two_step)
- Initial distribution: Uniform or sequence_starts

All emission calculations are vectorized (no for loops). Forward filtering is done
in log space to avoid underflow. The variance parameter beta is a scalar; the
variance-covariance matrix is never formed explicitly (diagonal with all variances = beta).
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Literal

# Type alias for transition types. All use no-wrap and support zeroing in forecast_gdc_style.
TransitionType = Literal['sequential', 'self_loop', 'self_loop_two_step']


def _logsumexp(x: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    """Numerically stable log-sum-exp. Preserves dtype of x."""
    x_max = np.max(x, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=keepdims))
    if keepdims:
        out = out + x_max
    else:
        out = out + np.squeeze(x_max, axis=axis)
    return np.asarray(out, dtype=x.dtype)


class GenerativeDenseChainTimeSeries:
    """
    Training-free generative HMM for time series: hidden states emit from
    N(state, beta*I). Transitions match GenerativeDenseChain (same speedups).
    """

    def __init__(
        self,
        sequences: Union[np.ndarray, List[np.ndarray]],
        beta: float,
        alpha: float = 0.8,
        theta: float = 0.1,
        gamma: float = 0.0,
        transition_type: TransitionType = 'sequential',
        initial_dist: str = 'uniform',
        dtype: Optional[np.dtype] = None
    ):
        """
        Parameters
        ----------
        sequences : np.ndarray or List[np.ndarray]
            States as n*k array or list of arrays (same as GDC).
        beta : float
            Variance of emissions per dimension. Variance-covariance is diagonal
            with all diagonal entries equal to beta (not formed explicitly).
        alpha, theta, gamma, transition_type, initial_dist
            Same as GenerativeDenseChain.
        dtype : np.dtype, optional
            Dtype for forward-pass buffers (log_dist, emissions). Default float64.
            Use np.float32 for very large n_states to halve memory and often speed up.
        """
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")

        if initial_dist not in ('uniform', 'sequence_starts'):
            raise ValueError(f"initial_dist must be 'uniform' or 'sequence_starts', got '{initial_dist}'")

        if transition_type not in ('sequential', 'self_loop', 'self_loop_two_step'):
            raise ValueError(f"transition_type must be 'sequential', 'self_loop', or 'self_loop_two_step', got '{transition_type}'")

        if transition_type == 'self_loop' and alpha + theta > 1:
            raise ValueError(f"alpha + theta must be <= 1, got {alpha} + {theta} = {alpha + theta}")

        if transition_type == 'self_loop_two_step' and alpha + theta + gamma > 1:
            raise ValueError(f"alpha + theta + gamma must be <= 1, got {alpha} + {theta} + {gamma} = {alpha + theta + gamma}")

        if isinstance(sequences, np.ndarray):
            self.states = np.asarray(sequences, dtype=float).copy()
            self.terminal_mask = np.zeros(len(sequences), dtype=bool)
            self.terminal_mask[-1] = True
            self.start_mask = np.zeros(len(sequences), dtype=bool)
            self.start_mask[0] = True
        else:
            self.states = np.vstack([np.asarray(s, dtype=float) for s in sequences])
            self.terminal_mask = np.zeros(len(self.states), dtype=bool)
            self.start_mask = np.zeros(len(self.states), dtype=bool)
            cumsum = 0
            for seq in sequences:
                self.start_mask[cumsum] = True
                cumsum += len(seq)
                self.terminal_mask[cumsum - 1] = True

        self.n_states = len(self.states)
        self.k = self.states.shape[1]
        self.beta = float(beta)
        self.alpha = alpha
        self.theta = theta
        self.gamma = gamma
        self.transition_type = transition_type
        self.initial_dist = initial_dist
        self._dtype = np.dtype(dtype if dtype is not None else np.float64)

        # Precompute log normalizer for univariate N(0, beta): -0.5*log(2*pi*beta); for k dims: -0.5*k*log(2*pi*beta)
        self._log_norm_const = np.asarray(-0.5 * self.k * np.log(2.0 * np.pi * self.beta), dtype=self._dtype)

    def _transition(
        self,
        dist: np.ndarray,
        alpha: Optional[float] = None,
        theta: Optional[float] = None,
        gamma: Optional[float] = None,
        transition_type: Optional[TransitionType] = None
    ) -> np.ndarray:
        """Apply transition to state distribution (same implementations as GDC)."""
        if alpha is None:
            alpha = self.alpha
        if theta is None:
            theta = self.theta
        if gamma is None:
            gamma = self.gamma
        if transition_type is None:
            transition_type = self.transition_type

        n = self.n_states
        if n == 1:
            return dist.copy()

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
        Sequential transition with no wrap: factor = (1-alpha)/n, state 0 gets only diffusion.
        Compatible with GDC-style zeroing in forecast_gdc_style.
        """
        n = self.n_states
        factor = (1 - alpha) / n
        S = np.sum(dist)
        new_dist = np.empty_like(dist)
        new_dist[0] = 0#factor * S
        new_dist[1:n] = factor * S + (alpha - factor) * dist[: n - 1]
        return new_dist

    def _transition_self_loop(self, dist: np.ndarray, alpha: float, theta: float) -> np.ndarray:
        n = self.n_states
        if n == 2:
            return theta * dist + (1 - theta) * np.roll(dist, 1)
        gamma_diff = 1 - alpha - theta
        beta_nt = gamma_diff / (n - 2)
        beta_t = (1 - theta) / (n - 1)
        non_terminal_dist = dist * (~self.terminal_mask).astype(float)
        terminal_dist = dist * self.terminal_mask.astype(float)
        non_terminal_sum = non_terminal_dist.sum()
        terminal_sum = terminal_dist.sum()
        # No-wrap shift: state 0 gets no inflow from chain; mass that would wrap goes to state 0 as wrap_to_zero
        shifted = np.zeros(n)
        shifted[1:n] = non_terminal_dist[: n - 1]
        last_nt_idx = np.where(~self.terminal_mask)[0][-1]
        self_loop = theta * dist
        sequential = alpha * shifted
        wrap_to_zero = np.zeros(n)
        #wrap_to_zero[0] = alpha * non_terminal_dist[last_nt_idx]
        nt_diffusion = beta_nt * non_terminal_sum - beta_nt * non_terminal_dist - beta_nt * shifted
        nt_diffusion[0] -= beta_nt * non_terminal_dist[last_nt_idx]
        t_diffusion = beta_t * terminal_sum - beta_t * terminal_dist
        return self_loop + sequential + nt_diffusion + t_diffusion + wrap_to_zero

    def _transition_self_loop_two_step(self, dist: np.ndarray, alpha: float, theta: float, gamma: float) -> np.ndarray:
        n = self.n_states
        if n == 2:
            return theta * dist + (1 - theta) * np.roll(dist, 1)
        terminal_dist = dist * self.terminal_mask.astype(float)
        terminal_sum = terminal_dist.sum()
        beta_t = (1 - theta) / (n - 1)
        pre_terminal_mask = np.roll(self.terminal_mask, -1)
        pre_terminal_dist = dist * (~self.terminal_mask).astype(float) * pre_terminal_mask.astype(float)
        pre_terminal_sum = pre_terminal_dist.sum()
        has_two_mask = (~self.terminal_mask) & (~pre_terminal_mask)
        has_two_dist = dist * has_two_mask.astype(float)
        has_two_sum = has_two_dist.sum()
        self_loop = theta * dist
        t_diffusion = beta_t * terminal_sum - beta_t * terminal_dist
        # No-wrap shifts instead of roll
        if pre_terminal_sum > 0 and n >= 3:
            beta_pre = (1 - theta - alpha) / (n - 2)
            shifted_pre = np.zeros(n)
            shifted_pre[1:n] = pre_terminal_dist[: n - 1]
            pre_contrib = alpha * shifted_pre + beta_pre * (pre_terminal_sum - pre_terminal_dist - shifted_pre)
        else:
            pre_contrib = np.zeros(n)
        if has_two_sum > 0:
            shifted_1_ht = np.zeros(n)
            shifted_1_ht[1:n] = has_two_dist[: n - 1]
            shifted_2_ht = np.zeros(n)
            shifted_2_ht[2:n] = has_two_dist[: n - 2]
            if n == 3:
                beta_ht = (1 - theta - alpha - gamma) / n
                has_two_contrib = (theta - beta_ht) * has_two_dist + (alpha - beta_ht) * shifted_1_ht + (gamma - beta_ht) * shifted_2_ht + beta_ht * has_two_sum
            else:
                beta_ht = (1 - theta - alpha - gamma) / (n - 3)
                has_two_contrib = (theta - beta_ht) * has_two_dist + (alpha - beta_ht) * shifted_1_ht + (gamma - beta_ht) * shifted_2_ht + beta_ht * has_two_sum
        else:
            has_two_contrib = np.zeros(n)
        return self_loop + t_diffusion + pre_contrib + has_two_contrib

    def _emission_log_likelihood(self, observation: np.ndarray) -> np.ndarray:
        """
        Log P(obs | state) for each state. Vectorized over states.
        Assumes N(obs; state, beta*I): log density = -0.5*k*log(2*pi*beta) - 0.5 * ||obs - state||^2 / beta.
        """
        obs = np.asarray(observation, dtype=float)
        if obs.shape != (self.k,):
            raise ValueError(f"observation shape must be ({self.k},), got {obs.shape}")
        # (n_states, k) - (k,) -> (n_states, k); sum of squares over last axis -> (n_states,)
        sum_sq = np.sum((self.states - obs) ** 2, axis=1, dtype=self._dtype)
        out = self._log_norm_const - np.multiply(0.5 / self.beta, sum_sq, dtype=self._dtype)
        return out

    def _emission_log_likelihood_batch(self, observations: np.ndarray) -> np.ndarray:
        """
        Log P(obs_t | state) for all t and all states. Shape (T, n_states).
        Vectorized over time and states (no for loops).
        """
        # observations (T, k), states (n_states, k)
        # diff[i,t,j] = states[i,j] - obs[t,j] -> we want (T, n_states, k)
        diff = observations[:, np.newaxis, :] - self.states[np.newaxis, :, :]  # (T, n_states, k)
        sum_sq = np.sum(diff ** 2, axis=2)  # (T, n_states)
        return self._log_norm_const - 0.5 * sum_sq / self.beta

    def _get_initial_distribution(self, initial_dist: Optional[str] = None) -> np.ndarray:
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
        transition_type: Optional[TransitionType] = None,
        initial_dist: Optional[str] = None,
        return_history: bool = False,
        precompute_emissions: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Forward pass (filtering) in log space. Same transition speedups; emissions
        applied as log-likelihoods and normalized with log-sum-exp.

        precompute_emissions : bool, default False
            If True, precompute full (T, n_states) log emission matrix (original
            behavior). Uses more memory but can be faster when T*n fits in cache.
            If False (default), compute emission per timestep to save memory.
        """
        T = observations.shape[0]
        n = self.n_states
        dtype = self._dtype
        tiny = np.finfo(dtype).tiny

        init_dist = self._get_initial_distribution(initial_dist)
        log_dist = np.log(np.asarray(init_dist, dtype=dtype) + tiny)

        if return_history:
            history = np.empty((T, n), dtype=dtype)

        if precompute_emissions:
            log_emission = self._emission_log_likelihood_batch(observations)
            if log_emission.dtype != dtype:
                log_emission = log_emission.astype(dtype)

        for t in range(T):
            if t > 0:
                dist = np.exp(log_dist - np.max(log_dist)).astype(dtype)
                dist = dist / dist.sum()
                dist = self._transition(dist, alpha, theta, gamma, transition_type)
                log_dist = np.log(np.asarray(dist, dtype=dtype) + tiny)

            if precompute_emissions:
                log_dist += log_emission[t]
            else:
                log_dist += self._emission_log_likelihood(observations[t])
            log_dist -= _logsumexp(log_dist)

            if return_history:
                history[t] = np.exp(log_dist)

        out = np.exp(log_dist)
        if return_history:
            return out, history
        return out

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
        """Project state distribution forward via transitions (unchanged from GDC)."""
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

    def forecast_gdc_style(
        self,
        observations: np.ndarray,
        n_steps: int,
        alpha: Optional[float] = None,
        theta: Optional[float] = None,
        gamma: Optional[float] = None,
        transition_type: Optional[TransitionType] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GDC-style forecast: forward pass, zero last state, then propagate with
        re-zeroing after each step. Uses the same zeroing for all transition types
        (sequential, self_loop, self_loop_two_step).

        Parameters
        ----------
        observations : np.ndarray
            Shape (T, k). Context sequence for filtering.
        n_steps : int
            Forecast horizon.
        alpha, theta, gamma, transition_type
            Optional overrides; default use instance values.

        Returns
        -------
        forecasts : np.ndarray
            Shape (n_steps, k). Point forecast E[obs] per step.
        state_distributions : np.ndarray
            Shape (n_steps, n_states). State distribution at each step (before zeroing).
        """
        if transition_type is None:
            transition_type = self.transition_type
        end_dist = self.forward_pass(observations, alpha=alpha, theta=theta, gamma=gamma, transition_type=transition_type)
        end_dist = end_dist.copy()
        end_dist[-1] = 0.0
        total = np.sum(end_dist)
        if total <= 0:
            raise ValueError("End state distribution has zero mass after zeroing last state.")
        end_dist = end_dist / total

        current_dist = end_dist.copy()
        state_distributions = []
        for _ in range(n_steps):
            current_dist = self._transition(current_dist, alpha=alpha, theta=theta, gamma=gamma, transition_type=transition_type)
            state_distributions.append(current_dist.copy())
            current_dist[-1] = 0.0
            total = np.sum(current_dist)
            if total > 0:
                current_dist = current_dist / total

        state_distributions = np.array(state_distributions)
        # forecasts[i] = E[obs] = sum_s state_distributions[i,s] * states[s]
        forecasts = np.dot(state_distributions, self.states)
        return forecasts, state_distributions

    def greedy_sample(
        self,
        state_dist: np.ndarray,
        mask: Optional[np.ndarray] = None,
        conditional: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Most likely state (optionally grouped by mask / conditional). Same as GDC."""
        states = self.states
        dist = state_dist

        if conditional is not None:
            conditional = np.asarray(conditional, dtype=float)
            constraint_mask = ~np.isnan(conditional)
            if constraint_mask.any():
                matches = (states[:, constraint_mask] == conditional[constraint_mask]).all(axis=1)
                if matches.any():
                    matching_indices = np.where(matches)[0]
                    states = states[matching_indices]
                    dist = state_dist[matching_indices]

        if mask is None:
            best_idx = np.argmax(dist)
            return states[best_idx].astype(float).copy()

        mask = np.asarray(mask, dtype=bool)
        # Vectorized: unique rows of states[:, mask], sum dist per group, argmax
        states_m = states[:, mask]
        unq, inv = np.unique(states_m, axis=0, return_inverse=True)
        group_mass = np.bincount(inv, minlength=unq.shape[0], weights=dist)
        best_idx = np.argmax(group_mass)
        result = np.full(self.k, np.nan, dtype=float)
        result[mask] = unq[best_idx].astype(float)
        return result

    def random_sample(
        self,
        state_dist: np.ndarray,
        mask: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Random state sample (optionally grouped by mask). Same as GDC."""
        if rng is None:
            rng = np.random.default_rng()
        if mask is None:
            total = state_dist.sum()
            if total > 0:
                probs = state_dist / total
            else:
                probs = np.ones(self.n_states) / self.n_states
            idx = rng.choice(self.n_states, p=probs)
            return self.states[idx].astype(float).copy()
        mask = np.asarray(mask, dtype=bool)
        states_m = self.states[:, mask]
        unq, inv = np.unique(states_m, axis=0, return_inverse=True)
        group_mass = np.bincount(inv, minlength=unq.shape[0], weights=state_dist)
        total = group_mass.sum()
        if total > 0:
            probs = group_mass / total
        else:
            probs = np.ones(unq.shape[0]) / unq.shape[0]
        chosen_idx = rng.choice(unq.shape[0], p=probs)
        result = np.full(self.k, np.nan, dtype=float)
        result[mask] = unq[chosen_idx].astype(float)
        return result


def _run_sandbox_speed_test():
    """Sandbox: compare on-the-fly vs precomputed emission with 50k states, 1000 obs."""
    import time
    n_states = 50_000
    T = 1_000
    k = 4
    beta = 0.1
    np.random.seed(123)
    states = np.random.randn(n_states, k).astype(np.float64)
    observations = np.random.randn(T, k).astype(np.float64)
    model = GenerativeDenseChainTimeSeries(states, beta=beta, alpha=0.8)

    n_runs = 5
    # On-the-fly (default)
    times_onthefly = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.forward_pass(observations, precompute_emissions=False)
        times_onthefly.append(time.perf_counter() - t0)
    # Precomputed (original)
    times_precomputed = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.forward_pass(observations, precompute_emissions=True)
        times_precomputed.append(time.perf_counter() - t0)

    mean_onthefly = np.mean(times_onthefly)
    mean_precomputed = np.mean(times_precomputed)
    print("Sandbox speed test: 50,000 states, 1,000 observations, 5 runs each")
    print(f"  On-the-fly emission (default): {mean_onthefly:.3f}s  (std {np.std(times_onthefly):.3f}s)")
    print(f"  Precomputed emission (original): {mean_precomputed:.3f}s  (std {np.std(times_precomputed):.3f}s)")
    if mean_precomputed > 0:
        ratio = mean_onthefly / mean_precomputed
        print(f"  Ratio (on-the-fly / precomputed): {ratio:.2f}x")
    print()


if __name__ == "__main__":
    print("=== Generative Dense Chain Time Series Demo ===\n")

    # States as sequence of 2D points
    sequence = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])

    beta = 0.1
    gdc = GenerativeDenseChainTimeSeries(sequence, beta=beta, alpha=0.8)
    print(f"States:\n{gdc.states}")
    print(f"Beta (emission variance): {gdc.beta}\n")

    # Noisy observations (slight perturbations around states)
    np.random.seed(42)
    observations = sequence[:3] + np.random.randn(3, 2) * np.sqrt(beta) * 0.5

    final_dist, history = gdc.forward_pass(observations, return_history=True)
    print("Observations (states + small noise):\n", observations)
    print("Final state distribution:", final_dist)
    print("History (per-timestep distributions):\n", history)

    forecast_dist = gdc.forecast(final_dist, n_steps=2)
    print("\nForecast (2 steps):", forecast_dist)

    greedy = gdc.greedy_sample(final_dist)
    print("Greedy sample:", greedy)

    rng = np.random.default_rng(43)
    random = gdc.random_sample(final_dist, rng=rng)
    print("Random sample:", random)

    print("\n" + "=" * 50)
    _run_sandbox_speed_test()
