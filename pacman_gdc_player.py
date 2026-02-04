"""
Pacman GDC Player
Uses a Generative Dense Chain (GDC) to learn and play Pacman.

The GDC learns from game histories and predicts the best action at each timestep
by computing expected rewards for each possible action.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from pacman_simulator import MiniPacmanSimulator, Cell
from generative_dense_chain import GenerativeDenseChain


def reverse_ewma(values: np.ndarray, halflife: float = 3.0) -> np.ndarray:
    """
    Apply exponential weighted moving average from right to left (reverse).
    
    This smooths rewards backwards in time, so future rewards influence
    past timesteps. Useful for credit assignment in RL.
    
    Args:
        values: 1D array of values to smooth
        halflife: Number of steps for weight to decay to half (default: 3)
        
    Returns:
        Smoothed values array of same shape
    """
    if len(values) == 0:
        return values.copy()
    
    # Calculate decay factor alpha from halflife
    # After 'halflife' steps, weight should be 0.5
    # (1 - alpha)^halflife = 0.5
    # alpha = 1 - 0.5^(1/halflife)
    alpha = 1.0 - (0.5 ** (1.0 / halflife))
    
    smoothed = np.zeros_like(values, dtype=np.float64)
    n = len(values)
    
    # Start from the last element and work backwards
    smoothed[n-1] = values[n-1]
    for i in range(n-2, -1, -1):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i+1]
    
    return smoothed.astype(values.dtype)


def play_game_with_gdc(
    gdc: Optional[GenerativeDenseChain],
    n_steps: int = 50,
    n_ghosts: int = 3,
    use_random: bool = False,
    verbose: bool = False,
    maze_type: str = 'standard',
    reward_halflife: float = 3.0,
    exploration_rate: float = 0.0
) -> np.ndarray:
    """
    Play a single game using the GDC to select actions.
    
    At each timestep:
    1. Get current 3x3 observation from game
    2. For each possible action (0-4), use GDC mean_sample conditioned on
       the action and current observation to predict expected reward
    3. Select action with highest expected reward (or random with probability exploration_rate)
    4. Execute action and record [action, obs_flat, reward]
    
    After the game, rewards are smoothed using reverse EWMA (from end to start)
    to propagate future rewards back to earlier timesteps.
    
    Args:
        gdc: Trained GDC model, or None for random play
        n_steps: Maximum steps per game
        n_ghosts: Number of ghosts
        use_random: If True, use random actions (for first game)
        verbose: Print debug information
        maze_type: Type of maze ('standard', 'random', or 'custom')
        reward_halflife: Halflife for reverse EWMA reward smoothing (default: 3.0)
        exploration_rate: Probability of choosing random action during training (default: 0.0)
        
    Returns:
        History array of shape (actual_steps, 11)
    """
    # Initialize game with 3x3 observation and standard maze
    game = MiniPacmanSimulator(
        n_ghosts=n_ghosts,
        obs_size=3,
        maze_type=maze_type
    )
    
    # Track observations for this game
    game_observations = []  # List of [action, obs_flat, reward] arrays
    
    # Initialize state distribution for incremental updates (O(n) instead of O(n²))
    state_dist = gdc._get_initial_distribution() if gdc is not None else None
    
    for step in range(n_steps):
        obs = game.get_observation()  # Current 3x3 observation
        obs_flat = obs.flatten()
        
        if use_random or gdc is None:
            # Use random action
            action = np.random.randint(0, 5)
        elif exploration_rate > 0 and np.random.random() < exploration_rate:
            # Exploration: random action with probability exploration_rate
            action = np.random.randint(0, 5)
        else:
            # Use GDC to select action (using maintained state_dist)
            action = select_action_with_gdc(gdc, state_dist, obs_flat, verbose)
        
        # Execute action
        _, reward, done, _ = game.step(action)
        
        # Record this step as [action, obs_flat, reward]
        record = np.zeros(11, dtype=np.float32)
        record[0] = action
        record[1:10] = obs_flat
        record[10] = reward
        game_observations.append(record)
        
        # Incrementally update state distribution with this observation
        if gdc is not None:
            state_dist = update_state_dist(gdc, state_dist, record)
        
        if verbose:
            action_names = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT']
            print(f"  Step {step}: {action_names[action]}, Reward: {reward}")
        
        if done:
            break
    
    # Convert to array
    history = np.array(game_observations, dtype=np.float32)
    
    # Apply reverse EWMA to rewards (smooth from end to start)
    # This propagates future rewards back to earlier timesteps
    history[:, 10] = reverse_ewma(history[:, 10], halflife=reward_halflife)
    
    if verbose:
        print(f"  Rewards smoothed with reverse EWMA (halflife={reward_halflife})")
    
    return history


def select_action_with_gdc(
    gdc: GenerativeDenseChain,
    state_dist: np.ndarray,
    current_obs_flat: np.ndarray,
    verbose: bool = False
) -> int:
    """
    Select the best action using GDC mean forecast.
    
    For each possible action:
    1. Apply transition to predict next state distribution
    2. Use mean_sample with conditional on [action, current_obs, nan] to
       get expected reward for that action
    3. Return action with highest expected reward
    
    Args:
        gdc: Trained GDC model
        state_dist: Current state distribution (maintained incrementally)
        current_obs_flat: Current flattened 3x3 observation (9 elements)
        verbose: Print debug info
        
    Returns:
        Selected action (0-4)
    """
    # Apply transition to predict next state distribution
    next_dist = gdc._transition(state_dist)
    
    # Try each action and compute expected reward
    expected_rewards = np.zeros(5)
    
    for action in range(5):
        # Create conditional: [action, current_obs, nan for reward]
        conditional = np.full(11, np.nan)
        conditional[0] = action
        conditional[1:10] = current_obs_flat
        
        # Get expected next state given this action and observation
        expected_state = gdc.mean_sample(next_dist, conditional=conditional)
        
        # Expected reward is last element
        expected_rewards[action] = expected_state[10]
    
    if verbose:
        action_names = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT']
        print(f"    Expected rewards: {dict(zip(action_names, expected_rewards.round(2)))}")
    
    # Select action with highest expected reward
    # Break ties randomly
    best_reward = expected_rewards.max()
    best_actions = np.where(expected_rewards == best_reward)[0]
    return np.random.choice(best_actions)


def update_state_dist(
    gdc: GenerativeDenseChain,
    state_dist: np.ndarray,
    observation: np.ndarray
) -> np.ndarray:
    """
    Incrementally update state distribution with a single observation.
    
    This is much faster than running forward_pass on entire history.
    Performs: transition -> emission update -> normalize
    
    Args:
        gdc: GDC model
        state_dist: Current state distribution
        observation: Single observation vector (length 11)
        
    Returns:
        Updated state distribution
    """
    # Apply transition
    dist = gdc._transition(state_dist)
    
    # Apply emission likelihood
    emission = gdc._emission_likelihood(observation)
    dist = dist * emission
    
    # Normalize
    total = dist.sum()
    if total > 0:
        dist = dist / total
    else:
        # Fall back to uniform if all zero
        dist = np.ones(gdc.n_states) / gdc.n_states
    
    return dist


def run_gdc_learning(
    n_games: int = 50,
    n_steps_per_game: int = 50,
    n_ghosts: int = 3,
    gdc_alpha: float = 0.8,
    gdc_beta: float = 0.1,
    seed: Optional[int] = None,
    verbose: bool = False,
    maze_type: str = 'standard',
    reward_halflife: float = 3.0,
    exploration_rate: float = 0.1
) -> tuple:
    """
    Run the GDC learning loop for multiple games.
    
    1. First game uses random actions
    2. After each game, rebuild GDC with all histories
    3. Subsequent games use GDC to select actions (with exploration)
    
    Uses the standard Pacman maze by default. Ghost and pacman starting 
    positions are randomized each game.
    
    Args:
        n_games: Number of games to play
        n_steps_per_game: Maximum steps per game
        n_ghosts: Number of ghosts
        gdc_alpha: GDC transition probability
        gdc_beta: GDC emission noise (for partial matching)
        seed: Random seed
        verbose: Print progress
        maze_type: Type of maze ('standard', 'random')
        reward_halflife: Halflife for reverse EWMA reward smoothing (default: 3.0)
        exploration_rate: Probability of random action during training (default: 0.1)
        
    Returns:
        (game_rewards, all_histories) - rewards per game and all history arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    all_histories = []  # List of history arrays from each game
    game_rewards = []   # Total reward for each game
    gdc = None
    
    if verbose:
        print(f"Using {maze_type} maze")
        print(f"Reward smoothing halflife: {reward_halflife}")
        print(f"Exploration rate: {exploration_rate:.0%}")
        print("(Ghost/Pacman positions randomized each game)")
    
    for game_idx in range(n_games):
        # First game is random, subsequent games use GDC
        use_random = (game_idx == 0)
        
        if verbose:
            mode = "RANDOM" if use_random else "GDC"
            print(f"\n=== Game {game_idx + 1}/{n_games} ({mode}) ===")
        
        # Play game (with exploration during training)
        history = play_game_with_gdc(
            gdc=gdc,
            n_steps=n_steps_per_game,
            n_ghosts=n_ghosts,
            use_random=use_random,
            verbose=verbose,
            maze_type=maze_type,
            reward_halflife=reward_halflife,
            exploration_rate=exploration_rate
        )
        
        # Record total reward for this game
        total_reward = history[:, 10].sum()
        game_rewards.append(total_reward)
        all_histories.append(history)
        
        if verbose:
            print(f"  Total steps: {len(history)}, Total reward: {total_reward:.0f}")
        
        # Rebuild GDC with all histories
        gdc = GenerativeDenseChain(
            sequences=all_histories,
            alpha=gdc_alpha,
            beta=gdc_beta,
            partial_match=True,
            initial_dist='sequence_starts'
        )
        
        if verbose:
            print(f"  GDC rebuilt with {gdc.n_states} states from {len(all_histories)} games")
    
    return np.array(game_rewards), all_histories


def plot_rewards(game_rewards: np.ndarray, title: str = "GDC Pacman Learning"):
    """Plot rewards vs game number."""
    plt.figure(figsize=(10, 6))
    
    # Raw rewards
    plt.plot(range(1, len(game_rewards) + 1), game_rewards, 
             'b-', alpha=0.5, label='Game Reward')
    
    # Moving average (window of 5)
    if len(game_rewards) >= 20:
        window = 20
        moving_avg = np.convolve(game_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window, len(game_rewards) + 1), moving_avg, 
                 'r-', linewidth=2, label=f'{window}-Game Moving Avg')
    
    plt.xlabel('Game Number', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pacman_gdc_rewards.png', dpi=150)
    plt.show()
    
    return plt.gcf()


def print_summary(game_rewards: np.ndarray):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("LEARNING SUMMARY")
    print("="*60)
    print(f"Total games played: {len(game_rewards)}")
    print(f"First game reward (random): {game_rewards[0]:.0f}")
    print(f"Last game reward: {game_rewards[-1]:.0f}")
    print(f"Best game reward: {game_rewards.max():.0f} (game {game_rewards.argmax() + 1})")
    print(f"Worst game reward: {game_rewards.min():.0f} (game {game_rewards.argmin() + 1})")
    print(f"Mean reward: {game_rewards.mean():.1f}")
    print(f"Std reward: {game_rewards.std():.1f}")
    
    # Compare first half vs second half
    half = len(game_rewards) // 2
    first_half_mean = game_rewards[:half].mean()
    second_half_mean = game_rewards[half:].mean()
    print(f"\nFirst {half} games mean: {first_half_mean:.1f}")
    print(f"Last {half} games mean: {second_half_mean:.1f}")
    print(f"Improvement: {second_half_mean - first_half_mean:+.1f}")


if __name__ == "__main__":
    print("="*60)
    print("GDC PACMAN PLAYER")
    print("="*60)
    print("Running 50 games with GDC learning...")
    print("First game uses random actions, subsequent games use GDC.\n")
    
    # Run learning
    game_rewards, all_histories = run_gdc_learning(
        n_games=1000,
        n_steps_per_game=20,
        n_ghosts=2,
        gdc_alpha=0.61,
        gdc_beta=0.1,
        seed=41,
        reward_halflife=1.0,
        verbose=False  # Set to True for detailed output
    )
    
    # Print summary
    print_summary(game_rewards)
    
    # Plot results
    print("\nPlotting rewards...")
    plot_rewards(game_rewards)
    
    print("\nDone! Plot saved to 'pacman_gdc_rewards.png'")
