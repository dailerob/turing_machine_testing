"""
Pacman Simulation Recorder
Simulates the Mini Pacman game and records agent perspective (5x5 grid around agent).
"""

import numpy as np
from enum import IntEnum
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field


class Cell(IntEnum):
    EMPTY = 0
    WALL = 1
    PELLET = 2
    GHOST = 3
    PACMAN = 4


@dataclass
class SimulationRecord:
    """Container for simulation data at a single timestep."""
    step: int
    observation: np.ndarray  # 5x5 grid around agent
    action: int
    reward: float
    done: bool
    pacman_pos: Tuple[int, int]
    ghost_positions: List[Tuple[int, int]]
    pellets_remaining: int
    score: int


@dataclass
class SimulationResult:
    """Complete simulation results."""
    records: List[SimulationRecord] = field(default_factory=list)
    total_reward: float = 0.0
    total_steps: int = 0
    won: bool = False
    final_score: int = 0
    
    def get_observations(self) -> np.ndarray:
        """Return all observations as a stacked array."""
        return np.array([r.observation for r in self.records])
    
    def get_actions(self) -> np.ndarray:
        """Return all actions as an array."""
        return np.array([r.action for r in self.records])
    
    def get_rewards(self) -> np.ndarray:
        """Return all rewards as an array."""
        return np.array([r.reward for r in self.records])


class MiniPacmanSimulator:
    """
    Pacman game simulator with configurable observation window.
    """
    
    ACTIONS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # stay, up, down, left, right
    ACTION_NAMES = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT']
    
    # Standard 15x15 Pacman-style maze (1 = wall, 0 = empty)
    # Designed to look like a classic Pacman maze with corridors
    STANDARD_MAZE_15x15 = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ], dtype=bool)
    
    # Default starting position for Pacman (center-bottom, like classic Pacman)
    DEFAULT_START_POS = (11, 7)
    
    def __init__(self, width=15, height=15, n_ghosts=3, wall_density=0.15, obs_size=5, 
                 fixed_walls=None, maze_type='standard', pacman_start=None):
        """
        Initialize the simulator.
        
        Args:
            width: Grid width (ignored if maze_type='standard')
            height: Grid height (ignored if maze_type='standard')
            n_ghosts: Number of ghosts
            wall_density: Probability of internal walls (only for maze_type='random')
            obs_size: Size of observation window (obs_size x obs_size)
            fixed_walls: Optional boolean array of wall positions (True=wall).
                        If provided, this map is used instead of random generation.
                        Shape should be (height, width).
            maze_type: Type of maze to use:
                      - 'standard': Use the hardcoded Pacman-style maze (default)
                      - 'random': Generate random walls based on wall_density
                      - 'custom': Use the provided fixed_walls array
            pacman_start: Fixed starting position (row, col) for Pacman.
                         If None, uses random position for all maze types.
        """
        self.maze_type = maze_type
        self.wall_density = wall_density
        self.obs_size = obs_size
        self.n_ghosts = n_ghosts
        
        # Set dimensions based on maze type
        if maze_type == 'standard':
            self.width = 15
            self.height = 15
            self.fixed_walls = self.STANDARD_MAZE_15x15.copy()
            # Use random start position unless explicitly specified
            self.pacman_start = pacman_start  # None means random
        elif maze_type == 'custom' and fixed_walls is not None:
            self.fixed_walls = fixed_walls
            self.height, self.width = fixed_walls.shape
            self.pacman_start = pacman_start  # None means random
        else:
            # Random maze
            self.width = width
            self.height = height
            self.fixed_walls = fixed_walls
            self.pacman_start = pacman_start  # None means random
        
        self.obs_radius = obs_size // 2
        self.score = 0
        self.reset()
    
    def generate_wall_map(self):
        """
        Generate a random wall map.
        
        Returns:
            Boolean array of shape (height, width) where True = wall.
        """
        walls = np.zeros((self.height, self.width), dtype=bool)
        # Border walls
        walls[0, :] = walls[-1, :] = True
        walls[:, 0] = walls[:, -1] = True
        # Random internal walls
        inner_mask = np.random.random((self.height - 2, self.width - 2)) < self.wall_density
        walls[1:-1, 1:-1] = inner_mask
        return walls
    
    def reset(self, regenerate_walls=False):
        """
        Reset the game to initial state.
        
        Args:
            regenerate_walls: If True, generate new random walls (only for maze_type='random').
                             If False (default), reuse existing wall layout.
        """
        # Create base grid with pellets
        self.grid = np.full((self.height, self.width), Cell.PELLET)
        
        # Apply walls based on maze type
        if self.maze_type == 'standard':
            # Always use the standard maze
            self.grid[self.STANDARD_MAZE_15x15] = Cell.WALL
        elif self.maze_type == 'custom' and self.fixed_walls is not None:
            # Use custom fixed walls
            self.grid[self.fixed_walls] = Cell.WALL
        elif self.fixed_walls is not None and not regenerate_walls:
            # Use previously generated random walls
            self.grid[self.fixed_walls] = Cell.WALL
        else:
            # Generate new random walls
            walls = self.generate_wall_map()
            self.grid[walls] = Cell.WALL
            # Save for future resets
            self.fixed_walls = walls
        
        # Place pacman (fixed position if specified, random otherwise)
        if self.pacman_start is not None:
            self.pacman_pos = self.pacman_start
        else:
            self.pacman_pos = self._random_empty()
        self.grid[self.pacman_pos] = Cell.EMPTY
        
        # Place ghosts (random positions each reset)
        self.ghost_positions = [self._random_empty() for _ in range(self.n_ghosts)]
        
        self.done = False
        self.won = False
        self.score = 0
        self.pellets_remaining = np.sum(self.grid == Cell.PELLET)
        
        # Track consecutive steps in same position (for stagnation penalty)
        self.steps_in_same_pos = 0
        self.last_pos = self.pacman_pos
        
        return self.get_observation()
    
    def _random_empty(self):
        """Find a random empty position on the grid."""
        while True:
            pos = (np.random.randint(1, self.height-1), np.random.randint(1, self.width-1))
            if self.grid[pos] != Cell.WALL and pos != getattr(self, 'pacman_pos', None):
                if not hasattr(self, 'ghost_positions') or pos not in self.ghost_positions:
                    return pos
    
    def get_observation(self) -> np.ndarray:
        """
        Return obs_size x obs_size grid centered on pacman.
        
        Returns:
            numpy array of shape (obs_size, obs_size) with cell values
        """
        obs = np.full((self.obs_size, self.obs_size), Cell.WALL, dtype=np.int8)
        py, px = self.pacman_pos
        
        for dy in range(-self.obs_radius, self.obs_radius + 1):
            for dx in range(-self.obs_radius, self.obs_radius + 1):
                ny, nx = py + dy, px + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    obs[dy + self.obs_radius, dx + self.obs_radius] = self.grid[ny, nx]
        
        # Mark ghosts in observation
        for gy, gx in self.ghost_positions:
            dy, dx = gy - py, gx - px
            if abs(dy) <= self.obs_radius and abs(dx) <= self.obs_radius:
                obs[dy + self.obs_radius, dx + self.obs_radius] = Cell.GHOST
        
        # Mark pacman position (center)
        obs[self.obs_radius, self.obs_radius] = Cell.PACMAN
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one game step.
        
        Args:
            action: Integer action (0=stay, 1=up, 2=down, 3=left, 4=right)
            
        Returns:
            observation, reward, done, info
        """
        if self.done:
            return self.get_observation(), 0, True, {}
        
        dy, dx = self.ACTIONS[action]
        ny, nx = self.pacman_pos[0] + dy, self.pacman_pos[1] + dx
        
        reward = 0
        old_pos = self.pacman_pos
        
        # Move if not wall
        if 0 <= ny < self.height and 0 <= nx < self.width and self.grid[ny, nx] != Cell.WALL:
            self.pacman_pos = (ny, nx)
            if self.grid[ny, nx] == Cell.PELLET:
                reward = 10
                self.score += 10
                self.grid[ny, nx] = Cell.EMPTY
                self.pellets_remaining -= 1
        
        # Track stagnation (staying in same position)
        if self.pacman_pos == self.last_pos:
            self.steps_in_same_pos += 1
        else:
            self.steps_in_same_pos = 0
            self.last_pos = self.pacman_pos
        
        # Apply stagnation penalty: -5 for each step after 4 consecutive steps in same spot
        if self.steps_in_same_pos > 4:
            reward -= 5
            self.score -= 5
        
        # Move ghosts randomly
        for i, (gy, gx) in enumerate(self.ghost_positions):
            moves = [(gy+mdy, gx+mdx) for mdy, mdx in self.ACTIONS 
                     if 0 <= gy+mdy < self.height and 0 <= gx+mdx < self.width 
                     and self.grid[gy+mdy, gx+mdx] != Cell.WALL]
            if moves:
                self.ghost_positions[i] = moves[np.random.randint(len(moves))]
        
        # Check collisions
        if self.pacman_pos in self.ghost_positions:
            reward = -100
            self.score -= 100
            self.done = True
            self.won = False
        elif self.pellets_remaining == 0:
            reward = 100
            self.score += 100
            self.done = True
            self.won = True
        
        info = {
            'pacman_pos': self.pacman_pos,
            'ghost_positions': list(self.ghost_positions),
            'pellets_remaining': self.pellets_remaining,
            'score': self.score,
            'won': self.won
        }
        
        return self.get_observation(), reward, self.done, info
    
    def render_observation(self, obs: np.ndarray) -> str:
        """Render an observation as a string for display."""
        symbols = {
            Cell.EMPTY: '.',
            Cell.WALL: '#',
            Cell.PELLET: 'o',
            Cell.GHOST: 'G',
            Cell.PACMAN: 'P'
        }
        return '\n'.join(''.join(symbols[c] for c in row) for row in obs)


def random_policy(observation: np.ndarray, game: MiniPacmanSimulator) -> int:
    """Random action policy."""
    return np.random.randint(0, 5)


def simulate_game(
    n_steps: int = 50,
    policy: Optional[Callable] = None,
    width: int = 15,
    height: int = 15,
    n_ghosts: int = 3,
    wall_density: float = 0.15,
    obs_size: int = 5,
    seed: Optional[int] = None,
    verbose: bool = False
) -> SimulationResult:
    """
    Simulate a Pacman game and record agent perspective.
    
    Args:
        n_steps: Maximum number of steps to simulate
        policy: Function(observation, game) -> action. If None, uses random policy
        width: Grid width
        height: Grid height
        n_ghosts: Number of ghosts
        wall_density: Probability of internal walls
        obs_size: Size of observation window (5 = 5x5 grid around agent)
        seed: Random seed for reproducibility
        verbose: If True, print game state at each step
        
    Returns:
        SimulationResult containing all recorded data
    """
    if seed is not None:
        np.random.seed(seed)
    
    if policy is None:
        policy = random_policy
    
    # Initialize game
    game = MiniPacmanSimulator(
        width=width,
        height=height,
        n_ghosts=n_ghosts,
        wall_density=wall_density,
        obs_size=obs_size
    )
    
    result = SimulationResult()
    
    # Get initial observation
    obs = game.get_observation()
    
    for step in range(n_steps):
        # Select action using policy
        action = policy(obs, game)
        
        # Record state BEFORE taking action
        record = SimulationRecord(
            step=step,
            observation=obs.copy(),
            action=action,
            reward=0,  # Will be updated after step
            done=game.done,
            pacman_pos=game.pacman_pos,
            ghost_positions=list(game.ghost_positions),
            pellets_remaining=game.pellets_remaining,
            score=game.score
        )
        
        # Take action
        obs, reward, done, info = game.step(action)
        
        # Update record with reward
        record.reward = reward
        record.done = done
        
        result.records.append(record)
        result.total_reward += reward
        
        if verbose:
            print(f"\n=== Step {step} ===")
            print(f"Action: {game.ACTION_NAMES[action]}")
            print(f"Reward: {reward}, Score: {game.score}")
            print(f"Pellets remaining: {game.pellets_remaining}")
            print("Observation (5x5):")
            print(game.render_observation(obs))
        
        if done:
            break
    
    result.total_steps = len(result.records)
    result.won = game.won
    result.final_score = game.score
    
    return result


def run_multiple_simulations(
    n_simulations: int = 10,
    n_steps: int = 50,
    policy: Optional[Callable] = None,
    **kwargs
) -> List[SimulationResult]:
    """
    Run multiple simulations and return results.
    
    Args:
        n_simulations: Number of simulations to run
        n_steps: Maximum steps per simulation
        policy: Action selection policy
        **kwargs: Additional arguments passed to simulate_game
        
    Returns:
        List of SimulationResult objects
    """
    results = []
    for i in range(n_simulations):
        result = simulate_game(n_steps=n_steps, policy=policy, **kwargs)
        results.append(result)
    return results


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


def prep_history(
    n_steps: int = 50,
    policy: Optional[Callable] = None,
    width: int = 15,
    height: int = 15,
    n_ghosts: int = 3,
    wall_density: float = 0.15,
    seed: Optional[int] = None,
    reward_halflife: float = 3.0
) -> np.ndarray:
    """
    Run a simulation and return history as an n_steps x 11 array.
    
    The array columns are:
        - Column 0: action (0-4)
        - Columns 1-9: flattened 3x3 observation grid (row-major order)
        - Column 10: reward (smoothed with reverse EWMA)
    
    Args:
        n_steps: Maximum number of steps to simulate (default: 50)
        policy: Function(observation, game) -> action. If None, uses random policy
        width: Grid width
        height: Grid height
        n_ghosts: Number of ghosts
        wall_density: Probability of internal walls
        seed: Random seed for reproducibility
        reward_halflife: Halflife for reverse EWMA reward smoothing (default: 3.0)
        
    Returns:
        numpy array of shape (actual_steps, 11) where actual_steps <= n_steps
        
    Example:
        >>> history = prep_history(n_steps=50, seed=42)
        >>> print(history.shape)  # (actual_steps, 11)
        >>> actions = history[:, 0]
        >>> observations = history[:, 1:10].reshape(-1, 3, 3)
        >>> rewards = history[:, 10]
    """
    # Run simulation with 3x3 observation window
    result = simulate_game(
        n_steps=n_steps,
        policy=policy,
        width=width,
        height=height,
        n_ghosts=n_ghosts,
        wall_density=wall_density,
        obs_size=3,  # Fixed at 3x3 for 9 elements
        seed=seed,
        verbose=False
    )
    
    actual_steps = len(result.records)
    
    # Create the history array: n_steps x 11
    history = np.zeros((actual_steps, 11), dtype=np.float32)
    
    for i, record in enumerate(result.records):
        # Column 0: action
        history[i, 0] = record.action
        
        # Columns 1-9: flattened 3x3 observation
        history[i, 1:10] = record.observation.flatten()
        
        # Column 10: reward
        history[i, 10] = record.reward
    
    # Apply reverse EWMA to rewards (smooth from end to start)
    # This propagates future rewards back to earlier timesteps
    history[:, 10] = reverse_ewma(history[:, 10], halflife=reward_halflife)
    
    return history


def print_simulation_summary(result: SimulationResult):
    """Print a summary of simulation results."""
    print("\n" + "="*50)
    print("SIMULATION SUMMARY")
    print("="*50)
    print(f"Total steps: {result.total_steps}")
    print(f"Total reward: {result.total_reward}")
    print(f"Final score: {result.final_score}")
    print(f"Won: {result.won}")
    
    # Compute action distribution
    actions = result.get_actions()
    action_names = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT']
    print("\nAction distribution:")
    for i, name in enumerate(action_names):
        count = np.sum(actions == i)
        pct = 100 * count / len(actions) if len(actions) > 0 else 0
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    # Reward summary
    rewards = result.get_rewards()
    print(f"\nReward stats:")
    print(f"  Total: {np.sum(rewards)}")
    print(f"  Mean: {np.mean(rewards):.2f}")
    print(f"  Max: {np.max(rewards)}")
    print(f"  Min: {np.min(rewards)}")


if __name__ == "__main__":
    print("="*60)
    print("PREP_HISTORY DEMO")
    print("="*60)
    print("Running prep_history with 50 steps, random actions, 3x3 obs...\n")
    
    # Demonstrate prep_history
    history = prep_history(n_steps=50, seed=42)
    
    print(f"History array shape: {history.shape}")
    print(f"  - Rows: {history.shape[0]} timesteps")
    print(f"  - Columns: {history.shape[1]} (1 action + 9 obs + 1 reward)")
    
    print("\nColumn layout:")
    print("  Column 0:    Action (0=stay, 1=up, 2=down, 3=left, 4=right)")
    print("  Columns 1-9: Flattened 3x3 observation grid")
    print("  Column 10:   Reward")
    
    print("\n" + "-"*60)
    print("First 5 rows of history:")
    print("-"*60)
    print("Action | Observation (flattened 3x3)              | Reward")
    print("-"*60)
    for i in range(min(5, len(history))):
        action = int(history[i, 0])
        obs_flat = history[i, 1:10].astype(int)
        reward = history[i, 10]
        action_name = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT'][action]
        print(f"{action_name:6} | {obs_flat} | {reward:6.0f}")
    
    print("\n" + "-"*60)
    print("Reconstructed 3x3 observations:")
    print("-"*60)
    
    # Show first observation as 3x3 grid
    obs_3x3 = history[0, 1:10].reshape(3, 3).astype(int)
    print("First observation (3x3):")
    symbols = {0: '.', 1: '#', 2: 'o', 3: 'G', 4: 'P'}
    for row in obs_3x3:
        print("  " + " ".join(symbols[c] for c in row))
    
    print(f"\nNumeric values: {obs_3x3.tolist()}")
    
    # Extract components
    print("\n" + "-"*60)
    print("Extracting components:")
    print("-"*60)
    actions = history[:, 0].astype(int)
    observations = history[:, 1:10].reshape(-1, 3, 3)
    rewards = history[:, 10]
    
    print(f"Actions shape: {actions.shape}")
    print(f"Observations shape: {observations.shape}")
    print(f"Rewards shape: {rewards.shape}")
    
    # Check if ghost death penalty was applied
    ghost_death = (rewards == -102).any()  # -100 - 2 = -102
    
    print(f"\nTotal reward: {rewards.sum():.0f}")
    print(f"Pellets collected: {(rewards == 10).sum() + (rewards == 8).sum()}")  # 10 or 10-2=8
    print(f"Game ended by ghost: {ghost_death}")
    if ghost_death:
        print("  -> Ghost death penalty applied: -2 to all rewards")
