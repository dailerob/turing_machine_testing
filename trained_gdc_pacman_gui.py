"""
Trained GDC Pacman GUI
Watch a trained Generative Dense Chain (GDC) agent play Pacman.
The GDC learns from previous games and selects actions based on predicted rewards.
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
from enum import IntEnum
from typing import Optional, List

from pacman_simulator import MiniPacmanSimulator, Cell
from pacman_gdc_player import run_gdc_learning, select_action_with_gdc, reverse_ewma, update_state_dist
from generative_dense_chain import GenerativeDenseChain


class TrainedGDCPacmanGUI:
    """GUI to watch a trained GDC agent play Pacman."""
    
    # Colors for each cell type
    COLORS = {
        Cell.EMPTY: '#1a1a2e',      # Dark blue background
        Cell.WALL: '#16213e',        # Darker blue walls
        Cell.PELLET: '#f0f0f0',      # White pellets
        Cell.GHOST: '#e94560',       # Red ghosts
        Cell.PACMAN: '#ffc107',      # Yellow pacman
    }
    
    GHOST_COLORS = ['#e94560', '#00ff88', '#00bfff', '#ff69b4']
    ACTION_NAMES = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT']
    
    def __init__(
        self,
        n_ghosts: int = 3,
        cell_size: int = 35,
        n_training_games: int = 50,
        training_seed: int = 42,
        reward_halflife: float = 1.0,
        exploration_rate: float = 0.1
    ):
        # Standard maze is 15x15
        self.width = 15
        self.height = 15
        self.n_ghosts = n_ghosts
        self.cell_size = cell_size
        self.reward_halflife = reward_halflife
        self.exploration_rate = exploration_rate  # Used during training only
        
        # Game state
        self.game: Optional[MiniPacmanSimulator] = None
        self.gdc: Optional[GenerativeDenseChain] = None
        self.current_observations: List[np.ndarray] = []
        self.state_dist: Optional[np.ndarray] = None  # Maintained incrementally for O(n) speed
        self.running = False
        self.step_delay = 200  # milliseconds between steps
        self.total_games = 0
        self.total_wins = 0
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("GDC Pacman Agent")
        self.root.configure(bg='#0f0f23')
        self.root.resizable(False, False)
        
        # Bind keyboard events
        self.root.bind('<space>', lambda e: self.toggle_running())
        self.root.bind('<r>', lambda e: self.reset_game())
        self.root.bind('<Escape>', lambda e: self.root.quit())
        
        # Show training progress
        self._create_loading_screen()
        self.root.update()
        
        # Train the GDC
        self._train_gdc(n_training_games, training_seed)
        
        # Remove loading screen and create main GUI
        self.loading_frame.destroy()
        self._create_widgets()
        
        # Initialize first game
        self._init_game()
        self._draw_game()
    
    def _create_loading_screen(self):
        """Create a loading screen shown during training."""
        self.loading_frame = tk.Frame(self.root, bg='#0f0f23', padx=50, pady=50)
        self.loading_frame.pack(expand=True, fill='both')
        
        title = tk.Label(
            self.loading_frame,
            text="GDC PACMAN AGENT",
            font=('Arial', 24, 'bold'),
            fg='#ffc107',
            bg='#0f0f23'
        )
        title.pack(pady=20)
        
        self.loading_label = tk.Label(
            self.loading_frame,
            text=f"Training GDC agent (exploration={self.exploration_rate:.0%})...",
            font=('Arial', 14),
            fg='#00ff88',
            bg='#0f0f23'
        )
        self.loading_label.pack(pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.loading_frame,
            variable=self.progress_var,
            maximum=100,
            length=300,
            mode='determinate'
        )
        self.progress_bar.pack(pady=10)
        
        self.progress_text = tk.Label(
            self.loading_frame,
            text="0 / 50 games",
            font=('Arial', 12),
            fg='#888888',
            bg='#0f0f23'
        )
        self.progress_text.pack(pady=5)
    
    def _train_gdc(self, n_games: int, seed: int):
        """Train the GDC by playing games."""
        np.random.seed(seed)
        
        all_histories = []
        
        gdc = None
        
        for game_idx in range(n_games):
            # Update progress
            progress = (game_idx / n_games) * 100
            self.progress_var.set(progress)
            self.progress_text.config(text=f"{game_idx} / {n_games} games")
            self.root.update()
            
            # Play game
            use_random = (game_idx == 0)
            history = self._play_training_game(gdc, use_random)
            all_histories.append(history)
            
            # Rebuild GDC
            gdc = GenerativeDenseChain(
                sequences=all_histories,
                alpha=0.6,
                beta=0.1,
                partial_match=True,
                initial_dist='sequence_starts'
            )
        
        self.gdc = gdc
        self.all_histories = all_histories
        
        # Final update
        self.progress_var.set(50)
        self.progress_text.config(text=f"{n_games} / {n_games} games - Complete!")
        self.root.update()
    
    def _play_training_game(
        self,
        gdc: Optional[GenerativeDenseChain],
        use_random: bool
    ) -> np.ndarray:
        """Play a single training game with incremental state updates."""
        game = MiniPacmanSimulator(
            n_ghosts=self.n_ghosts,
            obs_size=3,
            maze_type='standard'
        )
        
        game_observations = []
        # Initialize state distribution for incremental updates (O(n) instead of O(n²))
        state_dist = gdc._get_initial_distribution() if gdc is not None else None
        
        for step in range(20):
            obs = game.get_observation()
            obs_flat = obs.flatten()
            
            if use_random or gdc is None:
                action = np.random.randint(0, 5)
            elif self.exploration_rate > 0 and np.random.random() < self.exploration_rate:
                # Exploration: random action during training
                action = np.random.randint(0, 5)
            else:
                action = select_action_with_gdc(gdc, state_dist, obs_flat)
            
            _, reward, done, _ = game.step(action)
            
            record = np.zeros(11, dtype=np.float32)
            record[0] = action
            record[1:10] = obs_flat
            record[10] = reward
            game_observations.append(record)
            
            # Incrementally update state distribution
            if gdc is not None:
                state_dist = update_state_dist(gdc, state_dist, record)
            
            if done:
                break
        
        history = np.array(game_observations, dtype=np.float32)
        
        # Apply reverse EWMA to rewards (smooth from end to start)
        history[:, 10] = reverse_ewma(history[:, 10], halflife=self.reward_halflife)
        
        return history
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Title frame
        title_frame = tk.Frame(self.root, bg='#0f0f23')
        title_frame.pack(pady=10)
        
        title_label = tk.Label(
            title_frame,
            text="GDC PACMAN AGENT",
            font=('Arial', 24, 'bold'),
            fg='#ffc107',
            bg='#0f0f23'
        )
        title_label.pack()
        
        subtitle = tk.Label(
            title_frame,
            text="Watching trained GDC play",
            font=('Arial', 12),
            fg='#888888',
            bg='#0f0f23'
        )
        subtitle.pack()
        
        # Stats frame
        stats_frame = tk.Frame(self.root, bg='#0f0f23')
        stats_frame.pack(pady=5)
        
        self.score_label = tk.Label(
            stats_frame,
            text="Score: 0",
            font=('Arial', 16, 'bold'),
            fg='#00ff88',
            bg='#0f0f23'
        )
        self.score_label.pack(side=tk.LEFT, padx=20)
        
        self.pellets_label = tk.Label(
            stats_frame,
            text="Pellets: 0",
            font=('Arial', 16, 'bold'),
            fg='#00bfff',
            bg='#0f0f23'
        )
        self.pellets_label.pack(side=tk.LEFT, padx=20)
        
        self.step_label = tk.Label(
            stats_frame,
            text="Step: 0",
            font=('Arial', 16, 'bold'),
            fg='#ff69b4',
            bg='#0f0f23'
        )
        self.step_label.pack(side=tk.LEFT, padx=20)
        
        # Game stats frame
        game_stats_frame = tk.Frame(self.root, bg='#0f0f23')
        game_stats_frame.pack(pady=5)
        
        self.games_label = tk.Label(
            game_stats_frame,
            text="Games: 0 | Wins: 0",
            font=('Arial', 12),
            fg='#888888',
            bg='#0f0f23'
        )
        self.games_label.pack()
        
        # Game canvas
        canvas_width = self.width * self.cell_size
        canvas_height = self.height * self.cell_size
        
        canvas_frame = tk.Frame(self.root, bg='#0f0f23', padx=10, pady=10)
        canvas_frame.pack()
        
        self.canvas = tk.Canvas(
            canvas_frame,
            width=canvas_width,
            height=canvas_height,
            bg=self.COLORS[Cell.EMPTY],
            highlightthickness=2,
            highlightbackground='#ffc107'
        )
        self.canvas.pack()
        
        # Action display
        self.action_label = tk.Label(
            self.root,
            text="Action: --",
            font=('Arial', 14, 'bold'),
            fg='#ffc107',
            bg='#0f0f23'
        )
        self.action_label.pack(pady=5)
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Press SPACE to Start | R to Reset | ESC to Quit",
            font=('Arial', 11),
            fg='#888888',
            bg='#0f0f23'
        )
        self.status_label.pack(pady=5)
        
        # Game over label
        self.game_over_label = tk.Label(
            self.root,
            text="",
            font=('Arial', 20, 'bold'),
            fg='#e94560',
            bg='#0f0f23'
        )
        self.game_over_label.pack(pady=5)
        
        # Button frame
        button_frame = tk.Frame(self.root, bg='#0f0f23')
        button_frame.pack(pady=10)
        
        self.run_btn = tk.Button(
            button_frame,
            text="Start",
            font=('Arial', 12, 'bold'),
            fg='#0f0f23',
            bg='#00ff88',
            activebackground='#00cc6a',
            command=self.toggle_running,
            padx=20,
            pady=5,
            width=8
        )
        self.run_btn.pack(side=tk.LEFT, padx=10)
        
        reset_btn = tk.Button(
            button_frame,
            text="Reset",
            font=('Arial', 12, 'bold'),
            fg='#0f0f23',
            bg='#00bfff',
            activebackground='#0099cc',
            command=self.reset_game,
            padx=20,
            pady=5,
            width=8
        )
        reset_btn.pack(side=tk.LEFT, padx=10)
        
        quit_btn = tk.Button(
            button_frame,
            text="Quit",
            font=('Arial', 12, 'bold'),
            fg='#0f0f23',
            bg='#e94560',
            activebackground='#c93850',
            command=self.root.quit,
            padx=20,
            pady=5,
            width=8
        )
        quit_btn.pack(side=tk.LEFT, padx=10)
        
        # Speed control
        speed_frame = tk.Frame(self.root, bg='#0f0f23')
        speed_frame.pack(pady=5)
        
        tk.Label(
            speed_frame,
            text="Speed:",
            font=('Arial', 11),
            fg='#888888',
            bg='#0f0f23'
        ).pack(side=tk.LEFT, padx=5)
        
        self.speed_scale = tk.Scale(
            speed_frame,
            from_=50,
            to=500,
            orient=tk.HORIZONTAL,
            length=200,
            bg='#0f0f23',
            fg='#ffc107',
            troughcolor='#16213e',
            highlightthickness=0,
            command=self._update_speed
        )
        self.speed_scale.set(200)
        self.speed_scale.pack(side=tk.LEFT, padx=5)
        
        tk.Label(
            speed_frame,
            text="ms/step",
            font=('Arial', 11),
            fg='#888888',
            bg='#0f0f23'
        ).pack(side=tk.LEFT, padx=5)
    
    def _update_speed(self, value):
        """Update step delay from speed slider."""
        self.step_delay = int(value)
    
    def _init_game(self):
        """Initialize a new game."""
        self.game = MiniPacmanSimulator(
            n_ghosts=self.n_ghosts,
            obs_size=3,
            maze_type='standard'
        )
        self.current_observations = []
        self.current_step = 0
        # Initialize state distribution for incremental updates
        self.state_dist = self.gdc._get_initial_distribution() if self.gdc else None
    
    def _draw_game(self):
        """Draw the current game state."""
        self.canvas.delete("all")
        
        # Get display grid
        display = self.game.grid.copy()
        for gpos in self.game.ghost_positions:
            display[gpos] = Cell.GHOST
        display[self.game.pacman_pos] = Cell.PACMAN
        
        for y in range(self.height):
            for x in range(self.width):
                x1 = x * self.cell_size
                y1 = y * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                cell = display[y, x]
                
                # Draw cell background
                if cell == Cell.WALL:
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill=self.COLORS[Cell.WALL],
                        outline='#1a1a3e',
                        width=1
                    )
                else:
                    self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        fill=self.COLORS[Cell.EMPTY],
                        outline=''
                    )
                
                # Draw pellet
                if cell == Cell.PELLET:
                    cx = x1 + self.cell_size // 2
                    cy = y1 + self.cell_size // 2
                    r = 3
                    self.canvas.create_oval(
                        cx - r, cy - r, cx + r, cy + r,
                        fill=self.COLORS[Cell.PELLET],
                        outline=''
                    )
                
                # Draw Pacman
                elif cell == Cell.PACMAN:
                    cx = x1 + self.cell_size // 2
                    cy = y1 + self.cell_size // 2
                    r = self.cell_size // 2 - 4
                    self.canvas.create_arc(
                        cx - r, cy - r, cx + r, cy + r,
                        start=30, extent=300,
                        fill=self.COLORS[Cell.PACMAN],
                        outline='#e6a800',
                        width=2
                    )
                    # Eye
                    eye_r = 3
                    eye_x = cx + 2
                    eye_y = cy - r // 2
                    self.canvas.create_oval(
                        eye_x - eye_r, eye_y - eye_r,
                        eye_x + eye_r, eye_y + eye_r,
                        fill='#0f0f23',
                        outline=''
                    )
        
        # Draw ghosts
        for i, (gy, gx) in enumerate(self.game.ghost_positions):
            x1 = gx * self.cell_size
            y1 = gy * self.cell_size
            cx = x1 + self.cell_size // 2
            cy = y1 + self.cell_size // 2
            r = self.cell_size // 2 - 4
            
            ghost_color = self.GHOST_COLORS[i % len(self.GHOST_COLORS)]
            
            # Ghost body
            self.canvas.create_arc(
                cx - r, cy - r, cx + r, cy + r,
                start=0, extent=180,
                fill=ghost_color,
                outline=ghost_color
            )
            self.canvas.create_rectangle(
                cx - r, cy, cx + r, cy + r,
                fill=ghost_color,
                outline=ghost_color
            )
            
            # Ghost eyes
            eye_r = 4
            for ex in [-r//2, r//2]:
                self.canvas.create_oval(
                    cx + ex - eye_r, cy - r//3 - eye_r,
                    cx + ex + eye_r, cy - r//3 + eye_r,
                    fill='white',
                    outline=''
                )
                self.canvas.create_oval(
                    cx + ex - eye_r//2, cy - r//3 - eye_r//2,
                    cx + ex + eye_r//2, cy - r//3 + eye_r//2,
                    fill='#0f0f23',
                    outline=''
                )
        
        # Update labels
        self.score_label.config(text=f"Score: {self.game.score}")
        self.pellets_label.config(text=f"Pellets: {self.game.pellets_remaining}")
        self.step_label.config(text=f"Step: {self.current_step}")
        self.games_label.config(text=f"Games: {self.total_games} | Wins: {self.total_wins}")
        
        # Game over message
        if self.game.done:
            if self.game.won:
                self.game_over_label.config(text="WIN!", fg='#00ff88')
            else:
                self.game_over_label.config(text="GAME OVER", fg='#e94560')
        else:
            self.game_over_label.config(text="")
    
    def toggle_running(self):
        """Toggle auto-run mode."""
        self.running = not self.running
        if self.running:
            self.run_btn.config(text="Pause", bg='#ffc107')
            self._run_step()
        else:
            self.run_btn.config(text="Start", bg='#00ff88')
    
    def _run_step(self):
        """Execute one step of the GDC agent."""
        if not self.running or self.game.done:
            if self.game.done:
                self.running = False
                self.run_btn.config(text="Start", bg='#00ff88')
            return
        
        # Get current observation
        obs = self.game.get_observation()
        obs_flat = obs.flatten()
        
        # Select action using GDC (using maintained state_dist for O(n) speed)
        action = select_action_with_gdc(
            self.gdc,
            self.state_dist,
            obs_flat,
            verbose=False
        )
        
        # Execute action
        _, reward, done, _ = self.game.step(action)
        
        # Record observation
        record = np.zeros(11, dtype=np.float32)
        record[0] = action
        record[1:10] = obs_flat
        record[10] = reward
        self.current_observations.append(record)
        
        # Incrementally update state distribution
        self.state_dist = update_state_dist(self.gdc, self.state_dist, record)
        
        self.current_step += 1
        
        # Update action display
        self.action_label.config(text=f"Action: {self.ACTION_NAMES[action]}")
        
        # Check game end
        if done:
            self.total_games += 1
            if self.game.won:
                self.total_wins += 1
        
        # Redraw
        self._draw_game()
        
        # Schedule next step
        if not done and self.running:
            self.root.after(self.step_delay, self._run_step)
    
    def reset_game(self):
        """Reset the game (randomizes ghost positions)."""
        self.running = False
        self.run_btn.config(text="Start", bg='#00ff88')
        self._init_game()
        self.action_label.config(text="Action: --")
        self._draw_game()
    
    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


def main():
    print("Starting GDC Pacman GUI...")
    print("Training will begin first (50 games)")
    print("Using standard Pacman maze (15x15)")
    print()
    
    gui = TrainedGDCPacmanGUI(
        n_ghosts=2,
        cell_size=35,
        n_training_games=1000,
        training_seed=42,
        reward_halflife=1.0
    )
    gui.run()


if __name__ == "__main__":
    main()
