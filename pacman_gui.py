"""
Mini Pacman Game with GUI
A simple Pacman game using tkinter for the graphical interface.
Use arrow keys to move Pacman and collect all pellets while avoiding ghosts!
"""

import numpy as np
import tkinter as tk
from tkinter import messagebox
from enum import IntEnum


class Cell(IntEnum):
    EMPTY = 0
    WALL = 1
    PELLET = 2
    GHOST = 3
    PACMAN = 4


class MiniPacman:
    """Core game logic for Mini Pacman."""
    
    ACTIONS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]  # stay, up, down, left, right
    
    # Standard 15x15 Pacman-style maze (1 = wall, 0 = empty)
    STANDARD_MAZE = np.array([
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
    
    def __init__(self, width=15, height=15, n_ghosts=3, wall_density=0.15, maze_type='standard',
                 pacman_start=None):
        self.width = 15 if maze_type == 'standard' else width
        self.height = 15 if maze_type == 'standard' else height
        self.n_ghosts = n_ghosts
        self.wall_density = wall_density
        self.maze_type = maze_type
        # Use default start position for standard maze
        if maze_type == 'standard':
            self.pacman_start = pacman_start if pacman_start else self.DEFAULT_START_POS
        else:
            self.pacman_start = pacman_start
        self.score = 0
        self.reset()
    
    def reset(self):
        """Reset the game to initial state."""
        # Create grid with walls on border
        self.grid = np.full((self.height, self.width), Cell.PELLET)
        
        if self.maze_type == 'standard':
            # Use standard Pacman-style maze
            self.grid[self.STANDARD_MAZE] = Cell.WALL
        else:
            # Random maze
            self.grid[0, :] = self.grid[-1, :] = Cell.WALL
            self.grid[:, 0] = self.grid[:, -1] = Cell.WALL
            inner = self.grid[1:-1, 1:-1]
            wall_mask = np.random.random(inner.shape) < self.wall_density
            inner[wall_mask] = Cell.WALL
        
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
        self.initial_pellets = self.pellets_remaining
        
        # Track consecutive steps in same position (for stagnation penalty)
        self.steps_in_same_pos = 0
        self.last_pos = self.pacman_pos
        
        return self._get_obs()
    
    def _random_empty(self):
        """Find a random empty position on the grid."""
        while True:
            pos = (np.random.randint(1, self.height-1), np.random.randint(1, self.width-1))
            if self.grid[pos] != Cell.WALL and pos != getattr(self, 'pacman_pos', None):
                if not hasattr(self, 'ghost_positions') or pos not in self.ghost_positions:
                    return pos
    
    def _get_obs(self):
        """Return 3x3 grid centered on pacman."""
        obs = np.zeros((3, 3), dtype=np.int8)
        py, px = self.pacman_pos
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = py + dy, px + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    obs[dy+1, dx+1] = self.grid[ny, nx]
                else:
                    obs[dy+1, dx+1] = Cell.WALL
        # Mark ghosts in observation
        for gy, gx in self.ghost_positions:
            if abs(gy - py) <= 1 and abs(gx - px) <= 1:
                obs[gy - py + 1, gx - px + 1] = Cell.GHOST
        obs[1, 1] = Cell.PACMAN
        return obs
    
    def step(self, action):
        """Execute one game step."""
        if self.done:
            return self._get_obs(), 0, True, {}
        
        dy, dx = self.ACTIONS[action]
        ny, nx = self.pacman_pos[0] + dy, self.pacman_pos[1] + dx
        
        reward = 0
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
        
        return self._get_obs(), reward, self.done, {}
    
    def get_display_grid(self):
        """Return grid with all entities for rendering."""
        display = self.grid.copy()
        for gpos in self.ghost_positions:
            display[gpos] = Cell.GHOST
        display[self.pacman_pos] = Cell.PACMAN
        return display


class PacmanGUI:
    """Graphical User Interface for Mini Pacman."""
    
    # Colors for each cell type
    COLORS = {
        Cell.EMPTY: '#1a1a2e',      # Dark blue background
        Cell.WALL: '#16213e',        # Darker blue walls
        Cell.PELLET: '#f0f0f0',      # White pellets
        Cell.GHOST: '#e94560',       # Red ghosts
        Cell.PACMAN: '#ffc107',      # Yellow pacman
    }
    
    GHOST_COLORS = ['#e94560', '#00ff88', '#00bfff', '#ff69b4']  # Different ghost colors
    
    def __init__(self, width=15, height=15, n_ghosts=3, cell_size=35):
        self.cell_size = cell_size
        self.game = MiniPacman(width=width, height=height, n_ghosts=n_ghosts)
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Mini Pacman")
        self.root.configure(bg='#0f0f23')
        self.root.resizable(False, False)
        
        # Bind keyboard events
        self.root.bind('<Up>', lambda e: self.move(1))
        self.root.bind('<Down>', lambda e: self.move(2))
        self.root.bind('<Left>', lambda e: self.move(3))
        self.root.bind('<Right>', lambda e: self.move(4))
        self.root.bind('<space>', lambda e: self.restart_game())
        self.root.bind('<Escape>', lambda e: self.root.quit())
        
        self._create_widgets()
        self._draw_game()
        
        # Auto-move ghosts periodically
        self.ghost_move_delay = 300  # milliseconds
        self._schedule_ghost_move()
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Title frame
        title_frame = tk.Frame(self.root, bg='#0f0f23')
        title_frame.pack(pady=10)
        
        title_label = tk.Label(
            title_frame, 
            text="🎮 MINI PACMAN 🎮",
            font=('Arial', 24, 'bold'),
            fg='#ffc107',
            bg='#0f0f23'
        )
        title_label.pack()
        
        # Score frame
        score_frame = tk.Frame(self.root, bg='#0f0f23')
        score_frame.pack(pady=5)
        
        self.score_label = tk.Label(
            score_frame,
            text="Score: 0",
            font=('Arial', 16, 'bold'),
            fg='#00ff88',
            bg='#0f0f23'
        )
        self.score_label.pack(side=tk.LEFT, padx=20)
        
        self.pellets_label = tk.Label(
            score_frame,
            text=f"Pellets: {self.game.pellets_remaining}",
            font=('Arial', 16, 'bold'),
            fg='#00bfff',
            bg='#0f0f23'
        )
        self.pellets_label.pack(side=tk.LEFT, padx=20)
        
        # Game canvas
        canvas_width = self.game.width * self.cell_size
        canvas_height = self.game.height * self.cell_size
        
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
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Use Arrow Keys to Move | Space to Restart | Esc to Quit",
            font=('Arial', 11),
            fg='#888888',
            bg='#0f0f23'
        )
        self.status_label.pack(pady=5)
        
        # Game over label (hidden initially)
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
        
        restart_btn = tk.Button(
            button_frame,
            text="🔄 New Game",
            font=('Arial', 12, 'bold'),
            fg='#0f0f23',
            bg='#00ff88',
            activebackground='#00cc6a',
            command=self.restart_game,
            padx=20,
            pady=5
        )
        restart_btn.pack(side=tk.LEFT, padx=10)
        
        quit_btn = tk.Button(
            button_frame,
            text="❌ Quit",
            font=('Arial', 12, 'bold'),
            fg='#0f0f23',
            bg='#e94560',
            activebackground='#c93850',
            command=self.root.quit,
            padx=20,
            pady=5
        )
        quit_btn.pack(side=tk.LEFT, padx=10)
    
    def _draw_game(self):
        """Draw the current game state on the canvas."""
        self.canvas.delete("all")
        
        display = self.game.get_display_grid()
        
        for y in range(self.game.height):
            for x in range(self.game.width):
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
                    # Pacman body (circle with mouth)
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
        
        # Draw ghosts separately to use different colors
        for i, (gy, gx) in enumerate(self.game.ghost_positions):
            x1 = gx * self.cell_size
            y1 = gy * self.cell_size
            cx = x1 + self.cell_size // 2
            cy = y1 + self.cell_size // 2
            r = self.cell_size // 2 - 4
            
            ghost_color = self.GHOST_COLORS[i % len(self.GHOST_COLORS)]
            
            # Ghost body (rounded top, wavy bottom)
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
        
        # Show game over message
        if self.game.done:
            if self.game.won:
                self.game_over_label.config(
                    text="🎉 YOU WIN! 🎉",
                    fg='#00ff88'
                )
            else:
                self.game_over_label.config(
                    text="💀 GAME OVER 💀",
                    fg='#e94560'
                )
        else:
            self.game_over_label.config(text="")
    
    def move(self, action):
        """Handle player movement."""
        if not self.game.done:
            self.game.step(action)
            self._draw_game()
    
    def _schedule_ghost_move(self):
        """Schedule periodic ghost movement."""
        if not self.game.done:
            # Move ghosts without moving pacman (action 0 = stay)
            # But we only want ghosts to move, so we'll do a custom step
            self._move_ghosts_only()
            self._draw_game()
        self.root.after(self.ghost_move_delay, self._schedule_ghost_move)
    
    def _move_ghosts_only(self):
        """Move only the ghosts, not pacman."""
        if self.game.done:
            return
        
        # Move ghosts randomly
        for i, (gy, gx) in enumerate(self.game.ghost_positions):
            moves = [(gy+dy, gx+dx) for dy, dx in MiniPacman.ACTIONS 
                     if 0 <= gy+dy < self.game.height and 0 <= gx+dx < self.game.width 
                     and self.game.grid[gy+dy, gx+dx] != Cell.WALL]
            if moves:
                self.game.ghost_positions[i] = moves[np.random.randint(len(moves))]
        
        # Check collisions after ghost movement
        if self.game.pacman_pos in self.game.ghost_positions:
            self.game.score -= 20
            self.game.done = True
            self.game.won = False
    
    def restart_game(self):
        """Restart the game."""
        self.game.reset()
        self._draw_game()
    
    def run(self):
        """Start the game."""
        self.root.mainloop()


def main():
    """Main entry point."""
    print("Starting Mini Pacman...")
    print("Controls:")
    print("  Arrow Keys - Move Pacman")
    print("  Space      - Restart Game")
    print("  Escape     - Quit")
    print()
    
    # Create and run the game
    # You can customize: width, height, n_ghosts, cell_size
    gui = PacmanGUI(width=15, height=15, n_ghosts=3, cell_size=35)
    gui.run()


if __name__ == "__main__":
    main()
