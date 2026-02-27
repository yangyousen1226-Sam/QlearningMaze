import numpy as np
import matplotlib.pyplot as plt

# Action definitions: movement directions
RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3
ACTIONS = [RIGHT, UP, LEFT, DOWN]

# Reward configuration for different game states
REWARD = {"move": -0.05, "finish": 1.0, "wall": -0.85, "bound": -0.85, "repeat": -0.3, "dead": -2}


# REWARD = {"move":-0.05, "finish":1.0, "wall":-0.3, "bound":-0.3, "repeat":-0.3, "dead":-2}

class Maze(object):
    """Maze environment for rat navigation with traps, fire and dynamic rewards"""

    def __init__(self, period, maze_map=None):
        # maze_map: 1=free space, 0=wall, 2=trap, 3=fire
        # Fire appears in second half of each period (period must be even)
        if not isinstance(period, (int)):
            raise TypeError('period should be int var')
        if not period % 2 == 0:
            raise TypeError('period should be even')

        self.period = period
        self.time = 0  # Track current time step in fire period
        self.maze = np.array(maze_map, dtype=np.int)
        self.aim = (self.maze.shape[0] - 1, self.maze.shape[1] - 1)  # Target position (bottom-right)
        self.free_cells = [(i, j) for i in range(self.maze.shape[0]) for j in range(self.maze.shape[1]) if
                           self.maze[i, j] == 1 or self.maze[i, j] == 3]
        self.trap = [(i, j) for i in range(self.maze.shape[0]) for j in range(self.maze.shape[1]) if
                     self.maze[i, j] == 2]  # Mouse trap positions
        self.fire = [(i, j) for i in range(self.maze.shape[0]) for j in range(self.maze.shape[1]) if
                     self.maze[i, j] == 3]  # Fire positions
        if len(self.fire) == 0:
            self.period = 0  # Disable fire mechanics if no fire cells
        self.rat = (0, 0)  # Rat's initial position (top-left)
        self.score = 0  # Cumulative reward score
        self.min_reward = -0.5 * self.maze.shape[0] * self.maze.shape[1]  # Minimum reward threshold for game over
        self.visited = []  # Track visited positions
        self.reset()

    def random_generate(self, size, block_ratio):
        """Generate random maze (unused in main demo)"""
        random_maze = np.random.random_sample([size * size - 2])  # Exclude start and end points
        self.maze = np.zeros([size, size], dtype=np.int)
        for i in range(size):
            for j in range(size):
                if (i == 0 and j == 0) or (i == size - 1 and j == size - 1):
                    self.maze[i, j] = 1  # Keep start/end as free space
                    continue
                if random_maze[i * size + j - 1] > block_ratio:
                    self.maze[i, j] = 0  # Wall
                else:
                    self.maze[i, j] = 1  # Free space

    def act(self, action, get_state_temp):
        """Execute action, update state and return reward/game status"""
        rat_i, rat_j = self.rat
        next_i, next_j = self.move2next(rat_i, rat_j, action)
        nrow, ncol = self.maze.shape

        game_status = ""
        # Check boundary violation
        if next_i >= nrow or next_j >= ncol or next_i < 0 or next_j < 0:
            award = REWARD['bound']
            next_i = rat_i
            next_j = rat_j
            self.visited.append((next_i, next_j))
            game_status = "blocked"
        # Check wall collision
        elif self.maze[next_i, next_j] == 0:
            award = REWARD['wall']
            next_i = rat_i
            next_j = rat_j
            self.visited.append((next_i, next_j))
            game_status = "blocked"
        # Check goal reached
        elif next_i == self.aim[0] and next_j == self.aim[1]:
            award = REWARD['finish']
            game_status = "win"
        # Check repeated position
        elif (next_i, next_j) in self.visited:
            award = REWARD['repeat']
            self.visited.append((next_i, next_j))
            game_status = "normal"
        # Check trap collision
        elif (next_i, next_j) in self.trap:
            award = REWARD['dead']
            game_status = "lose"
        # Check fire damage (active in second half of period)
        elif (next_i, next_j) in self.fire and self.period != 0 and ((self.time + 1) % self.period) >= self.period / 2:
            award = REWARD['dead']
            game_status = "lose"
        # Normal movement
        else:
            award = REWARD['move']
            self.visited.append((next_i, next_j))
            game_status = "normal"

        # Additional fire check for blocked states
        if game_status == "blocked" and (rat_i, rat_j) in self.fire and self.period != 0 and (
                (self.time + 1) % self.period) >= self.period / 2:
            award = REWARD['dead']
            game_status = "lose"

        # Update time step if fire mechanics enabled
        if self.period != 0:
            self.time = (self.time + 1) % self.period
        self.score += award
        self.rat = (next_i, next_j)

        # Game over if score too low
        if self.score < self.min_reward:
            game_status = "lose"

        return get_state_temp(), award, game_status

    def move2next(self, i, j, action):
        """Calculate next position based on action"""
        if action == UP:
            next_i = i - 1
            next_j = j
        elif action == RIGHT:
            next_i = i
            next_j = j + 1
        elif action == LEFT:
            next_i = i
            next_j = j - 1
        elif action == DOWN:
            next_i = i + 1
            next_j = j
        return next_i, next_j

    def reset(self, rat=(0, 0), time=0):
        """Reset maze state to initial conditions"""
        self.time = time
        self.rat = rat
        self.score = 0
        self.visited = []

    def get_current_state(self):
        """Get full maze state (flattened array with rat position marked)"""
        maze_temp = np.copy(np.array(self.maze, dtype=np.float))
        maze_temp[self.rat[0], self.rat[1]] = 0.5  # Mark rat position with 0.5
        maze_temp = maze_temp.reshape([1, -1])
        return maze_temp

    def get_current_state_simple(self):
        """Get simplified state (rat position + time step)"""
        return (*self.rat, self.time)

    def valid_actions(self, cell=None):
        """Return list of valid actions for current/specified cell (avoid walls/boundaries)"""
        if cell is None:
            row, col = self.rat
        else:
            row, col = cell
        actions = [RIGHT, UP, LEFT, DOWN]
        nrows, ncols = self.maze.shape

        # Remove actions that go out of bounds
        if row == 0:
            actions.remove(UP)
        elif row == nrows - 1:
            actions.remove(DOWN)

        if col == 0:
            actions.remove(LEFT)
        elif col == ncols - 1:
            actions.remove(RIGHT)

        # Remove actions that hit walls
        if row > 0 and self.maze[row - 1, col] == 0:
            actions.remove(UP)
        if row < nrows - 1 and self.maze[row + 1, col] == 0:
            actions.remove(DOWN)

        if col > 0 and self.maze[row, col - 1] == 0:
            actions.remove(LEFT)
        if col < ncols - 1 and self.maze[row, col + 1] == 0:
            actions.remove(RIGHT)

        return actions


def show_maze(Maze):
    """Visualize maze with rat position, visited cells and goal"""
    plt.grid(True)
    nrows, ncols = Maze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Prepare visualization canvas
    canvas = np.copy(np.array(Maze.maze, dtype=np.float))
    for row, col in Maze.visited:
        canvas[row, col] = 0.6  # Mark visited cells
    rat_row, rat_col = Maze.rat
    canvas[rat_row, rat_col] = 0.3  # Mark rat position
    canvas[nrows - 1, ncols - 1] = 0.9  # Mark goal position
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    return img


if __name__ == "__main__":
    # Demo maze initialization
    maze = np.array([
        [1., 0., 1., 1., 1., 1., 1., 1.],
        [1., 0., 1., 0, 1., 0., 1., 1.],
        [1., 1., 1., 1., 0., 1., 0., 1.],
        [1., 1., 1., 0., 1., 1., 1., 1.],
        [0, 1., 0., 1., 1., 1., 1., 1.],
        [1., 1., 1., 0., 1., 0., 0., 0.],
        [1., 1., 1., 0., 1., 1., 1., 1.],
        [1., 1., 1., 1., 0., 1., 1., 1.]
    ])

    # Create maze instance and test actions
    my_maze = Maze(period=2, maze_map=maze)  # Added period (required parameter)
    my_maze.act(DOWN, my_maze.get_current_state)
    my_maze.act(LEFT, my_maze.get_current_state)
    my_maze.act(RIGHT, my_maze.get_current_state)
    my_maze.act(RIGHT, my_maze.get_current_state)
    my_maze.act(UP, my_maze.get_current_state)

    # Visualize and display maze
    show_maze(my_maze)
    plt.show()
    print()