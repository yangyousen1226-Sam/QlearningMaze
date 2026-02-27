from maze import Maze, ACTIONS
from maze_map import Mazes

import random
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# ===================== Global Configuration Parameters =====================
NUM_ACTION = 4
epoch_num = 6000  # Maximum number of training epochs
save_file = "saved_weight/first_simple"
data_size = 50  # Sliding window size for calculating win rate

# Directory to save win rate curves
plot_save_dir = "train_plots"
# Create save directory if it does not exist
if not os.path.exists(plot_save_dir):
    os.makedirs(plot_save_dir)


# ===================== Q-Table Learning Model Class =====================
class QTableModel(object):
    def __init__(self, my_maze, epsilon=0.1, learning_rate=0.1, gamma=0.9):
        self.Q_table = dict()  # Q-table, stores the value corresponding to (state, action)
        self.my_maze = my_maze  # Maze environment
        self.epsilon_ = epsilon  # Exploration rate (ε-greedy strategy)
        self.hsize = my_maze.maze.size // 2  # Auxiliary parameter for calculating win rate
        self.learning_rate = learning_rate  # Learning rate
        self.gamma = gamma  # Discount factor

        # New: Record win rate of each epoch (for plotting)
        self.epoch_win_rates = []  # Stores win rate of each training epoch
        self.epoch_list = []  # Stores corresponding epoch numbers
        self.stop_epoch = None  # Records the epoch when training stops

    def q_value(self, state):
        """Get Q-values of all actions in the current state"""
        return np.array([self.Q_table.get((state, action), 0.0) for action in ACTIONS])

    def predict(self, state):
        """Select optimal action based on Q-table (greedy strategy)"""
        return ACTIONS[np.argmax(self.q_value(state))]

    def train(self, output_line=None, main_ui=None, epoch_N=epoch_num):
        """Main training logic, returns the epoch when training terminates"""
        self.Q_table = dict()  # Reset Q-table
        win_history = []  # Record the outcome of each game (1=win, 0=lose)
        win_rate = 0.0  # Win rate
        self.stop_epoch = None  # Reset stop epoch

        # Reset win rate recording lists (ensure re-statistics for each maze training)
        self.epoch_win_rates = []
        self.epoch_list = []

        for epoch in range(epoch_N):
            # Initialize rat position
            if self.my_maze.period == 0:
                rat_cell = random.choice(self.my_maze.free_cells)
            else:
                rat_cell = (0, 0)
            self.my_maze.reset(rat_cell, 0)
            game_over = False
            state = self.my_maze.get_current_state_simple()
            n_episodes = 0  # Record steps of current epoch

            while not game_over:
                # Get valid actions in current state
                valid_actions = self.my_maze.valid_actions()
                if not valid_actions:
                    break
                state_now = state

                # ε-greedy strategy to select action (exploration + exploitation)
                if np.random.rand() < self.epsilon_:
                    action = random.choice(valid_actions)  # Random exploration
                else:
                    action = self.predict(state_now)  # Exploit Q-table to select optimal action

                # Execute action to get new state, reward, and game status
                state_next, reward, game_status = self.my_maze.act(action, self.my_maze.get_current_state_simple)

                # Ensure (state, action) exists in Q-table to avoid KeyError
                if (state, action) not in self.Q_table.keys():
                    self.Q_table[(state, action)] = 0.0

                # Q-Learning update formula
                max_next_Q = max([self.Q_table.get((state_next, a), 0.0) for a in ACTIONS])
                self.Q_table[(state, action)] += self.learning_rate * (
                        reward + self.gamma * max_next_Q - self.Q_table[(state, action)]
                )

                # Update game state and outcome record
                if game_status == 'win':
                    win_history.append(1)
                    game_over = True
                elif game_status == 'lose':
                    win_history.append(0)
                    game_over = True
                else:
                    game_over = False

                state = state_next
                n_episodes += 1

            # Calculate win rate in sliding window and record
            if len(win_history) > self.hsize:
                win_rate = sum(win_history[-self.hsize:]) / self.hsize
            self.epoch_win_rates.append(win_rate)
            self.epoch_list.append(epoch)

            # Print training progress
            template = "Epoch: {:03d}/{:d}    Episodes: {:d}  Win count: {:d} Win rate: {:.3f}"
            print(template.format(epoch, epoch_N - 1, n_episodes, sum(win_history), win_rate))

            # Compatible with UI display (retain original code logic)
            if output_line is not None and main_ui is not None:
                output_line.setText(template.format(epoch, epoch_N - 1, n_episodes, sum(win_history), win_rate))
                if epoch % 200 == 0:
                    main_ui.repaint()

            # Dynamically adjust exploration rate, reduce exploration in later training stages
            if win_rate > 0.9:
                self.epsilon_ = 0.05

            # Training convergence judgment (terminate early if 100% win rate is achieved)
            if self.my_maze.period == 0:
                if sum(win_history[-self.hsize:]) == self.hsize and self.completion_check():
                    # Fix: Escape 100% as 100%% to avoid formatting errors
                    print("Reached 100%% win rate at epoch: %d" % (epoch,))
                    self.stop_epoch = epoch  # Record stop epoch
                    break
            else:
                if win_rate > 0.8 and self.play_game((0, 0), 0):
                    print("Reached 100%% win rate at epoch: %d" % (epoch,))
                    self.stop_epoch = epoch  # Record stop epoch
                    break

        # If training terminates normally (not reaching max epoch), supplement win rate to 1.0
        if self.stop_epoch is not None:
            stop_idx = self.epoch_list.index(self.stop_epoch)
            # Set win rate to 1.0 from stop epoch onwards (if there is subsequent data)
            for i in range(stop_idx, len(self.epoch_win_rates)):
                self.epoch_win_rates[i] = 1.0

        return self.stop_epoch

    def completion_check(self):
        """Verify if all positions and all time steps can win"""
        period_temp = 1 if self.my_maze.period == 0 else self.my_maze.period
        for time_ in range(period_temp):
            for cell in self.my_maze.free_cells:
                if not self.my_maze.valid_actions(cell):
                    return False
                if not self.play_game(cell, time_):
                    return False
        return True

    def play_game(self, rat_cell, time):
        """Play game with trained Q-table to verify if it can win"""
        self.my_maze.reset(rat_cell, time)
        envstate = self.my_maze.get_current_state_simple()
        while True:
            prev_envstate = envstate
            action = self.predict(prev_envstate)  # Pure exploitation, no exploration
            envstate, reward, game_status = self.my_maze.act(action, self.my_maze.get_current_state_simple)
            if game_status == 'win':
                return True
            elif game_status == 'lose':
                return False

    def save_table(self, filename):
        """Save Q-table to file (overwrite existing file)"""
        # Delete existing file first (if it exists)
        if os.path.exists(f"{filename}.npy"):
            os.remove(f"{filename}.npy")
        np.save(filename, self.Q_table)

    def load_table(self, filename):
        """Load Q-table from file"""
        self.Q_table = np.load(filename, allow_pickle=True).item()
        return self.Q_table

    def plot_win_rate(self, maze_name):
        """Plot and save win rate change curve (mark stop point, overwrite existing image)"""
        # Set plot style
        plt.figure(figsize=(10, 6))

        # Plot win rate curve
        plt.plot(self.epoch_list, self.epoch_win_rates, color='blue', linewidth=1.5, label='Win Rate')
        # Plot 90% win rate reference line
        plt.axhline(y=0.9, color='red', linestyle='--', linewidth=1, label='90% Win Rate')

        # Add vertical annotation line if stop epoch exists
        if self.stop_epoch is not None:
            plt.axvline(x=self.stop_epoch, color='green', linestyle='-.', linewidth=1.5,
                        label=f'Stopped at Epoch {self.stop_epoch} (100% Win Rate)')

        # Set chart title and labels
        plt.title(f'Training Win Rate Curve - Maze: {maze_name}', fontsize=12)
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Win Rate', fontsize=10)
        plt.ylim(0, 1.05)  # Limit win rate range to 0~1.05 for better appearance
        plt.grid(True, alpha=0.3)  # Display grid to enhance readability
        plt.legend(loc='lower right')  # Place legend at bottom right to avoid blocking curve
        plt.tight_layout()  # Adaptive layout to prevent label truncation

        # Solve Chinese garbled character problem
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # Define save path
        save_path = os.path.join(plot_save_dir, f"{maze_name}_win_rate.png")

        # Core modification: Delete existing file (if exists) to ensure overwriting
        if os.path.exists(save_path):
            os.remove(save_path)

        # Save image (automatically overwrite files with the same name, double protection)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close canvas to release memory
        print(f"✅ Win rate curve saved to: {save_path} (overwrote existing file)")


# ===================== Main Program Entry =====================
if __name__ == "__main__":
    # Ensure directory for saving Q-table exists
    if not os.path.exists("saved_qtable"):
        os.makedirs("saved_qtable")

    # Traverse all mazes for training
    for maze_name in Mazes.keys():
        print(f"\n========== Start training maze: {maze_name} ==========")
        # Load maze and initialize maze environment
        maze = np.array(Mazes[maze_name])
        my_maze = Maze(maze_map=maze, period=2)
        # Initialize Q-table model and train
        model = QTableModel(my_maze)
        stop_epoch = model.train()  # Train and get stop epoch
        # Save trained Q-table (overwrite existing file)
        model.save_table(f"saved_qtable/{maze_name}")
        # Plot and save win rate curve (automatically mark stop point, overwrite existing file)
        model.plot_win_rate(maze_name)
        # Verify training effect
        model.play_game((0, 0), 0)
        print(f"========== Maze {maze_name} training completed ==========\n")