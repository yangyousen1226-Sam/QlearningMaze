# a Grid-Based Maze Game based on Q-Learning
This project is a grid-based maze pathfinding game implemented with the Q-Learning reinforcement learning algorithm. An agent is trained to navigate from the top-left start to the bottom-right end of the maze, avoiding static walls, traps, and periodically activated dynamic flames. It supports predefined mazes (6 types) and user-customized mazes, with functions including Q-table training/saving/loading, optimal path animation playback, and training win rate curve plotting.

## How to Run
- Train the Q-table model first
- Run the main program to launch the interactive UI

## Technical Stack
- Programming Language: Python 3.6
- Reinforcement Learning: Q-Learning (Q-table, ε-greedy, TD Learning)
- UI Framework: PyQt5 (interactive interface, animation)
- Visualization: Matplotlib (maze, Q-table, win rate curves)
- Numerical Computing: NumPy (array operations, Q-table storage)