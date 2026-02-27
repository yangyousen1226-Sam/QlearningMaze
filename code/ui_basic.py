from PyQt5.QtWidgets import QLabel, QComboBox, QPushButton, QMessageBox, QWidget, QGroupBox, QGridLayout
from PyQt5.QtCore import QBasicTimer, QRect
from PyQt5.QtGui import QFont
import numpy as np
from maze_map import Mazes
from maze import Maze
from train_qtable import QTableModel
import time

from draw_ui import Draw_ui

# Basic UI class for maze Q-learning visualization and interaction
class Ui_basic(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None  # QTableModel instance (None initially) for maze Q-learning training/playback

    # Initialize UI components and layout
    def initUI(self):
        self.resize(1200, 700)

        # List of available maze options
        self.pic_list=['maze7_1', 'maze7_2', 'maze7_3', 'maze10_1', 'maze10_2', 'maze10_3' ]
        self.timer = QBasicTimer()  # Timer for playback animation
        widget1 = QWidget(parent=self)
        widget1.setGeometry(QRect(30, 50, 800, 500))
        table_area = QGroupBox(parent=widget1) # Area for maze visualization
        table_area.setGeometry(QRect(widget1.x(), widget1.y(), widget1.width(), widget1.height()))

        # Initialize drawing component for maze display
        self.Plot = Draw_ui(width=3, height=3, dpi=100)
        gridlayout1 = QGridLayout(table_area)  # Layout for visualization area
        gridlayout1.addWidget(self.Plot, 0, 1)

        # Maze selection label and combobox
        pic_choose_label = QLabel(self)
        pic_choose_label.move(table_area.x()+table_area.width()+30, table_area.y()+20)
        pic_choose_label.setText("Choose the Maze:")
        self.pic_choose_combo = QComboBox(self)
        self.pic_choose_combo.move(pic_choose_label.geometry().x()+pic_choose_label.geometry().width()+30, pic_choose_label.geometry().y())
        self.pic_choose_combo.resize(150,self.pic_choose_combo.geometry().height())
        self.pic_choose_combo.addItems(self.pic_list)
        self.pic_choose_combo.currentIndexChanged.connect(self.pic_change)
        self.pic_change()

        middle_x = (pic_choose_label.geometry().x() + self.pic_choose_combo.geometry().x() + self.pic_choose_combo.geometry().width()) / 2

        self.playing_index = -1  # Index for playback animation frame
        self.problem_solving = False  # Flag for training status

        # Training button setup
        self.solve_problem_button = QPushButton(parent=self)
        self.solve_problem_button.setText("Training (optional)")
        self.solve_problem_button.move(middle_x - self.solve_problem_button.width() / 2, self.pic_choose_combo.y()+self.pic_choose_combo.height()+100)
        self.solve_problem_button.pressed.connect(self.solve_button_pressed)

        # Label for training status display
        self.solve_test = QLabel(parent=self)
        self.solve_test.setText("Currently in training...")
        self.solve_test.resize(400, self.solve_test.height())
        self.solve_test.setFont(QFont("Fixed",7))
        self.solve_test.move(middle_x - self.solve_test.geometry().width() / 2,
                             self.solve_problem_button.geometry().y() + self.solve_problem_button.geometry().height() + 20)
        self.solve_test.setHidden(True)

        # Playback speed selection
        speed_choose_label = QLabel(self)
        speed_choose_label.move(self.solve_test.x()+20, self.solve_test.geometry().y() + 40)
        speed_choose_label.setText("Playback speed:")
        self.play_speed_combo = QComboBox(self)
        self.play_speed_combo.move(speed_choose_label.geometry().x() + speed_choose_label.geometry().width() + 30,
                                   speed_choose_label.geometry().y())
        self.play_speed_combo.addItems(["High speed", "Medium Speed", "Slow speed"])

        # Playback button setup
        play_button = QPushButton(self)
        play_button.setText("Play")
        play_button.move(middle_x - play_button.geometry().width() / 2,
                         self.play_speed_combo.geometry().y() + self.play_speed_combo.geometry().height() + 40)
        play_button.pressed.connect(self.play_button_pressed)

    # Handle maze selection change: load maze and Q-table, update visualization
    def pic_change(self):
        self.timer.stop()
        current_text = self.pic_choose_combo.currentText()
        maze = Mazes[current_text]
        my_maze = Maze(maze_map=np.array(maze), period=2)
        self.model = QTableModel(my_maze)

        # Load saved Q-table if exists, show message if not
        try:
            self.model.load_table('saved_qtable/'+current_text+'.npy')
        except:
            QMessageBox.information(self, "Prompt", "Could not find the Q-table saved file", QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)

        self.model.play_game((0, 0), 0)
        self.Plot.draw_root(self.model.my_maze, (0, 0), 1, 0, False)
        self.Plot.draw_qtable(qtable_model=self.model, time_=self.model.my_maze.period-1 if self.model.my_maze.period!=0 else 0, fire_flag=True)

    # Handle play button click: start playback animation with selected speed
    def play_button_pressed(self):
        if self.model == None:
            QMessageBox.information(self, "Prompt", "Please select the maze first", QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            return

        self.model.play_game((0, 0), 0)
        speed_text = self.play_speed_combo.currentText()
        self.playing_index = 0
        # Set timer interval based on selected speed
        if speed_text == "High speed":
            self.timer.start(150, self)
        elif speed_text == "Moderate speed":
            self.timer.start(500, self)
        else:
            self.timer.start(1500, self)

    # Timer event handler: update maze visualization for playback frame
    def timerEvent(self, event):
        if event.timerId() == self.timer.timerId():
            period = self.model.my_maze.period
            # Determine fire flag based on maze period
            if period != 0 and (self.playing_index % period) >= period / 2:
                fire_flag = True
            else:
                fire_flag = False

            # Update Q-table and maze visualization
            self.Plot.draw_qtable(self.model, self.playing_index % period if period != 0 else 0, fire_flag)
            self.Plot.draw_root(self.model.my_maze, (0,0), self.playing_index, period, fire_flag)

            self.playing_index = self.playing_index + 1

            # Reset playback when end of path is reached
            if self.playing_index >= len(self.model.my_maze.visited) + 2:
                self.playing_index = 0
        else:
            super(Ui_basic, self).timerEvent(event)

    # Handle training button click: run Q-learning training for selected maze
    def solve_button_pressed(self):
        if self.problem_solving:
            return
        if type(self.model)==type(None):
            QMessageBox.information(self, "Prompt", "Please select the maze first", QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)
            return

        self.problem_solving = True
        self.playing_index = -1
        self.solve_test.setHidden(False)
        self.timer.stop()
        self.repaint()

        # Record training time and run Q-table training
        start_time = time.time()
        self.model.train(output_line = self.solve_test, main_ui=self)
        end_time = time.time()

        # Show training completion message with time spent
        QMessageBox.information(self, "Prompt", "Training completed, time elapsed: %.3f s" % (end_time - start_time),
                                QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)

        # Update visualization with trained Q-table
        self.Plot.draw_qtable(qtable_model=self.model,
                              time_=self.model.my_maze.period - 1 if self.model.my_maze.period != 0 else 0,
                              fire_flag=True)
        self.problem_solving = False
        self.solve_test.setHidden(True)