from PyQt5.QtWidgets import QLabel, QComboBox, QPushButton, QMessageBox, QWidget, QGroupBox, QGridLayout, QPlainTextEdit, QSpinBox
from PyQt5.QtCore import QBasicTimer, QRect
import numpy as np
from maze import Maze
from train_qtable import QTableModel
import time

from draw_ui import Draw_ui

# Custom widget for user-defined maze configuration and Q-table training/playback
class Ui_userDefine(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None  # QTableModel instance for maze training
        self.playing_index = -1  # Index for maze traversal playback
        self.problem_solving = False  # Flag to check if training is in progress

    # Initialize UI components and layout
    def initUI(self):
        self.resize(1200, 700)

        self.timer = QBasicTimer()  # Timer for maze playback animation
        widget1 = QWidget(parent=self)
        widget1.setGeometry(QRect(30, 50, 800, 500))
        table_area = QGroupBox(parent=widget1) # Graphic display area for maze
        table_area.setGeometry(QRect(widget1.x(), widget1.y(), widget1.width(), widget1.height()))

        # Initialize drawing component for maze visualization
        self.Plot = Draw_ui(width=3, height=3, dpi=100)
        gridlayout1 = QGridLayout(table_area)  # Layout for embedding drawing component
        gridlayout1.addWidget(self.Plot, 0, 1)

        # Label for maze input prompt
        pic_choose_label = QLabel(self)
        pic_choose_label.move(table_area.x()+table_area.width()+30, table_area.y()+20)
        pic_choose_label.setText("Input Maze:")
        # Label for maze value explanation (1:space, 0:wall, 2:trap, 3:fire)
        pic_choose_label2 = QLabel(self)
        pic_choose_label2.move(pic_choose_label.geometry().x(), pic_choose_label.y()+pic_choose_label.height()+20)
        pic_choose_label2.setText("(1 for space,0 for wall,2 for trap,\n3 for fire)")

        # Text edit for user to input maze matrix
        self.maze_input = QPlainTextEdit(parent=self)
        self.maze_input.setGeometry(QRect(pic_choose_label2.x(), pic_choose_label2.y()+pic_choose_label2.height()+20, 300, 200))
        # Default maze matrix example
        self.maze_input.setPlainText(
            '1, 0, 1, 1, 1, 1, 1,\r\n1, 1, 3, 0, 0, 1, 0,\r\n0, 0, 0, 1, 1, 1, 0,\r\n1, 1, 1, 1, 0, 0, 1,\r\n1, 0, 0, 0, 1, 1, 1,\r\n1, 0, 1, 1, 1, 2, 2,\r\n1, 1, 1, 0, 1, 1, 1,'
        )

        # Label for fire period input
        period_label = QLabel(parent=self)
        period_label.setText('Fire Period:')
        period_label.move(self.maze_input.x(), self.maze_input.height()+self.maze_input.y()+10)

        # Spin box for fire period value (multiplied by 2)
        self.period_input = QSpinBox(parent=self)
        self.period_input.setValue(1)
        self.period_input.move(period_label.x()+period_label.width()+15, period_label.y())

        # Label for fire period multiplier explanation
        period_label2 = QLabel(parent=self)
        period_label2.setText('*2')
        period_label2.move(self.period_input.x()+self.period_input.width()-40, self.period_input.y())

        # Button to confirm maze input
        maze_input_button = QPushButton(parent=self)
        maze_input_button.move(self.period_input.x()-50, self.period_input.y()+self.period_input.height()+10)
        maze_input_button.setText('Confirm Input')
        maze_input_button.pressed.connect(self.pic_change)

        middle_x = self.maze_input.geometry().x()+ self.maze_input.geometry().width()/2

        # Label for max training epochs input
        train_epoch_label = QLabel(parent=self)
        train_epoch_label.setText('Max Training Epochs:')
        train_epoch_label.move(self.maze_input.x(), maze_input_button.height() + maze_input_button.y() + 40)

        # Spin box for training epochs (multiplied by 1000)
        self.epoch_input = QSpinBox(parent=self)
        self.epoch_input.move(train_epoch_label.x() + train_epoch_label.width() + 76, train_epoch_label.y())
        self.epoch_input.setValue(30)

        # Label for epoch multiplier explanation
        train_epoch_label2 = QLabel(parent=self)
        train_epoch_label2.setText('*1000')
        train_epoch_label2.move(self.epoch_input.x() + self.epoch_input.width() - 40, self.epoch_input.y())

        # Button to start Q-table training
        self.solve_problem_button = QPushButton(parent=self)
        self.solve_problem_button.setText("Train")
        self.solve_problem_button.move(maze_input_button.x(), train_epoch_label.y()+train_epoch_label.height()+10)
        self.solve_problem_button.pressed.connect(self.solve_button_pressed)

        # Label to display training status
        self.solve_test = QLabel(parent=self)
        self.solve_test.setText("Training...")
        self.solve_test.resize(250, self.solve_test.height())
        self.solve_test.move(middle_x - self.solve_test.geometry().width() / 2,
                             self.solve_problem_button.y() + self.solve_problem_button.height() + 10)
        self.solve_test.setHidden(True)

        # Label for playback speed selection
        speed_choose_label = QLabel(self)
        speed_choose_label.move(train_epoch_label.x(), self.solve_test.y() + self.solve_test.height() + 10)
        speed_choose_label.setText("Playback Speed:")
        # Combo box for playback speed options
        self.play_speed_combo = QComboBox(self)
        self.play_speed_combo.move(speed_choose_label.geometry().x() + speed_choose_label.geometry().width() + 30,
                                   speed_choose_label.geometry().y())
        self.play_speed_combo.addItems(["High Speed", "Medium Speed", "Low Speed"])

        # Button to start maze traversal playback
        play_button = QPushButton(self)
        play_button.setText("Play Maze Traversal")
        play_button.move(speed_choose_label.x()+40,
                         self.play_speed_combo.geometry().y() + self.play_speed_combo.geometry().height() + 10)
        play_button.pressed.connect(self.play_button_pressed)

    # Process maze input and update visualization
    def pic_change(self):
        self.timer.stop()
        current_text = self.maze_input.toPlainText()
        rows = current_text.split('\n')
        maze_map = []
        try:
            # Parse input text to maze matrix
            for row in rows:
                row_sp = row.split(',')
                row_list= []
                for c in row_sp:
                    c = c.strip()
                    if len(c)==0:
                        continue
                    else:
                        row_list.append(int(c))
                maze_map.append(row_list)
        except:
            # Show error if input parsing fails
            QMessageBox.information(self, "Prompt", "Unable to parse input", QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)
            return

        # Check if all rows have same column count
        maze_len = len(maze_map[0])
        for i in range(1,len(maze_map)):
            if len(maze_map[i])!=maze_len:
                QMessageBox.information(self, "Prompt", "Error: Each row must have the same number of columns", QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)
                return

        # Initialize maze and QTableModel with input parameters
        my_maze = Maze(maze_map=np.array(maze_map), period=self.period_input.value()*2)
        self.model = QTableModel(my_maze)

        # Update maze visualization
        self.Plot.draw_root(self.model.my_maze, (0, 0), 1, 0, False)
        self.Plot.draw_qtable(qtable_model=self.model, time_=self.model.my_maze.period-1 if self.model.my_maze.period!=0 else 0, fire_flag=True)

    # Handle playback button click event
    def play_button_pressed(self):
        # Check if maze is initialized
        if self.model == None:
            QMessageBox.information(self, "Prompt", "Please input maze first", QMessageBox.Ok | QMessageBox.Close,
                                    QMessageBox.Close)
            return

        # Start maze traversal simulation
        self.model.play_game((0, 0), 0)
        speed_text = self.play_speed_combo.currentText()
        self.playing_index = 0
        # Set playback speed based on selection
        if speed_text == "High Speed":
            self.timer.start(150, self)
        elif speed_text == "Medium Speed":
            self.timer.start(500, self)
        else:
            self.timer.start(1500, self)

    # Timer event handler for playback animation
    def timerEvent(self, event):
        if event.timerId() == self.timer.timerId():
            period = self.model.my_maze.period
            # Determine fire flag based on current playback index
            if period != 0 and (self.playing_index % period) >= period / 2:
                fire_flag = True
            else:
                fire_flag = False

            # Update Q-table and maze visualization
            self.Plot.draw_qtable(self.model, self.playing_index % period if period != 0 else 0, fire_flag)
            self.Plot.draw_root(self.model.my_maze, (0,0), self.playing_index, period, fire_flag)

            self.playing_index = self.playing_index + 1

            # Reset playback index when end is reached
            if self.playing_index >= len(self.model.my_maze.visited) + 2:
                self.playing_index = 0
        else:
            super(Ui_userDefine, self).timerEvent(event)

    # Handle train button click event
    def solve_button_pressed(self):
        # Prevent duplicate training
        if self.problem_solving:
            return
        # Check if maze is initialized
        if type(self.model)==type(None):
            QMessageBox.information(self, "Prompt", "Please input maze first", QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)
            return

        # Set training status and update UI
        self.problem_solving = True
        self.playing_index = -1
        self.solve_test.setHidden(False)
        self.timer.stop()
        self.repaint()

        # Calculate total training epochs (spin box value * 1000)
        train_epoch = self.epoch_input.value()*1000
        start_time = time.time()
        # Start Q-table training
        self.model.train(output_line = self.solve_test, main_ui=self, epoch_N=train_epoch)
        end_time = time.time()

        # Show training completion message with elapsed time
        QMessageBox.information(self, "Prompt", "Training completed, time elapsed: %.3f s" % (end_time - start_time),
                                QMessageBox.Ok | QMessageBox.Close, QMessageBox.Close)

        # Update Q-table visualization after training
        self.Plot.draw_qtable(qtable_model=self.model,
                              time_=self.model.my_maze.period - 1 if self.model.my_maze.period != 0 else 0,
                              fire_flag=True)
        # Reset training status
        self.problem_solving = False
        self.solve_test.setHidden(True)