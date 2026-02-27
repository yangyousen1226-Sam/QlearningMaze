# Import necessary PyQt5 modules for GUI creation
from PyQt5.QtWidgets import QTabWidget, QMainWindow, QDesktopWidget, QApplication
import sys
# Import custom UI modules for maze interfaces
from ui_basic import Ui_basic
from ui_userDefine import Ui_userDefine

# Main window class for the Maze application
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set window size
        self.resize(1200, 750)
        # Center window on screen
        self.center()
        # Set window title
        self.setWindowTitle('Maze')

        # Create tab widget to hold different maze interfaces
        self.tabW = QTabWidget(parent=self)
        # Initialize custom UI instances
        ui_userD = Ui_userDefine()
        ui_basic = Ui_basic()

        # Add tabs for existing mazes and user-defined mazes
        self.tabW.addTab(ui_basic, "existing mazes")
        self.tabW.addTab(ui_userD, "User-defined")
        # Resize tab widget to match window size
        self.tabW.resize(1200,750)
        # Initialize UI components for each tab
        ui_basic.initUI()
        ui_userD.initUI()

        # Display the main window
        self.show()

    # Method to center the window on the screen
    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2)

# Run the application if this file is executed directly
if __name__ == "__main__":
    app = QApplication([])
    ui = MainWindow()
    sys.exit(app.exec_())