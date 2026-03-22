from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget,QWidget
from PyQt5 import uic
import sys
from PyQt5.QtCore import QTimer, Qt

from window_interaction_handler import WindowInteractionHandler

from manual_tab import ManualTab
from training_ai_tab import TrainingAiTab
from autoscan_tab import AutoScan
from a_eye_tab import A_Eye_Tab

from image_frame_manager import ImageFrameManager
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("main_window.ui", self)

        # transparent, frameless, always on top
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)

        # interaction handler (drag/resize since frameless)
        self.interaction_handler = WindowInteractionHandler(self)

        # screenshot of the microscope view region
        self.image_frame_manager = ImageFrameManager(self.image_frame)

        # tabs
        self.tab_widget: QTabWidget = self.findChild(QTabWidget, "all_tabWidget")
        self.manual_tab   = ManualTab()
        self.ai_tab       = TrainingAiTab()
        self.A_Eye_Tab    = A_Eye_Tab(self.image_frame_manager)
        self.autoscan_tab = AutoScan(self.manual_tab, self.image_frame_manager, self.A_Eye_Tab)
        self.tab_widget.addTab(self.manual_tab,    "Manual")
        self.tab_widget.addTab(self.ai_tab,        "Train")
        self.tab_widget.addTab(self.A_Eye_Tab,     "A-Eye")
        self.tab_widget.addTab(self.autoscan_tab,  "Auto")

        self.show()

    def mousePressEvent(self, event):
        self.interaction_handler.mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.interaction_handler.mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.interaction_handler.mouseReleaseEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
