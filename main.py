from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget,QWidget
from PyQt5 import uic
import sys
from PyQt5.QtCore import QTimer, Qt

from window_interaction_handler import WindowInteractionHandler

from manual_tab import ManualTab
from ai_tab import AiTab
from autoscan_tab import AutoScan

from image_frame_manager import ImageFrameManager
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("main_window.ui", self)

        #making the window transparent
        self.setWindowFlags(Qt.WindowStaysOnTopHint)  # Window stays on top
        self.setWindowFlags(Qt.FramelessWindowHint) 
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)      
        self.show()
        self.setMouseTracking(True)

        # Setup interaction handler for resizing/drags
        self.interaction_handler = WindowInteractionHandler(self)
        # to make the pickup tab know where is the star, we need:
        self.image_frame_manager = ImageFrameManager(self.image_frame)

        # import settings tab
        self.tab_widget: QTabWidget = self.findChild(QTabWidget, "all_tabWidget")
        self.settings_tab = ManualTab()
        self.tab_widget.addTab(self.settings_tab, "manual control") 
        #note it was a whole tab with title and I demoted it to widget for simplicity, so I gave it the title here

        #import ai_tab
        self.tab_widget: QTabWidget = self.findChild(QTabWidget, "all_tabWidget")
        self.ai_tab = AiTab()
        self.tab_widget.addTab(self.ai_tab, "Ai setting")

        #import ai_tab
        self.tab_widget: QTabWidget = self.findChild(QTabWidget, "all_tabWidget")
        self.autoscan_tab = AutoScan()
        self.tab_widget.addTab(self.autoscan_tab, "Auto Scan")

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
