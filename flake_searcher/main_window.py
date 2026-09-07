from PyQt5.QtWidgets import QApplication, QMainWindow, QStyle, QTabWidget
from PyQt5 import uic
import sys
from PyQt5.QtCore import QEvent, QPoint, QRect, QSize, Qt, QTimer

from .window_interaction_handler import WindowInteractionHandler

from .manual_tab import ManualTab
from .training_tab import TrainingAiTab
from .autoscan_tab import AutoScan
from .a_eye_tab import A_Eye_Tab

from .image_frame_manager import ImageFrameManager
from .paths import ui_path


INITIAL_CAPTURE_FRAME_SIZE = QSize(621, 611)
MIN_CAPTURE_FRAME_SIZE = QSize(46, 46)
PANEL_MIN_WIDTH = 300
PANEL_PREFERRED_WIDTH = 320
PANEL_MAX_WIDTH = 360
PANEL_MIN_HEIGHT = 300


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(str(ui_path("main_window.ui")), self)

        # transparent, frameless, always on top
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)

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

        self._normal_geometry = QRect()
        self._normal_capture_size = QSize(INITIAL_CAPTURE_FRAME_SIZE)
        self._external_handle_ready = False
        self._configure_panel()
        self._configure_window_controls()
        self._configure_external_move_handle()

        # Explicit handles avoid interpreting microscope-region clicks as resize drags.
        self.interaction_handler = WindowInteractionHandler(
            self,
            move_handles=(self.move_mark_2, self.window_title_label),
            resize_handle=self.resize_handle,
            drag_header=self.window_header,
        )

        self.resize_capture_frame(INITIAL_CAPTURE_FRAME_SIZE)

        self.show()
        self._show_external_move_handle()

    def _configure_panel(self):
        self.setStyleSheet(ui_path("instrument.qss").read_text(encoding="utf-8"))
        self.setLayoutDirection(Qt.LeftToRight)
        self.tab_widget.setLayoutDirection(Qt.LeftToRight)
        self.tab_widget.tabBar().setExpanding(True)
        self.tab_widget.tabBar().setUsesScrollButtons(False)
        requested = max(PANEL_PREFERRED_WIDTH, self.tab_widget.tabBar().sizeHint().width())
        self.panel_width = min(PANEL_MAX_WIDTH, max(PANEL_MIN_WIDTH, requested))
        # Lock the session width after font-aware selection so tab contents cannot
        # negotiate a different capture-frame size later.
        self.control_panel.setFixedWidth(self.panel_width)
        self.control_panel.setMinimumHeight(PANEL_MIN_HEIGHT)
        self.setMinimumSize(
            MIN_CAPTURE_FRAME_SIZE.width() + self.panel_width,
            MIN_CAPTURE_FRAME_SIZE.height(),
        )

    def _configure_window_controls(self):
        controls = (
            (self.minimize_btn, "Minimize", QStyle.SP_TitleBarMinButton),
            (self.maximize_restore_btn, "Maximize", QStyle.SP_TitleBarMaxButton),
            (self.close_btn, "Close", QStyle.SP_TitleBarCloseButton),
        )
        for button, name, standard_icon in controls:
            button.setIcon(self.style().standardIcon(standard_icon))
            button.setToolTip(name)
            button.setAccessibleName(name)
            button.setFocusPolicy(Qt.NoFocus)
        self.minimize_btn.clicked.connect(self.showMinimized)
        self.maximize_restore_btn.clicked.connect(self.toggle_maximize_restore)
        self.close_btn.clicked.connect(self.close)
        self._update_maximize_control()

    def _configure_external_move_handle(self):
        self.move_mark_2.setWindowFlags(
            Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.move_mark_2.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self._external_handle_ready = True

    def _position_external_move_handle(self):
        if not self._external_handle_ready:
            return
        frame_top_left = self.image_frame.mapToGlobal(QPoint())
        self.move_mark_2.move(
            frame_top_left.x(),
            frame_top_left.y() - self.move_mark_2.height(),
        )

    def _show_external_move_handle(self):
        if not self._external_handle_ready or not self.isVisible() or self.isMinimized():
            return
        self._position_external_move_handle()
        self.move_mark_2.show()

    def resize_capture_frame(self, requested_size):
        width = max(MIN_CAPTURE_FRAME_SIZE.width(), requested_size.width())
        height = max(MIN_CAPTURE_FRAME_SIZE.height(), requested_size.height())
        frame_size = QSize(width, height)
        self.image_frame.setFixedSize(frame_size)
        if not self.isMaximized():
            shell_height = max(PANEL_MIN_HEIGHT, self.control_panel.minimumSizeHint().height())
            self.setMinimumSize(
                MIN_CAPTURE_FRAME_SIZE.width() + self.panel_width,
                MIN_CAPTURE_FRAME_SIZE.height(),
            )
            self.resize(width + self.panel_width, max(height, shell_height))
        return frame_size

    def toggle_maximize_restore(self):
        if self.isMaximized():
            self.restore_window()
            return
        self._normal_geometry = QRect(self.geometry())
        self._normal_capture_size = QSize(self.image_frame.size())
        self.showMaximized()
        QTimer.singleShot(0, self._fit_capture_to_maximized_window)
        self._update_maximize_control()

    def restore_window(self):
        geometry = QRect(self._normal_geometry)
        frame_size = QSize(self._normal_capture_size)
        self.showNormal()
        if geometry.isValid():
            self.setGeometry(geometry)
        if frame_size.isValid():
            self.image_frame.setFixedSize(frame_size)
        self._update_maximize_control()

    def restore_for_drag(self, global_pos):
        maximized_width = max(1, self.width())
        horizontal_ratio = min(1.0, max(0.0, global_pos.x() / maximized_width))
        self.restore_window()
        target_x = global_pos.x() - round(self.width() * horizontal_ratio)
        target_y = global_pos.y() - self.window_header.height() // 2
        self.move(target_x, target_y)

    def _fit_capture_to_maximized_window(self):
        if not self.isMaximized():
            return
        central_size = self.centralWidget().size()
        width = max(MIN_CAPTURE_FRAME_SIZE.width(), central_size.width() - self.panel_width)
        height = max(MIN_CAPTURE_FRAME_SIZE.height(), central_size.height())
        self.image_frame.setFixedSize(width, height)

    def _update_maximize_control(self):
        restored = self.isMaximized()
        name = "Restore" if restored else "Maximize"
        icon = QStyle.SP_TitleBarNormalButton if restored else QStyle.SP_TitleBarMaxButton
        self.maximize_restore_btn.setIcon(self.style().standardIcon(icon))
        self.maximize_restore_btn.setToolTip(name)
        self.maximize_restore_btn.setAccessibleName(name)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._external_handle_ready:
            QTimer.singleShot(0, self._position_external_move_handle)
        if self.isMaximized():
            QTimer.singleShot(0, self._fit_capture_to_maximized_window)

    def moveEvent(self, event):
        super().moveEvent(event)
        if self._external_handle_ready:
            QTimer.singleShot(0, self._position_external_move_handle)

    def showEvent(self, event):
        super().showEvent(event)
        if self._external_handle_ready:
            QTimer.singleShot(0, self._show_external_move_handle)

    def hideEvent(self, event):
        if self._external_handle_ready:
            self.move_mark_2.hide()
        super().hideEvent(event)

    def closeEvent(self, event):
        if self._external_handle_ready:
            self.move_mark_2.close()
        super().closeEvent(event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            self._update_maximize_control()
            if self.isMinimized():
                self.move_mark_2.hide()
            else:
                QTimer.singleShot(0, self._show_external_move_handle)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
