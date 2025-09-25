# window_interaction_handler.py

from PyQt5.QtCore import Qt, QPoint

class WindowInteractionHandler:
    """
    A class to manage window dragging via markers and resizing via bottom-right corner.
    Connect this to your MainWindow and forward mouse events to it.

    Usage:
        self.interaction_handler = WindowInteractionHandler(self)
        self.mousePressEvent = self.interaction_handler.mousePressEvent
        ...
    """

    def __init__(self, main_window):
        self.main = main_window
        self.is_resizing = False
        self.is_global_move = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.globalPos()
            self.start_geometry = self.main.image_frame.geometry()
            self.start_window_geometry = self.main.geometry()

        if self.is_near_bottom_right(event.pos()):
            self.is_resizing = True

        if self.main.move_mark_2.geometry().contains(event.pos() - self.main.image_frame.pos()):
            #or \ self.main.move_mark_1.geometry().contains(event.pos() - self.main.all_tabWidget.pos()) or \
            self.is_global_move = True
            self.start_pos = event.globalPos() - self.main.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self.is_resizing:
            delta = event.globalPos() - self.start_pos
            new_width = max(20, self.start_geometry.width() + delta.x())
            new_height = max(20, self.start_geometry.height() + delta.y())

            self.main.image_frame.setGeometry(
                self.start_geometry.left(), self.start_geometry.top(),
                new_width, new_height
            )

            self.main.resize(
                self.start_window_geometry.width() + delta.x(),
                self.start_window_geometry.height() + delta.y()
            )

            self.main.all_tabWidget.move(new_width, new_height - self.main.all_tabWidget.height())

        if self.is_global_move:
            self.main.move(event.globalPos() - self.start_pos)

    def mouseReleaseEvent(self, event):
        self.is_resizing = False
        self.is_global_move = False

    def is_near_bottom_right(self, pos):
        return (20 <= pos.x() <= self.main.image_frame.width()) and \
               (20 <= pos.y() <= self.main.image_frame.height())
