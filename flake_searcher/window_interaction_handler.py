from PyQt5.QtCore import QEvent, QObject, QPoint, QSize, Qt


class WindowInteractionHandler(QObject):
    """Handle explicit frameless-window movement and capture-frame resizing."""

    def __init__(self, main_window, move_handles, resize_handle, drag_header):
        super().__init__(main_window)
        self.main = main_window
        self.move_handles = tuple(move_handles)
        self.resize_handle = resize_handle
        self.drag_header = drag_header
        self._moving = False
        self._resizing = False
        self._move_offset = QPoint()
        self._resize_start_global = QPoint()
        self._resize_start_size = QSize()

        for widget in (*self.move_handles, self.resize_handle, self.drag_header):
            widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        event_type = event.type()

        if watched is self.drag_header and event_type == QEvent.MouseButtonDblClick:
            if event.button() == Qt.LeftButton:
                self.main.toggle_maximize_restore()
                return True

        if event_type == QEvent.MouseButtonPress:
            if event.button() != Qt.LeftButton:
                return False
            if watched is self.resize_handle:
                self._begin_resize(event.globalPos())
                return True
            if watched in self.move_handles or watched is self.drag_header:
                self._begin_move(event.globalPos())
                return True

        if event_type == QEvent.MouseMove:
            if self._resizing and event.buttons() & Qt.LeftButton:
                self._resize(event.globalPos())
                return True
            if self._moving and event.buttons() & Qt.LeftButton:
                self.main.move(event.globalPos() - self._move_offset)
                return True

        if event_type == QEvent.MouseButtonRelease:
            if event.button() == Qt.LeftButton and (self._moving or self._resizing):
                self._moving = False
                self._resizing = False
                return True

        return super().eventFilter(watched, event)

    def _begin_move(self, global_pos):
        if self.main.isMaximized():
            self.main.restore_for_drag(global_pos)
        self._moving = True
        self._resizing = False
        self._move_offset = global_pos - self.main.frameGeometry().topLeft()

    def _begin_resize(self, global_pos):
        if self.main.isMaximized():
            self.main.restore_window()
        self._resizing = True
        self._moving = False
        self._resize_start_global = QPoint(global_pos)
        self._resize_start_size = QSize(self.main.image_frame.size())

    def _resize(self, global_pos):
        delta = global_pos - self._resize_start_global
        requested = QSize(
            self._resize_start_size.width() + delta.x(),
            self._resize_start_size.height() + delta.y(),
        )
        self.main.resize_capture_frame(requested)

    @property
    def is_moving(self):
        return self._moving

    @property
    def is_resizing(self):
        return self._resizing
