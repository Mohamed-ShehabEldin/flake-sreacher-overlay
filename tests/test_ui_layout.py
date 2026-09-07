import os
from pathlib import Path
import subprocess
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEvent, QPoint, QPointF, QRect, QSize, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from flake_searcher.main_window import MainWindow


class ResponsiveMainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.move(100, 80)
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()

    def test_window_keeps_overlay_flags_and_explicit_capture_size(self):
        flags = self.window.windowFlags()
        self.assertTrue(flags & Qt.FramelessWindowHint)
        self.assertTrue(flags & Qt.WindowStaysOnTopHint)

        for requested in (
            QSize(320, 240),
            QSize(640, 480),
            QSize(800, 600),
            QSize(1280, 720),
            QSize(360, 720),
        ):
            with self.subTest(size=(requested.width(), requested.height())):
                top_left = self.window.image_frame.mapToGlobal(QPoint())
                actual = self.window.resize_capture_frame(requested)
                self.app.processEvents()

                self.assertEqual(actual, requested)
                self.assertEqual(self.window.image_frame.size(), requested)
                self.assertEqual(self.window.image_frame.mapToGlobal(QPoint()), top_left)
                self.assertEqual(self.window.control_panel.x(), requested.width())
                self.assertEqual(self.window.control_panel.width(), self.window.panel_width)
                self.assertEqual(self.window.width(), requested.width() + self.window.panel_width)
                self.assertGreaterEqual(self.window.height(), requested.height())

    def test_panel_width_is_bounded_and_tabs_fit_without_tab_scrolling(self):
        self.assertGreaterEqual(self.window.panel_width, 300)
        self.assertLessEqual(self.window.panel_width, 360)

        tab_bar = self.window.all_tabWidget.tabBar()
        self.assertFalse(tab_bar.usesScrollButtons())
        for index in range(tab_bar.count()):
            with self.subTest(index=index):
                self.assertTrue(tab_bar.rect().contains(tab_bar.tabRect(index)))

    def test_manual_tab_uses_vertical_scrolling_without_horizontal_overflow(self):
        scroll = self.window.manual_tab.manual_scroll_area
        self.assertEqual(scroll.horizontalScrollBarPolicy(), Qt.ScrollBarAlwaysOff)
        self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
        self.assertGreater(scroll.verticalScrollBar().maximum(), 0)

    def test_window_controls_are_outside_capture_and_accessible(self):
        frame_right = self.window.image_frame.mapToGlobal(
            QPoint(self.window.image_frame.width(), 0)
        ).x()
        for button, name in (
            (self.window.minimize_btn, "Minimize"),
            (self.window.maximize_restore_btn, "Maximize"),
            (self.window.close_btn, "Close"),
        ):
            with self.subTest(name=name):
                self.assertGreaterEqual(button.mapToGlobal(QPoint()).x(), frame_right)
                self.assertEqual(button.accessibleName(), name)
                self.assertEqual(button.toolTip(), name)

    def test_external_move_handle_is_adjacent_and_outside_capture_geometry(self):
        frame_rect = QRect(
            self.window.image_frame.mapToGlobal(QPoint()),
            self.window.image_frame.size(),
        )
        handle_rect = QRect(
            self.window.move_mark_2.mapToGlobal(QPoint()),
            self.window.move_mark_2.size(),
        )
        screenshot_rect = QRect(
            self.window.image_frame.mapToGlobal(QPoint(15, 15)),
            self.window.image_frame.size() - QSize(30, 30),
        )

        self.assertFalse(handle_rect.intersects(frame_rect))
        self.assertFalse(handle_rect.intersects(screenshot_rect))
        self.assertEqual(handle_rect.left(), frame_rect.left())
        self.assertEqual(handle_rect.bottom() + 1, frame_rect.top())
        self.assertEqual(self.window.move_mark_2.accessibleName(), "Move overlay")
        self.assertEqual(
            self.window.move_mark_2.toolTip(), "Drag to move the overlay"
        )

    def test_external_move_handle_moves_overlay_without_resizing_capture(self):
        handle = self.window.move_mark_2
        handler = self.window.interaction_handler
        initial_frame_origin = self.window.image_frame.mapToGlobal(QPoint())
        initial_frame_size = QSize(self.window.image_frame.size())
        local = handle.rect().center()
        start = handle.mapToGlobal(local)

        right_press = QMouseEvent(
            QEvent.MouseButtonPress,
            QPointF(local),
            QPointF(start),
            Qt.RightButton,
            Qt.RightButton,
            Qt.NoModifier,
        )
        self.assertFalse(handler.eventFilter(handle, right_press))
        self.assertFalse(handler.is_moving)

        left_press = QMouseEvent(
            QEvent.MouseButtonPress,
            QPointF(local),
            QPointF(start),
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
        self.assertTrue(handler.eventFilter(handle, left_press))
        destination = start + QPoint(37, 29)
        left_move = QMouseEvent(
            QEvent.MouseMove,
            QPointF(local),
            QPointF(destination),
            Qt.NoButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
        self.assertTrue(handler.eventFilter(handle, left_move))
        self.app.processEvents()

        self.assertEqual(
            self.window.image_frame.mapToGlobal(QPoint()),
            initial_frame_origin + QPoint(37, 29),
        )
        self.assertEqual(self.window.image_frame.size(), initial_frame_size)

    def test_resize_handle_changes_only_capture_size_and_keeps_origin(self):
        handler = self.window.interaction_handler
        initial_origin = self.window.image_frame.mapToGlobal(QPoint())
        initial_size = QSize(self.window.image_frame.size())
        start = self.window.resize_handle.mapToGlobal(
            self.window.resize_handle.rect().center()
        )

        handler._begin_resize(start)
        handler._resize(start + QPoint(79, -36))
        self.app.processEvents()

        self.assertEqual(self.window.image_frame.mapToGlobal(QPoint()), initial_origin)
        self.assertEqual(
            self.window.image_frame.size(),
            QSize(initial_size.width() + 79, initial_size.height() - 36),
        )
        self.assertTrue(handler.is_resizing)

    def test_header_move_changes_origin_not_capture_size(self):
        handler = self.window.interaction_handler
        initial_size = QSize(self.window.image_frame.size())
        start = self.window.window_header.mapToGlobal(
            self.window.window_header.rect().center()
        )
        handler._begin_move(start)
        self.window.move(start + QPoint(44, 27) - handler._move_offset)
        self.app.processEvents()

        self.assertEqual(self.window.image_frame.size(), initial_size)
        self.assertEqual(self.window.pos(), QPoint(144, 107))
        self.assertTrue(handler.is_moving)

    def test_maximize_then_restore_recovers_capture_and_window_geometry(self):
        self.window.resize_capture_frame(QSize(640, 480))
        self.window.move(90, 70)
        self.app.processEvents()
        initial_geometry = self.window.geometry()
        initial_capture = QSize(self.window.image_frame.size())

        self.window.toggle_maximize_restore()
        QTest.qWait(1)
        self.app.processEvents()
        self.assertTrue(self.window.isMaximized())
        self.assertEqual(self.window.maximize_restore_btn.accessibleName(), "Restore")

        self.window.toggle_maximize_restore()
        QTest.qWait(1)
        self.app.processEvents()
        self.assertFalse(self.window.isMaximized())
        self.assertEqual(self.window.geometry(), initial_geometry)
        self.assertEqual(self.window.image_frame.size(), initial_capture)
        self.assertEqual(self.window.maximize_restore_btn.accessibleName(), "Maximize")

    def test_close_control_uses_normal_window_close_path(self):
        self.assertTrue(self.window.isVisible())
        QTest.mouseClick(self.window.close_btn, Qt.LeftButton)
        self.app.processEvents()
        self.assertFalse(self.window.isVisible())

    def test_minimize_control_uses_normal_window_state(self):
        QTest.mouseClick(self.window.minimize_btn, Qt.LeftButton)
        self.app.processEvents()
        self.assertTrue(self.window.isMinimized())

    def test_layout_contract_at_common_display_scale_factors(self):
        project_root = Path(__file__).resolve().parents[1]
        script = r"""
from PyQt5.QtCore import QPoint, QSize
from PyQt5.QtWidgets import QApplication
from flake_searcher.main_window import MainWindow
app = QApplication([])
window = MainWindow()
app.processEvents()
assert window.image_frame.size() == QSize(621, 611)
assert 300 <= window.control_panel.width() <= 360
frame_top_left = window.image_frame.mapToGlobal(QPoint())
handle_top_left = window.move_mark_2.mapToGlobal(QPoint())
assert handle_top_left.x() == frame_top_left.x()
assert handle_top_left.y() + window.move_mark_2.height() == frame_top_left.y()
bar = window.all_tabWidget.tabBar()
assert all(bar.rect().contains(bar.tabRect(i)) for i in range(bar.count()))
for index, scroll_name in enumerate((
    'manual_scroll_area', 'training_scroll_area',
    'a_eye_settings_scroll_area', 'auto_scroll_area',
)):
    window.all_tabWidget.setCurrentIndex(index)
    app.processEvents()
    tab = window.all_tabWidget.currentWidget()
    assert getattr(tab, scroll_name).horizontalScrollBar().maximum() == 0
window.resize_capture_frame(QSize(320, 240))
window.all_tabWidget.setCurrentIndex(3)
app.processEvents()
assert window.image_frame.size() == QSize(320, 240)
auto = window.autoscan_tab
assert all(widget.isVisibleTo(auto) for widget in (
    auto.coord_display, auto.scan_info, auto.scan_progress,
    auto.start_btn, auto.stop_btn,
))
window.close()
"""
        for factor in ("1", "1.25", "1.5", "2"):
            with self.subTest(scale=factor):
                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(project_root)
                environment["QT_QPA_PLATFORM"] = "offscreen"
                environment["QT_ENABLE_HIGHDPI_SCALING"] = "1"
                environment["QT_SCALE_FACTOR"] = factor
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    cwd=project_root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
