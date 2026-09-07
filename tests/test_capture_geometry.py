import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PyQt5.QtCore import QPoint, QRect, QSize
from PyQt5.QtWidgets import QApplication, QFrame, QWidget

from flake_searcher.image_frame_manager import ImageFrameManager
from flake_searcher.main_window import MainWindow


CAPTURE_TRIM = 15


class CaptureGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.screenshot_regions = []

    def _fake_screenshot(self, *, region):
        self.screenshot_regions.append(region)
        return Image.new("RGB", (region[2], region[3]), color=(12, 34, 56))

    def _capture(self, manager):
        with patch(
            "flake_searcher.image_frame_manager.pyautogui.screenshot",
            side_effect=self._fake_screenshot,
        ):
            image = manager.get_screenshot()
        return self.screenshot_regions[-1], image

    def test_initial_global_capture_rectangle_and_panel_relationship(self):
        window = MainWindow()
        try:
            window.move(100, 80)
            self.app.processEvents()

            self.assertEqual(window.image_frame.size(), QSize(621, 611))
            self.assertEqual(window.image_frame.mapToGlobal(QPoint(0, 0)), QPoint(100, 80))

            region, image = self._capture(window.image_frame_manager)
            self.assertEqual(region, (115, 95, 591, 581))
            self.assertEqual(image.shape, (581, 591, 3))

            capture_rect = QRect(QPoint(region[0], region[1]), QSize(region[2], region[3]))
            panel_rect = QRect(
                window.all_tabWidget.mapToGlobal(QPoint(0, 0)),
                window.all_tabWidget.size(),
            )
            self.assertFalse(capture_rect.intersects(panel_rect))
            self.assertTrue(window.rect().contains(window.image_frame.geometry()))
            self.assertTrue(window.rect().contains(window.all_tabWidget.geometry()))
        finally:
            window.close()

    def test_capture_uses_current_geometry_for_common_aspect_ratios(self):
        host = QWidget()
        frame = QFrame(host)
        manager = ImageFrameManager(frame)
        host.move(70, 45)
        frame.move(23, 17)

        sizes = (
            QSize(320, 240),
            QSize(640, 480),
            QSize(800, 600),
            QSize(1280, 720),
            QSize(360, 720),
        )
        try:
            host.show()
            for size in sizes:
                with self.subTest(size=(size.width(), size.height())):
                    host.resize(size.width() + 80, size.height() + 80)
                    frame.resize(size)
                    self.app.processEvents()
                    top_left = frame.mapToGlobal(QPoint(0, 0))

                    region, image = self._capture(manager)

                    self.assertEqual(frame.size(), size)
                    self.assertEqual(region[0], top_left.x() + CAPTURE_TRIM)
                    self.assertEqual(region[1], top_left.y() + CAPTURE_TRIM)
                    self.assertEqual(region[2], size.width() - 2 * CAPTURE_TRIM)
                    self.assertEqual(region[3], size.height() - 2 * CAPTURE_TRIM)
                    self.assertEqual(
                        image.shape,
                        (size.height() - 2 * CAPTURE_TRIM, size.width() - 2 * CAPTURE_TRIM, 3),
                    )
        finally:
            host.close()

    def test_moving_window_changes_capture_origin_not_capture_size(self):
        window = MainWindow()
        try:
            window.move(30, 40)
            self.app.processEvents()
            first_region, _ = self._capture(window.image_frame_manager)

            window.move(240, 190)
            self.app.processEvents()
            second_region, _ = self._capture(window.image_frame_manager)

            self.assertEqual(first_region[2:], second_region[2:])
            self.assertEqual(second_region[0] - first_region[0], 210)
            self.assertEqual(second_region[1] - first_region[1], 150)
            self.assertEqual(window.image_frame.size(), QSize(621, 611))
        finally:
            window.close()

    def test_tab_and_long_status_changes_do_not_change_capture_rectangle(self):
        window = MainWindow()
        try:
            window.move(100, 80)
            self.app.processEvents()
            initial_region, _ = self._capture(window.image_frame_manager)

            window.manual_tab.MController_status.setText(
                "A deliberately long controller status that must never resize the capture frame"
            )
            window.autoscan_tab.scan_info.setText(
                "A deliberately long scan status that must never resize the capture frame"
            )
            window.ai_tab.training_status.setText(
                "A deliberately long training status that must never resize the capture frame"
            )
            window.A_Eye_Tab.info.setText(
                "A deliberately long detector status that must never resize the capture frame"
            )
            for index in range(window.all_tabWidget.count()):
                window.all_tabWidget.setCurrentIndex(index)
                self.app.processEvents()
                region, _ = self._capture(window.image_frame_manager)
                self.assertEqual(region, initial_region)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
