import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QApplication

from flake_searcher.main_window import MainWindow


class AutoTabUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow()
        self.window.all_tabWidget.setCurrentWidget(self.window.autoscan_tab)
        self.app.processEvents()
        self.tab = self.window.autoscan_tab

    def tearDown(self):
        self.window.close()
        self.app.processEvents()

    def test_defaults_and_scan_geometry_inputs_are_preserved(self):
        self.assertAlmostEqual(self.tab.x_speed_bx.value(), 0.4)
        self.assertAlmostEqual(self.tab.y_speed_bx.value(), 0.4)
        self.assertAlmostEqual(self.tab.x_angle_bx.value(), 1000.0)
        self.assertAlmostEqual(self.tab.y_angle_bx.value(), 1000.0)
        self.assertEqual(self.tab.x_multible.value(), 100)
        self.assertEqual(self.tab.y_multible.value(), 100)
        self.assertTrue(self.tab.fast_x_rad.isChecked())
        self.assertFalse(self.tab.fast_y_rad.isChecked())
        self.assertTrue(self.tab.save_relevant_rad.isChecked())
        self.assertFalse(self.tab.save_all_rad.isChecked())
        self.assertFalse(self.tab.zigizag_chkbx.isChecked())

    def test_auto_settings_scroll_vertically_without_horizontal_overflow(self):
        scroll = self.tab.auto_scroll_area
        self.assertEqual(scroll.horizontalScrollBarPolicy(), Qt.ScrollBarAlwaysOff)
        self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
        self.assertGreater(scroll.verticalScrollBar().maximum(), 0)

    def test_safety_and_status_controls_remain_visible_at_small_capture_height(self):
        self.window.resize_capture_frame(QSize(320, 240))
        self.app.processEvents()

        self.assertEqual(self.window.image_frame.size(), QSize(320, 240))
        self.assertGreaterEqual(self.window.control_panel.height(), 300)
        for widget in (
            self.tab.coord_display,
            self.tab.scan_info,
            self.tab.scan_progress,
            self.tab.start_btn,
            self.tab.stop_btn,
        ):
            with self.subTest(widget=widget.objectName()):
                self.assertTrue(widget.isVisibleTo(self.tab))
        self.assertFalse(self.tab.stop_btn.isEnabled())

    def test_step_progress_and_scan_status_are_visible(self):
        info = {
            "done": 7,
            "total": 20,
            "slow_i": 1,
            "fast_j": 2,
            "x": 1000,
            "y": -500,
            "z": 0,
            "flake_found": True,
            "flake_size": 9,
        }
        self.tab.on_step_done(info)

        self.assertEqual(self.tab.scan_progress.maximum(), 20)
        self.assertEqual(self.tab.scan_progress.value(), 7)
        self.assertIn("Step 7/20", self.tab.scan_info.text())
        self.assertIn("flake=YES", self.tab.scan_info.text())
        self.assertEqual(self.tab.scan_info.property("statusState"), "busy")
        self.assertIn("Z: unavailable", self.tab.coord_display.text())

    def test_preconnection_scan_failure_is_shown_without_exception(self):
        self.tab.start_scan()
        self.assertIn("Connect stage first", self.tab.scan_info.text())
        self.assertEqual(self.tab.scan_info.property("statusState"), "unknown")

    def test_stage_failure_is_prominent_and_keeps_stop_available_until_finish(self):
        self.tab.stop_btn.setEnabled(True)
        self.tab.on_stage_failed("lost acknowledgement")

        self.assertIn("Scan aborted", self.tab.scan_info.text())
        self.assertEqual(self.tab.scan_info.property("statusState"), "error")
        self.assertIn("Stage error", self.window.manual_tab.MController_status.text())
        self.assertTrue(self.tab.stop_btn.isEnabled())


if __name__ == "__main__":
    unittest.main()
