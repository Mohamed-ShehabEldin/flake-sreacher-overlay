import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from flake_searcher.a_eye_tab import A_Eye_Tab
from flake_searcher.training_tab import TrainingAiTab


class _FrameManager:
    def get_screenshot(self):
        raise RuntimeError("capture unavailable")


class AnalysisTabsUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_training_defaults_and_vertical_only_scrolling_are_preserved(self):
        tab = TrainingAiTab()
        tab.resize(304, 500)
        tab.show()
        self.app.processEvents()

        self.assertEqual(tab.max_display_width_spin.value(), 1024)
        self.assertEqual(tab.grid_sample_size_spin.value(), 128)
        self.assertEqual(tab.epochs_spin.value(), 100)
        self.assertEqual(tab.batch_size_spin.value(), 32)
        self.assertAlmostEqual(tab.test_size_spin.value(), 0.2)
        self.assertEqual(tab.patience_spin.value(), 10)
        self.assertEqual(
            tab.training_scroll_area.horizontalScrollBarPolicy(), Qt.ScrollBarAlwaysOff
        )
        self.assertEqual(tab.training_scroll_area.horizontalScrollBar().maximum(), 0)
        self.assertTrue(tab.training_status.wordWrap())
        self.assertTrue(tab.train_btn.isVisibleTo(tab))
        self.assertTrue(tab.save_model_btn.isVisibleTo(tab))
        tab.close()

    def test_training_completion_and_failure_are_visible_and_restore_actions(self):
        tab = TrainingAiTab()
        tab._set_training_busy(True)
        tab._on_worker_done("done")
        self.assertTrue(tab.train_btn.isEnabled())
        self.assertTrue(tab.save_model_btn.isEnabled())
        self.assertIn("complete", tab.training_status.text().lower())
        self.assertEqual(tab.training_status.property("statusState"), "connected")

        tab._set_training_busy(True)
        tab._on_worker_done("error: no samples")
        self.assertTrue(tab.train_btn.isEnabled())
        self.assertIn("no samples", tab.training_status.text())
        self.assertEqual(tab.training_status.property("statusState"), "error")

    def test_a_eye_defaults_and_vertical_only_scrolling_are_preserved(self):
        tab = A_Eye_Tab(_FrameManager())
        tab.resize(304, 500)
        tab.show()
        self.app.processEvents()

        self.assertEqual(tab.ratio_spin.value(), 5)
        self.assertEqual(tab.pred_batch_size_spin.value(), 23000)
        self.assertEqual(tab.radius_spin.value(), 2)
        self.assertEqual(
            tab.a_eye_settings_scroll_area.horizontalScrollBarPolicy(),
            Qt.ScrollBarAlwaysOff,
        )
        self.assertEqual(tab.a_eye_settings_scroll_area.horizontalScrollBar().maximum(), 0)
        self.assertTrue(tab.info.wordWrap())
        self.assertTrue(tab.info.isVisibleTo(tab))
        self.assertTrue(tab.graphicsView.isVisibleTo(tab))
        tab.close()

    def test_a_eye_errors_are_visible_and_restore_actions(self):
        tab = A_Eye_Tab(_FrameManager())
        tab._run_inference("unused")
        self.assertIn("model", tab.info.text().lower())
        self.assertEqual(tab.info.property("statusState"), "warning")

        tab._set_busy(True)
        tab._on_inference_error("bad image")
        self.assertTrue(tab.check_an_img_btn.isEnabled())
        self.assertIn("bad image", tab.info.text())
        self.assertEqual(tab.info.property("statusState"), "error")

        tab._set_busy(True)
        tab._on_folder_done("Folder done. checked: 3 saved: 1")
        self.assertTrue(tab.check_fldr_btn.isEnabled())
        self.assertEqual(tab.info.property("statusState"), "connected")

    def test_capture_failure_is_reported_in_the_a_eye_panel(self):
        tab = A_Eye_Tab(_FrameManager())
        tab.check_current_window()
        self.assertIn("capture failed", tab.info.text().lower())
        self.assertEqual(tab.info.property("statusState"), "error")


if __name__ == "__main__":
    unittest.main()
